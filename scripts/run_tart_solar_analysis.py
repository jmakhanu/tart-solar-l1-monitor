import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from casacore.tables import table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import get_sun, EarthLocation, AltAz
import astropy.units as u

# 1. Automatically detect the MS and extract exact telescope EarthLocation
ms_list = sorted(glob.glob("msdir/*.ms"))
if not ms_list:
    raise FileNotFoundError("No Measurement Set found in msdir/!")
ms_path = ms_list[0]
node_name = os.path.basename(ms_path).replace(".ms", "")

ant_tab = table(f"{ms_path}/ANTENNA", readonly=True)
pos_itrf = np.mean(ant_tab.getcol("POSITION"), axis=0)
ant_tab.close()

site_loc = EarthLocation.from_geocentric(pos_itrf[0]*u.m, pos_itrf[1]*u.m, pos_itrf[2]*u.m)
print(f"Loaded Telescope: {node_name} | Lat: {site_loc.lat.deg:.3f} deg, Lon: {site_loc.lon.deg:.3f} deg")

# 2. Process all WSClean FITS snapshots
fits_files = sorted(glob.glob(f"img/{node_name}-t*-image.fits"))
if not fits_files:
    raise FileNotFoundError(f"No FITS files found in img/{node_name}-t*-image.fits!")

records = []
t0 = None

for idx, fpath in enumerate(fits_files):
    with fits.open(fpath) as hdul:
        hdr = hdul[0].header
        img = np.squeeze(hdul[0].data)
        ny, nx = img.shape

        t_obs = Time(hdr["DATE-OBS"])
        if t0 is None:
            t0 = t_obs
        dt_sec = (t_obs - t0).sec

        sun_icrs = get_sun(t_obs)
        sun_altaz = sun_icrs.transform_to(AltAz(obstime=t_obs, location=site_loc))
        alt_deg = float(sun_altaz.alt.deg)
        za_deg = 90.0 - alt_deg

        wcs = WCS(hdr).celestial
        px, py = wcs.world_to_pixel(sun_icrs)
        ix, iy = int(np.round(px)), int(np.round(py))

        finite_vals = img[np.isfinite(img)]
        med_bg = np.median(finite_vals)
        sigma_mad = 1.4826 * np.median(np.abs(finite_vals - med_bg))

        r_box = 2
        if r_box <= ix < (nx - r_box) and r_box <= iy < (ny - r_box):
            cutout = img[iy - r_box : iy + r_box + 1, ix - r_box : ix + r_box + 1]
            raw_peak = float(np.nanmax(cutout) - med_bg)
        else:
            raw_peak = np.nan

        pb_gain = np.cos(np.radians(za_deg))**1.5 if alt_deg > 10.0 else np.nan
        pb_flux = raw_peak / pb_gain if np.isfinite(pb_gain) else np.nan
        pb_err = sigma_mad / pb_gain if np.isfinite(pb_gain) else np.nan
        snr = raw_peak / sigma_mad if sigma_mad > 0 else np.nan

        records.append({
            "snapshot_id": idx, "utc_time": t_obs.isot, "elapsed_sec": dt_sec,
            "sun_ra_deg": float(sun_icrs.ra.deg), "sun_dec_deg": float(sun_icrs.dec.deg),
            "sun_alt_deg": alt_deg, "pixel_x": float(px), "pixel_y": float(py),
            "raw_flux_jy_beam": raw_peak, "rms_noise_jy_beam": sigma_mad,
            "pb_corrected_flux": pb_flux, "pb_corrected_err": pb_err, "snr": snr
        })

df = pd.DataFrame(records)
csv_file = f"{node_name}_solar_l1_timeseries.csv"
df.to_csv(csv_file, index=False)
print(f"[SUCCESS] Saved {csv_file} | Mean Solar Alt: {df['sun_alt_deg'].mean():.2f} deg | Mean SNR: {df['snr'].mean():.2f}")

# 3. Plot 3-Panel WSClean Solar Photometry Dashboard
fig = plt.figure(figsize=(16, 5.2))
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(df["elapsed_sec"], df["raw_flux_jy_beam"], "o-", color="#1f77b4", ms=3.5, label="Raw Peak Flux")
if df["pb_corrected_flux"].notna().any():
    ax1.plot(df["elapsed_sec"], df["pb_corrected_flux"], "s-", color="#d62728", ms=3.5, label="Beam-Corrected Flux")
    ax1.fill_between(df["elapsed_sec"], df["pb_corrected_flux"]-df["pb_corrected_err"],
                     df["pb_corrected_flux"]+df["pb_corrected_err"], color="#d62728", alpha=0.2, label=r"$\pm 1\sigma_{\mathrm{MAD}}$")
ax1.set_xlabel(f"Time Elapsed (s since {df['utc_time'].iloc[0][:19]} UTC)")
ax1.set_ylabel("Flux Density (Jy / beam)")
ax1.set_title(f"{node_name} 1.575 GHz Solar Light Curve")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(df["elapsed_sec"], df["snr"], ".-", color="#2ca02c", lw=1.5, label="Solar SNR")
ax2.axhline(5.0, color="red", ls="--", label=r"$5\sigma$ Threshold")
ax2.set_xlabel("Time Elapsed (s)")
ax2.set_ylabel("SNR")
ax2.set_title("Snapshot SNR Stability")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)

mid = len(fits_files) // 2
with fits.open(fits_files[mid]) as hdul_mid:
    wcs_mid = WCS(hdul_mid[0].header).celestial
    data_mid = np.squeeze(hdul_mid[0].data)
    t_mid = Time(hdul_mid[0].header["DATE-OBS"])
    sun_mid = get_sun(t_mid)

ax3 = fig.add_subplot(1, 3, 3, projection=wcs_mid)
im = ax3.imshow(data_mid, origin="lower", cmap="inferno")
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="Jy / beam")
ax3.scatter(sun_mid.ra.deg, sun_mid.dec.deg, s=280, fc="none", ec="yellow", lw=2.2,
            transform=ax3.get_transform("world"), label="Sun (Astropy WCS)")
ax3.set_title(f"{node_name} t={mid:04d} ({t_mid.isot[11:19]} UTC)")
ax3.legend(loc="upper right", fontsize=8)

plt.tight_layout()
png_file = f"{node_name}_solar_l1_dashboard.png"
plt.savefig(png_file, dpi=200)
print(f"[SUCCESS] Saved {png_file}")
