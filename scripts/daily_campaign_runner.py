#!/usr/bin/env python3
"""
Standard Multi-Week L1 Solar Flux Campaign Runner for TART (ke-kitui / za-rhodes)
Authors: Idriss James Makhanu & Winnie
"""
import os
import sys
import glob
import shutil
import subprocess
import urllib.request
import json
from datetime import datetime, timezone

import h5py
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

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = "/workspaces/tart-solar-l1-monitor"
CASA_DIR = os.path.join(BASE_DIR, "casa")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DAILY_PLOTS_DIR = os.path.join(RESULTS_DIR, "daily_dashboards")
MASTER_CSV = os.path.join(RESULTS_DIR, "master_solar_campaign.csv")

PRIORITY_NODES = ["ke-kitui", "ke-tuk", "za-rhodes", "mu-udm"]
MIN_SUN_ALT_DEG = 15.0  # Only record when Sun is >= 15 deg above horizon

os.makedirs(DAILY_PLOTS_DIR, exist_ok=True)


def run_cmd(cmd, cwd=CASA_DIR):
    print(f"\n[EXEC] {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd}")


def select_online_daytime_node():
    """Find the first online TART node where the Sun is currently > 15 deg."""
    now_t = Time(datetime.now(timezone.utc))
    for node in PRIORITY_NODES:
        url = f"https://api.elec.ac.nz/tart/{node}/api/v1/info"
        try:
            req = urllib.request.urlopen(url, timeout=5)
            info = json.loads(req.read().decode()).get("info", {})
            loc = info.get("location", {})
            site = EarthLocation(lat=loc["lat"]*u.deg, lon=loc["lon"]*u.deg, height=loc["alt"]*u.m)
            sun_alt = float(get_sun(now_t).transform_to(AltAz(obstime=now_t, location=site)).alt.deg)
            print(f"[CHECK] {node:<10} | ONLINE | Current Sun Elevation: {sun_alt:.2f} deg")
            if sun_alt >= MIN_SUN_ALT_DEG:
                return node
        except Exception:
            print(f"[CHECK] {node:<10} | OFFLINE")
    raise RuntimeError("No priority TART nodes are both ONLINE and in daytime (>15 deg solar elevation).")


def clean_workspace():
    """Purge heavy temporary HDF5, MS, and FITS files to protect disk space."""
    for folder in ["rawdata", "msdir", "img", "img_unpeeled"]:
        target = os.path.join(CASA_DIR, folder)
        if os.path.exists(target):
            shutil.rmtree(target)
        os.makedirs(target, exist_ok=True)


def peel_gnss_satellites(ms_path):
    """Subtract MODEL_DATA (GNSS satellites) from CORRECTED_DATA in-place."""
    print(f"[PEEL] Subtracting GNSS satellite MODEL_DATA from {ms_path}...")
    t = table(ms_path, readonly=False)
    corr = t.getcol("CORRECTED_DATA")
    model = t.getcol("MODEL_DATA")
    t.putcol("CORRECTED_DATA", corr - model)
    t.close()


def extract_photometry_and_append(node):
    """Extract WCS solar photometry across all 70 snapshots and append to Master CSV."""
    ms_path = os.path.join(CASA_DIR, "msdir", f"{node}.ms")
    ant_tab = table(f"{ms_path}/ANTENNA", readonly=True)
    pos_itrf = np.mean(ant_tab.getcol("POSITION"), axis=0)
    ant_tab.close()
    site_loc = EarthLocation.from_geocentric(pos_itrf[0]*u.m, pos_itrf[1]*u.m, pos_itrf[2]*u.m)

    fits_files = sorted(glob.glob(os.path.join(CASA_DIR, "img", f"{node}-t*-image.fits")))
    if not fits_files:
        raise FileNotFoundError("No WSClean FITS images found!")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
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
            az_deg = float(sun_altaz.az.deg)
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
                "run_id": run_id,
                "date_utc": t_obs.isot[:10],
                "utc_time": t_obs.isot,
                "node": node,
                "snapshot_id": idx,
                "elapsed_sec": dt_sec,
                "sun_ra_deg": float(sun_icrs.ra.deg),
                "sun_dec_deg": float(sun_icrs.dec.deg),
                "sun_alt_deg": alt_deg,
                "sun_az_deg": az_deg,
                "pixel_x": float(px),
                "pixel_y": float(py),
                "raw_flux_jy_beam": raw_peak,
                "rms_noise_jy_beam": sigma_mad,
                "pb_corrected_flux": pb_flux,
                "pb_corrected_err": pb_err,
                "snr": snr
            })

    df_run = pd.DataFrame(records)

    # Append to Master 3-Week Campaign CSV (deduplicating by utc_time + node)
    if os.path.exists(MASTER_CSV):
        df_master = pd.read_csv(MASTER_CSV)
        df_combined = pd.concat([df_master, df_run], ignore_index=True)
        df_combined.drop_duplicates(subset=["utc_time", "node"], keep="last", inplace=True)
    else:
        df_combined = df_run

    df_combined.sort_values("utc_time", inplace=True)
    df_combined.to_csv(MASTER_CSV, index=False)
    print(f"[SUCCESS] Appended {len(df_run)} rows. Master CSV now has {len(df_combined)} total rows.")

    # Save daily diagnostic dashboard
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(df_run["elapsed_sec"], df_run["pb_corrected_flux"], "o-", color="#d62728", ms=3.5, label="Beam-Corrected Flux")
    ax1.set_title(f"{node} Solar Flux ({run_id} UTC)")
    ax1.set_xlabel("Elapsed Time (s)")
    ax1.set_ylabel("Flux Density (Jy/beam)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(df_run["elapsed_sec"], df_run["snr"], ".-", color="#2ca02c", label="Solar SNR")
    ax2.axhline(5.0, color="red", ls="--", label="5-sigma")
    ax2.set_title(f"Solar Elevation: {df_run['sun_alt_deg'].mean():.1f} deg | Mean SNR: {df_run['snr'].mean():.2f}")
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_ylabel("SNR")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(DAILY_PLOTS_DIR, f"{node}_{run_id}_dashboard.png")
    plt.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"[SUCCESS] Saved daily dashboard: {plot_path}")


def update_multiweek_summary_plot():
    """Generate a multi-day campaign overview plot from master_solar_campaign.csv."""
    if not os.path.exists(MASTER_CSV):
        return
    df = pd.read_csv(MASTER_CSV)
    df["dt"] = pd.to_datetime(df["utc_time"])

    # Aggregate mean flux & peak flux per observation run
    daily = df.groupby(["run_id", "node", "date_utc"]).agg(
        mean_time=("dt", "mean"),
        mean_flux=("pb_corrected_flux", "mean"),
        max_flux=("pb_corrected_flux", "max"),
        std_flux=("pb_corrected_flux", "std"),
        mean_snr=("snr", "mean"),
        mean_alt=("sun_alt_deg", "mean")
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for node, grp in daily.groupby("node"):
        ax1.errorbar(grp["mean_time"], grp["mean_flux"], yerr=grp["std_flux"],
                     fmt="o-", capsize=3, label=f"{node} (Mean +/- 1 SD)")
        ax1.scatter(grp["mean_time"], grp["max_flux"], marker="^", s=45, alpha=0.7, label=f"{node} Peak Burst Flux")
        ax2.plot(grp["mean_time"], grp["mean_snr"], "s--", label=f"{node} Mean SNR")

    ax1.set_ylabel("Beam-Corrected Flux (Jy / beam)")
    ax1.set_title("Multi-Week TART 1.575 GHz Solar Radio Flux Monitoring Campaign")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.axhline(5.0, color="red", ls=":", label="5-sigma Threshold")
    ax2.set_ylabel("Detection SNR")
    ax2.set_xlabel("Observation Date (UTC)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    summary_png = os.path.join(RESULTS_DIR, "multiweek_campaign_trend.png")
    plt.savefig(summary_png, dpi=200)
    plt.close(fig)
    print(f"[SUCCESS] Updated multi-week campaign trend plot: {summary_png}")


def main():
    node = sys.argv[1] if len(sys.argv) > 1 else select_online_daytime_node()
    print(f"=== STARTING SOLAR CAMPAIGN RUN FOR NODE: {node} ===")

    clean_workspace()

    # 1. Run Stimela Download, MS creation, and RIME Gain Calibration
    stimela_base = f"stimela run tart_cal_casa.yml tart={node}"
    run_cmd(f"{stimela_base} -s download-hdf")
    run_cmd(f"{stimela_base} -s create-ms -N")
    run_cmd(f"{stimela_base} -s updateobservatory,flagsave,calibrate_amplitude,calibrate_phase,applycal")

    # 2. Peel GNSS Satellites in Visibility Space
    ms_path = os.path.join(CASA_DIR, "msdir", f"{node}.ms")
    peel_gnss_satellites(ms_path)

    # 3. Run WSClean 70-Snapshot Synthesis Imaging on Peeled Visibilities
    run_cmd(f"{stimela_base} -s snapshotimage")

    # 4. Extract Solar Photometry & Update Multi-Week Master CSV
    extract_photometry_and_append(node)
    update_multiweek_summary_plot()

    # 5. Purge heavy temporary files (.hdf, .ms, .fits) to save disk space
    clean_workspace()

    # 6. Auto-commit and push updated CSV & plots to GitHub for Winnie
    run_cmd("git add results/ scripts/", cwd=BASE_DIR)
    commit_msg = f"Campaign data update: {node} ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})"
    run_cmd(f'git commit -m "{commit_msg}" || true', cwd=BASE_DIR)
    run_cmd("git push origin main", cwd=BASE_DIR)
    print("=== CAMPAIGN RUN COMPLETE & PUSHED TO GITHUB ===")


if __name__ == "__main__":
    main()
