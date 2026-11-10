#!/usr/bin/env python3
"""
Fault-Tolerant Sequential Multi-Week TART L1 Solar Radio Flux & Space Weather Pipeline
Verifies live /api/v1/vis/data stream before running Stimela and falls back automatically.
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

BASE_DIR = "/workspaces/tart-solar-l1-monitor"
CASA_DIR = os.path.join(BASE_DIR, "casa")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DAILY_DIR = os.path.join(RESULTS_DIR, "campaign_runs")
MASTER_CSV = os.path.join(RESULTS_DIR, "master_solar_campaign.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "paper_summary_table_for_winnie.csv")

PRIORITY_NODES = ["ke-kitui", "za-rhodes", "mu-udm", "ke-tuk", "nz-25"]

os.makedirs(DAILY_DIR, exist_ok=True)


def clean_iso_string(ts_val):
    s = ts_val.decode("utf-8") if isinstance(ts_val, bytes) else str(ts_val)
    return s.replace("+00:00", "").replace("Z", "").strip()


def parse_astropy_time(ts_input):
    dt = pd.to_datetime(ts_input, utc=True, format="ISO8601")
    return Time(dt)


def run_cmd(cmd, cwd=CASA_DIR):
    print(f"\n[EXEC] {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed (exit {res.returncode}): {cmd}")


def backfill_day1_if_needed():
    day1_csv = os.path.join(RESULTS_DIR, "ke-kitui_solar_l1_timeseries.csv")
    if os.path.exists(day1_csv) and not os.path.exists(MASTER_CSV):
        print("[INIT] Importing Day 1 (2026-11-06) ke-kitui dataset into Master Campaign archive...")
        df1 = pd.read_csv(day1_csv)
        t_first = clean_iso_string(df1["utc_time"].iloc[0])
        run_id_1 = t_first[:10].replace("-", "") + "_" + t_first[11:16].replace(":", "")
        df1["run_id"] = run_id_1
        df1["date_utc"] = df1["utc_time"].astype(str).str[:10]
        df1["node"] = "ke-kitui"
        if "sun_az_deg" not in df1.columns:
            df1["sun_az_deg"] = np.nan
        df1["unpeeled_rms_noise"] = df1["rms_noise_jy_beam"]
        df1["unpeeled_snr"] = df1["snr"]
        df1["noise_reduction_pct"] = 0.0
        df1["goes_xray_wm2"] = np.nan
        df1["goes_flare_class"] = "Baseline (Day 1)"
        df1.to_csv(MASTER_CSV, index=False)
        day1_png = os.path.join(RESULTS_DIR, "ke-kitui_solar_l1_dashboard.png")
        if os.path.exists(day1_png):
            shutil.copy(day1_png, os.path.join(DAILY_DIR, f"ke-kitui_{run_id_1}_day1_dashboard.png"))
        print(f"[SUCCESS] Seeded Master CSV with {len(df1)} rows from {df1['date_utc'].iloc[0]} (Run {run_id_1}).")


def check_node_vis_stream(node):
    """Test both /info AND /vis/data so 502 Bad Gateway nodes are caught immediately."""
    base_url = f"https://api.elec.ac.nz/tart/{node}/api/v1"
    try:
        info_raw = json.loads(urllib.request.urlopen(f"{base_url}/info", timeout=5).read().decode())
        info = info_raw.get("info", info_raw)
        loc = info.get("location", {})
        # Crucial check: verify /vis/data actually responds without 502 error
        urllib.request.urlopen(f"{base_url}/vis/data", timeout=6).read(256)
        now_t = Time(datetime.now(timezone.utc))
        site = EarthLocation(lat=loc["lat"]*u.deg, lon=loc["lon"]*u.deg, height=loc.get("alt", 500.0)*u.m)
        sun_alt = float(get_sun(now_t).transform_to(AltAz(obstime=now_t, location=site)).alt.deg)
        print(f"[STREAM CHECK] {node:<10} | STREAMING OK | Sun Elevation: {sun_alt:.2f} deg")
        return True, sun_alt
    except Exception as e:
        print(f"[STREAM CHECK] {node:<10} | UNREACHABLE ({type(e).__name__}: 502/Offline)")
        return False, -90.0


def select_working_node(requested_node=None):
    candidates = [requested_node] + [n for n in PRIORITY_NODES if n != requested_node] if requested_node else PRIORITY_NODES
    for node in candidates:
        ok, sun_alt = check_node_vis_stream(node)
        if ok:
            return node
    raise RuntimeError("No TART nodes are currently streaming /vis/data.")


def clean_workspace():
    for folder in ["rawdata", "msdir", "img", "img_unpeeled"]:
        target = os.path.join(CASA_DIR, folder)
        if os.path.exists(target):
            shutil.rmtree(target)
        os.makedirs(target, exist_ok=True)


def generate_ksa_baseline_heatmap(node, site_loc, run_id):
    hdf_files = sorted(glob.glob(os.path.join(CASA_DIR, "rawdata", "*.hdf")))
    if not hdf_files:
        return None
    all_vis, all_ts = [], []
    for fpath in hdf_files:
        with h5py.File(fpath, "r") as f:
            all_vis.append(np.abs(f["vis"][:]))
            all_ts.extend([clean_iso_string(t) for t in f["timestamp"][:]])

    vis_matrix = np.vstack(all_vis)
    t_objs = parse_astropy_time(all_ts)
    elapsed_min = (t_objs - t_objs[0]).sec / 60.0
    sun_alt = get_sun(t_objs).transform_to(AltAz(obstime=t_objs, location=site_loc)).alt.deg
    total_power = np.mean(vis_matrix, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax1.plot(elapsed_min, total_power, color="#d62728", lw=1.4, label="Mean Baseline Visibility Amplitude |V_pq|")
    ax1.set_ylabel("Raw Visibility Power (arb.)", color="#d62728")
    ax1.grid(True, alpha=0.3)
    ax1_r = ax1.twinx()
    ax1_r.plot(elapsed_min, sun_alt, color="#1f77b4", ls="--", lw=1.4, label="Sun Elevation (deg)")
    ax1_r.set_ylabel("Sun Elevation (deg)", color="#1f77b4")
    ax1.set_title(f"{node} — Non-Imaging Baseline Power (Mutula & Dusabe 2025 Method) [{run_id} UTC]")

    im = ax2.imshow(vis_matrix.T, aspect="auto", origin="lower", cmap="viridis",
                    extent=[elapsed_min[0], elapsed_min[-1], 0, vis_matrix.shape[1]])
    ax2.set_xlabel(f"Elapsed Time (minutes since {t_objs[0].isot[:19]} UTC)")
    ax2.set_ylabel("Baseline Pair Index (0 - 275)")
    ax2.set_title("276-Baseline Power Distribution Heatmap")
    plt.colorbar(im, ax=ax2, label="|V_pq|")

    plt.tight_layout()
    out_png = os.path.join(DAILY_DIR, f"{node}_{run_id}_1_ksa_heatmap.png")
    plt.savefig(out_png, dpi=180)
    shutil.copy(out_png, os.path.join(RESULTS_DIR, "latest_ksa_baseline_heatmap.png"))
    plt.close(fig)
    print(f"[SUCCESS] Saved sequential KSA Heatmap: {out_png}")


def measure_fits_folder(folder_path, node, site_loc):
    fits_files = sorted(glob.glob(os.path.join(folder_path, f"{node}-t*-image.fits")))
    rows = []
    t0 = None
    for idx, fpath in enumerate(fits_files):
        with fits.open(fpath) as hdul:
            hdr = hdul[0].header
            img = np.squeeze(hdul[0].data)
            ny, nx = img.shape
            t_obs = parse_astropy_time(hdr["DATE-OBS"])
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

            rows.append({
                "snapshot_id": idx, "utc_time": t_obs.isot, "elapsed_sec": dt_sec,
                "sun_ra_deg": float(sun_icrs.ra.deg), "sun_dec_deg": float(sun_icrs.dec.deg),
                "sun_alt_deg": alt_deg, "sun_az_deg": az_deg,
                "pixel_x": float(px), "pixel_y": float(py),
                "raw_flux_jy_beam": raw_peak, "rms_noise_jy_beam": sigma_mad,
                "pb_corrected_flux": pb_flux, "pb_corrected_err": pb_err, "snr": snr
            })
    return pd.DataFrame(rows)


def fetch_goes_xray_flux():
    url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
    try:
        req = urllib.request.urlopen(url, timeout=8)
        raw_txt = req.read().decode("utf-8", errors="ignore").strip()
        data, _ = json.JSONDecoder().raw_decode(raw_txt)
        df_g = pd.DataFrame(data)
        df_g = df_g[df_g["energy"] == "0.1-0.8nm"].copy()
        df_g["dt"] = pd.to_datetime(df_g["time_tag"], utc=True)
        return df_g
    except Exception as e:
        print(f"[WARN] Could not fetch live NOAA GOES X-ray feed: {e}")
        return pd.DataFrame()


def classify_goes_flare(flux_wm2):
    if not np.isfinite(flux_wm2) or flux_wm2 <= 0:
        return "Quiet"
    if flux_wm2 >= 1e-4:
        return f"X{flux_wm2/1e-4:.1f}"
    if flux_wm2 >= 1e-5:
        return f"M{flux_wm2/1e-5:.1f}"
    if flux_wm2 >= 1e-6:
        return f"C{flux_wm2/1e-6:.1f}"
    if flux_wm2 >= 1e-7:
        return f"B{flux_wm2/1e-7:.1f}"
    return f"A{flux_wm2/1e-8:.1f}"


def update_multiweek_trend_plot(df_combined):
    df_combined["dt"] = pd.to_datetime(df_combined["utc_time"], format="ISO8601", utc=True)
    daily = df_combined.groupby(["run_id", "date_utc", "node"]).agg(
        mean_time=("dt", "mean"),
        mean_flux=("pb_corrected_flux", "mean"),
        max_flux=("pb_corrected_flux", "max"),
        std_flux=("pb_corrected_flux", "std"),
        mean_snr=("snr", "mean")
    ).reset_index().sort_values("mean_time")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for node_lbl, grp in daily.groupby("node"):
        ax1.errorbar(grp["mean_time"], grp["mean_flux"], yerr=grp["std_flux"],
                     fmt="o-", capsize=3, lw=1.5, label=f"{node_lbl} Mean Flux (+/- 1 SD)")
        ax1.scatter(grp["mean_time"], grp["max_flux"], marker="^", s=50, alpha=0.8, label=f"{node_lbl} Peak Snapshot Flux")
        ax2.plot(grp["mean_time"], grp["mean_snr"], "s--", lw=1.5, label=f"{node_lbl} Mean SNR")

    ax1.set_ylabel("Beam-Corrected Flux (Jy/beam)")
    ax1.set_title("Sequential Multi-Week TART 1.575 GHz Solar Monitoring Campaign (From Nov 6, 2026)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.axhline(5.0, color="red", ls=":", label="5-sigma Detection Threshold")
    ax2.set_ylabel("Mean Solar SNR")
    ax2.set_xlabel("Observation Date & Time (UTC)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    trend_png = os.path.join(RESULTS_DIR, "multiweek_campaign_trend.png")
    plt.savefig(trend_png, dpi=200)
    plt.close(fig)
    print(f"[SUCCESS] Updated multi-week sequential trend plot: {trend_png}")


def main():
    backfill_day1_if_needed()

    req_arg = sys.argv[1] if len(sys.argv) > 1 else None
    clean_workspace()

    # Check if user wants to process cached workshop ke-kitui HDF5 files
    workshop_hdfs = sorted(glob.glob("/workspaces/radio-imaging/casa/rawdata/*.hdf"))
    if req_arg == "--local-kitui" and workshop_hdfs:
        node = "ke-kitui"
        print(f"[LOCAL CACHE] Copying {len(workshop_hdfs)} saved ke-kitui HDF5 files from workshop folder...")
        for f in workshop_hdfs:
            shutil.copy(f, os.path.join(CASA_DIR, "rawdata"))
        stimela_base = f"stimela run tart_cal_casa.yml tart={node}"
    else:
        node = select_working_node(req_arg)
        stimela_base = f"stimela run tart_cal_casa.yml tart={node}"
        run_cmd(f"{stimela_base} -s download-hdf")

    # 1. MS Creation & Two-Stage RIME Gain Calibration
    run_cmd(f"{stimela_base} -s create-ms -N")
    run_cmd(f"{stimela_base} -s updateobservatory,flagsave,calibrate_amplitude,calibrate_phase,applycal")

    ms_path = os.path.join(CASA_DIR, "msdir", f"{node}.ms")
    ant_tab = table(f"{ms_path}/ANTENNA", readonly=True)
    pos_itrf = np.mean(ant_tab.getcol("POSITION"), axis=0)
    ant_tab.close()
    site_loc = EarthLocation.from_geocentric(pos_itrf[0]*u.m, pos_itrf[1]*u.m, pos_itrf[2]*u.m)

    hdf0 = sorted(glob.glob(os.path.join(CASA_DIR, "rawdata", "*.hdf")))[0]
    with h5py.File(hdf0, "r") as f:
        ts0 = clean_iso_string(f["timestamp"][0])
    run_id = ts0[:10].replace("-", "") + "_" + ts0[11:16].replace(":", "")
    print(f"=== PROCESSING OBSERVATION RUN_ID: {run_id} ({node}) ===")

    # 2. Generate KSA Baseline-Power Heatmap
    generate_ksa_baseline_heatmap(node, site_loc, run_id)

    # 3. Image UNPEELED Visibilities (With GNSS Satellites)
    run_cmd(f"{stimela_base} -s snapshotimage")
    shutil.copytree(os.path.join(CASA_DIR, "img"), os.path.join(CASA_DIR, "img_unpeeled"), dirs_exist_ok=True)
    df_unpeeled = measure_fits_folder(os.path.join(CASA_DIR, "img_unpeeled"), node, site_loc)

    # 4. Peel GNSS Satellites (uvsub) & Re-Image
    print("[PEEL] Projecting & subtracting GNSS MODEL_DATA from CORRECTED_DATA...")
    t = table(ms_path, readonly=False)
    corr = t.getcol("CORRECTED_DATA")
    mod = t.getcol("MODEL_DATA")
    # Least-squares complex visibility projection per polarization to match flux scales
    peeled = corr.copy()
    for pol in range(corr.shape[-1]):
        c_p = corr[:, :, pol]
        m_p = mod[:, :, pol]
        denom = np.sum(np.abs(m_p)**2)
        if denom > 0:
            alpha = np.sum(c_p * np.conj(m_p)) / denom
            peeled[:, :, pol] = c_p - alpha * m_p
    t.putcol("CORRECTED_DATA", peeled)
    t.close()

    for f in glob.glob(os.path.join(CASA_DIR, "img", "*.fits")):
        os.remove(f)
    run_cmd(f"{stimela_base} -s snapshotimage")
    df_peeled = measure_fits_folder(os.path.join(CASA_DIR, "img"), node, site_loc)

    # 5. Merge Unpeeled vs. Peeled Metrics & Match NOAA GOES X-Ray Flux
    df_peeled["run_id"] = run_id
    df_peeled["date_utc"] = df_peeled["utc_time"].str[:10]
    df_peeled["node"] = node
    df_peeled["unpeeled_rms_noise"] = df_unpeeled["rms_noise_jy_beam"]
    df_peeled["unpeeled_snr"] = df_unpeeled["snr"]
    df_peeled["noise_reduction_pct"] = 100.0 * (df_unpeeled["rms_noise_jy_beam"] - df_peeled["rms_noise_jy_beam"]) / df_unpeeled["rms_noise_jy_beam"]

    df_goes = fetch_goes_xray_flux()
    if not df_goes.empty:
        t_tart = pd.to_datetime(df_peeled["utc_time"], format="ISO8601", utc=True)
        goes_interp = np.interp(
            t_tart.astype("int64"),
            df_goes["dt"].astype("int64"),
            df_goes["flux"]
        )
        df_peeled["goes_xray_wm2"] = goes_interp
        df_peeled["goes_flare_class"] = [classify_goes_flare(val) for val in goes_interp]
    else:
        df_peeled["goes_xray_wm2"] = np.nan
        df_peeled["goes_flare_class"] = "N/A"

    run_csv_path = os.path.join(DAILY_DIR, f"{node}_{run_id}_timeseries.csv")
    df_peeled.to_csv(run_csv_path, index=False)

    # 6. Plot Sequential Run Dashboard
    fig = plt.figure(figsize=(16, 5.2))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(df_peeled["elapsed_sec"], df_unpeeled["rms_noise_jy_beam"], "--", color="#7f7f7f", lw=1.4, label="Unpeeled RMS Noise")
    ax1.plot(df_peeled["elapsed_sec"], df_peeled["rms_noise_jy_beam"], "-", color="#1f77b4", lw=1.8, label="Peeled RMS Noise (uvsub)")
    ax1.set_xlabel("Elapsed Time (s)")
    ax1.set_ylabel(r"Background Noise $\sigma_{\mathrm{MAD}}$ (Jy/beam)")
    mean_red = df_peeled["noise_reduction_pct"].mean()
    ax1.set_title(f"GNSS Peeling Ablation (Noise Change: {mean_red:+.1f}%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(df_peeled["elapsed_sec"], df_peeled["pb_corrected_flux"], "o-", color="#d62728", ms=3.5, lw=1.5, label="TART 1.575 GHz Flux")
    ax2.fill_between(df_peeled["elapsed_sec"],
                     df_peeled["pb_corrected_flux"] - df_peeled["pb_corrected_err"],
                     df_peeled["pb_corrected_flux"] + df_peeled["pb_corrected_err"],
                     color="#d62728", alpha=0.2)
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_ylabel("TART L1 Flux Density (Jy/beam)", color="#d62728")
    ax2.grid(True, alpha=0.3)

    if df_peeled["goes_xray_wm2"].notna().any():
        ax2_r = ax2.twinx()
        ax2_r.plot(df_peeled["elapsed_sec"], df_peeled["goes_xray_wm2"], "s--", color="#ff7f0e", ms=3, lw=1.2, label="NOAA GOES 1-8A")
        ax2_r.set_yscale("log")
        ax2_r.set_ylabel(r"GOES X-Ray Flux ($\mathrm{W\,m^{-2}}$)", color="#ff7f0e")
        flare_lbl = df_peeled["goes_flare_class"].iloc[len(df_peeled)//2]
        ax2.set_title(f"{run_id} UTC | TART L1 vs. GOES ({flare_lbl})")
    else:
        ax2.set_title(f"{run_id} UTC | TART 1.575 GHz Light Curve")

    fits_peeled = sorted(glob.glob(os.path.join(CASA_DIR, "img", f"{node}-t*-image.fits")))
    mid = len(fits_peeled) // 2
    with fits.open(fits_peeled[mid]) as hdul_mid:
        wcs_mid = WCS(hdul_mid[0].header).celestial
        data_mid = np.squeeze(hdul_mid[0].data)
        t_mid = parse_astropy_time(hdul_mid[0].header["DATE-OBS"])
        sun_mid = get_sun(t_mid)

    ax3 = fig.add_subplot(1, 3, 3, projection=wcs_mid)
    im = ax3.imshow(data_mid, origin="lower", cmap="inferno")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="Jy/beam")
    ax3.scatter(sun_mid.ra.deg, sun_mid.dec.deg, s=280, fc="none", ec="yellow", lw=2.2,
                transform=ax3.get_transform("world"), label="Sun (Astropy WCS)")
    ax3.set_title(f"{node} Peeled Map ({t_mid.isot[11:19]} UTC)")
    ax3.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    dash_png = os.path.join(DAILY_DIR, f"{node}_{run_id}_2_ablation_goes_dashboard.png")
    plt.savefig(dash_png, dpi=200)
    shutil.copy(dash_png, os.path.join(RESULTS_DIR, "latest_ablation_goes_dashboard.png"))
    plt.close(fig)

    # 7. Append to Sequential Master CSV & Update Winnie's Summary Table
    if os.path.exists(MASTER_CSV):
        df_master = pd.read_csv(MASTER_CSV)
        df_combined = pd.concat([df_master, df_peeled], ignore_index=True)
        df_combined.drop_duplicates(subset=["utc_time", "node"], keep="last", inplace=True)
    else:
        df_combined = df_peeled

    df_combined.sort_values("utc_time", inplace=True)
    df_combined.to_csv(MASTER_CSV, index=False)

    summary_stats = df_combined.groupby(["run_id", "date_utc", "node"]).agg(
        snapshots=("snapshot_id", "count"),
        mean_sun_alt_deg=("sun_alt_deg", "mean"),
        unpeeled_rms_jy=("unpeeled_rms_noise", "mean"),
        peeled_rms_jy=("rms_noise_jy_beam", "mean"),
        noise_reduction_pct=("noise_reduction_pct", "mean"),
        mean_pb_flux_jy=("pb_corrected_flux", "mean"),
        peak_pb_flux_jy=("pb_corrected_flux", "max"),
        mean_snr=("snr", "mean"),
        goes_class=("goes_flare_class", "first")
    ).reset_index().sort_values("run_id")
    summary_stats.to_csv(SUMMARY_CSV, index=False)
    print("\n=== SEQUENTIAL CAMPAIGN SUMMARY TABLE ===")
    print(summary_stats.to_string(index=False))

    update_multiweek_trend_plot(df_combined)

    # 8. Purge Heavy Temporary Files & Push All Sequential Results to GitHub
    clean_workspace()
    run_cmd("git add results/ scripts/", cwd=BASE_DIR)
    run_cmd(f'git commit -m "Sequential campaign update {run_id} ({node}): appended to master archive" || true', cwd=BASE_DIR)
    run_cmd("git push origin main", cwd=BASE_DIR)
    print(f"=== RUN {run_id} COMPLETE & ARCHIVED IN SEQUENCE ON GITHUB! ===")


if __name__ == "__main__":
    main()
