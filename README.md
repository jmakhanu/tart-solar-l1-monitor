# L-Band (1.57542 GHz) Solar Radio Flux Monitoring with TART
**Authors:** Idriss James Makhanu & Winnie  
**Origin:** East Africa TART Training Workshop (SARAO / Rhodes University / Kenyatta University / KSA)

## Overview
An automated radio interferometry calibration, visibility peeling (`uvsub`), and World Coordinate System (WCS) aperture photometry pipeline for tracking 1.57542 GHz (GPS L1) solar radio flux density and space-weather bursts using the 24-element Transient Array Radio Telescope (`ke-kitui`, `ke-tuk`, and `mu-udm`).

## Pipeline Architecture
1. **Daytime Visibility Ingestion & Fringe-Stopping (`tart2ms`):** Converts 24-antenna HDF5 visibility archives (276 baselines) into rephased CASA Measurement Sets (`rephase: obs-midpoint`) with topocentric GNSS ephemeris sky models.
2. **RIME Complex Gain Calibration (`CASA`):** Two-stage calibration—slow amplitude calibration (`solint: 300s`) and rapid ionospheric/electronic phase tracking (`solint: 10s`, `refant: 0`).
3. **GNSS Visibility Peeling (`uvsub`):** Subtracts bright GNSS satellite model visibilities (`MODEL_DATA`) from `CORRECTED_DATA` in visibility space to eliminate satellite sidelobes across the solar disk.
4. **Snapshot Synthesis & WCS Solar Photometry (`WSClean` + `Astropy`):** Generates 70-frame all-sky snapshot maps (`1024x1024`, Briggs `0.0`), projects topocentric solar ephemerides `(RA, Dec)` to pixel coordinates `(x, y)` via `astropy.wcs`, applies primary beam attenuation correction ($\cos^{1.5}(\mathrm{ZA})$), and extracts 1.575 GHz solar light curves.
