#!/usr/bin/env python3
"""
Driver script for fast-radial-scan analysis.
Processes one FRS at a time and prompts user between intervals.
"""
import functions as fn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import shutil
import sys

def cleanup_previous_outputs():
    """Delete any previous outputs before next FRS run."""
    # delete all per-window CSVs
    for f in glob.glob("derived_range_*.csv"):
        os.remove(f)
    # optionally also remove summary CSV
    if os.path.exists("high_sigma_low_beta_windows.csv"):
        os.remove("high_sigma_low_beta_windows.csv")
    print("🧹 Previous outputs removed.")

if __name__ == "__main__":

    time_ranges = fn.parse_time_ranges_from_file("PSP_fastRadialScans.txt")
    fields_files, spc_files = fn.download_psp_data(time_ranges[38:41])
    fields_xrs , spc_xrs    = fn.convert_all_cdf_to_xarray(fields_files, spc_files)

    frs_sec = [(fn.datetime64_to_j2000_seconds(s),
                fn.datetime64_to_j2000_seconds(e)) for s,e in time_ranges]
     
    for (l),(fxr,sxr) in enumerate(zip(fields_xrs, spc_xrs)):
        i = l+38
        if fxr is None or sxr is None: continue
        print(f"\n🔹 Processing range {i}: {time_ranges[i]}")

        selected = []
        cleanup_previous_outputs()  # wipe any outputs before this FRS

        B,V,n,wp,tB,tSC = fn.extract_psp_data(fxr,sxr)

        B_int,Bmag,Vmag,t = fn.process_range_wavelet(B,V,n,wp,tB,tSC)

        wins = fn.generate_valid_windows_aligned(
            t, B_int, V, t[0], t[-1]
        )
                
        for w_idx,(ws,we) in enumerate(wins):
            mask = (t>=ws)&(t<we)
            t_win,B_win,V_win,n_win = t[mask],B_int[mask],V[mask],n[mask]
            fs = 1/np.median(np.diff(t_win))
            fig,ax = plt.subplots(1,2)
            ax[0].plot(Vmag[mask])
            ax[1].plot(Bmag[mask])
            plt.show()


            csv_name = f"derived_range_{i:02d}_win_{w_idx:02d}.csv"
            fn.calculate_derived_parameters_to_csv(
                B_win, V_win, n_win, wp[mask], t_win,
                fn.j2000_to_iso(ws), fn.j2000_to_iso(we), csv_name)

            # PSD panel
            fB,PB = fn.trace_psd(B_win, fs)

            
            if np.any(np.isnan(V_win)):
                mask_V = np.isnan(np.linalg.norm(V_win,axis=1))
                V_win = np.array([np.interp(t_win,t_win[~mask_V],V_win[~mask_V,i]) for i in range(3) ]).T
            

            fV,PV = fn.trace_psd(V_win, fs)
            Zp,Zm = fn.compute_elsasser(V_win-np.nanmean(V_win,axis=0),B_win-np.nanmean(B_win,axis=0),n_win)
            fZp,PZp = fn.trace_psd(Zp,fs)
            fZm,PZm = fn.trace_psd(Zm,fs)

            plt.figure(figsize=(9,5))
            plt.loglog(fB,PB,label="Trace B")
            plt.loglog(fV,PV,label="Trace V")
            plt.loglog(fZp,PZp,label="Trace Z+")
            plt.loglog(fZm,PZm,label="Trace Z−")
            plt.title(f"Trace PSD  {fn.j2000_to_iso(ws)} – {fn.j2000_to_iso(we)}")
            plt.xlabel("f [Hz]"); plt.ylabel("PSD [(km/s)² Hz⁻¹]")
            plt.grid(True,which="both",ls="--",lw=0.4); plt.legend()
            plt.tight_layout(); plt.show()

            # Wavelet diagnostics
            print("Shapes: ", B_win.shape, V_win.shape)

            Bmag_win = np.linalg.norm(B_win, axis=1) if B_win.ndim == 2 else B_win
            Vmag_win = np.linalg.norm(V_win, axis=1) if V_win.ndim == 2 else V_win
            try:
                fn.analyze_wavelet_for_range(
                    t_win, B_win, V_win,
                    np.linalg.norm(B_win,axis=1),
                    np.linalg.norm(V_win,axis=1),
                    (ws,we))
            except Exception as e:
                print("⚠️ wavelet skipped:", e)
                
            # Wavelet-based scalograms
            try:
                dt_seconds = np.median(np.diff(t_win))
                if not np.isfinite(dt_seconds) or dt_seconds <= 0:
                    raise ValueError("Invalid dt_seconds for scalogram")

                # Compute Elsasser fields for the window
                Zp_win, Zm_win = fn.compute_elsasser(
                    V_win - np.nanmean(V_win, axis=0),
                    B_win - np.nanmean(B_win, axis=0),
                    n_win
                )

                # Compute wavelets of |Z+|, |Z-|, |V|, |B|
                freqs, power_Zp, _, _ = fn.compute_wavelet(np.linalg.norm(Zp_win, axis=1), dt_seconds)
                _,     power_Zm, _, _ = fn.compute_wavelet(np.linalg.norm(Zm_win, axis=1), dt_seconds)
                _,     power_V,  _, _ = fn.compute_wavelet(np.linalg.norm(V_win,  axis=1), dt_seconds)
                _,     power_B,  _, _ = fn.compute_wavelet(np.linalg.norm(B_win,  axis=1), dt_seconds)

                # Compute σ_c and σ_r
                denom_Z  = np.clip(power_Zp + power_Zm, 1e-20, None)
                sigma_c_w = (power_Zp - power_Zm) / denom_Z

                denom_VB = np.clip(power_V + power_B, 1e-20, None)
                sigma_r_w = (power_V - power_B) / denom_VB

                # Plot
                fn.plot_scalograms(
                    t_win,
                    freqs,
                    sigma_c_w,
                    sigma_r_w,
                    fn.j2000_to_iso(ws),
                    fn.j2000_to_iso(we)
                )

            except Exception as e:
                print("⚠️ scalogram plot skipped:", e)

            
            beta_mean = np.nanmean(0.03948*n_win*wp[mask]/np.nanmean(Bmag[mask]**2))
            if fn.compute_sigma_c(V_win,B_win) > .5 and beta_mean < .4:
                selected.append({
                    "range_idx":i,"window_idx":w_idx,
                    "start_iso":fn.j2000_to_iso(ws),
                    "end_iso":fn.j2000_to_iso(we),
                    "beta_p":round(beta_mean,3)
                })
        if selected:
            pd.DataFrame(selected).to_csv("high_sigma_low_beta_windows.csv",index=False)
            print("✅ Wrote summary of qualifying windows.")
        else:
            print("⚠️ No windows satisfied selection criteria.")

        # ───── Ask user whether to continue ─────
        resp = input(f"➡️  Finished FRS {i}. Continue to next interval? [y/N]: ").strip().lower()
        if resp != "y":
            print("👋 Exiting after current FRS.")
            break
