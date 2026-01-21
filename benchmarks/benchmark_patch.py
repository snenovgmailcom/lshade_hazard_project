# Add this import at the top (after other imports)
import lzma

# Replace the PKL saving section in main() (around line 340-350)
# Find this block and replace it:

        # ---------------------------------------------------------------------
        # Save PKL immediately after function completes
        # ---------------------------------------------------------------------
        func_dir = os.path.join(args.outdir, func_short)
        os.makedirs(func_dir, exist_ok=True)
        
        # For high dimensions, save history separately to reduce main PKL size
        if args.dim >= 100:
            # Save each seed's history to separate compressed file
            for r in fn_results:
                seed = r["seed"]
                history = r.get("history", {})
                if history:
                    hist_path = os.path.join(func_dir, f"history_seed_{seed}.pkl.xz")
                    with lzma.open(hist_path, "wb") as f:
                        pickle.dump(history, f)
                # Clear history from main results
                r["history"] = {}
            print(f"  Histories saved separately (compressed)")
        
        pkl_path = os.path.join(func_dir, f"{func_short}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({fname: fn_results}, f)
            f.flush()
            os.fsync(f.fileno())
        
        size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
        print(f"  PKL saved   : {pkl_path} ({size_mb:.1f} MB)")
