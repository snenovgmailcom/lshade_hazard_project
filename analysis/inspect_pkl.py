#!/usr/bin/env python3
"""
Inspect the structure of pkl files to understand data format.

Usage:
    python inspect_pkl.py experiments/r_lshade_D10_nfev_100000
"""

import sys
import pickle
from pathlib import Path
import numpy as np

def inspect_pkl(pkl_path: Path, depth: int = 0, max_depth: int = 3):
    """Recursively inspect a pkl file or object."""
    indent = "  " * depth
    
    if depth == 0:
        print(f"\n{'='*60}")
        print(f"File: {pkl_path}")
        print(f"{'='*60}")
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = pkl_path  # Already loaded object
        
    dtype = type(data).__name__
    
    if isinstance(data, dict):
        print(f"{indent}dict with {len(data)} keys:")
        for key in list(data.keys())[:20]:  # Show first 20 keys
            val = data[key]
            vtype = type(val).__name__
            
            if isinstance(val, (np.ndarray, list)):
                shape = np.shape(val) if hasattr(val, '__len__') else '?'
                if isinstance(val, np.ndarray):
                    print(f"{indent}  '{key}': ndarray{shape}, dtype={val.dtype}")
                    if val.size > 0 and val.size <= 10:
                        print(f"{indent}    values: {val}")
                    elif val.size > 0:
                        print(f"{indent}    first 5: {val.flat[:5]}")
                        print(f"{indent}    last 5:  {val.flat[-5:]}")
                else:
                    print(f"{indent}  '{key}': list[{len(val)}]")
                    if len(val) > 0:
                        print(f"{indent}    first element type: {type(val[0]).__name__}")
                        if len(val) <= 5:
                            print(f"{indent}    values: {val}")
            elif isinstance(val, dict) and depth < max_depth:
                print(f"{indent}  '{key}': dict[{len(val)}]")
                inspect_pkl(val, depth + 2, max_depth)
            elif isinstance(val, (int, float, str, bool, type(None))):
                print(f"{indent}  '{key}': {vtype} = {val}")
            else:
                print(f"{indent}  '{key}': {vtype}")
                
        if len(data) > 20:
            print(f"{indent}  ... and {len(data) - 20} more keys")
            
    elif isinstance(data, list):
        print(f"{indent}list with {len(data)} elements")
        if len(data) > 0:
            print(f"{indent}  element type: {type(data[0]).__name__}")
            if isinstance(data[0], dict) and depth < max_depth:
                print(f"{indent}  first element:")
                inspect_pkl(data[0], depth + 2, max_depth)
            elif len(data) <= 10:
                for i, item in enumerate(data):
                    print(f"{indent}  [{i}]: {item}")
                    
    elif isinstance(data, np.ndarray):
        print(f"{indent}ndarray: shape={data.shape}, dtype={data.dtype}")
        if data.size <= 20:
            print(f"{indent}  values: {data}")
        else:
            print(f"{indent}  first 5: {data.flat[:5]}")
            print(f"{indent}  last 5:  {data.flat[-5:]}")
            print(f"{indent}  min={data.min():.6g}, max={data.max():.6g}, mean={data.mean():.6g}")
            
    else:
        print(f"{indent}{dtype}: {data}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_pkl.py <path_to_pkl_or_directory>")
        sys.exit(1)
        
    path = Path(sys.argv[1])
    
    if path.is_file() and path.suffix == '.pkl':
        inspect_pkl(path)
    elif path.is_dir():
        pkl_files = sorted(path.glob("*.pkl"))
        print(f"Found {len(pkl_files)} pkl files in {path}")
        
        # Inspect first few files
        for pkl_path in pkl_files[:3]:
            inspect_pkl(pkl_path)
            
        if len(pkl_files) > 3:
            print(f"\n... and {len(pkl_files) - 3} more files")
    else:
        print(f"Error: {path} is not a pkl file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
    
