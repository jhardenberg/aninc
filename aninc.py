import argparse
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Animate a horizontal slice from a 3D NetCDF file.")

    # Required arguments
    parser.add_argument("files", nargs="+", help="Path to one or more NetCDF files (supports wildcards)")
    parser.add_argument("-o", "--output",required=False, default="animation.mp4", help="Output filename (default: animation.mp4)")
    parser.add_argument("-v", "--var", required=False, help="Variable name to animate")
    parser.add_argument("--level", type=int, default=0, help="Index of the vertical level (default: 0)")

    # Optional styling
    parser.add_argument("--cmin", type=float, help="Minimum value for colorbar")
    parser.add_argument("--cmax", type=float, help="Maximum value for colorbar")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap (default: viridis)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--no-progress", action="store_true", help="Suppress the progress bar")

    args = parser.parse_args()

    # Load dataset
    try:
        # Use open_mfdataset to support multiple files and dask for lazy loading
        # chunks={} ensures dask is used
        ds = xr.open_mfdataset(args.files, chunks={})
        
        if args.var:
            var_name = args.var
        else:
            if not ds.data_vars:
                print("Error: No data variables found in the NetCDF file.")
                sys.exit(1)
            var_name = list(ds.data_vars)[0]
            print(f"No variable specified. Using the first available variable: {var_name}")
        
        data = ds[var_name]
    except Exception as e:
        print(f"Error loading file or variable: {e}")
        sys.exit(1)

    # Dimensionality check: assuming (time, lev, lat, lon) or (time, lev, y, x)
    if data.ndim != 4:
        print(f"Error: Variable must have 4 dimensions (Time, Level, Y, X). Found: {data.dims}")
        sys.exit(1)

    # Select the specific level
    # We use .isel to select by index on the second dimension (usually level)
    lev_dim = data.dims[1]
    slice_2d = data.isel({lev_dim: args.level})

    # Determine color limits
    vmin = args.cmin if args.cmin is not None else float(slice_2d.min())
    vmax = args.cmax if args.cmax is not None else float(slice_2d.max())

    # Setup the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')

    # Initial frame
    im = slice_2d.isel(time=0).plot(
        ax=ax,
        add_colorbar=True,
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax
    )

    title = ax.set_title(f"Var: {var_name} | Level Index: {args.level} | Time: {slice_2d.time.values[0]}")

    def update(frame):
        # Update the image data
        current_step = slice_2d.isel(time=frame)
        im.set_array(current_step.values.flatten())
        title.set_text(f"Var: {var_name} | Level Index: {args.level} | Time: {current_step.time.values}")
        return im, title

    # Create animation
    num_frames = len(slice_2d.time)
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=1000/args.fps, blit=False
    )

    # Save
    print(f"Processing {num_frames} frames...")
    
    # Progress bar is disabled if --no-progress is used
    pbar = tqdm(total=num_frames, disable=args.no_progress)

    def progress_callback(i, n):
        pbar.update(1)

    try:
        ani.save(args.output, writer='ffmpeg', fps=args.fps, progress_callback=progress_callback)
        pbar.close()
        print(f"\nSuccess! Animation saved to {args.output}")
    except Exception as e:
        pbar.close()
        print(f"\nError saving animation: {e}. Ensure 'ffmpeg' is installed.")

if __name__ == "__main__":
    main()
