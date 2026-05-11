import argparse
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser(description="Animate a horizontal slice from a 3D NetCDF file.")

    # Required arguments
    parser.add_argument("file", help="Path to the NetCDF file")
    parser.add_argument("output", nargs='?', default="animation.mp4", help="Output filename (default: animation.mp4)")
    parser.add_argument("-v", "--var", required=True, help="Variable name to animate")
    parser.add_argument("--level", type=int, default=0, help="Index of the vertical level (default: 0)")

    # Optional styling
    parser.add_argument("--cmin", type=float, help="Minimum value for colorbar")
    parser.add_argument("--cmax", type=float, help="Maximum value for colorbar")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap (default: viridis)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")

    args = parser.parse_args()

    # Load dataset
    try:
        ds = xr.open_dataset(args.file)
        data = ds[args.var]
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

    title = ax.set_title(f"Var: {args.var} | Level Index: {args.level} | Time: {slice_2d.time.values[0]}")

    def update(frame):
        # Update the image data
        current_step = slice_2d.isel(time=frame)
        im.set_array(current_step.values.flatten())
        title.set_text(f"Var: {args.var} | Level Index: {args.level} | Time: {current_step.time.values}")
        return im, title

    # Create animation
    print(f"Processing {len(slice_2d.time)} frames...")
    ani = animation.FuncAnimation(
        fig, update, frames=len(slice_2d.time), interval=1000/args.fps, blit=False
    )

    # Save
    try:
        ani.save(args.output, writer='ffmpeg', fps=args.fps)
        print(f"Success! Animation saved to {args.output}")
    except Exception as e:
        print(f"Error saving animation: {e}. Ensure 'ffmpeg' is installed.")

if __name__ == "__main__":
    main()
