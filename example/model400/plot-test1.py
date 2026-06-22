from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from netCDF4 import num2date


REFERENCE_DIR = Path("felipe-test1/reference/test1")
LINK_ID = 391450


def load_reference_series(
    reference_dir: Path,
    link_id: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    files = sorted(
        reference_dir.glob("test_*.h5"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    if not files:
        raise FileNotFoundError(f"No reference files found in {reference_dir}")

    times: list[int] = []
    series: dict[str, list[float]] = {
        "q": [],
        "h0": [],
        "h1": [],
        "h2": [],
        "h3": [],
        "h4": [],
    }

    # ASYNCH snapshot order:
    #   State0=q, State1=static, State2=surface, State3=subsurf, State4=gw, State5=snow
    # Python model400 output names:
    #   q, h0=snow, h1=static, h2=surface, h3=subsurf, h4=gw
    state_to_var = {
        "state_0": "q",
        "state_1": "h1",
        "state_2": "h2",
        "state_3": "h3",
        "state_4": "h4",
        "state_5": "h0",
    }

    for path in files:
        unix_time = int(path.stem.split("_")[1])
        with h5py.File(path, "r") as f:
            snapshot = f["snapshot"]
            matches = np.flatnonzero(snapshot["link_id"] == link_id)
            if matches.size == 0:
                raise KeyError(f"link_id {link_id} not found in {path}")
            row = snapshot[matches[0]]

        times.append(unix_time)
        for state_name, var_name in state_to_var.items():
            series[var_name].append(float(row[state_name]))

    time_coord = np.array(times, dtype="datetime64[s]")
    ref_series = {
        name: np.array(values, dtype=np.float64)
        for name, values in series.items()
    }
    return time_coord, ref_series


def load_model_outputs(pattern: str = "*.felipe-test1.nc") -> xr.Dataset:
    paths = sorted(Path(".").glob(pattern), key=lambda path: int(path.stem.split(".")[0]))
    if not paths:
        raise FileNotFoundError(f"No model output files matched {pattern!r}")

    times: list[np.ndarray] = []
    variables: dict[str, list[np.ndarray]] = {}
    node_coord: np.ndarray | None = None

    for path in paths:
        with h5py.File(path, "r") as f:
            time_var = f["time"]
            units = time_var.attrs["units"].decode("utf-8")
            calendar = time_var.attrs.get("calendar", b"standard").decode("utf-8")
            decoded_time = num2date(
                np.asarray(time_var[:]),
                units=units,
                calendar=calendar,
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
            time = np.array(decoded_time, dtype="datetime64[ns]")
            node = np.array(f["node"][:], copy=True)
            if node_coord is None:
                node_coord = node
            elif not np.array_equal(node_coord, node):
                raise ValueError(f"Node coordinate mismatch in {path}")

            times.append(time)

            for name in ("q", "h0", "h1", "h2", "h3", "h4"):
                data = np.array(f[name][:], copy=True)
                if data.ndim != 2 or data.shape[0] != time.size:
                    raise ValueError(
                        f"Expected {name} in {path} to have shape (time, node), got {data.shape}"
                    )
                variables.setdefault(name, []).append(data)

    if node_coord is None:
        raise ValueError("No node coordinate found in model outputs")

    time_coord = np.concatenate(times, axis=0)
    data_vars = {
        name: xr.DataArray(
            data=np.concatenate(series, axis=0),
            dims=("time", "node"),
            coords={"time": time_coord, "node": node_coord},
        )
        for name, series in variables.items()
    }

    return xr.Dataset(data_vars=data_vars).sortby("time")


def main():
    # Load and stitch together all chunked model outputs.
    ds = load_model_outputs()

    if "node" not in ds.coords:
        raise KeyError("Expected a 'node' coordinate in the NetCDF output")
    if LINK_ID not in ds["node"].values:
        raise KeyError(f"link_id {LINK_ID} not found in dataset node coordinate")

    ds_link = ds.sel(node=LINK_ID)
    ref_time, ref_series = load_reference_series(REFERENCE_DIR, LINK_ID)
    variables = list(ds_link.data_vars)
    nvars = len(variables)

    fig, axes = plt.subplots(
        nvars,
        1,
        figsize=(14, 2.2 * nvars),
        sharex=True,
        constrained_layout=True,
    )

    if nvars == 1:
        axes = [axes]

    for ax, var in zip(axes, variables):
        ax.plot(ds_link["time"].values, ds_link[var].values, linewidth=1.2, label="model")
        if var in ref_series:
            ax.plot(
                ref_time,
                ref_series[var],
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
                label="reference",
            )
        ax.set_title(var)
        ax.set_ylabel(var)
        if var == "q":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("time")
    fig.suptitle(f"Model outputs for node {LINK_ID}", fontsize=14)
    output_path = Path(f"felipe-test1/link_{LINK_ID}_comparison.png")
    fig.savefig(output_path, dpi=150)
    print(f"saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
