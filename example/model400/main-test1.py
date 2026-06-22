import argparse
from typing import Any, Protocol

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from tqdm import tqdm

from ifc_hlm.vectorized.model400 import Model400

MM_TO_M = 1000.0
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_30_DAY_MONTH = 30 * 24 * SECONDS_PER_HOUR

# MONTHLY_FORCINGS_NC_FILE = "data/monthly_evaporation.nc"
# DAILY_FORCINGS_NC_FILE = "data/combined_data.nc"
HOURLY_FORCINGS_NC_FILE = "data/felipe-tests.nc"

OUTPUT_STEP = 3600  # 1 hour
WRITE_STEP = 90 * 86400  # 90 days


class ProgressBar(Protocol):
    def update(self, n: float | None = 1) -> bool | None: ...
    def set_description(
        self, desc: str | None = None, refresh: bool | None = True
    ) -> None: ...


class Model400Runner:
    def __init__(
        self,
        model: Model400,
        # monthly_ds: xr.Dataset,
        # daily_ds: xr.Dataset,
        hourly_ds: xr.Dataset,
        *,
        output_step: int,
        write_step: int,
        output_suffix: str,
    ) -> None:
        self.model = model
        # self.monthly_ds = monthly_ds
        # self.daily_ds = daily_ds
        self.hourly_ds = hourly_ds
        self.output_step = output_step
        self.write_step = write_step
        self.output_suffix = output_suffix
        self.outputs_history: dict[str, list[NDArray[np.float64]]] = {
            name: [] for name in model.get_output_var_names()
        }
        self.time_history: list[float] = []
        self.previous_time = model.get_current_time()
        self.node_ids = np.array(hourly_ds["nodeid"].data, dtype=np.uint32)

    def elapsed_seconds(self) -> int:
        return int(self.model.get_current_time() - self.model.get_start_time())

    def current_time(self) -> np.datetime64:
        return np.datetime64(int(self.model.get_current_time()), "s")

    def is_hourly_input_step(self) -> bool:
        current_time = self.current_time()
        return current_time == current_time.astype("datetime64[h]").astype(
            "datetime64[s]"
        )

    def is_daily_input_step(self) -> bool:
        current_time = self.current_time()
        return current_time == current_time.astype("datetime64[D]").astype(
            "datetime64[s]"
        )

    def is_monthly_input_step(self) -> bool:
        current_time = self.current_time()
        return current_time == current_time.astype("datetime64[M]").astype(
            "datetime64[s]"
        )

    def is_output_step(self) -> bool:
        return self.elapsed_seconds() % self.output_step == 0

    def is_write_step(self) -> bool:
        return self.elapsed_seconds() % self.write_step == 0

    # def get_current_monthly_forcings(self) -> dict[str, NDArray[np.float64]]:
    #     input_var_names = self.model.get_input_var_names()
    #     monthly_time = self.current_time().astype("datetime64[M]")
    #     ds_monthly_t = self.monthly_ds.sel(datetime=monthly_time)
    #     return self._monthly_forcings_dict(ds_monthly_t, input_var_names)

    # def get_current_daily_forcings(self) -> dict[str, NDArray[np.float64]]:
    #     input_var_names = self.model.get_input_var_names()
    #     daily_time = self.current_time().astype("datetime64[D]")
    #     ds_daily_t = self.daily_ds.sel(datetime=daily_time)
    #     return {
    #         str(name): np.array(ds_daily_t[name].data, dtype=np.float64)
    #         for name in ds_daily_t.data_vars
    #         if name in input_var_names
    #     }

    def get_current_hourly_forcings(self) -> dict[str, NDArray[np.float64]]:
        input_var_names = self.model.get_input_var_names()
        ds_hourly_t = self.hourly_ds.sel(time=self.current_time())
        return {
            str(name): np.array(ds_hourly_t[name].data, dtype=np.float64)
            for name in ds_hourly_t.data_vars
            if name in input_var_names
        }

    @staticmethod
    def _monthly_forcings_dict(
        monthly_ds_t: xr.Dataset,
        input_var_names: tuple[str, ...],
    ) -> dict[str, NDArray[np.float64]]:
        monthly_forcings: dict[str, NDArray[np.float64]] = {}

        var_aliases = {
            "e_pot": ("e_pot", "evaporation"),
        }

        for input_name, candidate_names in var_aliases.items():
            if input_name not in input_var_names:
                continue

            for candidate_name in candidate_names:
                if candidate_name in monthly_ds_t.data_vars:
                    monthly_forcings[input_name] = np.array(
                        monthly_ds_t[candidate_name].data,
                        dtype=np.float64,
                    )
                    break

        for name in monthly_ds_t.data_vars:
            if name in input_var_names and name not in monthly_forcings:
                monthly_forcings[str(name)] = np.array(
                    monthly_ds_t[name].data,
                    dtype=np.float64,
                )

        return monthly_forcings

    @staticmethod
    def _convert_forcing_units(
        forcings: dict[str, NDArray[np.float64]],
    ) -> dict[str, NDArray[np.float64]]:
        converted = {name: values.copy() for name, values in forcings.items()}

        if "rainfall" in converted:
            converted["rainfall"] /= MM_TO_M * SECONDS_PER_HOUR

        if "e_pot" in converted:
            converted["e_pot"] /= MM_TO_M * SECONDS_PER_30_DAY_MONTH

        return converted

    def update_inputs(self, forcings: dict[str, NDArray[np.float64]]) -> None:
        forcings = self._convert_forcing_units(forcings)
        for name, array in forcings.items():
            self.model.set_value(name, array)
            # print(f"Updated input '{name}' at time {self.current_time()}")
            # print(f"Values: {array}")

    def store_outputs(self) -> None:
        self.time_history.append(self.model.get_current_time())
        for name in self.model.get_output_var_names():
            self.outputs_history[name].append(self.model.get_value_ptr(name).copy())

    def write_netcdf(self, fname: str) -> None:
        if not self.time_history:
            return

        time_coord = np.array(self.time_history, dtype="float64").astype(
            "datetime64[s]"
        )

        ds = xr.Dataset(
            data_vars={
                name: xr.DataArray(
                    data=np.stack(values, axis=0),
                    dims=("time", "node"),
                    coords={"time": time_coord, "node": self.node_ids},
                )
                for name, values in self.outputs_history.items()
            }
        )

        encoding: dict[str, dict[str, Any]] = {
            name: {
                "dtype": "float32",
                "zlib": True,
                "_FillValue": np.float32(np.nan),
            }
            for name in self.outputs_history
        }

        ds.to_netcdf(
            fname, encoding=encoding
        )  # pyright: ignore[reportUnknownMemberType]

    def clear_buffers(self) -> None:
        self.time_history.clear()
        for values in self.outputs_history.values():
            values.clear()

    @staticmethod
    def datetime_str(timestamp: float) -> str:
        return str(np.datetime64(int(timestamp), "s"))

    def update_progress_bar(self, pbar: ProgressBar) -> None:
        curr_time = self.model.get_current_time()
        dt = curr_time - self.previous_time
        if dt > 0:
            pbar.set_description(self.datetime_str(curr_time))
            pbar.update(dt)
            self.previous_time = curr_time

    def run(self) -> None:
        total_time = self.model.get_end_time() - self.model.get_start_time()

        with tqdm(
            total=total_time,
            unit="s",
            unit_scale=True,
            desc=self.datetime_str(self.previous_time),
        ) as pbar:
            while self.model.get_current_time() < self.model.get_end_time():
                # if self.is_monthly_input_step():
                #     self.update_inputs(self.get_current_monthly_forcings())

                # if self.is_daily_input_step():
                #     self.update_inputs(self.get_current_daily_forcings())

                if self.is_hourly_input_step():
                    self.update_inputs(self.get_current_hourly_forcings())

                if self.is_output_step():
                    self.store_outputs()
                    self.update_progress_bar(pbar)

                    if self.is_write_step():
                        fname = f"{int(self.time_history[0])}.{self.output_suffix}.nc"
                        self.write_netcdf(fname)
                        self.clear_buffers()

                # print(f"Updating model at time {self.current_time()}")
                # print(self.outputs_history)
                self.model.update()
                # self.model.current_time += self.model.config.time_step

            if self.time_history:
                fname = f"{int(self.time_history[0])}.{self.output_suffix}.nc"
                self.write_netcdf(fname)
                self.clear_buffers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default="testcase/config.toml")
    parser.add_argument("--config", default="felipe-test1/config.toml")
    parser.add_argument("--suffix", default="felipe-test1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = Model400()
    model.initialize(args.config)

    # df = pd.read_csv("felipe-test1/edges.csv")
    # src = set(value.item() for value in df["src"].values)
    # dst = set(value.item() for value in df["dst"].values)
    # links = np.array(sorted(src | dst), dtype=np.uint32)
    # print(len(links))

    # times = np.array(
    #     range(int(model.get_start_time()), int(model.get_end_time()) + 1, 3600),
    #     dtype=int,
    # ).astype("datetime64[s]")
    # rain = np.array([0 for _ in times])
    # rain[0:24] = 10.0

    # temperature = np.array([0 for _ in times])
    # frozen_ground = np.array([0 for _ in times])
    # evaporation = np.array([0 for _ in times])

    # forcings_df = pd.DataFrame(
    #     {
    #         "time": times,
    #         "rainfall": rain,
    #         "temperature": temperature,
    #         "frozen_ground": frozen_ground,
    #         "e_pot": evaporation,
    #     }
    # )

    # node_coord = links
    # time_coord = forcings_df["time"].values

    # xr_ds = xr.Dataset(
    #     data_vars={
    #         "rainfall": xr.DataArray(
    #             data=np.broadcast_to(
    #                 forcings_df["rainfall"].to_numpy()[:, None],
    #                 (len(forcings_df), len(node_coord)),
    #             ),
    #             dims=("time", "nodeid"),
    #             coords={"time": time_coord, "nodeid": node_coord},
    #         ),
    #         "temperature": xr.DataArray(
    #             data=np.broadcast_to(
    #                 forcings_df["temperature"].to_numpy()[:, None],
    #                 (len(forcings_df), len(node_coord)),
    #             ),
    #             dims=("time", "nodeid"),
    #             coords={"time": time_coord, "nodeid": node_coord},
    #         ),
    #         "frozen_ground": xr.DataArray(
    #             data=np.broadcast_to(
    #                 forcings_df["frozen_ground"].to_numpy()[:, None],
    #                 (len(forcings_df), len(node_coord)),
    #             ),
    #             dims=("time", "nodeid"),
    #             coords={"time": time_coord, "nodeid": node_coord},
    #         ),
    #         "e_pot": xr.DataArray(
    #             data=np.broadcast_to(
    #                 forcings_df["e_pot"].to_numpy()[:, None],
    #                 (len(forcings_df), len(node_coord)),
    #             ),
    #             dims=("time", "nodeid"),
    #             coords={"time": time_coord, "nodeid": node_coord},
    #         ),
    #     }
    # )

    # xr_ds.to_netcdf("data/felipe-tests.nc")  # pyright: ignore[reportUnknownMemberType]
    # return 0

    with (
        # xr.open_dataset(MONTHLY_FORCINGS_NC_FILE) as monthly_ds,
        # xr.open_dataset(DAILY_FORCINGS_NC_FILE) as daily_ds,
        xr.open_dataset(HOURLY_FORCINGS_NC_FILE) as hourly_ds
    ):
        runner = Model400Runner(
            model,
            # monthly_ds,
            # daily_ds,
            hourly_ds,
            output_step=OUTPUT_STEP,
            write_step=WRITE_STEP,
            output_suffix=args.suffix,
        )
        runner.run()

    model.finalize()


if __name__ == "__main__":
    main()
