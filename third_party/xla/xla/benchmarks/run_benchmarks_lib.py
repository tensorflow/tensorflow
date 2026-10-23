# Copyright 2026 The OpenXLA Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for running JAX and Pallas microbenchmarks on TPU."""

from typing import Sequence
from absl import flags
from absl import logging
from jax.experimental.pallas import tpu as pltpu
import pandas as pd

from xla.benchmarks import benchmark_configs
from xla.benchmarks import results_utils

_BENCHMARKS = flags.DEFINE_multi_enum(
    "benchmarks",
    None,
    list(benchmark_configs.BENCHMARK_FACTORIES.keys()),
    "List of benchmarks to run (e.g. dense_matmul, subchannel_matmul, "
    "jax_matmul). If not specified, runs all registered benchmarks.",
)
_CSV_PATH = flags.DEFINE_string(
    "csv_path",
    None,
    "Destination path for writing CSV benchmark results table.",
)
_REPEAT = flags.DEFINE_integer(
    "repeat",
    1,
    "Number of times target_fn is executed per benchmark run.",
)
_RUNS = flags.DEFINE_integer(
    "runs",
    1,
    "Number of end-to-end benchmark iterations.",
)
_USE_RANDOM_DATA = flags.DEFINE_bool(
    "use_random_data",
    True,
    "Use random data for input tensors, or zeros if false.",
)
_CHECK_NUMERICS = flags.DEFINE_bool(
    "check_numerics",
    False,
    "Whether to verify output accuracy against reference_fn.",
)


def run_benchmark_suite(
    benchmark_name: str,
    repeat: int = 1,
    runs: int = 1,
    use_random_data: bool = True,
    check_numerics: bool = False,
) -> pd.DataFrame:
  """Runs all configurations for a given benchmark suite serially."""
  config_factory = benchmark_configs.BENCHMARK_FACTORIES[benchmark_name]
  chip_version = pltpu.get_tpu_info().chip_version
  configs = config_factory(chip_version)
  logging.info(
      "=== Running %s: %d configuration(s) ===",
      benchmark_name,
      len(configs),
  )

  results = []
  for i, cfg in enumerate(configs):
    logging.info(
        "[%s %d/%d] Starting benchmark: %s",
        benchmark_name,
        i + 1,
        len(configs),
        cfg,
    )
    bm = cfg.get_benchmark()
    prof_results = bm.run(
        repeat=repeat,
        runs=runs,
        use_random_data=use_random_data,
        check_numerics=check_numerics,
    )
    results.append((cfg, prof_results))

  df = results_utils.create_results_table(results)
  results_utils.log_results_table(benchmark_name, df)
  return df


def main(argv: Sequence[str] | None = None) -> None:
  del argv
  if _BENCHMARKS.value is None or len(_BENCHMARKS.value) == 0:
    benchmarks_to_run = list(benchmark_configs.BENCHMARK_FACTORIES.keys())
  else:
    benchmarks_to_run = list(_BENCHMARKS.value)

  all_tables = {}
  for bm_name in benchmarks_to_run:
    df = run_benchmark_suite(
        benchmark_name=bm_name,
        repeat=_REPEAT.value,
        runs=_RUNS.value,
        use_random_data=_USE_RANDOM_DATA.value,
        check_numerics=_CHECK_NUMERICS.value,
    )
    all_tables[bm_name] = df

  if _CSV_PATH.value:
    results_utils.save_all_results_to_csv(all_tables, _CSV_PATH.value)

