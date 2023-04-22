# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Analyzes the latency and numerics information of sample model inference."""

import itertools
import json
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from absl import logging
import numpy as np

from tensorflow.python.compiler.tensorrt.model_tests import model_handler
import tensorflow.python.compiler.tensorrt.trt_convert as trt


# pylint: disable=bad-whitespace
class DataFrame:
  """Lightweight immutable Dataframe similar to Pandas Dataframe."""

  def __init__(self,
               column_names: Sequence[str],
               rows: Sequence[Sequence[Any]] = None,
               columns: Sequence[Sequence[Any]] = None):
    self._column_names = column_names
    if not rows and not columns:
      raise ValueError("Cannot initialize with empty data!")
    self._rows = rows
    self._columns = columns

  @property
  def n_rows(self) -> int:
    return len(self._rows) if self._rows else len(self._columns[0])

  @property
  def n_columns(self) -> int:
    return len(self._columns) if self._columns else len(self._rows[0])

  @property
  def column_names(self) -> Sequence[str]:
    return self._column_names

  @property
  def rows(self) -> Sequence[Sequence[Any]]:
    return self._rows if self._rows else [
        [c[i] for c in self._columns] for i in range(len(self._columns[0]))
    ]

  @property
  def columns(self) -> Sequence[Sequence[Any]]:
    return self._columns if self._columns else [
        [r[i] for r in self._rows] for i in range(len(self._rows[0]))
    ]

  def __add__(self, other: "DataFrame") -> "DataFrame":
    if (not set(self.column_names).intersection(other.column_names) and
        len(self.rows) == len(other.rows)):
      return DataFrame(
          column_names=list(
              itertools.chain(self.column_names, other.column_names)),
          columns=list(itertools.chain(self.columns, other.columns)))
    if self.column_names == other.column_names:
      return DataFrame(
          column_names=self.column_names,
          rows=list(itertools.chain(self.rows, other.rows)))
    raise ValueError("Cannot combine two DataFrame")

  def __iadd__(self, other: "DataFrame") -> "DataFrame":
    tmp = self + other
    self._column_names = tmp._column_names
    self._rows, self._columns = tmp._rows, tmp._columns
    return self

  def __call__(self, r: int, c: Optional[Union[int, str]] = None) -> Any:
    if c is None:
      return dict(zip(self.column_names, self.rows[r]))
    c = self._column_names.index(c) if isinstance(c, str) else c
    return self._rows[r][c] if self._rows else self._columns[c][r]

  def __str__(self) -> str:
    return ",".join(self.column_names) + "\n" + "\n".join(",".join(
        "N/A" if v is None else str(v) for v in row) for row in self.rows)

  def to_csv(self, path: str):
    with open(path, "w") as file:
      file.write(str(self))

  def to_json(self, path: str):
    with open(path, "w") as file:
      json.dump([dict(zip(self.column_names, r)) for r in self.rows], file)


def extract_test_info(
    test_results: model_handler.TestResultCollection) -> DataFrame:
  """Extracts the test infomation."""
  column_names = list(
      itertools.chain(model_handler.ModelConfig._fields,
                      ["enable_gpu", "trt_model"],
                      trt.TrtConversionParams._fields))
  rows = []
  for result in test_results.results:
    r = list(result.model_config) + [result.enable_gpu]
    if result.trt_convert_params is not None:
      r += [True] + list(result.trt_convert_params)
    else:
      r += [False] + [None for _ in trt.TrtConversionParams._fields]
    rows.append(r)
  return DataFrame(column_names=column_names, rows=rows)


def analyze_test_latency(test_results: model_handler.TestResultCollection,
                         use_cpu_baseline: bool) -> DataFrame:
  """Analyzes test latency."""
  base_result = (
      test_results.cpu_base_result
      if use_cpu_baseline else test_results.gpu_base_result)
  if base_result is None:
    raise ValueError(
        f"No {'CPU' if use_cpu_baseline else 'GPU'} baseline found!")
  base_mean_time = np.asscalar(np.mean(base_result.model_latency))
  column_names = ["time(ms)", "speedup"]
  rows = []
  for result in test_results.results:
    mean_time = np.asscalar(np.mean(result.model_latency))
    rows.append([mean_time * 1000.0, base_mean_time / mean_time])
  return DataFrame(column_names=column_names, rows=rows)


def analyze_test_numerics(test_results: model_handler.TestResultCollection,
                          use_cpu_baseline: bool) -> DataFrame:
  """Analyzes test numerics."""
  preprocess_funcs = {
      "diff": lambda x, y: np.fabs(x - y),
      # Ensures dividends are not zero to avoid exceptions/NaNs.
      "rel_diff": lambda x, y: np.fabs(x - y) / np.fmax(np.fabs(y), 1.0e-6)
  }
  postprocess_funcs = {"mean": np.mean, "std": np.std}
  column_names = []
  columns = []
  base_result = (
      test_results.cpu_base_result
      if use_cpu_baseline else test_results.gpu_base_result)
  if base_result is None:
    raise ValueError(
        f"No {'CPU' if use_cpu_baseline else 'GPU'} baseline found!")
  for fn0, fn1 in itertools.product(preprocess_funcs, postprocess_funcs):
    func0, func1 = preprocess_funcs[fn0], postprocess_funcs[fn1]
    column_names.append("{}_{}".format(fn0, fn1))
    columns.append([])
    for result in test_results.results:
      columns[-1].append(dict())
      for idx, tensor in enumerate(result.output_tensors):
        name = base_result.output_names[idx]
        cpu_tensor = base_result.output_tensors[idx]
        metric_value = np.asscalar(func1(func0(tensor, cpu_tensor)))
        columns[-1][-1][name] = metric_value
  return DataFrame(column_names=column_names, columns=columns)


def check_column(df: DataFrame, name: str, fn: Callable[[float], bool]) -> bool:
  """Checks the values of a column using a custom function and logs abnormals.

  The check is only performed on TensorRT models, not native CPU/GPU models.

  Args:
    df: The DataFrame to be checked.
    name: The name of the column to be checked.
    fn: The function that takes a value of at the specified column and returns
      if the value statisfies the check.

  Returns:
    Whether all the values of the specified column satisfies the provided check.
  """
  is_ok = True
  for r in range(df.n_rows):
    if df(r, "trt_model"):
      if not fn(df(r, name)):
        logging.error("Unsatisfied %s found at: %s", name, df(r))
        is_ok = False
  return is_ok


class ResultAnalyzer:
  """Analyzes ModelHandlerManager results."""

  def __init__(
      self,
      use_cpu_latency_baseline: bool,
      use_cpu_numerics_baseline: bool,
      checkers: Sequence[Callable[[DataFrame], bool]],
  ):
    self._use_cpu_latency_baseline = use_cpu_latency_baseline
    self._use_cpu_numerics_baseline = use_cpu_numerics_baseline
    self._checkers = checkers

  def analysis(
      self, test_results: model_handler.TestResultCollection
  ) -> Tuple[DataFrame, Sequence[bool]]:
    df = extract_test_info(test_results)
    df += analyze_test_latency(test_results, self._use_cpu_latency_baseline)
    df += analyze_test_numerics(test_results, self._use_cpu_numerics_baseline)
    checks = [c(df) for c in self._checkers]
    return df, checks
