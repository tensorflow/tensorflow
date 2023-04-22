# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Experimental API for controlling optimizations in `tf.data` pipelines."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum

from tensorflow.core.framework import dataset_options_pb2
from tensorflow.python.data.util import options
from tensorflow.python.util.tf_export import tf_export


class _AutotuneAlgorithm(enum.Enum):
  """Controls what algorithm is used in the autotune implementation."""
  HILL_CLIMB = 0
  GRADIENT_DESCENT = 1


@tf_export("data.experimental.OptimizationOptions")
class OptimizationOptions(options.OptionsBase):
  """Represents options for dataset optimizations.

  You can set the optimization options of a dataset through the
  `experimental_optimization` property of `tf.data.Options`; the property is
  an instance of `tf.data.experimental.OptimizationOptions`.

  ```python
  options = tf.data.Options()
  options.experimental_optimization.noop_elimination = True
  options.experimental_optimization.apply_default_optimizations = False
  dataset = dataset.with_options(options)
  ```
  """
  apply_default_optimizations = options.create_option(
      name="apply_default_optimizations",
      ty=bool,
      docstring=
      "Whether to apply default graph optimizations. If False, only graph "
      "optimizations that have been explicitly enabled will be applied.")

  autotune = options.create_option(
      name="autotune",
      ty=bool,
      docstring=
      "Whether to automatically tune performance knobs. If None, defaults to "
      "True.")

  autotune_buffers = options.create_option(
      name="autotune_buffers",
      ty=bool,
      docstring=
      "When autotuning is enabled (through `autotune`), determines whether to "
      "also autotune buffer sizes for datasets with parallelism. If None,"
      " defaults to False.")

  autotune_cpu_budget = options.create_option(
      name="autotune_cpu_budget",
      ty=int,
      docstring=
      "When autotuning is enabled (through `autotune`), determines the CPU "
      "budget to use. Values greater than the number of schedulable CPU cores "
      "are allowed but may result in CPU contention. If None, defaults to the "
      "number of schedulable CPU cores.")

  autotune_ram_budget = options.create_option(
      name="autotune_ram_budget",
      ty=int,
      docstring=
      "When autotuning is enabled (through `autotune`), determines the RAM "
      "budget to use. Values greater than the available RAM in bytes may "
      "result in OOM. If None, defaults to half of the available RAM in bytes.")

  filter_fusion = options.create_option(
      name="filter_fusion",
      ty=bool,
      docstring=
      "Whether to fuse filter transformations. If None, defaults to False.")

  map_and_batch_fusion = options.create_option(
      name="map_and_batch_fusion",
      ty=bool,
      docstring=
      "Whether to fuse map and batch transformations. If None, defaults to "
      "True.")

  map_and_filter_fusion = options.create_option(
      name="map_and_filter_fusion",
      ty=bool,
      docstring=
      "Whether to fuse map and filter transformations. If None, defaults to "
      "False.")

  map_fusion = options.create_option(
      name="map_fusion",
      ty=bool,
      docstring="Whether to fuse map transformations. If None, defaults to "
      "False.")

  map_parallelization = options.create_option(
      name="map_parallelization",
      ty=bool,
      docstring=
      "Whether to parallelize stateless map transformations. If None, defaults "
      "to True.")

  noop_elimination = options.create_option(
      name="noop_elimination",
      ty=bool,
      docstring=
      "Whether to eliminate no-op transformations. If None, defaults to True.")

  parallel_batch = options.create_option(
      name="parallel_batch",
      ty=bool,
      docstring="Whether to parallelize copying of batch elements. This "
      "optimization is highly experimental and can cause performance "
      "degradation (e.g. when the parallelization overhead exceeds the "
      "benefits of performing the data copies in parallel). You should only "
      "enable this optimization if a) your input pipeline is bottlenecked on "
      "batching and b) you have validated that this optimization improves "
      "performance. If None, defaults to False.")

  shuffle_and_repeat_fusion = options.create_option(
      name="shuffle_and_repeat_fusion",
      ty=bool,
      docstring="Whether to fuse shuffle and repeat transformations. If None, "
      "defaults to True.")

  def _to_proto(self):
    pb = dataset_options_pb2.OptimizationOptions()
    if self.apply_default_optimizations is not None:
      pb.apply_default_optimizations = self.apply_default_optimizations
    if self.autotune is not None:
      pb.autotune = self.autotune
    if self.autotune_buffers is not None:
      pb.autotune_buffers = self.autotune_buffers
    if self.autotune_cpu_budget is not None:
      pb.autotune_cpu_budget = self.autotune_cpu_budget
    if self.autotune_ram_budget is not None:
      pb.autotune_ram_budget = self.autotune_ram_budget
    if self.filter_fusion is not None:
      pb.filter_fusion = self.filter_fusion
    if self.map_and_batch_fusion is not None:
      pb.map_and_batch_fusion = self.map_and_batch_fusion
    if self.map_and_filter_fusion is not None:
      pb.map_and_filter_fusion = self.map_and_filter_fusion
    if self.map_fusion is not None:
      pb.map_fusion = self.map_fusion
    if self.map_parallelization is not None:
      pb.map_parallelization = self.map_parallelization
    if self.noop_elimination is not None:
      pb.noop_elimination = self.noop_elimination
    if self.parallel_batch is not None:
      pb.parallel_batch = self.parallel_batch
    if self.shuffle_and_repeat_fusion is not None:
      pb.shuffle_and_repeat_fusion = self.shuffle_and_repeat_fusion
    return pb

  def _from_proto(self, pb):
    if pb.WhichOneof("optional_apply_default_optimizations") is not None:
      self.apply_default_optimizations = pb.apply_default_optimizations
    if pb.WhichOneof("optional_autotune") is not None:
      self.autotune = pb.autotune
    if pb.WhichOneof("optional_autotune_buffers") is not None:
      self.autotune_buffers = pb.autotune_buffers
    if pb.WhichOneof("optional_autotune_cpu_budget") is not None:
      self.autotune_cpu_budget = pb.autotune_cpu_budget
    if pb.WhichOneof("optional_autotune_ram_budget") is not None:
      self.autotune_ram_budget = pb.autotune_ram_budget
    if pb.WhichOneof("optional_filter_fusion") is not None:
      self.filter_fusion = pb.filter_fusion
    if pb.WhichOneof("optional_map_and_batch_fusion") is not None:
      self.map_and_batch_fusion = pb.map_and_batch_fusion
    if pb.WhichOneof("optional_map_and_filter_fusion") is not None:
      self.map_and_filter_fusion = pb.map_and_filter_fusion
    if pb.WhichOneof("optional_map_fusion") is not None:
      self.map_fusion = pb.map_fusion
    if pb.WhichOneof("optional_map_parallelization") is not None:
      self.map_parallelization = pb.map_parallelization
    if pb.WhichOneof("optional_noop_elimination") is not None:
      self.noop_elimination = pb.noop_elimination
    if pb.WhichOneof("optional_parallel_batch") is not None:
      self.parallel_batch = pb.parallel_batch
    if pb.WhichOneof("optional_shuffle_and_repeat_fusion") is not None:
      self.shuffle_and_repeat_fusion = pb.shuffle_and_repeat_fusion

  def _set_mutable(self, mutable):
    """Change the mutability value to `mutable` on this options and children."""
    # pylint: disable=protected-access
    object.__setattr__(self, "_mutable", mutable)
