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

# Do not modify.
_ENABLE_AUTOTUNE_BUFFERS_BY_DEFAULT = False


class _AutotuneAlgorithm(enum.Enum):
  """Controls what algorithm is used in the autotune implementation."""
  HILL_CLIMB = 0
  GRADIENT_DESCENT = 1


@tf_export("data.experimental.MapVectorizationOptions")
class MapVectorizationOptions(options.OptionsBase):
  """Represents options for the MapVectorization optimization."""
  # TODO(rachelim): Other configuration parameters can go here, for example,
  # how many "experiments" to run with ChooseFastestBranchDataset.
  enabled = options.create_option(
      name="enabled",
      ty=bool,
      docstring=
      "Whether to vectorize map transformations. If None, defaults to False."
  )

  use_choose_fastest = options.create_option(
      name="use_choose_fastest",
      ty=bool,
      docstring="Whether to use ChooseFastestBranchDataset with this "
      "transformation. If True, the pipeline picks between the vectorized and "
      "original segment at runtime based on their iterations speed. If None, "
      "defaults to False.")

  def _graph_rewrites(self):
    graph_rewrites = options.graph_rewrites()
    result = graph_rewrites(enabled=[], disabled=[], default=[])
    if self.enabled is True:  # pylint: disable=g-bool-id-comparison
      result.enabled.append("map_vectorization")
    elif self.enabled is False:  # pylint: disable=g-bool-id-comparison
      result.disabled.append("map_vectorization")
    return result

  def _graph_rewrite_configs(self):
    if not self.enabled:
      return []
    if self.use_choose_fastest:
      return ["map_vectorization:use_choose_fastest:true"]
    else:
      return ["map_vectorization:use_choose_fastest:false"]

  def _to_proto(self):
    pb = dataset_options_pb2.MapVectorization()
    if self.enabled is not None:
      pb.enabled = self.enabled
    if self.use_choose_fastest is not None:
      pb.use_choose_fastest = self.use_choose_fastest
    return pb

  def _from_proto(self, pb):
    if pb.WhichOneof("optional_enabled") is not None:
      self.enabled = pb.enabled
    if pb.WhichOneof("optional_use_choose_fastest") is not None:
      self.use_choose_fastest = pb.use_choose_fastest


@tf_export("data.experimental.OptimizationOptions")
class OptimizationOptions(options.OptionsBase):
  """Represents options for dataset optimizations.

  You can set the optimization options of a dataset through the
  `experimental_optimization` property of `tf.data.Options`; the property is
  an instance of `tf.data.experimental.OptimizationOptions`.

  ```python
  options = tf.data.Options()
  options.experimental_optimization.noop_elimination = True
  options.experimental_optimization.map_vectorization.enabled = True
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

  filter_with_random_uniform_fusion = options.create_option(
      name="filter_with_random_uniform_fusion",
      ty=bool,
      docstring=
      "Whether to fuse filter dataset that predicts random_uniform < rate into "
      "a sampling dataset. If None, defaults to False.")

  hoist_random_uniform = options.create_option(
      name="hoist_random_uniform",
      ty=bool,
      docstring=
      "Whether to hoist `tf.random_uniform()` ops out of map transformations. "
      "If None, defaults to False.")

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
      "to False.")

  map_vectorization = options.create_option(
      name="map_vectorization",
      ty=MapVectorizationOptions,
      docstring=
      "The map vectorization options associated with the dataset. See "
      "`tf.data.experimental.MapVectorizationOptions` for more details.",
      default_factory=MapVectorizationOptions)

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

  reorder_data_discarding_ops = options.create_option(
      name="reorder_data_discarding_ops",
      ty=bool,
      docstring="Whether to reorder ops that will discard data to the front of "
      "unary cardinality preserving transformations, e.g. "
      "dataset.map(...).take(3) will be optimized to dataset.take(3).map(...). "
      "For now this optimization will move `skip`, `shard` and `take` to the "
      "front of `map` and `prefetch`. This optimization is only for "
      "performance; it will not affect the output of the dataset. "
      "If None, defaults to True.")

  shuffle_and_repeat_fusion = options.create_option(
      name="shuffle_and_repeat_fusion",
      ty=bool,
      docstring="Whether to fuse shuffle and repeat transformations. If None, "
      "defaults to True.")

  def _autotune_buffers(self):
    if self.autotune_buffers is not None:
      return self.autotune_buffers
    # The default setting for autotune_buffers is based on
    # _ENABLE_AUTOTUNE_BUFFERS_BY_DEFAULT
    return _ENABLE_AUTOTUNE_BUFFERS_BY_DEFAULT

  def _autotune_settings(self):
    # Default autotune settings
    autotune = True

    # If autotune_buffers is enabled, we use the GRADIENT_DESCENT algorithm by
    # default, which is more performant for tuning heterogeneous parameters.
    algorithm = (
        _AutotuneAlgorithm.GRADIENT_DESCENT
        if self._autotune_buffers() else _AutotuneAlgorithm.HILL_CLIMB)
    cpu_budget = 0  # Indicates that all CPU cores should be used by default.
    ram_budget = 0  # Indicates that default value of RAM budget should be used.

    # Set these options if they are explicitly set by the user.
    if self.autotune is False:  # pylint: disable=g-bool-id-comparison
      autotune = False
    if self.autotune_cpu_budget is not None:
      cpu_budget = self.autotune_cpu_budget
    if self.autotune_ram_budget is not None:
      ram_budget = self.autotune_ram_budget

    return autotune, algorithm, cpu_budget, ram_budget

  def _graph_rewrites(self):
    """Produces lists of enabled, disabled and default graph optimizations.

    Returns:
      result: a namedtuple with three attributes. `result.enabled` is the list
        of user enabled optimizations. `result.disabled` is the list of user
        disabled optimizations. `result.default` is the list of optimizations
        that are enabled by default (the user has not explicitly enabled or
        disabled them).
    """
    if self.map_vectorization is not None:
      result = self.map_vectorization._graph_rewrites()  # pylint: disable=protected-access
    else:
      result = MapVectorizationOptions()._graph_rewrites()  # pylint: disable=protected-access

    all_optimizations = [
        "filter_fusion",
        "filter_with_random_uniform_fusion",
        "hoist_random_uniform",
        "map_and_batch_fusion",
        "map_and_filter_fusion",
        "map_parallelization",
        "map_fusion",
        "noop_elimination",
        "parallel_batch",
        "reorder_data_discarding_ops",
        "shuffle_and_repeat_fusion",
    ]

    if self.apply_default_optimizations is not False:  # pylint: disable=g-bool-id-comparison
      # The following optimizations are turned on by default, unless the user
      # explicitly disables them.
      optimizations_to_disable = [
          "map_and_batch_fusion",
          "map_parallelization",
          "noop_elimination",
          "shuffle_and_repeat_fusion",
      ]
      for optimization in optimizations_to_disable:
        if getattr(self, optimization) is None:
          result.default.append(optimization)

    # Each of these attributes on the Options object is either True (explicitly
    # enabled), False (explicitly disabled), or None (default).
    for optimization in all_optimizations:
      if getattr(self, optimization) is True:  # pylint: disable=g-bool-id-comparison
        result.enabled.append(optimization)
      elif getattr(self, optimization) is False:  # pylint: disable=g-bool-id-comparison
        result.disabled.append(optimization)

    autotune_buffers = self._autotune_buffers()
    if self.autotune is not False and autotune_buffers is True:  # pylint: disable=g-bool-id-comparison
      # When autotuning buffer sizes is enabled, we inject a `prefetch`
      # transformation after asynchronous dataset ops. Only the buffer sizes of
      # prefetch transformations will be autotuned, though this is practically
      # equivalent to tuning the buffer sizes of the other asynchronous
      # transformations.
      result.enabled.append("autotune_buffer_sizes")
      result.enabled.append("disable_prefetch_legacy_autotune")

    if self.autotune is False:  # pylint: disable=g-bool-id-comparison
      result.disabled.append("autotune_buffer_sizes")
      result.disabled.append("disable_prefetch_legacy_autotune")

    return result

  def _graph_rewrite_configs(self, autotune):
    if self.map_vectorization is not None:
      graph_rewrite_configs = self.map_vectorization._graph_rewrite_configs()  # pylint: disable=protected-access
    else:
      graph_rewrite_configs = []
    autotune_only_optimizations = [
        "autotune_buffer_sizes",
        "batch_parallelization",
        "disable_prefetch_legacy_autotune",
        "enable_gradient_descent",
        "map_parallelization"
    ]
    if autotune is False:  # pylint: disable=g-bool-id-comparison
      for optimization in autotune_only_optimizations:
        graph_rewrite_configs.append(optimization + ":autotune:false")
    else:
      for optimization in autotune_only_optimizations:
        graph_rewrite_configs.append(optimization + ":autotune:true")

    return graph_rewrite_configs

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
    if self.filter_with_random_uniform_fusion is not None:
      pb.filter_with_random_uniform_fusion = (
          self.filter_with_random_uniform_fusion)
    if self.hoist_random_uniform is not None:
      pb.hoist_random_uniform = self.hoist_random_uniform
    if self.map_and_batch_fusion is not None:
      pb.map_and_batch_fusion = self.map_and_batch_fusion
    if self.map_and_filter_fusion is not None:
      pb.map_and_filter_fusion = self.map_and_filter_fusion
    if self.map_fusion is not None:
      pb.map_fusion = self.map_fusion
    if self.map_parallelization is not None:
      pb.map_parallelization = self.map_parallelization
    pb.map_vectorization.CopyFrom(self.map_vectorization._to_proto())  # pylint: disable=protected-access
    if self.noop_elimination is not None:
      pb.noop_elimination = self.noop_elimination
    if self.parallel_batch is not None:
      pb.parallel_batch = self.parallel_batch
    if self.reorder_data_discarding_ops is not None:
      pb.reorder_data_discarding_ops = self.reorder_data_discarding_ops
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
    if pb.WhichOneof("optional_filter_with_random_uniform_fusion") is not None:
      self.filter_with_random_uniform_fusion = (
          pb.filter_with_random_uniform_fusion)
    if pb.WhichOneof("optional_hoist_random_uniform") is not None:
      self.hoist_random_uniform = pb.hoist_random_uniform
    if pb.WhichOneof("optional_map_and_batch_fusion") is not None:
      self.map_and_batch_fusion = pb.map_and_batch_fusion
    if pb.WhichOneof("optional_map_and_filter_fusion") is not None:
      self.map_and_filter_fusion = pb.map_and_filter_fusion
    if pb.WhichOneof("optional_map_fusion") is not None:
      self.map_fusion = pb.map_fusion
    if pb.WhichOneof("optional_map_parallelization") is not None:
      self.map_parallelization = pb.map_parallelization
    self.map_vectorization._from_proto(pb.map_vectorization)  # pylint: disable=protected-access
    if pb.WhichOneof("optional_noop_elimination") is not None:
      self.noop_elimination = pb.noop_elimination
    if pb.WhichOneof("optional_parallel_batch") is not None:
      self.parallel_batch = pb.parallel_batch
    if pb.WhichOneof("optional_reorder_data_discarding_ops") is not None:
      self.reorder_data_discarding_ops = pb.reorder_data_discarding_ops
    if pb.WhichOneof("optional_shuffle_and_repeat_fusion") is not None:
      self.shuffle_and_repeat_fusion = pb.shuffle_and_repeat_fusion

  def _set_mutable(self, mutable):
    """Change the mutability value to `mutable` on this options and children."""
    # pylint: disable=protected-access
    object.__setattr__(self, "_mutable", mutable)
    self.map_vectorization._set_mutable(mutable)
