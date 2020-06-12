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
    if self.enabled:
      return ["map_vectorization"]
    return []

  def _graph_rewrite_configs(self):
    if not self.enabled:
      return []
    if self.use_choose_fastest:
      return ["map_vectorization:use_choose_fastest:true"]
    else:
      return ["map_vectorization:use_choose_fastest:false"]


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

  hoist_data_discarding_ops = options.create_option(
      name="hoist_data_discarding_ops",
      ty=bool,
      docstring=
      "Whether to hoist ops that will discard data (such as skip, take, shard)"
      "out of map transformations. If None, defaults to False.")

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
      docstring="Whether to parallelize copying of batch elements. If None, "
      "defaults to False.")

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

    # Set these options if they are explicitly set by the user.
    if self.autotune is False:  # pylint: disable=g-bool-id-comparison
      autotune = False
    if self.autotune_cpu_budget is not None:
      cpu_budget = self.autotune_cpu_budget

    return autotune, algorithm, cpu_budget

  def _graph_rewrites(self):
    """Produces the list of enabled graph optimizations."""
    result = set()
    all_optimizations = [
        "filter_fusion",
        "filter_with_random_uniform_fusion",
        "hoist_data_discarding_ops",
        "hoist_random_uniform",
        "map_and_batch_fusion",
        "map_and_filter_fusion",
        "map_parallelization",
        "map_fusion",
        "noop_elimination",
        "parallel_batch",
        "shuffle_and_repeat_fusion",
    ]
    for optimization in all_optimizations:
      if getattr(self, optimization):
        result.add(optimization)

    if self.apply_default_optimizations is not False:
      # The following optimizations are turned on by default, unless the user
      # explicitly disables them.
      optimizations_to_disable = [
          "map_and_batch_fusion",
          "noop_elimination",
          "shuffle_and_repeat_fusion",
      ]
      for optimization in optimizations_to_disable:
        if getattr(self, optimization) is not False:
          result.add(optimization)

    if self.map_vectorization is not None:
      result.update(self.map_vectorization._graph_rewrites())  # pylint: disable=protected-access

    autotune_buffers = self._autotune_buffers()
    if self.autotune is not False and autotune_buffers:  # pylint: disable=g-bool-id-comparison
      # When autotuning buffer sizes is enabled, we inject a `prefetch`
      # transformation after asynchronous dataset ops. Only the buffer sizes of
      # prefetch transformations will be autotuned, though this is practically
      # equivalent to tuning the buffer sizes of the other asynchronous
      # transformations.
      result.add("inject_prefetch")
    return sorted(list(result))

  def _graph_rewrite_configs(self):
    if self.map_vectorization is not None:
      return self.map_vectorization._graph_rewrite_configs()  # pylint: disable=protected-access
    return []
