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
"""StatsOptions to configure stats aggregation options for `tf.data` pipelines.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops import stats_aggregator
from tensorflow.python.data.util import options
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.StatsOptions")
class StatsOptions(options.OptionsBase):
  """Represents options for collecting dataset stats using `StatsAggregator`.

  To apply `StatsOptions` with a `tf.data.Dataset` object, use the following
  pattern:

  ```python
  aggregator = tf.data.experimental.StatsAggregator()

  options = tf.data.Options()
  options.experimental_stats = tf.data.experimental.StatsOptions()
  options.experimental_stats.aggregator = aggregator
  dataset = dataset.with_options(options)
  ```

  Note: a `StatsAggregator` object can be attached either duing construction or
  can be provided later like in above example.

  ```python
  aggretator = tf.data.experimental.StatsAggregator()
  # attach aggregator during construction
  options.experimental_stats = tf.data.experimental.StatsOptions(aggregator)
  .....
  ```
  """

  aggregator = options.create_option(
      name="aggregator",
      ty=stats_aggregator.StatsAggregator,
      docstring=
      "Associates the given statistics aggregator with the dataset pipeline.")

  prefix = options.create_option(
      name="prefix",
      ty=str,
      docstring=
      "Prefix to prepend all statistics recorded for the input `dataset` with.",
      default="")

  counter_prefix = options.create_option(
      name="counter_prefix",
      ty=str,
      docstring=
      "Prefix for the statistics recorded as counter.",
      default="")

  latency_all_edges = options.create_option(
      name="latency_all_edges",
      ty=bool,
      docstring=
      "Whether to add latency measurements on all edges.",
      default=True)
