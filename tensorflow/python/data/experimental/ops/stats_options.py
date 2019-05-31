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

  You can set the stats options of a dataset through the `experimental_stats`
  property of `tf.data.Options`; the property is an instance of
  `tf.data.experimental.StatsOptions`. For example, to collect latency stats
  on all dataset edges, use the following pattern:

  ```python
  aggregator = tf.data.experimental.StatsAggregator()

  options = tf.data.Options()
  options.experimental_stats.aggregator = aggregator
  options.experimental_stats.latency_all_edges = True
  dataset = dataset.with_options(options)
  ```
  """

  aggregator = options.create_option(
      name="aggregator",
      ty=(stats_aggregator.StatsAggregatorV2,
          stats_aggregator.StatsAggregatorV1),
      docstring=
      "Associates the given statistics aggregator with the dataset pipeline.")

  prefix = options.create_option(
      name="prefix",
      ty=str,
      docstring=
      "Prefix to prepend all statistics recorded for the input `dataset` with.",
      default_factory=lambda: "")

  counter_prefix = options.create_option(
      name="counter_prefix",
      ty=str,
      docstring="Prefix for the statistics recorded as counter.",
      default_factory=lambda: "")

  latency_all_edges = options.create_option(
      name="latency_all_edges",
      ty=bool,
      docstring=
      "Whether to add latency measurements on all edges. Defaults to False.")
