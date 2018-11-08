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
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.StatsOptions")
class StatsOptions(object):
  """Represents options for collecting dataset stats using `StatsAggregator`.

  To apply `StatsOptions` with a `tf.data.Dataset` object, use the following
  pattern:

  ```python
  aggretator = tf.data.experimental.StatsAggregator()

  options = dataset_ops.Options()
  options.experimental_stats = tf.data.experimental.StatsOptions()
  options.experimental_stats.aggregator = aggregator
  dataset = dataset.with_options(options)

  iterator = dataset.make_one_shot_iterator()
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

  for _name, _ty, _default, _docstring in [
      ("aggregator", stats_aggregator.StatsAggregator, None,
       "Associate the given statistics options with the dataset pipeline."),
      ("prefix", str, "",
       "Prefix to prepend all statistics recorded for the input `dataset` with."
      ),
      ("counter_prefix", str, "",
       "Prefix for the statistics recorded as counter."),
      ("latency_all_edges", bool, True,
       "Whether to add latency measurements on all edges."),
  ]:

    def _make_getter(name):  # pylint: disable=no-self-argument

      def getter(self):
        return getattr(self, "_" + name)

      return getter

    def _make_setter(name, ty):  # pylint: disable=no-self-argument

      def setter(self, value):
        if not isinstance(value, ty):
          raise TypeError(
              "Attempting to set the option %s to incompatible value: %r when "
              "it expects  %r" % (name, value, ty))
        setattr(self, "_" + name, value)

      return setter

    vars()["_" + _name] = _default
    vars()[_name] = property(
        _make_getter(_name), _make_setter(_name, _ty), _default, _docstring)

  def __init__(self, aggregator=None):
    if aggregator:
      self.aggregator = aggregator

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.__dict__ == other.__dict__
    else:
      return False

  def __ne__(self, other):
    return not self.__eq__(other)

  def __str__(self):
    return str(self.__dict__)
