# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for evaluation using Keras model and ParameterServerStrategy."""

import time

from tensorflow.python import keras
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.util import nest


# TODO(yuefengz): move the following implementation to Keras core.
class KerasMetricTypeSpec(type_spec.TypeSpec):

  def __init__(self, cls, config, weights):
    self._cls = cls
    self._config = config
    self._weights = weights

  def _serialize(self):
    return (self._cls.__name__, self._config)

  @property
  def value_type(self):
    return self._cls

  def most_specific_compatible_type(self, other):
    if (type(self) is not type(other) or self._cls != other._cls or
        self._config != other._config):
      raise ValueError("No TypeSpec is compatible with both %s and %s" %
                       (self, other))
    return KerasMetricTypeSpec(self._cls, self._config, self._weights)

  @property
  def _component_specs(self):
    ret = []
    for w in self._weights:
      ret.append(
          resource_variable_ops.VariableSpec(
              w.shape, w.dtype, w.name.split(":")[0], trainable=False))
    return ret

  def _to_components(self, value):
    return value.weights

  def _from_components(self, weights):
    counter = [0]

    def fetch_variable(next_creator, **kwargs):
      del next_creator, kwargs
      # TODO(yuefengz): verify the var creation order matches the weights
      # property
      var = weights[counter[0]]
      counter[0] += 1
      return var

    with variable_scope.variable_creator_scope(fetch_variable):
      ret = self._cls.from_config(self._config)
    assert len(weights) == len(ret.weights)
    return ret


class MeanMetricAsCompositeTensor(keras.metrics.Mean,
                                  composite_tensor.CompositeTensor):

  def element_spec(self):
    raise NotImplementedError("element_spec not implemented")

  @property
  def _type_spec(self):
    return KerasMetricTypeSpec(self.__class__, self.get_config(), self.weights)


class EvaluationTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(EvaluationTest, cls).setUpClass()
    cls._cluster = multi_worker_test_base.create_multi_process_cluster(
        num_workers=3, num_ps=2, rpc_layer="grpc")
    cls._cluster_def = cls._cluster.cluster_resolver.cluster_spec().as_dict()
    cluster_resolver = SimpleClusterResolver(
        server_lib.ClusterSpec(cls._cluster_def), rpc_layer="grpc")

    cls.strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        cluster_resolver)
    cls.cluster_coord = coordinator_lib.ClusterCoordinator(cls.strategy)

  @classmethod
  def tearDownClass(cls):
    cls._cluster.stop()
    cls._cluster = None
    super(EvaluationTest, cls).tearDownClass()

  def testPassMetricToTfFunction(self):
    metric1 = MeanMetricAsCompositeTensor()
    metric2 = MeanMetricAsCompositeTensor()

    self.assertEqual(metric1.result(), 0.0)
    self.assertEqual(metric2.result(), 0.0)

    nest.assert_same_structure(
        metric1, metric2._type_spec, expand_composites=True)
    nest.assert_same_structure(
        metric1._type_spec, metric2, expand_composites=True)

    @def_function.function
    def func(m):
      m.update_state([1.0, 2.0])

    func(metric1)
    self.assertEqual(metric1.result(), 1.5)
    self.assertEqual(metric2.result(), 0.0)

    concrete_f = func.get_concrete_function(metric1._type_spec)
    concrete_f(metric2)
    self.assertEqual(metric1.result(), 1.5)
    self.assertEqual(metric2.result(), 1.5)

  def testModelEvaluatePrototype(self):

    def metric_fn():
      return MeanMetricAsCompositeTensor()

    # TODO(yuefengz): make _create_per_worker_resources public and get rid of
    # the type_spec hack.
    per_worker_metric = self.cluster_coord._create_per_worker_resources(
        metric_fn)

    metric_on_coordinator = metric_fn()

    for metric_remote_value in per_worker_metric._values:
      metric_remote_value._type_spec = metric_on_coordinator._type_spec

    def dataset_fn():
      return dataset_ops.DatasetV2.range(1024)

    # TODO(yuefengz): integrate it into model.evaluate.

    @def_function.function
    def eval_fn(total_shard, shard_id, metric):
      metric.reset_states()
      dataset_shard = dataset_fn().shard(total_shard, shard_id)
      for i in dataset_shard:
        metric.update_state(i)

      # TODO(yuefengz): we should return the internal state of the metric and
      # then use the combiner API.
      return metric.result()

    total_shards = 128
    result_remote_values = []
    for i in range(total_shards):
      result_remote_values.append(
          self.cluster_coord.schedule(
              eval_fn, args=(total_shards, i, per_worker_metric)))

    self._cluster.kill_task("worker", 0)
    self._cluster.kill_task("worker", 1)
    time.sleep(1)
    self._cluster.start_task("worker", 0)
    self._cluster.start_task("worker", 1)

    results = [r.fetch() for r in result_remote_values]
    result = sum(results) / len(results)
    self.assertEqual(result, 511.5)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
