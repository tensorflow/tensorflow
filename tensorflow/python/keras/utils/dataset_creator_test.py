# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for dataset_creator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.layers import core as core_layers
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.utils import dataset_creator
from tensorflow.python.platform import test
from tensorflow.python.training.server_lib import ClusterSpec


class DatasetCreatorTest(test.TestCase):

  def test_dataset_creator(self):
    with self.assertRaisesRegex(
        TypeError, "`dataset_fn` for `DatasetCreator` must be a `callable`."):
      dataset_creator.DatasetCreator(2)

    dataset_fn = lambda: 3
    with self.assertRaisesRegex(
        TypeError, "The `callable` provided to `DatasetCreator` must return "
        "a Dataset."):
      dataset_creator.DatasetCreator(dataset_fn)()

    dataset_fn = lambda: dataset_ops.DatasetV2.from_tensor_slices([1, 1])
    got = dataset_creator.DatasetCreator(dataset_fn)()
    self.assertEqual(
        next(iter(got)),
        next(iter(dataset_ops.DatasetV2.from_tensor_slices([1, 1]))))

  def test_dataset_creator_usage_in_parameter_server_model_fit(self):
    cluster_def = multi_worker_test_base.create_in_process_cluster(
        num_workers=2, num_ps=1, rpc_layer="grpc")
    cluster_def["chief"] = [
        "localhost:%d" % multi_worker_test_base.pick_unused_port()
    ]
    strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        SimpleClusterResolver(ClusterSpec(cluster_def), rpc_layer="grpc"))
    with strategy.scope():
      model = sequential.Sequential([core_layers.Dense(10)])
    model.compile(gradient_descent.SGD(), loss="mse")

    def dataset_fn(input_context):
      global_batch_size = 64
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      dataset = dataset_ops.DatasetV2.from_tensors(([1.], [1.])).repeat()
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
      dataset = dataset.batch(batch_size)
      dataset = dataset.prefetch(2)
      return dataset

    history = model.fit(
        dataset_creator.DatasetCreator(dataset_fn),
        epochs=10,
        steps_per_epoch=10,
        verbose=0)
    self.assertLen(history.history["loss"], 10)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
