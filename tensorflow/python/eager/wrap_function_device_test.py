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

from absl.testing import parameterized
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer as graph_def_importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


def _dataset_reduce_sum(dataset):
  return dataset.reduce(
      constant_op.constant(0, dtype=dtypes.int64), lambda x, y: x + y)


def _loop_dataset_sum(dataset):
  value = constant_op.constant(0, dtype=dtypes.int64)
  for d in dataset:
    value += d
  return value


def _iter_dataset_sum(dataset):
  value = constant_op.constant(0, dtype=dtypes.int64)
  for d in iter(dataset):
    value += d
  return value


class WrappedGraphTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('cpu_reduce', 'CPU', _dataset_reduce_sum),
      ('gpu_reduce', 'GPU', _dataset_reduce_sum),
      ('cpu_loop', 'CPU', _loop_dataset_sum),
      ('gpu_loop', 'GPU', _loop_dataset_sum),
      ('cpu_iter', 'CPU', _iter_dataset_sum),
      ('gpu_iter', 'GPU', _iter_dataset_sum),
  )
  def testWrapFuncDatasetDevice(self, device_type, dataset_reduce_fn):

    devices = config.list_logical_devices(device_type=device_type)
    if not devices:
      self.skipTest('Skip when {} is not detected by TF'.format(device_type))

    @def_function.function
    def comp():
      return dataset_reduce_fn(dataset_ops.Dataset.range(10))

    graph = comp.get_concrete_function().graph

    def function_to_wrap():
      with ops.device(devices[0].name):
        return graph_def_importer.import_graph_def(graph.as_graph_def())

    with ops.device(devices[0].name):
      wrapped_noarg_fn = wrap_function.wrap_function(
          function_to_wrap, signature=[])

    wrapped_noarg_fn()


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
