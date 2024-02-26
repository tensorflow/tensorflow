# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.data.experimental.ops.iterator_ops."""

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import iterator_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import test


class IteratorOpsTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.v2_eager_only_combinations())
  def test_get_model_proto_not_empty(self):
    options = dataset_ops.options_lib.Options()
    options.autotune.enabled = True

    dataset = dataset_ops.Dataset.range(100000)
    dataset = dataset.map(lambda x: ("tf.data", "is the best ML data loader."))
    dataset = dataset.batch(5, drop_remainder=True, num_parallel_calls=1)
    dataset = dataset.with_options(options)

    iterator = iter(dataset)
    model_proto = iterator_ops.get_model_proto(iterator)
    dataset_names = set(
        model_proto.nodes[key].name for key in model_proto.nodes
    )

    self.assertNotEmpty(
        dataset_names,
        "The model proto from the iterator should contain at least 1"
        " dataset op.",
    )

  @combinations.generate(test_base.v2_eager_only_combinations())
  def test_get_model_proto_error_when_autotune_not_enabled(self):
    options = dataset_ops.options_lib.Options()
    options.autotune.enabled = False

    dataset = dataset_ops.Dataset.range(100000)
    dataset = dataset.map(lambda x: ("tf.data", "is the best ML data loader."))
    dataset = dataset.batch(5, drop_remainder=True, num_parallel_calls=1)
    dataset = dataset.with_options(options)

    iterator = iter(dataset)

    with self.assertRaisesRegex(
        errors_impl.NotFoundError,
        "Did you disable autotune",
    ):
      _ = iterator_ops.get_model_proto(iterator)

  @combinations.generate(test_base.graph_only_combinations())
  def test_get_model_proto_unsupported_in_graph_mode(self):
    dataset = dataset_ops.Dataset.range(100000)

    iterator = dataset_ops.make_one_shot_iterator(dataset)
    with self.session():
      with self.assertRaises(ValueError):
        _ = iterator_ops.get_model_proto(iterator)


if __name__ == "__main__":
  test.main()
