# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for StructuredTensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.structured import structured_tensor
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class StructuredTensorTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):

  def assertAllEqual(self, a, b, msg=None):
    if not (isinstance(a, structured_tensor.StructuredTensor) or
            isinstance(b, structured_tensor.StructuredTensor)):
      return super(StructuredTensorTest, self).assertAllEqual(a, b, msg)
    if not (isinstance(a, structured_tensor.StructuredTensor) and
            isinstance(b, structured_tensor.StructuredTensor)):
      # TODO(edloper) Add support for this once structured_factory_ops is added.
      raise ValueError("Not supported yet")

    self.assertEqual(repr(a.shape), repr(b.shape))
    self.assertEqual(set(a.field_names()), set(b.field_names()))
    for field in a.field_names():
      self.assertAllEqual(a.field_value(field), b.field_value(field))

  @parameterized.parameters([
      {
          "shape": [],
          "fields": {},
      },
      {
          "shape": [None],
          "fields": {},
      },
      {
          "shape": [1, 5, 3],
          "fields": {},
      },
      {
          "shape": [],
          "fields": {"Foo": 5, "Bar": [1, 2, 3]},
      },
      {
          "shape": [2],
          "fields": {"x": [1, 2], "y": [[1, 2], [3, 4]]},
      },
      {
          "shape": [None],
          "fields": {"x": [1, 2], "y": [[1, 2], [3, 4]]},
          "expected_shape": [2],  # inferred from field values.
      },
      {
          "shape": [],
          "fields": {
              "r": ragged_factory_ops.constant_value([[1, 2], [3]]),
          },
      },
      {
          "shape": [2],
          "fields": {
              "r": ragged_factory_ops.constant_value([[1, 2], [3]]),
          },
      },
  ])  # pyformat: disable
  def testConstruction(self, shape, fields, expected_shape=None):
    struct = structured_tensor.StructuredTensor.from_fields(shape, fields)
    if expected_shape is None:
      expected_shape = shape
    self.assertEqual(struct.shape.as_list(), expected_shape)
    self.assertLen(expected_shape, struct.rank)
    self.assertEqual(struct.field_names(), tuple(fields.keys()))
    for field, value in fields.items():
      self.assertIsInstance(
          struct.field_value(field),
          (ops.Tensor, structured_tensor.StructuredTensor,
           ragged_tensor.RaggedTensor))
      self.assertAllEqual(struct.field_value(field), value)

  def testNestedStructConstruction(self):
    rt = ragged_factory_ops.constant([[1, 2], [3]])
    struct1 = structured_tensor.StructuredTensor.from_fields([], {"x": [1, 2]})
    struct2 = structured_tensor.StructuredTensor.from_fields([2], {"x": [1, 2]})
    struct3 = structured_tensor.StructuredTensor.from_fields([], {
        "r": rt,
        "s": struct1
    })
    struct4 = structured_tensor.StructuredTensor.from_fields([2], {
        "r": rt,
        "s": struct2
    })

    self.assertEqual(struct3.shape.as_list(), [])
    self.assertEqual(struct3.rank, 0)
    self.assertEqual(set(struct3.field_names()), set(["r", "s"]))
    self.assertAllEqual(struct3.field_value("r"), rt)
    self.assertAllEqual(struct3.field_value("s"), struct1)

    self.assertEqual(struct4.shape.as_list(), [2])
    self.assertEqual(struct4.rank, 1)
    self.assertEqual(set(struct4.field_names()), set(["r", "s"]))
    self.assertAllEqual(struct4.field_value("r"), rt)
    self.assertAllEqual(struct4.field_value("s"), struct2)

  @parameterized.parameters([
      (object(), {}, TypeError),
      ([], object(), TypeError, "fields must be a dictionary"),
      ([], {1: 2}, TypeError, "Unexpected type for key"),
      ([], {"x": object()}, TypeError, "Unexpected type for value"),
      (None, {}, ValueError, "StructuredTensor's shape must have known rank"),
      ([5], {"f": 5}, ValueError, r"Shapes \(5,\) and \(\) are not compatible"),
      ([None], {"x": [1], "y": []}, ValueError,
       r"Shapes \([01],\) and \([01],\) are not compatible"),
      ([], {"": 5}, ValueError, "Field name '' is not currently allowed."),
      ([], {"_": 5}, ValueError, "Field name '_' is not currently allowed."),
  ])  # pyformat: disable
  def testConstructionErrors(self, shape, fields, err, msg=None):
    with self.assertRaisesRegexp(err, msg):
      structured_tensor.StructuredTensor.from_fields(shape, fields)


if __name__ == "__main__":
  googletest.main()
