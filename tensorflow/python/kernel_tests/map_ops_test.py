# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for TensorMap ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import map_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class MapOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testEmptyTensorMapSize(self):
    m = map_ops.empty_tensor_map()
    s = map_ops.tensor_map_size(m)
    self.assertAllEqual(s, 0)

  def testTensorMapInsert(self):
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    v = constant_op.constant(2.0)
    m = map_ops.tensor_map_insert(m, k, v)
    s = map_ops.tensor_map_size(m)
    self.assertAllEqual(s, 1)

  def testTensorMapLookup(self):
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    v = constant_op.constant(2.0)
    m = map_ops.tensor_map_insert(m, k, v)
    l = map_ops.tensor_map_lookup(m, k, dtypes.float32)
    self.assertAllClose(l, v)

  def testTensorMapLookupFromEmptyMapFails(self):
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Trying to lookup non-existent key."):
      l = map_ops.tensor_map_lookup(m, k, dtypes.float32)
      self.evaluate(l)

  def testTensorMapLookupMissingKeyFails(self):
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    k2 = constant_op.constant(2.0)
    v = constant_op.constant(11.0)
    m = map_ops.tensor_map_insert(m, k, v)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Trying to lookup non-existent key."):
      l = map_ops.tensor_map_lookup(m, k2, dtypes.float32)
      self.evaluate(l)

  def testTensorMapErase(self):
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    v = constant_op.constant(2.0)
    m = map_ops.tensor_map_insert(m, k, v)
    s = map_ops.tensor_map_size(m)
    self.assertAllEqual(s, 1)

    m, e = map_ops.tensor_map_erase(m, k, dtypes.float32)
    s = map_ops.tensor_map_size(m)
    self.assertAllEqual(s, 0)
    self.assertAllClose(e, v)

  def testTensorMapEraseFromEmptyMapFails(self):
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Trying to erase non-existent item."):
      m, e = map_ops.tensor_map_erase(m, k, dtypes.float32)
      self.evaluate(e)

  def testTensorMapEraseMissingKeyFails(self):
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    k2 = constant_op.constant(2.0)
    v = constant_op.constant(2.0)
    m = map_ops.tensor_map_insert(m, k2, v)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Trying to erase non-existent item."):
      m, e = map_ops.tensor_map_erase(m, k, dtypes.float32)
      self.evaluate(e)

  def testTensorMapHasKey(self):
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    k2 = constant_op.constant(2.0)
    v = constant_op.constant(2.0)
    m = map_ops.tensor_map_insert(m, k, v)

    b = map_ops.tensor_map_has_key(m, k)
    b2 = map_ops.tensor_map_has_key(m, k2)
    self.assertAllEqual(b, True)
    self.assertAllEqual(b2, False)

  def testIfHasKeyLookup(self):
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    k2 = constant_op.constant(2.0)
    v = constant_op.constant(2.0)
    m = map_ops.tensor_map_insert(m, k, v)

    default_value = array_ops.zeros_like(v)
    l = control_flow_ops.cond(
        map_ops.tensor_map_has_key(m, k),
        lambda: map_ops.tensor_map_lookup(m, k, dtypes.float32),
        lambda: default_value)
    l2 = control_flow_ops.cond(
        map_ops.tensor_map_has_key(m, k2),
        lambda: map_ops.tensor_map_lookup(m, k, dtypes.float32),
        lambda: default_value)
    self.assertAllClose(l, v)
    self.assertAllClose(l2, default_value)

  def testInsertLookupGrad(self):
    with backprop.GradientTape() as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      v = constant_op.constant(11.0)
      tape.watch(v)
      m = map_ops.tensor_map_insert(m, k, v)
      l = map_ops.tensor_map_lookup(m, k, dtypes.float32)
      l *= 5
      g = tape.gradient(l, v)
      self.assertAllEqual(g, 5)

  def testMultipleInsertLookupGrad(self):
    with backprop.GradientTape(persistent=True) as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(2.0)
      k3 = constant_op.constant(3.0)
      v = constant_op.constant(11.0)
      v2 = constant_op.constant(12.0)
      v3 = constant_op.constant(13.0)
      tape.watch(v)
      tape.watch(v2)
      tape.watch(v3)
      m = map_ops.tensor_map_insert(m, k, v)
      m = map_ops.tensor_map_insert(m, k2, v2)
      m = map_ops.tensor_map_insert(m, k3, v3)
      l = map_ops.tensor_map_lookup(m, k, v.dtype)
      l2 = map_ops.tensor_map_lookup(m, k2, v2.dtype)
      l3 = map_ops.tensor_map_lookup(m, k3, v3.dtype)
      g = tape.gradient(l * 5, v)
      g2 = tape.gradient(l2 * 6, v2)
      g3 = tape.gradient(l3 * 7, v3)
      self.assertAllEqual(g, 5)
      self.assertAllEqual(g2, 6)
      self.assertAllEqual(g3, 7)
    del tape

  def testInsertLookupComposeGrad(self):
    with backprop.GradientTape() as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(2.0)
      v = constant_op.constant(11.0)
      tape.watch(v)
      m = map_ops.tensor_map_insert(m, k, v)
      l = map_ops.tensor_map_lookup(m, k, v.dtype)
      m = map_ops.tensor_map_insert(m, k2, l)
      l2 = map_ops.tensor_map_lookup(m, k2, l.dtype)
      g = tape.gradient(l2 * 5, v)
      self.assertAllEqual(g, 5)

  def testReplaceLookupGrad(self):
    with backprop.GradientTape(persistent=True) as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      v = constant_op.constant(11.0)
      v2 = constant_op.constant(22.0)
      tape.watch(v)
      tape.watch(v2)
      m = map_ops.tensor_map_insert(m, k, v)
      l = map_ops.tensor_map_lookup(m, k, v.dtype)
      self.assertAllClose(l, v)
      g = tape.gradient(l * 5, v)
      self.assertAllEqual(g, 5)
      m = map_ops.tensor_map_insert(m, k, v2)
      l2 = map_ops.tensor_map_lookup(m, k, v2.dtype)
      self.assertAllClose(l2, v2)
      g2 = tape.gradient(l2 * 6, v)
      g3 = tape.gradient(l2 * 7, v2)
      self.assertAllClose(g2, array_ops.zeros_like(v))
      self.assertAllClose(g3, 7)
    del tape

  def testDiffKeySameValueGrad(self):
    with backprop.GradientTape(persistent=True) as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(11.0)
      v = constant_op.constant(2.0)
      v2 = constant_op.constant(2.0)
      tape.watch(v)
      tape.watch(v2)
      m = map_ops.tensor_map_insert(m, k, v)
      m = map_ops.tensor_map_insert(m, k2, v)
      l = map_ops.tensor_map_lookup(m, k, v.dtype)
      l2 = map_ops.tensor_map_lookup(m, k2, v.dtype)
      g = tape.gradient(l + l2, v)
      self.assertAllEqual(g, 2)
      m = map_ops.tensor_map_insert(m, k2, v2)
      l2 = map_ops.tensor_map_lookup(m, k2, v2.dtype)
      g2 = tape.gradient(l + l2, v2)
      self.assertAllEqual(g2, 1)
    del tape

  def testLookupAddGrad(self):
    with backprop.GradientTape(persistent=True) as tape:
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(2.0)
      v = constant_op.constant(11.0)
      v2 = constant_op.constant(22.0)
      tape.watch(v)
      tape.watch(v2)
      m = map_ops.empty_tensor_map()
      m = map_ops.tensor_map_insert(m, k, v)
      m = map_ops.tensor_map_insert(m, k2, v2)
      l1 = map_ops.tensor_map_lookup(m, k, v.dtype)
      l2 = map_ops.tensor_map_lookup(m, k2, v2.dtype)
      g = tape.gradient(l1 + l2, [l1, l2])
      self.assertAllClose(g, [1, 1])
      g2 = tape.gradient(l1 + l2, [v, v2])
      self.assertAllClose(g2, [1, 1])
      g3 = tape.gradient(l1 + l2 * 4, v2)
      self.assertAllEqual(g3, 4)
    del tape

  def testLookupMultiplyGrad(self):
    with backprop.GradientTape(persistent=True) as tape:
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(2.0)
      v = constant_op.constant(11.0)
      v2 = constant_op.constant(22.0)
      tape.watch(v)
      tape.watch(v2)
      m = map_ops.empty_tensor_map()
      m = map_ops.tensor_map_insert(m, k, v)
      m = map_ops.tensor_map_insert(m, k2, v2)
      l1 = map_ops.tensor_map_lookup(m, k, v.dtype)
      l2 = map_ops.tensor_map_lookup(m, k2, v2.dtype)
      g = tape.gradient(l1 * l2, [v, v2])
      self.assertAllClose(g, [v2, v])
      g2 = tape.gradient(l1 * l1, v)
      self.assertAllClose(g2, 2 * v)
    del tape

  def testEraseSecondGrad(self):
    with backprop.GradientTape(persistent=True) as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(2.0)
      v = constant_op.constant(11.0)
      v2 = constant_op.constant(22.0)
      tape.watch(v)
      tape.watch(v2)
      m = map_ops.tensor_map_insert(m, k, v)
      m = map_ops.tensor_map_insert(m, k2, v2)
      m, e = map_ops.tensor_map_erase(m, k2, v2.dtype)
      l = map_ops.tensor_map_lookup(m, k, v.dtype)
      self.assertAllClose(l, v)
      self.assertAllClose(e, v2)
      g = tape.gradient(l * 5, v)
      self.assertAllEqual(g, 5)
      g2 = tape.gradient(e * 6, v2)
      self.assertAllEqual(g2, 6)
    del tape

  def testEraseFirstGrad(self):
    with backprop.GradientTape(persistent=True) as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(2.0)
      v = constant_op.constant(11.0)
      v2 = constant_op.constant(22.0)
      tape.watch(v)
      tape.watch(v2)
      m = map_ops.tensor_map_insert(m, k, v)
      l = map_ops.tensor_map_lookup(m, k, v.dtype)
      m = map_ops.tensor_map_insert(m, k2, v2)
      m, e = map_ops.tensor_map_erase(m, k, v.dtype)
      l2 = map_ops.tensor_map_lookup(m, k2, v2.dtype)
      self.assertAllClose(l2, v2)
      self.assertAllClose(e, v)
      g = tape.gradient(l * 5, v)
      self.assertAllEqual(g, 5)
      g2 = tape.gradient(l2 * 6, v2)
      self.assertAllEqual(g2, 6)
      g3 = tape.gradient(e * 7, v)
      self.assertAllEqual(g3, 7)
      m, e2 = map_ops.tensor_map_erase(m, k2, v2.dtype)
      g4 = tape.gradient(e2 * 8, v2)
      self.assertAllEqual(g4, 8)
    del tape

  def testEraseInsertComposedGrad(self):
    with backprop.GradientTape(persistent=True) as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(2.0)
      v = constant_op.constant(11.0)
      v2 = constant_op.constant(22.0)
      tape.watch(v)
      tape.watch(v2)
      m = map_ops.tensor_map_insert(m, k, v)
      m, e = map_ops.tensor_map_erase(m, k, v.dtype)
      m = map_ops.tensor_map_insert(m, k2, e)
      l = map_ops.tensor_map_lookup(m, k2, e.dtype)
      self.assertAllClose(e, v)
      self.assertAllClose(l, e)
      g = tape.gradient(l * 5, v)
      self.assertAllEqual(g, 5)
      g2 = tape.gradient(e * 6, v)
      self.assertAllEqual(g2, 6)
    del tape

  def testStringKeyGrad(self):
    with backprop.GradientTape(persistent=True) as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant("key")
      k2 = constant_op.constant("key2")
      v = constant_op.constant(2.0)
      v2 = constant_op.constant(22.0)
      tape.watch(v2)
      m = map_ops.tensor_map_insert(m, k2, v2)
      m = map_ops.tensor_map_insert(m, k, v)
      s = map_ops.tensor_map_size(m)
      self.assertAllEqual(s, 2)
      l = map_ops.tensor_map_lookup(m, k, v.dtype)
      self.assertAllClose(l, v)
      m = map_ops.tensor_map_insert(m, k, v2)
      l2 = map_ops.tensor_map_lookup(m, k, v2.dtype)
      self.assertAllClose(l2, v2)
      g = tape.gradient(l2 * 5, v2)
      self.assertAllEqual(g, 5)

      m, e = map_ops.tensor_map_erase(m, k, v2.dtype)
      s = map_ops.tensor_map_size(m)
      self.assertAllEqual(s, 1)
      self.assertAllClose(e, v2)
      g2 = tape.gradient(e * 6, v2)
      self.assertAllEqual(g2, 6)
    del tape

  def testStringValue(self):
    m = map_ops.empty_tensor_map()
    k = constant_op.constant("key")
    v = constant_op.constant("value")
    k2 = constant_op.constant(1.0)
    v2 = constant_op.constant(2.0)
    m = map_ops.tensor_map_insert(m, k, v)
    m = map_ops.tensor_map_insert(m, k2, v2)
    l = map_ops.tensor_map_lookup(m, k, v.dtype)
    self.assertAllEqual(l, v)
    l2 = map_ops.tensor_map_lookup(m, k2, v2.dtype)
    self.assertAllClose(l2, v2)
    m, e = map_ops.tensor_map_erase(m, k, v.dtype)
    self.assertAllEqual(e, v)

  def testVectorValue(self):
    m = map_ops.empty_tensor_map()
    k = constant_op.constant([1.0, 2.0])
    v = constant_op.constant([11.0, 22.0])
    m = map_ops.tensor_map_insert(m, k, v)
    s = map_ops.tensor_map_size(m)
    self.assertAllEqual(s, 1)
    l = map_ops.tensor_map_lookup(m, k, v.dtype)
    self.assertAllEqual(l, v)

    m, e = map_ops.tensor_map_erase(m, k, v.dtype)
    s = map_ops.tensor_map_size(m)
    self.assertAllEqual(s, 0)
    self.assertAllClose(e, v)


if __name__ == "__main__":
  test.main()
