# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for c_api utils."""
import gc
from unittest import mock
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class ApiDefMapTest(test_util.TensorFlowTestCase):

  def testApiDefMapOpNames(self):
    api_def_map = c_api_util.ApiDefMap()
    self.assertIn("Add", api_def_map.op_names())

  def testApiDefMapGet(self):
    api_def_map = c_api_util.ApiDefMap()
    op_def = api_def_map.get_op_def("Add")
    self.assertEqual(op_def.name, "Add")
    api_def = api_def_map.get_api_def("Add")
    self.assertEqual(api_def.graph_op_name, "Add")

  def testApiDefMapGetInvalidOp(self):
    api_def_map = c_api_util.ApiDefMap()
    with self.assertRaises(ValueError):
      api_def_map.get_api_def("InvalidOperationName")

  def testApiDefMapPutThenGet(self):
    api_def_map = c_api_util.ApiDefMap()
    api_def_text = """
op {
  graph_op_name: "Add"
  summary: "Returns x + y element-wise."
  description: <<END
*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
END
}
"""
    api_def_map.put_api_def(api_def_text)
    api_def = api_def_map.get_api_def("Add")
    self.assertEqual(api_def.graph_op_name, "Add")
    self.assertEqual(api_def.summary, "Returns x + y element-wise.")


class UniquePtrTest(test_util.TensorFlowTestCase):

  def setUp(self):

    super(UniquePtrTest, self).setUp()

    class MockClass:

      def __init__(self):
        self.deleted = False

    def deleter(obj):
      obj.deleted = True

    self.obj = MockClass()
    self.deleter = deleter

  def testLifeCycle(self):
    self.assertFalse(self.obj.deleted)

    a = c_api_util.UniquePtr(name="mock", deleter=self.deleter, obj=self.obj)

    with a.get() as obj:
      self.assertIs(obj, self.obj)

    del a
    gc.collect()
    self.assertTrue(self.obj.deleted)

  def testSafeUnderRaceCondition(self):
    self.assertFalse(self.obj.deleted)

    a = c_api_util.UniquePtr(name="mock", deleter=self.deleter, obj=self.obj)

    with a.get() as obj:
      self.assertIs(obj, self.obj)
      # The del below mimics a potential race condition.
      # 'a' could be owned by a different thread, and this thread not
      # necessarily hold a long-term reference to a.
      del a
      gc.collect()
      self.assertFalse(obj.deleted)

    gc.collect()
    self.assertTrue(self.obj.deleted)

  def testRaiseAfterDeleted(self):
    self.assertFalse(self.obj.deleted)

    a = c_api_util.UniquePtr(name="mock", deleter=self.deleter, obj=self.obj)

    # The __del__ below mimics a partially started deletion, potentially
    # started from another thread.
    # 'a' could be owned by a different thread, and this thread not
    # necessarily hold a long-term reference to a.
    a.__del__()
    self.assertTrue(self.obj.deleted)

    with self.assertRaisesRegex(c_api_util.AlreadyGarbageCollectedError,
                                "MockClass"):
      with a.get():
        pass

    gc.collect()
    self.assertTrue(self.obj.deleted)


class ScopedTFBufferTest(test_util.TensorFlowTestCase):

  def testDeleteRunsUnderNormalConditions(self):
    # Mock TF_DeleteBuffer rather than letting __del__ actually free the
    # buffer here: calling __del__() directly on a live ScopedTFBuffer frees
    # the real C buffer immediately, and the object's own garbage collection
    # later calls __del__() again for the same (already-freed) buffer,
    # double-freeing it. Mocking out the delete call lets us call __del__()
    # directly (CPython prints and swallows any exception raised inside a
    # GC-triggered __del__ instead of propagating it, so a plain
    # `del buf; gc.collect()` would let a broken destructor pass silently)
    # while leaving the real deletion to happen exactly once, naturally,
    # when the object is actually collected.
    with mock.patch.object(c_api_util.c_api, "TF_DeleteBuffer") as mock_delete:
      buf = c_api_util.ScopedTFBuffer(b"test")
      buf.__del__()  # should not raise
      mock_delete.assert_called_once()

  def testDeleteGuardsAgainstNoneCApi(self):
    buf = c_api_util.ScopedTFBuffer(b"test")
    # Regression test for the __del__ fix: during interpreter shutdown,
    # `c_api_util.c_api` can already have been torn down to None by the time
    # this object is collected, so __del__ must guard against that rather
    # than unconditionally calling into it.
    with mock.patch.object(c_api_util, "c_api", None):
      buf.__del__()  # should not raise

  def testDeletePartiallyInitializedObjectDoesNotRaise(self):
    # Regression test: if __init__ raises before `self.buffer` is ever
    # assigned (e.g. compat.as_bytes failing on bad input), __del__ still
    # runs on the partially-constructed object. It must not raise an
    # AttributeError from accessing the missing `self.buffer`, which would
    # mask whatever real exception caused __init__ to fail.
    buf = c_api_util.ScopedTFBuffer.__new__(c_api_util.ScopedTFBuffer)
    buf.__del__()  # should not raise

if __name__ == "__main__":
  googletest.main()
