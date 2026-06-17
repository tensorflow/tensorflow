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

if __name__ == "__main__":
  googletest.main()
