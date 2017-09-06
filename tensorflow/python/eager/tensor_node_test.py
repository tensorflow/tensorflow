# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import tensor
from tensorflow.python.eager import tensor_node
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util


def public_or_operator(n):
  if not n.startswith("_"):
    return True
  return n.startswith("__") and n.endswith("__")


class TensorNodeTest(test_util.TensorFlowTestCase):

  # TensorNode must implement autograd core's Node interface, which
  # it does via inheritance. It also needs to be duck-typeable as
  # a tensorflow.python.framework.ops.EagerTensor.
  #
  # This is a "test" to help ensure interface compatibility.
  def testCanBeATensor(self):
    # TODO(ashankar,apassos): This list of "exceptions" - list of
    # Tensor methods not implemented by TensorNode needs to be
    # trimmed.
    exceptions = set([
        "OVERLOADABLE_OPERATORS",
        "__and__",
        "__del__",
        "__dict__",
        "__iter__",
        "__len__",
        "__matmul__",
        "__or__",
        "__rand__",
        "__rmatmul__",
        "__ror__",
        "__rxor__",
        "__weakref__",
        "__xor__",
        # BEGIN: Methods of Tensor that EagerTensor raises exceptions on.
        # But perhaps TensorNode should defer to "self.value.<method>" for
        # them?
        "consumers",
        "eval",
        "graph",
        "name",
        "op",
        "set_shape",
        "value_index",
        # END: Methods of Tensor that EagerTensor raises exceptions on.
    ])

    tensor_dir = dir(tensor.Tensor)
    tensor_dir = filter(public_or_operator, tensor_dir)
    tensor_dir = set(tensor_dir).difference(exceptions)

    tensor_node_dir = set(dir(tensor_node.TensorNode))

    missing = tensor_dir.difference(tensor_node_dir)
    self.assertEqual(
        0,
        len(missing),
        msg="Methods/properties missing in TensorNode: {}".format(missing))


if __name__ == "__main__":
  test.main()
