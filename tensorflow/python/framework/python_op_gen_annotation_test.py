# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Test that type annotations are generated."""

import inspect
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.platform import googletest


class PythonOpGetTest(googletest.TestCase):

  def test_type_annotation_not_empty_for_internal_op(self):
    for internal_op in [
        data_flow_ops.dynamic_stitch,
        gen_nn_ops._fused_batch_norm,
        gen_math_ops.add,
    ]:
      sig = inspect.signature(internal_op)
      for key in sig.parameters:
        if key == "name":
          continue
        assert sig.parameters[key].annotation != inspect.Signature.empty

  def test_type_annotation_empty_for_imported_op(self):
    for imported_op in [
        data_flow_ops.DynamicStitch,
        gen_nn_ops.FusedBatchNorm,
        gen_math_ops.Add,
    ]:
      sig = inspect.signature(imported_op)
      for key in sig.parameters:
        if key == "name":
          continue
        assert sig.parameters[key].annotation == inspect.Signature.empty


if __name__ == "__main__":
  googletest.main()
