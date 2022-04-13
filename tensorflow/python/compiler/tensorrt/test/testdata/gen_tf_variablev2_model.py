# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Saves a TensorFlow model containing VariableV2 nodes.

   The saved model is loaded and executed by tests to check that TF-TRT can
   successfully convert and execute models with VariableV2 without freezing.
"""

from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

ops.disable_eager_execution()
variable_scope.disable_resource_variables()

def GenerateModelWithVariableV2(tf_saved_model_dir):
  """Generate a model with a VariableV2 node using TFv1 API."""

  def SimpleModel():
    """Define model with a TF graph."""

    def GraphFn():
      input1 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input1")
      input2 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input2")
      var1 = variable_scope.get_variable(
          'var1', shape=[1, 1, 1], initializer=init_ops.constant_initializer(
              [[[13.]]]),
          dtype=dtypes.float32)
      var2 = variable_scope.get_variable(
          'var2', shape=[1, 1, 1], initializer=init_ops.constant_initializer(
              [[[37.]]]),
          dtype=dtypes.float32)
      mul1 = input1 * var1
      mul2 = input2 * var2
      add = mul1 + mul2
      sub = add - 45.
      out = array_ops.identity(sub, name="output")
      return g, input1, input2, out

    g = ops.Graph()
    with g.as_default():
      return GraphFn()

  g, input1, input2, out = SimpleModel()
  signature_def = signature_def_utils.build_signature_def(
      inputs={
          "input1": utils.build_tensor_info(input1),
          "input2": utils.build_tensor_info(input2)
      },
      outputs={"output": utils.build_tensor_info(out)},
      method_name=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
  saved_model_builder = builder.SavedModelBuilder(tf_saved_model_dir)
  with session.Session(graph=g) as sess:
    variables.global_variables_initializer().run()
    saved_model_builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
        })
  saved_model_builder.save()

if __name__ == "__main__":
  GenerateModelWithVariableV2(
      tf_saved_model_dir="tf_variablev2_saved_model")
