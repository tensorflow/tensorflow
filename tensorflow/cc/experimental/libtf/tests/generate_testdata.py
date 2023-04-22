# /* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================*/
r"""Generate models in testdata for use in tests.

If this script is being run via `<build-cmd> run`, pass an absolute path.
Otherwise, this script will attempt to write to a non-writable directory.

Example:
<build-cmd> run //third_party/tensorflow/cc/experimental/libtf:generate_testdata
 -- \
 --path`pwd`/third_party/tensorflow/cc/experimental/libtf/tests/testdata/ \
 --model_name=simple-model
"""
import os

from absl import app
from absl import flags

from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import saved_model

TESTDATA_PATH = flags.DEFINE_string(
    "path", None, help="Path to testdata directory.")

MODEL_NAME = flags.DEFINE_string(
    "model_name", None, help="Name of model to generate.")


class DataStructureModel(module.Module):
  """Model used for testing data structures in the C++ API."""

  def __init__(self):
    self.arr1 = [1.]
    self.const_arr = [constant_op.constant(1.)]
    self.var_arr = [variables.Variable(1.), variables.Variable(2.)]
    self.dict1 = {"a": 1.}
    self.var_dict = {"a": variables.Variable(1.), "b": variables.Variable(2.)}


class SimpleModel(module.Module):
  """A simple model used for exercising the C++ API."""

  @def_function.function(input_signature=[
      tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
  ])
  def test_float(self, x):
    return constant_op.constant(3.0) * x

  @def_function.function(input_signature=[
      tensor_spec.TensorSpec(shape=(), dtype=dtypes.int32),
  ])
  def test_int(self, x):
    return constant_op.constant(3) * x

  @def_function.function(input_signature=[
      tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
      tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32),
  ])
  def test_add(self, x, y):
    # Test a function with multiple arguments.
    return x + y


TEST_MODELS = {
    "simple-model": SimpleModel,
    "data-structure-model": DataStructureModel
}


def get_model(name):
  if name not in TEST_MODELS:
    raise ValueError("Model name '{}' not in TEST_MODELS")
  return TEST_MODELS[name]()


def main(unused_argv):

  model = get_model(MODEL_NAME.value)
  path = os.path.join(TESTDATA_PATH.value, MODEL_NAME.value)
  saved_model.save(model, path)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  flags.mark_flag_as_required("path")
  flags.mark_flag_as_required("model_name")
  app.run(main)
