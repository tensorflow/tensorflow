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
# =============================================================================
"""Saves a SavedModel after TensorRT conversion.
   The saved model is loaded and executed by tests to ensure backward
   compatibility across TF versions.
   The script may not work in TF1.x.

   Instructions on how to use this script:
   - Execute the script as follows:
       python gen_tftrt_model
   - Rename tftrt_saved_model to what makes sense for your test.
   - Delete directory tf_saved_model unless you want to use it.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.training.tracking import tracking
from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables
from tensorflow.python.ops import array_ops
from tensorflow.python.saved_model import save
from tensorflow.python.compiler.tensorrt import trt_convert

def GetGraph(inp1, inp2, var):
  """Define graph."""
  add = inp1 + var
  mul = inp1 * add
  add = mul + add
  add = add + inp2
  out = array_ops.identity(add, name="output")
  return out

class SimpleModel(tracking.AutoTrackable):
  """Define model with a TF function."""

  def __init__(self):
    self.v = None

  @def_function.function(input_signature=[
      tensor_spec.TensorSpec(shape=[None, 1, 1], dtype=dtypes.float32),
      tensor_spec.TensorSpec(shape=[None, 1, 1], dtype=dtypes.float32)
  ])
  def run(self, inp1, inp2):
    if self.v is None:
      self.v = variables.Variable([[[1.0]]], dtype=dtypes.float32)
    return GetGraph(inp1, inp2, self.v)

root = SimpleModel()

input_saved_model_dir = "tf_saved_model"
output_saved_model_dir = "tftrt_saved_model"
_SAVED_MODEL_SIGNATURE_KEY = "tftrt_test_predict"

# Saved TF model
save.save(root, input_saved_model_dir,
          {_SAVED_MODEL_SIGNATURE_KEY: root.run})

# Convert TF model to TensorRT
converter = trt_convert.TrtGraphConverterV2(
    input_saved_model_dir=input_saved_model_dir,
    input_saved_model_signature_key=_SAVED_MODEL_SIGNATURE_KEY)
converter.convert()
def my_input_fn():
    np_input1 = np.random.random_sample([4, 1, 1]).astype(np.float32)
    np_input2 = np.random.random_sample([4, 1, 1]).astype(np.float32)
    yield np_input1, np_input2,
converter.build(input_fn=my_input_fn)
# Convert TensorRT model
converter.save(output_saved_model_dir)

