# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# Lint as: python3
"""Generates a toy v1 saved model for testing."""

import shutil
from absl import app
from absl import flags
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

flags.DEFINE_string('saved_model_path', '', 'Path to save the model to.')
FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  shutil.rmtree(FLAGS.saved_model_path)

  # Create the graph

  bf16 = variables.Variable(name='bf16', dtype=dtypes.bfloat16, initial_value=1)
  f16 = variables.Variable(name='f16', dtype=dtypes.float16, initial_value=1)
  f32 = variables.Variable(name='f32', dtype=dtypes.float32, initial_value=1)
  f64 = variables.Variable(name='f64', dtype=dtypes.float64, initial_value=1)

  ui8 = variables.Variable(name='ui8', dtype=dtypes.uint8, initial_value=1)
  ui16 = variables.Variable(name='ui16', dtype=dtypes.uint16, initial_value=1)
  ui32 = variables.Variable(name='ui32', dtype=dtypes.uint32, initial_value=1)
  ui64 = variables.Variable(name='ui64', dtype=dtypes.uint64, initial_value=1)

  i1 = variables.Variable(name='i1', dtype=dtypes.bool, initial_value=True)
  i8 = variables.Variable(name='i8', dtype=dtypes.uint8, initial_value=1)
  i16 = variables.Variable(name='i16', dtype=dtypes.uint16, initial_value=1)
  i32 = variables.Variable(name='i32', dtype=dtypes.uint32, initial_value=1)
  i64 = variables.Variable(name='i64', dtype=dtypes.uint64, initial_value=1)

  complex64 = variables.Variable(
      name='complex64', dtype=dtypes.complex64, initial_value=1)
  complex128 = variables.Variable(
      name='complex128', dtype=dtypes.complex128, initial_value=1)

  string = variables.Variable(
      name='string', dtype=dtypes.string, initial_value='str')

  sess = session.Session()

  sess.run(variables.global_variables_initializer())

  sm_builder = builder.SavedModelBuilder(FLAGS.saved_model_path)

  r_bf16 = utils.build_tensor_info(bf16.read_value())
  r_f16 = utils.build_tensor_info(f16.read_value())
  r_f32 = utils.build_tensor_info(f32.read_value())
  r_f64 = utils.build_tensor_info(f64.read_value())

  r_ui8 = utils.build_tensor_info(ui8.read_value())
  r_ui16 = utils.build_tensor_info(ui16.read_value())
  r_ui32 = utils.build_tensor_info(ui32.read_value())
  r_ui64 = utils.build_tensor_info(ui64.read_value())

  r_i1 = utils.build_tensor_info(i1.read_value())
  r_i8 = utils.build_tensor_info(i8.read_value())
  r_i16 = utils.build_tensor_info(i16.read_value())
  r_i32 = utils.build_tensor_info(i32.read_value())
  r_i64 = utils.build_tensor_info(i64.read_value())

  r_complex64 = utils.build_tensor_info(complex64.read_value())
  r_complex128 = utils.build_tensor_info(complex128.read_value())

  r_string = utils.build_tensor_info(string.read_value())

  toy_signature = (
      signature_def_utils.build_signature_def(
          outputs={
              'r_bf16': r_bf16,
              'r_f16': r_f16,
              'r_f32': r_f32,
              'r_f64': r_f64,
              'r_ui8': r_ui8,
              'r_ui16': r_ui16,
              'r_ui32': r_ui32,
              'r_ui64': r_ui64,
              'r_i1': r_i1,
              'r_i8': r_i8,
              'r_i16': r_i16,
              'r_i32': r_i32,
              'r_i64': r_i64,
              'r_complex64': r_complex64,
              'r_complex128': r_complex128,
              'r_string': r_string,
          },
          method_name=signature_constants.PREDICT_METHOD_NAME))

  sm_builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: toy_signature,
      },
      strip_default_attrs=True)
  sm_builder.save()


if __name__ == '__main__':
  app.run(main)
