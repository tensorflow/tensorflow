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
from tensorflow.python.framework import test_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
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
  x1 = array_ops.placeholder(dtypes.int32, shape=(1, 3), name='input1')
  # 'y' is a read-only Reference Variable in this test case, which will be
  # converted to Resource Variable in the MLIR lowering pass.
  y = variable_scope.get_variable(name='y', initializer=[[1], [2], [3]])

  r1 = math_ops.matmul(x1, y, name='result1')

  x2 = array_ops.placeholder(dtypes.int32, shape=(1, 3), name='input2')
  r21 = math_ops.matmul(x2, y, name='result21')
  t2 = math_ops.add(x2, 1)
  r22 = math_ops.matmul(t2, y, name='result22')

  x3 = array_ops.placeholder(dtypes.int32, shape=(1, 3), name='input3')
  r31 = math_ops.matmul(x3, y, name='result31')
  r32 = math_ops.add(x3, r31, name='result32')
  r33 = math_ops.add(x3, r32, name='result33')

  # Sleep for 1 second.
  sleep_op = test_ops.sleep_identity_op(1, x1, name='sleep')

  sess = session.Session()

  sess.run(variables.global_variables_initializer())

  sm_builder = builder.SavedModelBuilder(FLAGS.saved_model_path)
  tensor_info_x1 = utils.build_tensor_info(x1)
  tensor_info_x2 = utils.build_tensor_info(x2)
  tensor_info_x3 = utils.build_tensor_info(x3)
  tensor_info_r1 = utils.build_tensor_info(r1)
  tensor_info_r21 = utils.build_tensor_info(r21)
  tensor_info_r22 = utils.build_tensor_info(r22)
  tensor_info_r31 = utils.build_tensor_info(r31)
  tensor_info_r32 = utils.build_tensor_info(r32)
  tensor_info_r33 = utils.build_tensor_info(r33)
  tensor_info_sleep = utils.build_tensor_info(sleep_op)

  toy_signature = (
      signature_def_utils.build_signature_def(
          inputs={'x1': tensor_info_x1},
          outputs={'r1': tensor_info_r1},
          method_name=signature_constants.PREDICT_METHOD_NAME))
  another_toy_signature = (
      signature_def_utils.build_signature_def(
          inputs={'x2': tensor_info_x2},
          outputs={
              'r21': tensor_info_r21,
              'r22': tensor_info_r22,
          },
          method_name=signature_constants.PREDICT_METHOD_NAME))
  yet_another_toy_signature = (
      signature_def_utils.build_signature_def(
          inputs={'x3': tensor_info_x3},
          outputs={
              'r31': tensor_info_r31,
              'r32': tensor_info_r32,
              'r33': tensor_info_r33,
          },
          method_name=signature_constants.PREDICT_METHOD_NAME))
  sleep_signature = (
      signature_def_utils.build_signature_def(
          inputs={'x1': tensor_info_x1},
          outputs={'sleep': tensor_info_sleep},
          method_name=signature_constants.PREDICT_METHOD_NAME))

  sm_builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
          'toy': toy_signature,
          'another_toy': another_toy_signature,
          'yet_another_toy': yet_another_toy_signature,
          'sleep': sleep_signature,
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: toy_signature,
      },
      strip_default_attrs=True)
  sm_builder.save()


if __name__ == '__main__':
  app.run(main)
