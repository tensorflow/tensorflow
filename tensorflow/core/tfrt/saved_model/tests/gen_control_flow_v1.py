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
"""Generates a toy v1 saved model for testing."""

import shutil
from absl import app
from absl import flags
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
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

  # The following test creates two signatures, each of which contains a
  # switch-merge construct that will be functionalized to the tf.If op. However,
  # `then` and `else` branches' arguments are deliberately made different
  # between these two model signatures, in order to trigger error in cases these
  # branches are functionalized to functions with the same function name.

  data_0 = array_ops.constant([1, 2, 3, 4, 5, 6])
  data_1 = array_ops.constant([2, 3, 4, 5, 6, 7])
  # Create placeholders to prevent constant folding.
  x_op = array_ops.placeholder(dtype=dtypes.int32)
  y_op = array_ops.placeholder(dtype=dtypes.int32)
  less_op = math_ops.less(x_op, y_op)
  switch_0_op = control_flow_ops.switch(data_0, less_op)
  switch_1_op = control_flow_ops.switch(data_1, less_op)

  # merge_0_op will be functionalized to a tf.If op with only one argument
  # `data_0` in addition to the condition `less_op`.
  merge_0_op = control_flow_ops.merge(switch_0_op)[0]

  # merge_1_op will be functionalized to a tf.If op with two arguments, `data_0`
  # and `data_1` in addition to the condition `less_op`.
  merge_1_op = control_flow_ops.merge([switch_0_op[0], switch_1_op[1]])[0]

  result = merge_0_op
  result_1 = merge_1_op

  sess = session.Session()

  sm_builder = builder.SavedModelBuilder(FLAGS.saved_model_path)
  tensor_info_x = utils.build_tensor_info(x_op)
  tensor_info_y = utils.build_tensor_info(y_op)
  tensor_info_result = utils.build_tensor_info(result)
  tensor_info_result_1 = utils.build_tensor_info(result_1)

  signature = (
      signature_def_utils.build_signature_def(
          inputs={
              'x': tensor_info_x,
              'y': tensor_info_y
          },
          outputs={'result': tensor_info_result},
          method_name=signature_constants.PREDICT_METHOD_NAME))

  signature_1 = (
      signature_def_utils.build_signature_def(
          inputs={
              'x': tensor_info_x,
              'y': tensor_info_y
          },
          outputs={'result_1': tensor_info_result_1},
          method_name=signature_constants.PREDICT_METHOD_NAME))

  sm_builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
          'sig': signature,
          'sig_1': signature_1,
      },
      strip_default_attrs=True)
  sm_builder.save()


if __name__ == '__main__':
  app.run(main)
