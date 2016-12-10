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
"""print_model_analysis test."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.contrib.tfprof.python.tools.tfprof import pywrap_tensorflow_print_model_analysis_lib as print_mdl
from tensorflow.tools.tfprof import tfprof_options_pb2
from tensorflow.tools.tfprof import tfprof_output_pb2

# pylint: disable=bad-whitespace
# pylint: disable=bad-continuation
TEST_OPTIONS = {
    'max_depth': 10000,
    'min_bytes': 0,
    'min_micros': 0,
    'min_params': 0,
    'min_float_ops': 0,
    'device_regexes': ['.*'],
    'order_by': 'name',
    'account_type_regexes': ['.*'],
    'start_name_regexes': ['.*'],
    'trim_name_regexes': [],
    'show_name_regexes': ['.*'],
    'hide_name_regexes': [],
    'account_displayed_op_only': True,
    'select': ['params'],
    'viz': False
}

# pylint: enable=bad-whitespace
# pylint: enable=bad-continuation


class PrintModelAnalysisTest(tf.test.TestCase):

  def _BuildSmallModel(self):
    image = tf.zeros([2, 6, 6, 3])
    kernel = tf.get_variable(
        'DW', [6, 6, 3, 6],
        tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.001))
    x = tf.nn.conv2d(image, kernel, [1, 2, 2, 1], padding='SAME')
    return x

  def testPrintModelAnalysis(self):
    opts = tfprof_options_pb2.OptionsProto()
    opts.max_depth = TEST_OPTIONS['max_depth']
    opts.min_bytes = TEST_OPTIONS['min_bytes']
    opts.min_micros = TEST_OPTIONS['min_micros']
    opts.min_params = TEST_OPTIONS['min_params']
    opts.min_float_ops = TEST_OPTIONS['min_float_ops']
    for p in TEST_OPTIONS['device_regexes']:
      opts.device_regexes.append(p)
    opts.order_by = TEST_OPTIONS['order_by']
    for p in TEST_OPTIONS['account_type_regexes']:
      opts.account_type_regexes.append(p)
    for p in TEST_OPTIONS['start_name_regexes']:
      opts.start_name_regexes.append(p)
    for p in TEST_OPTIONS['trim_name_regexes']:
      opts.trim_name_regexes.append(p)
    for p in TEST_OPTIONS['show_name_regexes']:
      opts.show_name_regexes.append(p)
    for p in TEST_OPTIONS['hide_name_regexes']:
      opts.hide_name_regexes.append(p)
    opts.account_displayed_op_only = TEST_OPTIONS['account_displayed_op_only']
    for p in TEST_OPTIONS['select']:
      opts.select.append(p)
    opts.viz = TEST_OPTIONS['viz']

    with tf.Session() as sess, tf.device('/cpu:0'):
      _ = self._BuildSmallModel()
      tfprof_pb = tfprof_output_pb2.TFProfNode()
      tfprof_pb.ParseFromString(
          print_mdl.PrintModelAnalysis(sess.graph.as_graph_def(
          ).SerializeToString(), b'', b'', b'scope', opts.SerializeToString()))

      expected_pb = tfprof_output_pb2.TFProfNode()
      text_format.Merge(r"""name: "_TFProfRoot"
      exec_micros: 0
      requested_bytes: 0
      total_exec_micros: 0
      total_requested_bytes: 0
      total_parameters: 648
      children {
      name: "Conv2D"
      exec_micros: 0
      requested_bytes: 0
      total_exec_micros: 0
      total_requested_bytes: 0
      total_parameters: 0
      device: "/device:CPU:0"
      float_ops: 0
      total_float_ops: 0
      }
      children {
      name: "DW"
      exec_micros: 0
      requested_bytes: 0
      parameters: 648
      total_exec_micros: 0
      total_requested_bytes: 0
      total_parameters: 648
      device: "/device:CPU:0"
      children {
      name: "DW/Assign"
      exec_micros: 0
      requested_bytes: 0
      total_exec_micros: 0
      total_requested_bytes: 0
      total_parameters: 0
      device: "/device:CPU:0"
      float_ops: 0
      total_float_ops: 0
      }
      children {
      name: "DW/Initializer"
      exec_micros: 0
      requested_bytes: 0
      total_exec_micros: 0
      total_requested_bytes: 0
      total_parameters: 0
      children {
      name: "DW/Initializer/random_normal"
      exec_micros: 0
      requested_bytes: 0
      total_exec_micros: 0
      total_requested_bytes: 0
      total_parameters: 0
      children {
      name: "DW/Initializer/random_normal/RandomStandardNormal"
      exec_micros: 0
      requested_bytes: 0
      total_exec_micros: 0
      total_requested_bytes: 0
      total_parameters: 0
      float_ops: 0
      total_float_ops: 0
      }
      children {
      name: "DW/Initializer/random_normal/mean"
      exec_micros: 0
      requested_bytes: 0
      total_exec_micros: 0
      total_requested_bytes: 0
      total_parameters: 0
      float_ops: 0
      total_float_ops: 0
      }
      children {
      name: "DW/Initializer/random_normal/mul"
      exec_micros: 0
      requested_bytes: 0
      total_exec_micros: 0
      total_requested_bytes: 0
      total_parameters: 0
      float_ops: 0
      total_float_ops: 0
      }
      children {
      name: "DW/Initializer/random_normal/shape"
      exec_micros: 0
      requested_bytes: 0
      total_exec_micros: 0
      total_requested_bytes: 0
      total_parameters: 0
      float_ops: 0
      total_float_ops: 0
      }
      children {
      name: "DW/Initializer/random_normal/stddev"
      exec_micros: 0
      requested_bytes: 0
      total_exec_micros: 0
      total_requested_bytes: 0
      total_parameters: 0
      float_ops: 0
      total_float_ops: 0
      }
      float_ops: 0
      total_float_ops: 0
      }
      float_ops: 0
      total_float_ops: 0
      }
      children {
      name: "DW/read"
      exec_micros: 0
      requested_bytes: 0
      total_exec_micros: 0
      total_requested_bytes: 0
      total_parameters: 0
      device: "/device:CPU:0"
      float_ops: 0
      total_float_ops: 0
      }
      float_ops: 0
      total_float_ops: 0
      }
      children {
      name: "zeros"
      exec_micros: 0
      requested_bytes: 0
      total_exec_micros: 0
      total_requested_bytes: 0
      total_parameters: 0
      device: "/device:CPU:0"
      float_ops: 0
      total_float_ops: 0
      }
      float_ops: 0
      total_float_ops: 0""", expected_pb)
      self.assertEqual(expected_pb, tfprof_pb)


if __name__ == '__main__':
  tf.test.main()
