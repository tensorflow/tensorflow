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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


class PrintModelAnalysisTest(tf.test.TestCase):

  def _BuildSmallModel(self):
    image = tf.zeros([2, 6, 6, 3])
    kernel = tf.get_variable(
        'DW', [3, 3, 3, 6],
        tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.001))
    x = tf.nn.conv2d(image, kernel, [1, 2, 2, 1], padding='SAME')
    kernel = tf.get_variable(
        'DW2', [2, 2, 6, 12],
        tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.001))
    x = tf.nn.conv2d(x, kernel, [1, 2, 2, 1], padding='SAME')
    return x

  def testDumpToFile(self):
    opts = tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
    opts['dump_to_file'] = os.path.join(tf.test.get_temp_dir(), 'dump')

    with tf.Session() as sess, tf.device('/cpu:0'):
      _ = self._BuildSmallModel()
      tf.contrib.tfprof.model_analyzer.print_model_analysis(
          sess.graph, tfprof_options=opts)

      with tf.gfile.Open(opts['dump_to_file'], 'r') as f:
        self.assertEqual(u'_TFProfRoot (--/450 params)\n'
                         '  DW (3x3x3x6, 162/162 params)\n'
                         '  DW2 (2x2x6x12, 288/288 params)\n',
                         f.read().decode('utf-8'))

  def testSelectEverything(self):
    opts = tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
    opts['dump_to_file'] = os.path.join(tf.test.get_temp_dir(), 'dump')
    opts['account_type_regexes'] = ['.*']
    opts['select'] = [
        'bytes', 'params', 'float_ops', 'num_hidden_ops', 'device', 'op_types'
    ]

    with tf.Session() as sess, tf.device('/cpu:0'):
      x = self._BuildSmallModel()

      sess.run(tf.global_variables_initializer())
      run_meta = tf.RunMetadata()
      _ = sess.run(x,
                   options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                   run_metadata=run_meta)

      tf.contrib.tfprof.model_analyzer.print_model_analysis(
          sess.graph, run_meta, tfprof_options=opts)

      with tf.gfile.Open(opts['dump_to_file'], 'r') as f:
        # pylint: disable=line-too-long
        self.assertEqual(
            '_TFProfRoot (0/450 params, 0/10.44k flops, 0B/5.28KB, _kTFScopeParent)\n  Conv2D (0/0 params, 5.83k/5.83k flops, 432B/432B, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Conv2D)\n  Conv2D_1 (0/0 params, 4.61k/4.61k flops, 384B/384B, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Conv2D)\n  DW (3x3x3x6, 162/162 params, 0/0 flops, 648B/1.30KB, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Variable|_trainable_variables)\n    DW/Assign (0/0 params, 0/0 flops, 0B/0B, /device:CPU:0, /device:CPU:0|Assign)\n    DW/Initializer (0/0 params, 0/0 flops, 0B/0B, _kTFScopeParent)\n      DW/Initializer/random_normal (0/0 params, 0/0 flops, 0B/0B, Add)\n        DW/Initializer/random_normal/RandomStandardNormal (0/0 params, 0/0 flops, 0B/0B, RandomStandardNormal)\n        DW/Initializer/random_normal/mean (0/0 params, 0/0 flops, 0B/0B, Const)\n        DW/Initializer/random_normal/mul (0/0 params, 0/0 flops, 0B/0B, Mul)\n        DW/Initializer/random_normal/shape (0/0 params, 0/0 flops, 0B/0B, Const)\n        DW/Initializer/random_normal/stddev (0/0 params, 0/0 flops, 0B/0B, Const)\n    DW/read (0/0 params, 0/0 flops, 648B/648B, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Identity)\n  DW2 (2x2x6x12, 288/288 params, 0/0 flops, 1.15KB/2.30KB, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Variable|_trainable_variables)\n    DW2/Assign (0/0 params, 0/0 flops, 0B/0B, /device:CPU:0, /device:CPU:0|Assign)\n    DW2/Initializer (0/0 params, 0/0 flops, 0B/0B, _kTFScopeParent)\n      DW2/Initializer/random_normal (0/0 params, 0/0 flops, 0B/0B, Add)\n        DW2/Initializer/random_normal/RandomStandardNormal (0/0 params, 0/0 flops, 0B/0B, RandomStandardNormal)\n        DW2/Initializer/random_normal/mean (0/0 params, 0/0 flops, 0B/0B, Const)\n        DW2/Initializer/random_normal/mul (0/0 params, 0/0 flops, 0B/0B, Mul)\n        DW2/Initializer/random_normal/shape (0/0 params, 0/0 flops, 0B/0B, Const)\n        DW2/Initializer/random_normal/stddev (0/0 params, 0/0 flops, 0B/0B, Const)\n    DW2/read (0/0 params, 0/0 flops, 1.15KB/1.15KB, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Identity)\n  init (0/0 params, 0/0 flops, 0B/0B, /device:CPU:0, /device:CPU:0|NoOp)\n  zeros (0/0 params, 0/0 flops, 864B/864B, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Const)\n',
            f.read().decode('utf-8'))
        # pylint: enable=line-too-long


if __name__ == '__main__':
  tf.test.main()
