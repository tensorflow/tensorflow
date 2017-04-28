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

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test

# XXX: this depends on pywrap_tensorflow and must come later
from tensorflow.contrib.tfprof.python.tools.tfprof import model_analyzer


class PrintModelAnalysisTest(test.TestCase):

  def _BuildSmallModel(self):
    image = array_ops.zeros([2, 6, 6, 3])
    _ = variable_scope.get_variable(
        'ScalarW', [],
        dtypes.float32,
        initializer=init_ops.random_normal_initializer(stddev=0.001))
    kernel = variable_scope.get_variable(
        'DW', [3, 3, 3, 6],
        dtypes.float32,
        initializer=init_ops.random_normal_initializer(stddev=0.001))
    x = nn_ops.conv2d(image, kernel, [1, 2, 2, 1], padding='SAME')
    kernel = variable_scope.get_variable(
        'DW2', [2, 2, 6, 12],
        dtypes.float32,
        initializer=init_ops.random_normal_initializer(stddev=0.001))
    x = nn_ops.conv2d(x, kernel, [1, 2, 2, 1], padding='SAME')
    return x

  def testDumpToFile(self):
    opts = model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
    opts['dump_to_file'] = os.path.join(test.get_temp_dir(), 'dump')

    with session.Session() as sess, ops.device('/cpu:0'):
      _ = self._BuildSmallModel()
      model_analyzer.print_model_analysis(sess.graph, tfprof_options=opts)

      with gfile.Open(opts['dump_to_file'], 'r') as f:
        self.assertEqual(u'_TFProfRoot (--/451 params)\n'
                         '  DW (3x3x3x6, 162/162 params)\n'
                         '  DW2 (2x2x6x12, 288/288 params)\n'
                         '  ScalarW (1, 1/1 params)\n',
                         f.read())

  def testSelectEverything(self):
    opts = model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
    opts['dump_to_file'] = os.path.join(test.get_temp_dir(), 'dump')
    opts['account_type_regexes'] = ['.*']
    opts['select'] = [
        'bytes', 'params', 'float_ops', 'num_hidden_ops', 'device', 'op_types'
    ]

    with session.Session() as sess, ops.device('/cpu:0'):
      x = self._BuildSmallModel()

      sess.run(variables.global_variables_initializer())
      run_meta = config_pb2.RunMetadata()
      _ = sess.run(x,
                   options=config_pb2.RunOptions(
                       trace_level=config_pb2.RunOptions.FULL_TRACE),
                   run_metadata=run_meta)

      model_analyzer.print_model_analysis(
          sess.graph, run_meta, tfprof_options=opts)

      with gfile.Open(opts['dump_to_file'], 'r') as f:
        # pylint: disable=line-too-long
        self.assertEqual(
            '_TFProfRoot (0/451 params, 0/10.44k flops, 0B/5.28KB, _kTFScopeParent)\n  Conv2D (0/0 params, 5.83k/5.83k flops, 432B/432B, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Conv2D)\n  Conv2D_1 (0/0 params, 4.61k/4.61k flops, 384B/384B, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Conv2D)\n  DW (3x3x3x6, 162/162 params, 0/0 flops, 648B/1.30KB, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|VariableV2|_trainable_variables)\n    DW/Assign (0/0 params, 0/0 flops, 0B/0B, /device:CPU:0, /device:CPU:0|Assign)\n    DW/Initializer (0/0 params, 0/0 flops, 0B/0B, _kTFScopeParent)\n      DW/Initializer/random_normal (0/0 params, 0/0 flops, 0B/0B, Add)\n        DW/Initializer/random_normal/RandomStandardNormal (0/0 params, 0/0 flops, 0B/0B, RandomStandardNormal)\n        DW/Initializer/random_normal/mean (0/0 params, 0/0 flops, 0B/0B, Const)\n        DW/Initializer/random_normal/mul (0/0 params, 0/0 flops, 0B/0B, Mul)\n        DW/Initializer/random_normal/shape (0/0 params, 0/0 flops, 0B/0B, Const)\n        DW/Initializer/random_normal/stddev (0/0 params, 0/0 flops, 0B/0B, Const)\n    DW/read (0/0 params, 0/0 flops, 648B/648B, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Identity)\n  DW2 (2x2x6x12, 288/288 params, 0/0 flops, 1.15KB/2.30KB, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|VariableV2|_trainable_variables)\n    DW2/Assign (0/0 params, 0/0 flops, 0B/0B, /device:CPU:0, /device:CPU:0|Assign)\n    DW2/Initializer (0/0 params, 0/0 flops, 0B/0B, _kTFScopeParent)\n      DW2/Initializer/random_normal (0/0 params, 0/0 flops, 0B/0B, Add)\n        DW2/Initializer/random_normal/RandomStandardNormal (0/0 params, 0/0 flops, 0B/0B, RandomStandardNormal)\n        DW2/Initializer/random_normal/mean (0/0 params, 0/0 flops, 0B/0B, Const)\n        DW2/Initializer/random_normal/mul (0/0 params, 0/0 flops, 0B/0B, Mul)\n        DW2/Initializer/random_normal/shape (0/0 params, 0/0 flops, 0B/0B, Const)\n        DW2/Initializer/random_normal/stddev (0/0 params, 0/0 flops, 0B/0B, Const)\n    DW2/read (0/0 params, 0/0 flops, 1.15KB/1.15KB, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Identity)\n  ScalarW (1, 1/1 params, 0/0 flops, 0B/0B, /device:CPU:0, /device:CPU:0|VariableV2|_trainable_variables)\n    ScalarW/Assign (0/0 params, 0/0 flops, 0B/0B, /device:CPU:0, /device:CPU:0|Assign)\n    ScalarW/Initializer (0/0 params, 0/0 flops, 0B/0B, _kTFScopeParent)\n      ScalarW/Initializer/random_normal (0/0 params, 0/0 flops, 0B/0B, Add)\n        ScalarW/Initializer/random_normal/RandomStandardNormal (0/0 params, 0/0 flops, 0B/0B, RandomStandardNormal)\n        ScalarW/Initializer/random_normal/mean (0/0 params, 0/0 flops, 0B/0B, Const)\n        ScalarW/Initializer/random_normal/mul (0/0 params, 0/0 flops, 0B/0B, Mul)\n        ScalarW/Initializer/random_normal/shape (0/0 params, 0/0 flops, 0B/0B, Const)\n        ScalarW/Initializer/random_normal/stddev (0/0 params, 0/0 flops, 0B/0B, Const)\n    ScalarW/read (0/0 params, 0/0 flops, 0B/0B, /device:CPU:0, /device:CPU:0|Identity)\n  init (0/0 params, 0/0 flops, 0B/0B, /device:CPU:0, /device:CPU:0|NoOp)\n  zeros (0/0 params, 0/0 flops, 864B/864B, /job:localhost/replica:0/task:0/cpu:0, /job:localhost/replica:0/task:0/cpu:0|Const)\n',
            f.read())
        # pylint: enable=line-too-long


if __name__ == '__main__':
  test.main()
