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
# ==============================================================================
"""Test file to display the error message and verify it with FileCheck."""

# RUN: %p/concrete_function_error | FileCheck %s

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import app

import tensorflow.compat.v2 as tf
if hasattr(tf, 'enable_v2_behavior'):
  tf.enable_v2_behavior()


class TestGraphDebugInfo(object):
  """Test stack trace can be displayed."""

  def testConcreteFunctionDebugInfo(self):
    """Create a concrete func with unsupported ops, and convert it."""
    @tf.function(
        input_signature=[tf.TensorSpec(shape=[3, 3], dtype=tf.float32)])
    def model(x):
      y = tf.math.betainc(x, 0.5, 1.0)  # Not supported
      return y + y

    func = model.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([func])
    converter.convert()

# pylint: disable=line-too-long

# CHECK-LABEL: testConcreteFunctionDebugInfo
# CHECK: error: 'tf.Betainc' op is neither a custom op nor a flex op
# CHECK:                                  attrs=attr_protos, op_def=op_def)
# CHECK:                                  ^
# CHECK: {{.*tensorflow/python/ops/gen_math_ops.py:[0-9]+:[0-9]+: note: called from}}
# CHECK:         "Betainc", a=a, b=b, x=x, name=name)
# CHECK:         ^
# CHECK: {{.*tensorflow/compiler/mlir/lite/tests/debuginfo/concrete_function_error.py:[0-9]+:[0-9]+: note: called from}}
# CHECK:     y = tf.math.betainc(x, 0.5, 1.0)  # Not supported
# CHECK:     ^
# CHECK: <unknown>:0: error: failed while converting: 'main'

# pylint: enable=line-too-long


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  try:
    TestGraphDebugInfo().testConcreteFunctionDebugInfo()
  except Exception as e:  # pylint: disable=broad-except
    sys.stdout.write('testConcreteFunctionDebugInfo')
    sys.stdout.write(str(e))


if __name__ == '__main__':
  app.run(main)
