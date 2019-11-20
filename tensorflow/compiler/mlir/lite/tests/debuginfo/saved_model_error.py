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

# RUN: %p/saved_model_error | FileCheck %s

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import app

import tensorflow.compat.v2 as tf
if hasattr(tf, 'enable_v2_behavior'):
  tf.enable_v2_behavior()


class TestModule(tf.Module):
  """The test model has unsupported op."""

  @tf.function(input_signature=[tf.TensorSpec(shape=[3, 3], dtype=tf.float32)])
  def model(self, x):
    y = tf.math.reciprocal(x)  # Not supported
    return y + y


class TestGraphDebugInfo(object):
  """Test stack trace can be displayed."""

  def testSavedModelDebugInfo(self):
    """Save a saved model with unsupported ops, and then load and convert it."""
    # saved the model
    test_model = TestModule()
    saved_model_path = '/tmp/test.saved_model'
    save_options = tf.saved_model.SaveOptions(save_debug_info=True)
    tf.saved_model.save(test_model, saved_model_path, options=save_options)

    # load the model and convert
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.experimental_new_converter = True
    converter.convert()

# pylint: disable=line-too-long

# CHECK-LABEL: testSavedModelDebugInfo
# CHECK: error: 'tf.Reciprocal' op is neither a custom op nor a flex op
# CHECK:                                  attrs=attr_protos, op_def=op_def)
# CHECK:                                  ^
# CHECK: {{.*tensorflow/python/ops/gen_math_ops.py:[0-9]+:[0-9]+: note: called from}}
# CHECK:         "Reciprocal", x=x, name=name)
# CHECK:         ^
# CHECK: {{.*tensorflow/compiler/mlir/lite/tests/debuginfo/saved_model_error.py:[0-9]+:[0-9]+: note: called from}}
# CHECK:     y = tf.math.reciprocal(x)  # Not supported
# CHECK:     ^
# CHECK: <unknown>:0: error: failed while converting: 'main'

# pylint: enable=line-too-long


def main(argv):
  """test driver method writes the error message to stdout."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  try:
    TestGraphDebugInfo().testSavedModelDebugInfo()
  except Exception as e:  # pylint: disable=broad-except
    sys.stdout.write('testSavedModelDebugInfo')
    sys.stdout.write(str(e))


if __name__ == '__main__':
  app.run(main)
