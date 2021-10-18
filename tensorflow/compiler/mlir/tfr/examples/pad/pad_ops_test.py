# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.compiler.mlir.tfr.examples.pad.ops_defs."""

import os
from absl.testing import parameterized
import tensorflow as tf

from tensorflow.compiler.mlir.tfr.examples.pad import gen_pad_ops
from tensorflow.compiler.mlir.tfr.examples.pad import ops_defs
from tensorflow.compiler.mlir.tfr.python import test_utils
from tensorflow.python.framework import load_library
from tensorflow.python.platform import test

_lib_dir = os.path.dirname(gen_pad_ops.__file__)
_lib_name = os.path.basename(gen_pad_ops.__file__)[4:].replace('.py', '.so')
load_library.load_op_library(os.path.join(_lib_dir, _lib_name))


class PadOpsDefsTest(test_utils.OpsDefsTest, parameterized.TestCase):

  @parameterized.named_parameters(('ReflectMode', 'REFLECT'),
                                  ('SymmetricMode', 'SYMMETRIC'))
  def test_mirror_pad(self, mode):
    input_ = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    paddings = tf.constant([[
        1,
        1,
    ], [2, 2]])
    kwargs = {
        'input': input_,
        'paddings': paddings,
        'mode': mode,
    }
    kwargs_ = {
        'input_': input_,
        'paddings': paddings,
        'mode': mode,
    }
    # Make sure the composition python function is correct
    self._assertOpAndComposite([input_], tf.raw_ops.MirrorPad,
                               ops_defs._composite_mirror_pad, kwargs_, kwargs)
    # Make sure the translation and decomposition is correct
    self._assertOpAndComposite([input_],
                               tf.function(gen_pad_ops.new_mirror_pad),
                               ops_defs._composite_mirror_pad, kwargs_)

  @parameterized.named_parameters(('ReflectMode', 'REFLECT'),
                                  ('SymmetricMode', 'SYMMETRIC'))
  def test_mirror_pad_grad(self, mode):
    input_ = tf.constant([[2, 1, 1, 2, 3, 3, 2], [2, 1, 1, 2, 3, 3, 2],
                          [5, 4, 4, 5, 6, 6, 5], [5, 4, 4, 5, 6, 6, 5]],
                         dtype=tf.float32)
    paddings = tf.constant([[
        1,
        1,
    ], [2, 2]])
    kwargs = {
        'input': input_,
        'paddings': paddings,
        'mode': mode,
    }
    kwargs_ = {
        'input_': input_,
        'paddings': paddings,
        'mode': mode,
    }
    # Make sure the composition python function is correct
    self._assertOpAndComposite([input_], tf.raw_ops.MirrorPadGrad,
                               ops_defs._composite_mirror_pad_grad, kwargs_,
                               kwargs)
    # Make sure the translation and decomposition is correct
    self._assertOpAndComposite([input_],
                               tf.function(gen_pad_ops.new_mirror_pad_grad),
                               ops_defs._composite_mirror_pad_grad, kwargs_)


if __name__ == '__main__':
  os.environ[
      'TF_MLIR_TFR_LIB_DIR'] = 'tensorflow/compiler/mlir/tfr/examples/pad'
  test.main()
