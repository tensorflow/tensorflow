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
"""Demonstrates how the composition overrides the behavior of an existing op."""

# pylint: disable=g-direct-tensorflow-import
# pylint: disable=missing-function-docstring

import os
import sys
from absl import app

from tensorflow.compiler.mlir.tfr.python import composite
from tensorflow.compiler.mlir.tfr.python.op_reg_gen import gen_register_op
from tensorflow.compiler.mlir.tfr.python.tfr_gen import tfr_gen_from_module
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_array_ops as array_ops
from tensorflow.python.platform import flags


Composite = composite.Composite
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output', None,
    'Path to write the genereated register op file and MLIR file.')

flags.DEFINE_bool('gen_register_op', True,
                  'Generate register op cc file or tfr mlir file.')


# The original kernel is defined in 'tensorflow/python/framework/ops_test.py'
# and prints out the current graph def version.
@Composite('TestAttr')
def _override_test_attr_op():
  ret = array_ops.Const(value=100.0, dtype=dtypes.float32)
  return ret


def main(_):
  if FLAGS.gen_register_op:
    assert FLAGS.output.endswith('.cc')
    generated_code = gen_register_op(sys.modules[__name__], '_override_')
  else:
    assert FLAGS.output.endswith('.mlir')
    generated_code = tfr_gen_from_module(sys.modules[__name__], '_override_')

  dirname = os.path.dirname(FLAGS.output)
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  with open(FLAGS.output, 'w') as f:
    f.write(generated_code)


if __name__ == '__main__':
  app.run(main=main)
