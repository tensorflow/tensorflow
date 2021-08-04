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
"""This is a Python API fuzzer for tf.raw_ops.ImmutableConst."""
import sys
import atheris_no_libfuzzer as atheris
from python_fuzzing import FuzzingHelper
import tensorflow as tf

_DEFAULT_FILENAME = '/tmp/test.txt'


def TestOneInput(input_bytes):
  """Test randomized integer fuzzing input for tf.raw_ops.ImmutableConst."""
  fh = FuzzingHelper(input_bytes)

  dtype = fh.get_tf_dtype()
  shape = fh.get_int_list()
  try:
    with open(_DEFAULT_FILENAME, 'w') as f:
      f.write(fh.get_string())
    _ = tf.raw_ops.ImmutableConst(
        dtype=dtype, shape=shape, memory_region_name=_DEFAULT_FILENAME)
  except (tf.errors.InvalidArgumentError, tf.errors.InternalError,
          UnicodeEncodeError, UnicodeDecodeError):
    pass


def main():
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()


if __name__ == '__main__':
  main()
