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
"""This is a Python API fuzzer for tf.raw_ops.DataFormatVecPermute."""
import sys
import atheris_no_libfuzzer as atheris
from python_fuzzing import FuzzingHelper
import tensorflow as tf


def TestOneInput(input_bytes):
  """Test randomized integer fuzzing input for tf.raw_ops.DataFormatVecPermute."""
  fh = FuzzingHelper(input_bytes)

  dtype = fh.get_tf_dtype()
  # Max shape can be 8 in length and randomized from 0-8 without running into
  # a OOM error.
  shape = fh.get_int_list(min_length=0, max_length=8, min_int=0, max_int=8)
  seed = fh.get_int()
  try:
    x = tf.random.uniform(shape=shape, dtype=dtype, seed=seed)
    src_format_digits = str(fh.get_int(min_int=0, max_int=999999999))
    dest_format_digits = str(fh.get_int(min_int=0, max_int=999999999))
    _ = tf.raw_ops.DataFormatVecPermute(
        x,
        src_format=src_format_digits,
        dst_format=dest_format_digits,
        name=fh.get_string())
  except (tf.errors.InvalidArgumentError, ValueError, TypeError):
    pass


def main():
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()


if __name__ == '__main__':
  main()
