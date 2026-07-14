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
"""This is a Python API fuzzer for v1 vs v2 API comparison."""
import atheris
with atheris.instrument_imports():
  import sys
  from python_fuzzing import FuzzingHelper
  import tensorflow as tf


@atheris.instrument_func
def TestOneInput(input_bytes):
  """Test randomized integer fuzzing input for v1 vs v2 APIs."""
  fh = FuzzingHelper(input_bytes)

  # Comparing tf.math.angle with tf.compat.v1.angle.
  input_supported_dtypes = [tf.float32, tf.float64]
  random_dtype_index = fh.get_int(min_int=0, max_int=1)
  input_dtype = input_supported_dtypes[random_dtype_index]
  input_shape = fh.get_int_list(
      min_length=0, max_length=6, min_int=0, max_int=10)
  seed = fh.get_int()
  input_tensor = tf.random.uniform(
      shape=input_shape, dtype=input_dtype, seed=seed, maxval=10)
  name = fh.get_string(5)
  v2_output = tf.math.angle(input=input_tensor, name=name)
  v1_output = tf.compat.v1.angle(input=input_tensor, name=name)
  try:
    tf.debugging.assert_equal(v1_output, v2_output)
    tf.debugging.assert_equal(v1_output.shape, v2_output.shape)
  except Exception as e:  # pylint: disable=broad-except
    print("Input tensor: {}".format(input_tensor))
    print("Input dtype: {}".format(input_dtype))
    print("v1_output: {}".format(v1_output))
    print("v2_output: {}".format(v2_output))
    raise e

  # Comparing tf.debugging.assert_integer with tf.compat.v1.assert_integer.
  x_supported_dtypes = [
      tf.float16, tf.float32, tf.float64, tf.int32, tf.int64, tf.string
  ]
  random_dtype_index = fh.get_int(min_int=0, max_int=5)
  x_dtype = x_supported_dtypes[random_dtype_index]
  x_shape = fh.get_int_list(min_length=0, max_length=6, min_int=0, max_int=10)
  seed = fh.get_int()
  try:
    x = tf.random.uniform(shape=x_shape, dtype=x_dtype, seed=seed, maxval=10)
  except ValueError:
    x = tf.constant(["test_string"])
  message = fh.get_string(128)
  name = fh.get_string(128)
  try:
    v2_output = tf.debugging.assert_integer(x=x, message=message, name=name)
  except Exception as e:  # pylint: disable=broad-except
    v2_output = e
  try:
    v1_output = tf.compat.v1.assert_integer(x=x, message=message, name=name)
  except Exception as e:  # pylint: disable=broad-except
    v1_output = e

  if v1_output and v2_output:
    assert type(v2_output) == type(v1_output)  # pylint: disable=unidiomatic-typecheck
    assert v2_output.args == v1_output.args


def main():
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()


if __name__ == "__main__":
  main()
