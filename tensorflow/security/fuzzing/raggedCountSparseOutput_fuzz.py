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
"""This is a Python API fuzzer for tf.raw_ops.RaggedCountSparseOutput."""
import sys
import atheris_no_libfuzzer as atheris
import tensorflow as tf


def TestOneInput(input_bytes):
  """Test randomized integer fuzzing input for tf.raw_ops.RaggedCountSparseOutput."""
  fdp = atheris.FuzzedDataProvider(input_bytes)
  random_split_length = fdp.ConsumeIntInRange(0, 500)
  random_length = fdp.ConsumeIntInRange(501, 1000)

  splits = fdp.ConsumeIntListInRange(random_split_length, 1, 100000)

  # First value of splits has to be 0.
  splits.insert(0, 0)
  # Last value of splits has to be length of the values/weights.
  splits.append(random_length)
  values = fdp.ConsumeIntListInRange(random_length, 0, 100000)
  weights = fdp.ConsumeIntListInRange(random_length, 0, 100000)
  _, _, _, = tf.raw_ops.RaggedCountSparseOutput(
            splits=splits, values=values, weights=weights, binary_output=False)


def main():
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()


if __name__ == "__main__":
  main()
