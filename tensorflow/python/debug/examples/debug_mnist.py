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
"""Redirect script that points to corresponding example based on tf version."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import absl
import tensorflow

import tensorflow.python.debug.examples.v1.debug_mnist_v1 as debug_mnist_v1
import tensorflow.python.debug.examples.v2.debug_mnist_v2 as debug_mnist_v2

tf = tensorflow.compat.v1


def main():
  if tf.__version__.startswith("1."):
    flags, unparsed = debug_mnist_v1.parse_args()
    debug_mnist_v1.FLAGS = flags

    with tf.Graph().as_default():
      tf.app.run(main=debug_mnist_v1.main, argv=[sys.argv[0]] + unparsed)
  else:
    flags, unparsed = debug_mnist_v2.parse_args()
    debug_mnist_v2.FLAGS = flags
    absl.app.run(main=debug_mnist_v2.main, argv=[sys.argv[0]] + unparsed)


if __name__ == "__main__":
  main()
