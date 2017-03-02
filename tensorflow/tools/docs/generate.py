# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Generate docs for the TensorFlow Python API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import inspect
import os
import sys

import tensorflow as tf

from tensorflow.python import debug as tf_debug
from tensorflow.tools.docs import generate_lib


if __name__ == '__main__':
  argument_parser = argparse.ArgumentParser()
  argument_parser.add_argument(
      '--output_dir',
      type=str,
      default=None,
      required=True,
      help='Directory to write docs to.'
  )

  argument_parser.add_argument(
      '--src_dir',
      type=str,
      default=None,
      required=True,
      help='Directory with the source docs.'
  )

  # This doc generator works on the TensorFlow codebase. Since this script lives
  # at tensorflow/tools/docs, and all code is defined somewhere inside
  # tensorflow/, we can compute the base directory (two levels up), which is
  # valid unless we're trying to apply this to a different code base, or are
  # moving the script around.
  script_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
  default_base_dir = os.path.join(script_dir, '..', '..')

  argument_parser.add_argument(
      '--base_dir',
      type=str,
      default=default_base_dir,
      help=('Base directory to to strip from file names referenced in docs. '
            'Defaults to two directories up from the location of this file.')
  )

  flags, _ = argument_parser.parse_known_args()

  # tf_debug is not imported with tf, it's a separate module altogether
  modules = [('tf', tf), ('tfdbg', tf_debug)]

  # Access something in contrib so tf.contrib is properly loaded (it's hidden
  # behind lazy loading)
  _ = tf.contrib.__name__

  sys.exit(generate_lib.main(
      flags.src_dir, flags.output_dir, flags.base_dir, modules))
