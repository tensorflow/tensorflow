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

import inspect
import os
import sys

import tensorflow as tf

from tensorflow.python import debug as tf_debug
from tensorflow.tools.docs import generate_lib


if __name__ == '__main__':
  doc_generator = generate_lib.DocGenerator()
  doc_generator.add_output_dir_argument()
  doc_generator.add_src_dir_argument()

  # This doc generator works on the TensorFlow codebase. Since this script lives
  # at tensorflow/tools/docs, and all code is defined somewhere inside
  # tensorflow/, we can compute the base directory (two levels up), which is
  # valid unless we're trying to apply this to a different code base, or are
  # moving the script around.
  script_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
  default_base_dir = os.path.join(script_dir, '..', '..')
  doc_generator.add_base_dir_argument(default_base_dir)

  flags = doc_generator.parse_known_args()

  # tf_debug is not imported with tf, it's a separate module altogether
  doc_generator.set_py_modules([('tf', tf), ('tfdbg', tf_debug)])

  doc_generator.load_contrib()
  doc_generator.set_do_not_descend_map({
      '': ['cli', 'lib', 'wrappers'],
      'contrib': [
          'compiler',
          'factorization',
          'grid_rnn',
          'labeled_tensor',
          'ndlstm',
          'quantization',
          'session_bundle',
          'slim',
          'solvers',
          'specs',
          'tensor_forest',
          'tensorboard',
          'testing',
          'training',
          'tfprof',
      ],
      'contrib.bayesflow': [
          'entropy', 'monte_carlo',
          'special_math', 'stochastic_gradient_estimators',
          'stochastic_graph', 'stochastic_tensor',
          'stochastic_variables', 'variational_inference'
      ],
      'contrib.distributions': ['bijector'],
      'contrib.ffmpeg': ['ffmpeg_ops'],
      'contrib.graph_editor': [
          'edit',
          'match',
          'reroute',
          'subgraph',
          'transform',
          'select',
          'util'
      ],
      'contrib.layers': ['feature_column', 'summaries'],
      'contrib.learn': [
          'datasets',
          'head',
          'graph_actions',
          'io',
          'models',
          'monitors',
          'ops',
          'preprocessing',
          'utils',
      ],
      'contrib.util': ['loader'],
  })

  sys.exit(doc_generator.build(flags))
