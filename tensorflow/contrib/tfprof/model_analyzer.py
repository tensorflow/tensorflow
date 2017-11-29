# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Model Analyzer.

Analyze model, including shape, params, time, memory, structure, etc.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Import the names here for existing users.
# pylint: disable=unused-import
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.profiler.model_analyzer import advise as _advise
from tensorflow.python.profiler.model_analyzer import ALL_ADVICE
from tensorflow.python.profiler.model_analyzer import profile as _profile
from tensorflow.python.profiler.model_analyzer import Profiler
from tensorflow.python.util.deprecation import deprecated

_DEFAULT_PROFILE_OPTIONS = 0
_DEFAULT_ADVISE_OPTIONS = 0

# pylint: disable=bad-whitespace
# pylint: disable=bad-continuation
# options examples for profiling API.
#
# Show the parameter statistics of trainable variables.
TRAINABLE_VARS_PARAMS_STAT_OPTIONS = {
    'max_depth': 10000,
    'min_bytes': 0,
    'min_micros': 0,
    'min_params': 0,
    'min_float_ops': 0,
    'order_by': 'name',
    'account_type_regexes': [tfprof_logger.TRAINABLE_VARIABLES],
    'start_name_regexes': ['.*'],
    'trim_name_regexes': [],
    'show_name_regexes': ['.*'],
    'hide_name_regexes': [],
    'account_displayed_op_only': True,
    'select': ['params'],
    'output': 'stdout',
    'dump_to_file': ''  # Deprecated, use 'output': 'file:outfile=<name>'
}

# Show the number float operations.
FLOAT_OPS_OPTIONS = {
    'max_depth': 10000,
    'min_bytes': 0,
    'min_micros': 0,
    'min_params': 0,
    'min_float_ops': 1,
    'order_by': 'float_ops',
    'account_type_regexes': ['.*'],
    'start_name_regexes': ['.*'],
    'trim_name_regexes': [],
    'show_name_regexes': ['.*'],
    'hide_name_regexes': [],
    'account_displayed_op_only': True,
    'select': ['float_ops'],
    'output': 'stdout',
    'dump_to_file': ''  # Deprecated, use 'output': 'file:outfile=<name>'
}


# Show the timing stats and memory demands.
PRINT_ALL_TIMING_MEMORY = {
    'max_depth': 10000,
    'min_bytes': 1,  # Only >=1
    'min_micros': 1,  # Only >=1
    'min_params': 0,
    'min_float_ops': 0,
    'order_by': 'name',
    'account_type_regexes': ['.*'],
    'start_name_regexes': ['.*'],
    'trim_name_regexes': [],
    'show_name_regexes': ['.*'],
    'hide_name_regexes': [],
    'account_displayed_op_only': True,
    'select': ['micros', 'bytes'],
    'output': 'stdout',
    'dump_to_file': ''  # Deprecated, use 'output': 'file:outfile=<name>'
}

# pylint: enable=bad-whitespace
# pylint: enable=bad-continuation


@deprecated('2018-01-01',
            'Use `tf.profiler.advise(graph, run_meta, options)`. See README.md')
def advise(graph, run_meta=None, tfprof_options=_DEFAULT_ADVISE_OPTIONS):
  return _advise(graph, run_meta, tfprof_options)


@deprecated('2018-01-01',
            'Use `tf.profiler.profile(graph, run_meta, op_log, cmd, options)`. '
            'Build `options` with `tf.profiler.ProfileOptionBuilder`. '
            'See README.md for details')
def print_model_analysis(graph,
                         run_meta=None,
                         op_log=None,
                         tfprof_cmd='scope',
                         tfprof_options=_DEFAULT_PROFILE_OPTIONS):
  return _profile(graph, run_meta, op_log, tfprof_cmd, tfprof_options)
