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
# ==============================================================================
"""Utils for creating PerfZero benchmarks from 
https://github.com/tensorflow/models/blob/master/official/benchmark/perfzero_benchmark.py"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl import logging
from absl.testing import flagsaver
import tensorflow as tf

FLAGS = flags.FLAGS


class PerfZeroBenchmark(tf.test.Benchmark):
  """Common methods used in PerfZero Benchmarks.

     Handles the resetting of flags between tests, loading of default_flags,
     overriding of defaults.  PerfZero (OSS) runs each test in a separate
     process reducing some need to reset the flags.
  """
  local_flags = None

  def __init__(self,
               output_dir=None,
               default_flags=None,
               root_data_dir=None,
               flag_methods=None,
               tpu=None):
    """Initialize class.

    Args:
      output_dir: Base directory to store all output for the test.
      default_flags: Set of flags to pass to model.
      root_data_dir: Optional param used by child classes to look for the
        dataset.
      flag_methods: Set of flag methods to run during setup.
      tpu: (optional) TPU name to use in a TPU benchmark.
    """
    if os.getenv('BENCHMARK_OUTPUT_DIR'):
      self.output_dir = os.getenv('BENCHMARK_OUTPUT_DIR')
    elif output_dir:
      self.output_dir = output_dir
    else:
      self.output_dir = '/tmp'
    self.default_flags = default_flags or {}
    self.flag_methods = flag_methods or {}

    if os.getenv('BENCHMARK_TPU'):
      resolved_tpu = os.getenv('BENCHMARK_TPU')
    elif tpu:
      resolved_tpu = tpu
    else:
      resolved_tpu = None

    if resolved_tpu:
      # TPU models are expected to accept a --tpu=name flag. PerfZero creates
      # the TPU at runtime and passes the TPU's name to this flag.
      self.default_flags['tpu'] = resolved_tpu

    logging.info('root_data_dir: %s', root_data_dir)

  @property
  def tpu(self):
    return self.default_flags.get('tpu', None)

  def _get_model_dir(self, folder_name):
    """Returns directory to store info, e.g. saved model and event log."""
    return os.path.join(self.output_dir, folder_name)

  def _setup(self):
    """Sets up and resets flags before each test."""
    logging.set_verbosity(logging.INFO)
    if PerfZeroBenchmark.local_flags is None:
      for flag_method in self.flag_methods:
        flag_method()
      # Loads flags to get defaults to then override. List cannot be empty.
      flags.FLAGS(['foo'])
      # Overrides flag values with defaults for the class of tests.
      for k, v in self.default_flags.items():
        setattr(FLAGS, k, v)
      saved_flag_values = flagsaver.save_flag_values()
      PerfZeroBenchmark.local_flags = saved_flag_values
    else:
      flagsaver.restore_flag_values(PerfZeroBenchmark.local_flags)
