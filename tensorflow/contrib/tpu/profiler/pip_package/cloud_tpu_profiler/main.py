# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Wraps capture_tpu_profile binary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import sys

import tensorflow as tf

tf.flags.DEFINE_string('service_addr', '',
                       'Address of TPU profiler service e.g. localhost:8466')
tf.flags.DEFINE_string('logdir', '',
                       'Path of TensorBoard log directory e.g. /tmp/tb_log')
tf.flags.DEFINE_integer('duration_ms', 2000, 'Duration of tracing in ms.')

FLAGS = tf.flags.FLAGS
EXECUTABLE = 'data/capture_tpu_profile'


def run_main():
  tf.app.run(main)


def main(unused_argv=None):
  if not FLAGS.service_addr or not FLAGS.logdir:
    sys.exit('service_addr and logdir must be provided.')
  executable_path = os.path.join(os.path.dirname(__file__), EXECUTABLE)
  logdir = os.path.expandvars(os.path.expanduser(FLAGS.logdir))
  cmd = [executable_path]
  cmd.append('--logdir='+logdir)
  cmd.append('--service_addr='+FLAGS.service_addr)
  cmd.append('--duration_ms='+str(FLAGS.duration_ms))
  subprocess.call(cmd)


if __name__ == '__main__':
  run_main()
