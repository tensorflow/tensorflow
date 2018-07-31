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
from absl import flags
from distutils.version import LooseVersion
import tensorflow as tf

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'gcp_project', None,
    'Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone',
    None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu', None, 'Name of the Cloud TPU for Cluster Resolvers. You must '
    'specify either this flag or --service_addr.')

# Tool specific parameters
flags.DEFINE_string(
    'service_addr', None, 'Address of TPU profiler service e.g. '
    'localhost:8466, you must specify either this flag or --tpu.')
flags.DEFINE_string(
    'workers_list', None, 'The list of worker TPUs that we are about to profile'
    ' e.g. 10.0.1.2, 10.0.1.3. You can specify this flag with --tpu or '
    '--service_addr to profile a subset of tpu nodes. You can also use only'
    '--tpu and leave this flag unspecified to profile all the tpus.')
flags.DEFINE_string(
    'logdir', None, 'Path of TensorBoard log directory e.g. /tmp/tb_log, '
    'gs://tb_bucket')
flags.DEFINE_integer('duration_ms', 2000, 'Duration of tracing in ms.')
flags.DEFINE_integer(
    'num_tracing_attempts', 3, 'Automatically retry N times when no trace '
    'event is collected.')
flags.DEFINE_boolean('include_dataset_ops', True,
                     'Set to false to profile longer TPU '
                     'device traces.')

FLAGS = flags.FLAGS
EXECUTABLE = 'data/capture_tpu_profile'
JOB_NAME = 'worker'


def get_workers_list(cluster_resolver):
  cluster_spec = cluster_resolver.cluster_spec()
  task_indices = cluster_spec.task_indices(JOB_NAME)
  workers_list = [
      cluster_spec.task_address(JOB_NAME, i).split(':')[0] for i in task_indices
  ]
  return ','.join(workers_list)


def run_main():
  tf.app.run(main)


def main(unused_argv=None):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf_version = tf.__version__
  print('TensorFlow version %s detected' % tf_version)

  if FLAGS.service_addr is None and FLAGS.tpu is None:
    sys.exit('You must specify either --service_addr or --tpu.')

  tpu_cluster_resolver = None
  if FLAGS.service_addr is not None:
    if FLAGS.tpu is not None:
      tf.logging.warn('Both --service_addr and --tpu are set. Ignoring '
                      '--tpu and using --service_addr.')
    service_addr = FLAGS.service_addr
  else:
    tpu_cluster_resolver = (
        tf.contrib.cluster_resolver.TPUClusterResolver(
            [FLAGS.tpu], zone=FLAGS.tpu_zone, project=FLAGS.gcp_project))
    service_addr = tpu_cluster_resolver.get_master()
  service_addr = service_addr.replace('grpc://', '').replace(':8470', ':8466')

  workers_list = ''
  if LooseVersion(tf_version) < LooseVersion('1.9'):
    tf.logging.warn('Attempt to profile with legacy support under TensorFlow '
                    'version %s' % tf_version)
  else:
    if FLAGS.workers_list is not None:
      workers_list = FLAGS.workers_list
    elif tpu_cluster_resolver is not None:
      workers_list = get_workers_list(tpu_cluster_resolver)

  if not FLAGS.logdir:
    sys.exit('logdir must be provided.')
  executable_path = os.path.join(os.path.dirname(__file__), EXECUTABLE)
  logdir = os.path.expandvars(os.path.expanduser(FLAGS.logdir))
  cmd = [executable_path]
  cmd.append('--logdir=' + logdir)
  cmd.append('--service_addr=' + service_addr)
  cmd.append('--workers_list=' + workers_list)
  cmd.append('--duration_ms=' + str(FLAGS.duration_ms))
  cmd.append('--num_tracing_attempts=' + str(FLAGS.num_tracing_attempts))
  cmd.append('--include_dataset_ops=' + str(FLAGS.include_dataset_ops).lower())
  subprocess.call(cmd)


if __name__ == '__main__':
  run_main()
