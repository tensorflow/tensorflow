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
# =============================================================================
"""Cloud TPU profiler client."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

from absl import app
from absl import flags
from distutils.version import LooseVersion

from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver as resolver
from tensorflow.python.profiler import profiler_client
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.framework import errors
from tensorflow.python.framework import versions
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.profiler import version as profiler_version

FLAGS = flags.FLAGS

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
    ' e.g. 10.0.1.2:8466, 10.0.1.3:8466. You can specify this flag with --tpu '
    'or --service_addr to profile a subset of tpu nodes. You can also use only'
    '--tpu and leave this flag unspecified to profile all the tpus.')
flags.DEFINE_string(
    'logdir', None, 'Path of TensorBoard log directory e.g. /tmp/tb_log, '
    'gs://tb_bucket')
flags.DEFINE_integer('duration_ms', 0,
                     'Duration of tracing or monitoring in ms.')
flags.DEFINE_integer(
    'num_tracing_attempts', 3, 'Automatically retry N times when no trace '
    'event is collected.')
flags.DEFINE_boolean('include_dataset_ops', True, 'Deprecated.')
flags.DEFINE_integer(
    'host_tracer_level', 2, 'Adjust host tracer level to control the verbosity '
    ' of the TraceMe event being collected.')

# Monitoring parameters
flags.DEFINE_integer(
    'monitoring_level', 0, 'Choose a monitoring level between '
    '1 and 2 to monitor your TPU job continuously. Level 2 is more verbose than'
    ' level 1 and shows more metrics.')
flags.DEFINE_integer(
    'num_queries', 100,
    'This script will run monitoring for num_queries before it stops.')
flags.DEFINE_boolean('display_timestamp', True, 'Deprecated.')


def get_workers_list(cluster_resolver):
  """Returns a comma separated list of TPU worker host:port pairs.

  Gets cluster_spec from cluster_resolver. Use the worker's task indices to
  obtain and return a list of host:port pairs.

  Args:
    cluster_resolver: TensorFlow TPUClusterResolver instance.

  Returns:
    A string of comma separated list of host:port pairs. For example:
    '10.2.0.1:8466,10.2.0.2:8466,10.2.0.3:8466,10.2.0.4:8466'

  Raises:
    UnavailableError: cluster_resolver doesn't contain a valid cluster_spec.
  """
  worker_job_name = 'worker'
  cluster_spec = cluster_resolver.cluster_spec()
  if not cluster_spec:
    raise errors.UnavailableError(
        'None', 'None',
        'Cluster spec not found, your client must run in GCE environment.')
  task_indices = cluster_spec.task_indices(worker_job_name)
  workers_list = [
      cluster_spec.task_address(worker_job_name, i).replace(':8470', ':8466')
      for i in task_indices
  ]
  return ','.join(workers_list)


def monitoring_helper(service_addr, duration_ms, monitoring_level, num_queries):
  """Helper function to print monitoring results.

  Helper function to print monitoring results for num_queries times.

  Args:
    service_addr: Address of the TPU profiler service.
    duration_ms: Duration of one monitoring sample in milliseconds.
    monitoring_level: An integer between 1 and 2. Level 2 is more verbose than
      level 1 and shows more metrics.
    num_queries: Number of monitoring samples to collect.
  """
  if monitoring_level <= 0 or monitoring_level > 2:
    sys.exit('Please choose a monitoring level between 1 and 2.')

  for query in range(0, num_queries):
    res = profiler_client.monitor(service_addr, duration_ms, monitoring_level)
    print('Cloud TPU Monitoring Results (Sample ', query, '):\n\n', res)


def run_main():
  app.run(main)


def main(unused_argv=None):
  logging.set_verbosity(logging.INFO)
  tf_version = versions.__version__
  print('TensorFlow version %s detected' % tf_version)
  print('Welcome to the Cloud TPU Profiler v%s' % profiler_version.__version__)

  if LooseVersion(tf_version) < LooseVersion('2.2.0'):
    sys.exit('You must install tensorflow >= 2.2.0 to use this plugin.')

  if not FLAGS.service_addr and not FLAGS.tpu:
    sys.exit('You must specify either --service_addr or --tpu.')

  tpu_cluster_resolver = None
  if FLAGS.service_addr:
    if FLAGS.tpu:
      logging.warn('Both --service_addr and --tpu are set. Ignoring '
                   '--tpu and using --service_addr.')
    service_addr = FLAGS.service_addr
  else:
    try:
      tpu_cluster_resolver = (
          resolver.TPUClusterResolver([FLAGS.tpu],
                                      zone=FLAGS.tpu_zone,
                                      project=FLAGS.gcp_project))
      service_addr = tpu_cluster_resolver.get_master()
    except (ValueError, TypeError):
      sys.exit('Failed to find TPU %s in zone %s project %s. You may use '
               '--tpu_zone and --gcp_project to specify the zone and project of'
               ' your TPU.' % (FLAGS.tpu, FLAGS.tpu_zone, FLAGS.gcp_project))
  service_addr = service_addr.replace('grpc://', '').replace(':8470', ':8466')

  workers_list = ''
  if FLAGS.workers_list is not None:
    workers_list = FLAGS.workers_list
  elif tpu_cluster_resolver is not None:
    workers_list = get_workers_list(tpu_cluster_resolver)

  # If profiling duration was not set by user or set to a non-positive value,
  # we set it to a default value of 1000ms.
  duration_ms = FLAGS.duration_ms if FLAGS.duration_ms > 0 else 1000

  if FLAGS.monitoring_level > 0:
    print('Since monitoring level is provided, profile', service_addr, ' for ',
          FLAGS.duration_ms, ' ms and show metrics for ', FLAGS.num_queries,
          ' time(s).')
    monitoring_helper(service_addr, duration_ms, FLAGS.monitoring_level,
                      FLAGS.num_queries)
  else:
    if not FLAGS.logdir:
      sys.exit('You must specify either --logdir or --monitoring_level.')

    if not gfile.Exists(FLAGS.logdir):
      gfile.MakeDirs(FLAGS.logdir)

    try:
      if LooseVersion(tf_version) < LooseVersion('2.3.0'):
        profiler_client.trace(service_addr, os.path.expanduser(FLAGS.logdir),
                              duration_ms, workers_list,
                              FLAGS.num_tracing_attempts)
      else:
        options = profiler.ProfilerOptions(
            host_tracer_level=FLAGS.host_tracer_level)
        profiler_client.trace(service_addr, os.path.expanduser(FLAGS.logdir),
                              duration_ms, workers_list,
                              FLAGS.num_tracing_attempts, options)
    except errors.UnavailableError:
      sys.exit(0)


if __name__ == '__main__':
  run_main()
