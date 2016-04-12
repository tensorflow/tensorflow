#!/usr/bin/python
# Copyright 2016 Google Inc. All Rights Reserved.
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

"""Python-based TensorFlow GRPC server.

Takes input arguments cluster_spec, job_name and task_id, and start a blocking
TensorFlow GRPC server.

Usage:
    grpc_tensorflow_server.py --cluster_spec=SPEC --job_name=NAME --task_id=ID

Where:
    SPEC is <JOB>(,<JOB>)*
    JOB  is <NAME>|<HOST:PORT>(;<HOST:PORT>)*
    NAME is a valid job name ([a-z][0-9a-z]*)
    HOST is a hostname or IP address
    PORT is a port number
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("cluster_spec", "",
                           """Cluster spec: SPEC.
    SPEC is <JOB>(,<JOB>)*,"
    JOB  is <NAME>|<HOST:PORT>(;<HOST:PORT>)*,"
    NAME is a valid job name ([a-z][0-9a-z]*),"
    HOST is a hostname or IP address,"
    PORT is a port number."
E.g., local|localhost:2222;localhost:2223, ps|ps0:2222;ps1:2222""")
tf.app.flags.DEFINE_string("job_name", "", "Job name: e.g., local")
tf.app.flags.DEFINE_integer("task_id", 0, "Task index, e.g., 0")
tf.app.flags.DEFINE_boolean("verbose", False, "Verbose mode")


def parse_cluster_spec(cluster_spec, cluster):
  """Parse content of cluster_spec string and inject info into cluster protobuf.

  Args:
    cluster_spec: cluster specification string, e.g.,
          "local|localhost:2222;localhost:2223"
    cluster: cluster protobuf.

  Raises:
    ValueError: if the cluster_spec string is invalid.
  """

  job_strings = cluster_spec.split(",")

  if not cluster_spec:
    raise ValueError("Empty cluster_spec string")

  for job_string in job_strings:
    job_def = cluster.job.add()

    if job_string.count("|") != 1:
      raise ValueError("Not exactly one instance of '|' in cluster_spec")

    job_name = job_string.split("|")[0]

    if not job_name:
      raise ValueError("Empty job_name in cluster_spec")

    job_def.name = job_name

    if FLAGS.verbose:
      print("Added job named \"%s\"" % job_name)

    job_tasks = job_string.split("|")[1].split(";")
    for i in range(len(job_tasks)):
      if not job_tasks[i]:
        raise ValueError("Empty task string at position %d" % i)

      job_def.tasks[i] = job_tasks[i]

      if FLAGS.verbose:
        print("  Added task \"%s\" to job \"%s\"" % (job_tasks[i], job_name))


def main(unused_args):
  # Create Protobuf ServerDef
  server_def = tf.train.ServerDef(protocol="grpc")

  # Cluster info
  parse_cluster_spec(FLAGS.cluster_spec, server_def.cluster)

  # Job name
  if not FLAGS.job_name:
    raise ValueError("Empty job_name")
  server_def.job_name = FLAGS.job_name

  # Task index
  if FLAGS.task_id < 0:
    raise ValueError("Invalid task_id: %d" % FLAGS.task_id)
  server_def.task_index = FLAGS.task_id

  # Create GRPC Server instance
  server = tf.train.Server(server_def)

  # join() is blocking, unlike start()
  server.join()


if __name__ == "__main__":
  tf.app.run()
