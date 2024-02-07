# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-import-not-at-top
"""Python-based TPU Worker GRPC server.

Start a blocking TPU Worker GRPC server.

Usage:
    python3 grpc_tpu_worker.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import requests

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.training import server_lib


def get_metadata(key):
  return requests.get(
      'http://metadata.google.internal/computeMetadata'
      '/v1/instance/attributes/{}'.format(key),
      headers={
          'Metadata-Flavor': 'Google'
      }).text


def get_host_ip():
  return requests.get(
      'http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip',
      headers={
          'Metadata-Flavor': 'Google'
      }).text


def setup_env_vars():
  """Set environment variables."""
  worker_id = get_metadata('agent-worker-number')
  accelerator_type = get_metadata('accelerator-type')
  worker_network_endpoints = get_metadata('worker-network-endpoints')
  os.environ['TPU_STDERR_LOG_LEVEL'] = '0'
  os.environ['CLOUD_TPU_TASK_ID'] = worker_id
  os.environ['TPU_LOCK_DEVICE'] = 'true'
  os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '2,2,1'
  accelerator_type_to_host_bounds = {
      # v2
      'v2-8': '1,1,1',
      'v2-32': '2,2,1',
      'v2-128': '4,4,1',
      'v2-256': '4,8,1',
      'v2-512': '8,8,1',
      # v3
      'v3-8': '1,1,1',
      'v3-32': '2,2,1',
      'v3-64': '2,4,1',
      'v3-128': '4,4,1',
      'v3-256': '4,8,1',
      'v3-512': '8,8,1',
      'v3-1024': '8,16,1',
      'v3-2048': '16,16,1',
      # v4
      'v4-8': '1,1,1',
      'v4-16': '1,1,2',
      'v4-32': '1,1,4',
      'v4-64': '1,2,4',
      'v4-128': '2,2,4',
      'v4-256': '2,2,8',
      'v4-512': '2,4,8',
      'v4-1024': '4,4,8',
      'v4-2048': '4,4,16',
      'v4-4096': '4,8,16',
  }

  os.environ['TPU_HOST_BOUNDS'] = accelerator_type_to_host_bounds[
      accelerator_type]
  os.environ['TPU_MESH_CONTROLLER_ADDRESS'] = worker_network_endpoints.split(
      ',')[0].split(':')[2] + ':8476'
  os.environ['TPU_MESH_CONTROLLER_PORT'] = '8476'

  os.environ['TPU_STDERR_LOG_LEVEL'] = '0'

  if accelerator_type not in ['v4-8', 'v4-16', 'v4-32', 'v4-64']:
    os.environ['TPU_TOPOLOGY_WRAP'] = 'true,true,true'

  # Set the hostname override.
  os.environ['TPU_HOSTNAME_OVERRIDE'] = get_host_ip()


def main(unused_args):
  # Create Protobuf ServerDef.
  server_def = tensorflow_server_pb2.ServerDef(protocol='grpc')
  job_def = server_def.cluster.job.add()
  job_def.name = 'tpu_worker'
  job_def.tasks[0] = 'localhost:8470'
  server_def.job_name = 'tpu_worker'
  server_def.task_index = 0

  config = config_pb2.ConfigProto()

  # Create GRPC Server instance
  server = server_lib.Server(server_def, config=config)

  # join() is blocking, unlike start()
  server.join()


def run():
  parser = argparse.ArgumentParser()

  _, unparsed = parser.parse_known_args()
  # Must set environment variables before importing tensorflow.
  setup_env_vars()
  from tensorflow.python.platform import app
  app.run(main=main, argv=[sys.argv[0]] + unparsed)


if __name__ == '__main__':
  run()
