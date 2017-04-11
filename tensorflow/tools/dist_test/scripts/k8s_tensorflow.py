#!/usr/bin/python
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

"""Generates YAML configuration files for distributed TensorFlow workers.

The workers will be run in a Kubernetes (k8s) container cluster.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import k8s_tensorflow_lib

# Note: It is intentional that we do not import tensorflow in this script. The
# machine that launches a TensorFlow k8s cluster does not have to have the
# Python package of TensorFlow installed on it.


DEFAULT_DOCKER_IMAGE = 'tensorflow/tf_grpc_test_server'
DEFAULT_PORT = 2222


def main():
  """Do arg parsing."""
  parser = argparse.ArgumentParser()
  parser.register(
      'type', 'bool', lambda v: v.lower() in ('true', 't', 'y', 'yes'))
  parser.add_argument('--num_workers',
                      type=int,
                      default=2,
                      help='How many worker pods to run')
  parser.add_argument('--num_parameter_servers',
                      type=int,
                      default=1,
                      help='How many paramater server pods to run')
  parser.add_argument('--grpc_port',
                      type=int,
                      default=DEFAULT_PORT,
                      help='GRPC server port (Default: %d)' % DEFAULT_PORT)
  parser.add_argument('--request_load_balancer',
                      type='bool',
                      default=False,
                      help='To request worker0 to be exposed on a public IP '
                      'address via an external load balancer, enabling you to '
                      'run client processes from outside the cluster')
  parser.add_argument('--docker_image',
                      type=str,
                      default=DEFAULT_DOCKER_IMAGE,
                      help='Override default docker image for the TensorFlow '
                      'GRPC server')
  parser.add_argument('--name_prefix',
                      type=str,
                      default='tf',
                      help='Prefix for job names. Jobs will be named as '
                      '<name_prefix>_worker|ps<task_id>')
  parser.add_argument('--use_shared_volume',
                      type='bool',
                      default=True,
                      help='Whether to mount /shared directory from host to '
                      'the pod')
  args = parser.parse_args()

  if args.num_workers <= 0:
    sys.stderr.write('--num_workers must be greater than 0; received %d\n'
                     % args.num_workers)
    sys.exit(1)
  if args.num_parameter_servers <= 0:
    sys.stderr.write(
        '--num_parameter_servers must be greater than 0; received %d\n'
        % args.num_parameter_servers)
    sys.exit(1)

  # Generate contents of yaml config
  yaml_config = k8s_tensorflow_lib.GenerateConfig(
      args.num_workers,
      args.num_parameter_servers,
      args.grpc_port,
      args.request_load_balancer,
      args.docker_image,
      args.name_prefix,
      env_vars=None,
      use_shared_volume=args.use_shared_volume)
  print(yaml_config)  # pylint: disable=superfluous-parens


if __name__ == '__main__':
  main()
