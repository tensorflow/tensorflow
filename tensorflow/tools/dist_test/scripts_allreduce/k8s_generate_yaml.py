#!/usr/bin/python
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Generates YAML configuration file for allreduce-based distributed TensorFlow.

The workers will be run in a Kubernetes (k8s) container cluster.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import k8s_generate_yaml_lib

# Note: It is intentional that we do not import tensorflow in this script. The
# machine that launches a TensorFlow k8s cluster does not have to have the
# Python package of TensorFlow installed on it.

DEFAULT_DOCKER_IMAGE = 'tensorflow/tensorflow:latest-devel'
DEFAULT_PORT = 22

DEFAULT_CONFIG_MAP = 'k8s-config-map'
DEFAULT_DEPLOYMENT = 'k8s-ml-deployment'


def main():
  """Do arg parsing."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--docker_image',
      type=str,
      default=DEFAULT_DOCKER_IMAGE,
      help='Override default docker image for the TensorFlow')
  parser.add_argument(
      '--num_containers',
      type=int,
      default=0,
      help='How many docker containers to launch')
  parser.add_argument(
      '--config_map',
      type=str,
      default=DEFAULT_CONFIG_MAP,
      help='Override default config map')
  parser.add_argument(
      '--deployment',
      type=str,
      default=DEFAULT_DEPLOYMENT,
      help='Override default deployment')
  parser.add_argument(
      '--ssh_port',
      type=int,
      default=DEFAULT_PORT,
      help='Override default ssh port (Default: %d)' % DEFAULT_PORT)
  parser.add_argument(
      '--use_hostnet',
      type=int,
      default=0,
      help='Used to enable host network mode (Default: 0)')
  parser.add_argument(
      '--use_shared_volume',
      type=int,
      default=0,
      help='Used to mount shared volume (Default: 0)')
  args = parser.parse_args()

  if args.num_containers <= 0:
    sys.stderr.write('--num_containers must be greater than 0; received %d\n' %
                     args.num_containers)
    sys.exit(1)

  # Generate contents of yaml config
  yaml_config = k8s_generate_yaml_lib.GenerateConfig(
      args.docker_image, args.num_containers, args.config_map, args.deployment,
      args.ssh_port, args.use_hostnet, args.use_shared_volume)
  print(yaml_config)  # pylint: disable=superfluous-parens


if __name__ == '__main__':
  main()
