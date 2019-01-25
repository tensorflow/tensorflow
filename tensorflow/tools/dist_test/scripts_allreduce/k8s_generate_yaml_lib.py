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

from Crypto.PublicKey import RSA

# Note: It is intentional that we do not import tensorflow in this script. The
# machine that launches a TensorFlow k8s cluster does not have to have the
# Python package of TensorFlow installed on it.

CONFIG_MAP = ("""apiVersion: v1
kind: ConfigMap
metadata:
  name: {config_map}
data:
  privatekey: |+
    {private_key}

  publickey: |+
    {public_key}

  start: |+
    mkdir /root/.ssh
    mkdir /var/run/sshd
    cp /tmp/configs/* /root/.ssh
    cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
    chmod 600 -R /root/.ssh
    {change_ssh_port}
    /usr/bin/ssh-keygen -A
    /usr/sbin/sshd -De

  sshconfig: |+
    Host *
      Port {port}
      StrictHostKeyChecking no

""")

DEPLOYMENT = ("""apiVersion: apps/v1beta1
kind: Deployment
metadata:
  name: {deployment}
  labels:
    app: k8s-ml
spec:
  replicas: {num_containers}
  selector:
    matchLabels:
      app: k8s-ml
  template:
    metadata:
      labels:
        app: k8s-ml
    spec: {hostnet}
      securityContext:
        runAsUser: 0
      containers:
      - name: ml
        image: {docker_image}
        command:
        - /bin/bash
        - -x
        - /tmp/scripts/start.sh
        ports:
        - containerPort: {port}
        env: [{env_vars}]
        securityContext:
          privileged: true
        volumeMounts: {volume_mounts}
        - name: dshm
          mountPath: /dev/shm
        - name: sshkeys
          mountPath: /tmp/configs
        - name: scripts
          mountPath: /tmp/scripts
      volumes: {volumes}
      - name: dshm
        emptyDir:
          medium: Memory
      - name: sshkeys
        configMap:
          name: {config_map}
          items:
          - key: publickey
            path: id_rsa.pub
          - key: privatekey
            path: id_rsa
          - key: sshconfig
            path: config
      - name: scripts
        configMap:
          name: {config_map}
          items:
          - key: start
            path: start.sh
""")
_ENV_VAR_TEMPLATE = '{name: "%s", value: "%s"}'


def GenerateConfig(docker_image,
                   num_containers,
                   config_map,
                   deployment,
                   port,
                   use_hostnet,
                   use_shared_volume,
                   env_vars=None):
  """Generate configuration strings.

  Args:
    docker_image: docker image to use.
    num_containers: number of containers.
    config_map: config map.
    deployment: deployment.
    port: ssh port.
    use_hostnet: Used to enable host network mode.
    use_shared_volume: Used to mount shared volume.
    env_vars: dictionary of environment variables to set.

  Returns:
    Kubernetes yaml config.
  """

  if env_vars is None:
    env_vars = {}
  env_str = ', '.join(
      [_ENV_VAR_TEMPLATE % (name, value) for name, value in env_vars.items()])

  private_key, public_key = generate_RSA(2048)

  CHANGE_SSH_PORT = get_change_ssh_port(use_hostnet, port)

  config = CONFIG_MAP.format(
      port=port,
      config_map=config_map,
      private_key=private_key,
      public_key=public_key,
      change_ssh_port=CHANGE_SSH_PORT,
      env_vars=env_str)
  config += '---\n\n'

  HOST_NET = get_hostnet(use_hostnet)
  VOLUME_MOUNTS = get_volume_mounts(use_shared_volume)
  VOLUMES = get_volumes(use_shared_volume)

  config += DEPLOYMENT.format(
      deployment=deployment,
      num_containers=num_containers,
      docker_image=docker_image,
      port=port,
      config_map=config_map,
      hostnet=HOST_NET,
      volume_mounts=VOLUME_MOUNTS,
      volumes=VOLUMES,
      env_vars=env_str)

  return config


def generate_RSA(bits=2048, exponent=65537):
  key = RSA.generate(bits, e=exponent)
  pubkey = key.publickey()

  private_key = key.exportKey('PEM')
  public_key = pubkey.exportKey('OpenSSH')

  # Format private_key in yaml file
  space_before = ' ' * 4
  private_key_split = private_key.split('\n')
  private_key = ''.join(('' if index == 0 else space_before) + line.strip() \
        + ('\n' if index != len(private_key_split) - 1 else '') \
        for index, line in enumerate(private_key_split))

  return private_key, public_key


def get_change_ssh_port(use_hostnet, port):
  if use_hostnet == 1:
    return r"sed -i '/Port 22/c\Port {}' /etc/ssh/sshd_config".format(port)

  return ''


def get_hostnet(use_hostnet):
  if use_hostnet == 1:
    return """
      hostNetwork: true
      hostIPC: true"""

  return ''


def get_volume_mounts(use_shared_volume):
  if use_shared_volume == 1:
    return """
        - name: shared
          mountPath: /shared"""

  return ''


def get_volumes(use_shared_volume):
  if use_shared_volume == 1:
    return """
       - name: shared
         hostPath:
           path: /shared"""

  return ''
