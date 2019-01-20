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

# Note: It is intentional that we do not import tensorflow in this script. The
# machine that launches a TensorFlow k8s cluster does not have to have the
# Python package of TensorFlow installed on it.

# TODO(cais): Consider adding resource requests/limits to the pods.

# Worker pods will mount host volume /shared, as a convenient way to create
# shared storage among workers during local tests.
WORKER_RC = (
    """apiVersion: v1
kind: ReplicationController
metadata:
  name: {name_prefix}-worker{worker_id}
spec:
  replicas: 1
  template:
    metadata:
      labels:
        tf-worker: "{worker_id}"
        name-prefix: "{name_prefix}"
        job: "worker"
    spec:
      containers:
      - name: tf-worker{worker_id}
        image: {docker_image}
        args: [{args}]
        ports:
        - containerPort: {port}
        env: [{env_vars}]
        volumeMounts: [{volume_mounts}]
      volumes: [{volumes}]
""")
WORKER_SVC = (
    """apiVersion: v1
kind: Service
metadata:
  name: {name_prefix}-worker{worker_id}
  labels:
    tf-worker: "{worker_id}"
spec:
  ports:
  - port: {port}
    targetPort: {port}
  selector:
    tf-worker: "{worker_id}"
""")
WORKER_LB_SVC = (
    """apiVersion: v1
kind: Service
metadata:
  name: {name_prefix}-worker{worker_id}
  labels:
    tf-worker: "{worker_id}"
spec:
  type: LoadBalancer
  ports:
  - port: {port}
  selector:
    tf-worker: "{worker_id}"
""")
PARAM_SERVER_RC = (
    """apiVersion: v1
kind: ReplicationController
metadata:
  name: {name_prefix}-ps{param_server_id}
spec:
  replicas: 1
  template:
    metadata:
      labels:
        tf-ps: "{param_server_id}"
        name-prefix: "{name_prefix}"
        job: "ps"
    spec:
      containers:
      - name: tf-ps{param_server_id}
        image: {docker_image}
        args: [{args}]
        ports:
        - containerPort: {port}
        env: [{env_vars}]
        volumeMounts: [{volume_mounts}]
      volumes: [{volumes}]
""")
PARAM_SERVER_SVC = (
    """apiVersion: v1
kind: Service
metadata:
  name: {name_prefix}-ps{param_server_id}
  labels:
    tf-ps: "{param_server_id}"
spec:
  ports:
  - port: {port}
  selector:
    tf-ps: "{param_server_id}"
""")
PARAM_LB_SVC = ("""apiVersion: v1
kind: Service
metadata:
  name: {name_prefix}-ps{param_server_id}
  labels:
    tf-ps: "{param_server_id}"
spec:
  type: LoadBalancer
  ports:
  - port: {port}
  selector:
    tf-ps: "{param_server_id}"
""")
VOLUME_MOUNTS = '{name: shared, mountPath: /shared}'
VOLUMES = '{name: shared, hostPath: {path: /shared}}'
_ENV_VAR_TEMPLATE = '{name: "%s", value: "%s"}'
_ARG_TEMPLATE = '"--%s=%s"'


def GenerateConfig(num_workers,
                   num_param_servers,
                   port,
                   request_load_balancer,
                   docker_image,
                   name_prefix,
                   env_vars=None,
                   use_shared_volume=True,
                   use_cluster_spec=True):
  """Generate configuration strings.

  Args:
    num_workers: number of worker jobs.
    num_param_servers: number of ps server jobs.
    port: GRPC server port.
    request_load_balancer: request worker0 to be exposed on a public IP
      address via an external load balancer.
    docker_image: docker image to use.
    name_prefix: name to prepend to pod job names.
    env_vars: dictionary of environment variables to set.
    use_shared_volume: whether to add hostPath to /shared directory
      to the kubernetes config.
    use_cluster_spec: if true, pass --cluster_spec to worker and ps jobs.
      If false, pass --worker_hosts and --ps_hosts to worker and ps jobs.

  Returns:
    Kubernetes yaml config.
  """
  if env_vars is None:
    env_vars = {}
  env_str = ', '.join([_ENV_VAR_TEMPLATE % (name, value)
                       for name, value in env_vars.items()])
  config = ''
  common_args = GetCommonArgs(
      num_workers, num_param_servers, port, name_prefix, use_cluster_spec)
  for worker in range(num_workers):
    worker_args = {
        'job_name': 'worker',
        'task_id': worker
    }
    worker_args.update(common_args)
    arg_str = ', '.join([_ARG_TEMPLATE % (name, value)
                         for name, value in worker_args.items()])
    config += WORKER_RC.format(
        port=port,
        worker_id=worker,
        docker_image=docker_image,
        name_prefix=name_prefix,
        volume_mounts=VOLUME_MOUNTS if use_shared_volume else '',
        volumes=VOLUMES if use_shared_volume else '',
        args=arg_str,
        env_vars=env_str)
    config += '---\n'
    if request_load_balancer:
      config += WORKER_LB_SVC.format(port=port,
                                     worker_id=worker,
                                     name_prefix=name_prefix)
    else:
      config += WORKER_SVC.format(port=port,
                                  worker_id=worker,
                                  name_prefix=name_prefix)
    config += '---\n'

  for param_server in range(num_param_servers):
    ps_args = {
        'job_name': 'ps',
        'task_id': param_server
    }
    ps_args.update(common_args)
    arg_str = ', '.join([_ARG_TEMPLATE % (name, value)
                         for name, value in ps_args.items()])
    config += PARAM_SERVER_RC.format(
        port=port,
        param_server_id=param_server,
        docker_image=docker_image,
        name_prefix=name_prefix,
        volume_mounts=VOLUME_MOUNTS if use_shared_volume else '',
        volumes=VOLUMES if use_shared_volume else '',
        args=arg_str,
        env_vars=env_str)
    config += '---\n'
    if request_load_balancer:
      config += PARAM_LB_SVC.format(
          port=port, param_server_id=param_server, name_prefix=name_prefix)
    else:
      config += PARAM_SERVER_SVC.format(
          port=port, param_server_id=param_server, name_prefix=name_prefix)
    config += '---\n'

  return config


def WorkerClusterSpecString(num_workers,
                            num_param_servers,
                            port,
                            name_prefix):
  """Generates worker cluster spec."""
  return ClusterSpecString(num_workers, num_param_servers, port, name_prefix)


def ParamServerClusterSpecString(num_workers,
                                 num_param_servers,
                                 port,
                                 name_prefix):
  """Generates parameter server spec."""
  return ClusterSpecString(num_workers, num_param_servers, port,
                           name_prefix)


def ClusterSpecString(num_workers,
                      num_param_servers,
                      port,
                      name_prefix):
  """Generates general cluster spec."""
  spec = 'worker|'
  for worker in range(num_workers):
    spec += '%s-worker%d:%d' % (name_prefix, worker, port)
    if worker != num_workers-1:
      spec += ';'

  spec += ',ps|'
  for param_server in range(num_param_servers):
    spec += '%s-ps%d:%d' % (name_prefix, param_server, port)
    if param_server != num_param_servers-1:
      spec += ';'

  return spec


def GetCommonArgs(num_workers,
                  num_param_servers,
                  port,
                  name_prefix,
                  use_cluster_spec):
  """Get arguments common to both worker and ps jobs.

  Args:
    num_workers: number of workers.
    num_param_servers: number of ps servers.
    port: worker and ps port number.
    name_prefix: prefix to prepend to job names.
    use_cluster_spec: if true, pass --cluster_spec argument.
      If false, parse --worker_hosts and --ps_hosts arguments.

  Returns:
    A dictionary of argument names mapping to argument values.
  """
  common_args = {}
  if use_cluster_spec:
    common_args['cluster_spec'] = WorkerClusterSpecString(
        num_workers,
        num_param_servers,
        port,
        name_prefix)
  else:
    common_args['worker_hosts'] = WorkerHosts(num_workers, port, name_prefix)
    common_args['ps_hosts'] = PsHosts(num_param_servers, port, name_prefix)
  return common_args


def WorkerHosts(num_workers, port, name_prefix):
  worker_hosts = ['%s-worker%d:%d' % (name_prefix, i, port)
                  for i in range(num_workers)]
  return ','.join(worker_hosts)


def PsHosts(num_ps, port, name_prefix):
  ps_hosts = ['%s-ps%d:%d' % (name_prefix, i, port)
              for i in range(num_ps)]
  return ','.join(ps_hosts)
