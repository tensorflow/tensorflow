# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of ClusterResolvers for GCE instance groups."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export


_GOOGLE_API_CLIENT_INSTALLED = True
try:
  from googleapiclient import discovery  # pylint: disable=g-import-not-at-top
  from oauth2client.client import GoogleCredentials  # pylint: disable=g-import-not-at-top
except ImportError:
  _GOOGLE_API_CLIENT_INSTALLED = False


@tf_export('distribute.cluster_resolver.GCEClusterResolver')
class GCEClusterResolver(ClusterResolver):
  """ClusterResolver for Google Compute Engine.

  This is an implementation of cluster resolvers for the Google Compute Engine
  instance group platform. By specifying a project, zone, and instance group,
  this will retrieve the IP address of all the instances within the instance
  group and return a ClusterResolver object suitable for use for distributed
  TensorFlow.

  Note: this cluster resolver cannot retrieve `task_type`, `task_id` or
  `rpc_layer`. To use it with some distribution strategies like
  `tf.distribute.experimental.MultiWorkerMirroredStrategy`, you will need to
  specify `task_type` and `task_id` in the constructor.

  Usage example with tf.distribute.Strategy:

    ```Python
    # On worker 0
    cluster_resolver = GCEClusterResolver("my-project", "us-west1",
                                          "my-instance-group",
                                          task_type="worker", task_id=0)
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)

    # On worker 1
    cluster_resolver = GCEClusterResolver("my-project", "us-west1",
                                          "my-instance-group",
                                          task_type="worker", task_id=1)
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=cluster_resolver)
    ```
  """

  def __init__(self,
               project,
               zone,
               instance_group,
               port,
               task_type='worker',
               task_id=0,
               rpc_layer='grpc',
               credentials='default',
               service=None):
    """Creates a new GCEClusterResolver object.

    This takes in a few parameters and creates a GCEClusterResolver project. It
    will then use these parameters to query the GCE API for the IP addresses of
    each instance in the instance group.

    Args:
      project: Name of the GCE project.
      zone: Zone of the GCE instance group.
      instance_group: Name of the GCE instance group.
      port: Port of the listening TensorFlow server (default: 8470)
      task_type: Name of the TensorFlow job this GCE instance group of VM
        instances belong to.
      task_id: The task index for this particular VM, within the GCE
        instance group. In particular, every single instance should be assigned
        a unique ordinal index within an instance group manually so that they
        can be distinguished from each other.
      rpc_layer: The RPC layer TensorFlow should use to communicate across
        instances.
      credentials: GCE Credentials. If nothing is specified, this defaults to
        GoogleCredentials.get_application_default().
      service: The GCE API object returned by the googleapiclient.discovery
        function. (Default: discovery.build('compute', 'v1')). If you specify a
        custom service object, then the credentials parameter will be ignored.

    Raises:
      ImportError: If the googleapiclient is not installed.
    """
    self._project = project
    self._zone = zone
    self._instance_group = instance_group
    self._task_type = task_type
    self._task_id = task_id
    self._rpc_layer = rpc_layer
    self._port = port
    self._credentials = credentials

    if credentials == 'default':
      if _GOOGLE_API_CLIENT_INSTALLED:
        self._credentials = GoogleCredentials.get_application_default()

    if service is None:
      if not _GOOGLE_API_CLIENT_INSTALLED:
        raise ImportError('googleapiclient must be installed before using the '
                          'GCE cluster resolver')
      self._service = discovery.build(
          'compute', 'v1',
          credentials=self._credentials)
    else:
      self._service = service

  def cluster_spec(self):
    """Returns a ClusterSpec object based on the latest instance group info.

    This returns a ClusterSpec object for use based on information from the
    specified instance group. We will retrieve the information from the GCE APIs
    every time this method is called.

    Returns:
      A ClusterSpec containing host information retrieved from GCE.
    """
    request_body = {'instanceState': 'RUNNING'}
    request = self._service.instanceGroups().listInstances(
        project=self._project,
        zone=self._zone,
        instanceGroups=self._instance_group,
        body=request_body,
        orderBy='name')

    worker_list = []

    while request is not None:
      response = request.execute()

      items = response['items']
      for instance in items:
        instance_name = instance['instance'].split('/')[-1]

        instance_request = self._service.instances().get(
            project=self._project,
            zone=self._zone,
            instance=instance_name)

        if instance_request is not None:
          instance_details = instance_request.execute()
          ip_address = instance_details['networkInterfaces'][0]['networkIP']
          instance_url = '%s:%s' % (ip_address, self._port)
          worker_list.append(instance_url)

      request = self._service.instanceGroups().listInstances_next(
          previous_request=request,
          previous_response=response)

    worker_list.sort()
    return ClusterSpec({self._task_type: worker_list})

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    task_type = task_type if task_type is not None else self._task_type
    task_id = task_id if task_id is not None else self._task_id

    if task_type is not None and task_id is not None:
      master = self.cluster_spec().task_address(task_type, task_id)
      if rpc_layer or self._rpc_layer:
        return '%s://%s' % (rpc_layer or self._rpc_layer, master)
      else:
        return master

    return ''

  @property
  def task_type(self):
    return self._task_type

  @property
  def task_id(self):
    return self._task_id

  @task_type.setter
  def task_type(self, task_type):
    raise RuntimeError(
        'You cannot reset the task_type of the GCEClusterResolver after it has '
        'been created.')

  @task_id.setter
  def task_id(self, task_id):
    self._task_id = task_id

  @property
  def rpc_layer(self):
    return self._rpc_layer

  @rpc_layer.setter
  def rpc_layer(self, rpc_layer):
    self._rpc_layer = rpc_layer
