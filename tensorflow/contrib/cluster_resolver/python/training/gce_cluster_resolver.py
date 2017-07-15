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
"""Implementation of Cluster Resolvers for GCE Instance Groups."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.contrib.cluster_resolver.python.training.cluster_resolver import ClusterResolver
from tensorflow.python.training.server_lib import ClusterSpec

_GOOGLE_API_CLIENT_INSTALLED = True
try:
  from googleapiclient import discovery  # pylint: disable=g-import-not-at-top
except ImportError:
  _GOOGLE_API_CLIENT_INSTALLED = False


class GceClusterResolver(ClusterResolver):
  """Cluster Resolver for Google Compute Engine.

  This is an implementation of cluster resolvers for the Google Compute Engine
  instance group platform. By specifying a project, zone, and instance group,
  this will retrieve the IP address of all the instances within the instance
  group and return a Cluster Resolver object suitable for use for distributed
  TensorFlow.
  """

  def __init__(self,
               project,
               zone,
               instance_group,
               port,
               job_name='worker',
               credentials=None,
               service=None):
    """Creates a new GceClusterResolver object.

    This takes in a few parameters and creates a GceClusterResolver project. It
    will then use these parameters to query the GCE API for the IP addresses of
    each instance in the instance group.

    Args:
      project: Name of the GCE project
      zone: Zone of the GCE instance group
      instance_group: Name of the GCE instance group
      port: Port of the listening TensorFlow server (default: 8470)
      job_name: Name of the TensorFlow job this set of instances belongs to
      credentials: GCE Credentials. This defaults to
        GoogleCredentials.get_application_default()
      service: The GCE API object returned by the googleapiclient.discovery
        function. (Default: discovery.build('compute', 'v1')). If you specify a
        custom service object, then the credentials parameter will be ignored.

    Raises:
      ImportError: If the googleapiclient is not installed.
    """
    self._project = project
    self._zone = zone
    self._instance_group = instance_group
    self._job_name = job_name
    self._port = port
    if service is None:
      if _GOOGLE_API_CLIENT_INSTALLED is True:
        self._service = discovery.build('compute', 'v1',
                                        credentials=credentials)
      else:
        raise ImportError('googleapiclient must be installed before using the '
                          'GCE cluster resolver')
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
    return ClusterSpec({self._job_name: worker_list})
