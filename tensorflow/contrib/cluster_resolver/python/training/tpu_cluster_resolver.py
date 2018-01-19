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
"""Implementation of Cluster Resolvers for Cloud TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from six.moves.urllib.request import Request
from six.moves.urllib.request import urlopen

from tensorflow.contrib.cluster_resolver.python.training.cluster_resolver import ClusterResolver
from tensorflow.python.training.server_lib import ClusterSpec

_GOOGLE_API_CLIENT_INSTALLED = True
try:
  from googleapiclient import discovery  # pylint: disable=g-import-not-at-top
  from oauth2client.client import GoogleCredentials  # pylint: disable=g-import-not-at-top
except ImportError:
  _GOOGLE_API_CLIENT_INSTALLED = False


class TPUClusterResolver(ClusterResolver):
  """Cluster Resolver for Google Cloud TPUs.

  This is an implementation of cluster resolvers for the Google Cloud TPU
  service. As Cloud TPUs are in alpha, you will need to specify a API definition
  file for this to consume, in addition to a list of Cloud TPUs in your Google
  Cloud Platform project.
  """

  def _requestComputeMetadata(self, path):
    req = Request('http://metadata/computeMetadata/v1/%s' % path,
                  headers={'Metadata-Flavor': 'Google'})
    resp = urlopen(req)
    return resp.read()

  def __init__(self,
               tpu_names,
               zone=None,
               project=None,
               job_name='tpu_worker',
               credentials='default',
               service=None):
    """Creates a new TPUClusterResolver object.

    The ClusterResolver will then use the parameters to query the Cloud TPU APIs
    for the IP addresses and ports of each Cloud TPU listed.

    Args:
      tpu_names: A list of names of the target Cloud TPUs.
      zone: Zone where the TPUs are located. If omitted or empty, we will assume
        that the zone of the TPU is the same as the zone of the GCE VM, which we
        will try to discover from the GCE metadata service.
      project: Name of the GCP project containing Cloud TPUs. If omitted or
        empty, we will try to discover the project name of the GCE VM from the
        GCE metadata service.
      job_name: Name of the TensorFlow job the TPUs belong to.
      credentials: GCE Credentials. If None, then we use default credentials
        from the oauth2client
      service: The GCE API object returned by the googleapiclient.discovery
        function. If you specify a custom service object, then the credentials
        parameter will be ignored.

    Raises:
      ImportError: If the googleapiclient is not installed.
    """

    if not project:
      project = self._requestComputeMetadata('/project/project-id')

    if not zone:
      zone_path = self._requestComputeMetadata('/instance/zone')
      zone = zone_path.split('/')[-1]

    self._project = project
    self._zone = zone
    self._tpu_names = tpu_names
    self._job_name = job_name
    self._credentials = credentials

    if credentials == 'default':
      if _GOOGLE_API_CLIENT_INSTALLED:
        self._credentials = GoogleCredentials.get_application_default()

    if service is None:
      if not _GOOGLE_API_CLIENT_INSTALLED:
        raise ImportError('googleapiclient must be installed before using the '
                          'TPU cluster resolver')

      self._service = discovery.build(
          'tpu', 'v1alpha1',
          credentials=self._credentials)
    else:
      self._service = service

  def get_master(self):
    """Get the ClusterSpec grpc master path.

    This returns the grpc path (grpc://1.2.3.4:8470) of first instance in the
    ClusterSpec returned by the cluster_spec function. This is suitable for use
    for the `master` argument in tf.Session() when you are using one TPU.

    Returns:
      string, the grpc path of the first instance in the ClusterSpec.

    Raises:
      ValueError: If none of the TPUs specified exists.
    """
    job_tasks = self.cluster_spec().job_tasks(self._job_name)
    if not job_tasks:
      raise ValueError('No TPUs exists with the specified names exist.')

    return 'grpc://' + job_tasks[0]

  def cluster_spec(self):
    """Returns a ClusterSpec object based on the latest TPU information.

    We retrieve the information from the GCE APIs every time this method is
    called.

    Returns:
      A ClusterSpec containing host information returned from Cloud TPUs.
    """
    worker_list = []

    for tpu_name in self._tpu_names:
      full_name = 'projects/%s/locations/%s/nodes/%s' % (
          self._project, self._zone, tpu_name)
      request = self._service.projects().locations().nodes().get(name=full_name)
      response = request.execute()

      instance_url = '%s:%s' % (response['ipAddress'], response['port'])
      worker_list.append(instance_url)

    return ClusterSpec({self._job_name: worker_list})
