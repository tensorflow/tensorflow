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

import os

from six.moves.urllib.request import Request
from six.moves.urllib.request import urlopen

from tensorflow.contrib.cluster_resolver.python.training.cluster_resolver import ClusterResolver
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

_GOOGLE_API_CLIENT_INSTALLED = True
try:
  from googleapiclient import discovery  # pylint: disable=g-import-not-at-top
  from oauth2client.client import GoogleCredentials  # pylint: disable=g-import-not-at-top
except ImportError:
  _GOOGLE_API_CLIENT_INSTALLED = False


_GKE_ENV_VARIABLE = 'KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS'
_DEFAULT_ENV_VARIABLE = 'TPU_NAME'
_DISCOVERY_SERVICE_URL_ENV_VARIABLE = 'TPU_API_DISCOVERY_URL'


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
    return compat.as_bytes(resp.read())

  def _shouldResolve(self):
    if (self._tpu == compat.as_bytes('') or
        self._tpu == compat.as_bytes('local') or
        self._tpu.startswith(compat.as_bytes('/bns')) or
        self._tpu.startswith(compat.as_bytes('grpc://'))):
      return False
    return True

  @staticmethod
  def _inGke():
    """When running in GKE, the environment variable will be set."""
    return _GKE_ENV_VARIABLE in os.environ

  @staticmethod
  def _gkeMaster():
    return os.environ[_GKE_ENV_VARIABLE].split(',')[0]

  @staticmethod
  def _envVarFallback():
    if _DEFAULT_ENV_VARIABLE in os.environ:
      return os.environ[_DEFAULT_ENV_VARIABLE]
    return None

  @staticmethod
  def _discoveryUrl():
    return os.environ.get(_DISCOVERY_SERVICE_URL_ENV_VARIABLE)

  def __init__(self,
               tpu=None,
               zone=None,
               project=None,
               job_name='worker',
               coordinator_name=None,
               coordinator_address=None,
               credentials='default',
               service=None,
               discovery_url=None):
    """Creates a new TPUClusterResolver object.

    The ClusterResolver will then use the parameters to query the Cloud TPU APIs
    for the IP addresses and ports of each Cloud TPU listed.

    Args:
      tpu: Either a string, or a list of strings corresponding to the TPUs to
        use. If the single string is the empty string, the string 'local', or a
        string that begins with 'grpc://' or '/bns', then it is assumed to not
        correspond with a Cloud TPU and will instead be passed as the session
        master and no ClusterSpec propagation will be done.
      zone: Zone where the TPUs are located. If omitted or empty, we will assume
        that the zone of the TPU is the same as the zone of the GCE VM, which we
        will try to discover from the GCE metadata service.
      project: Name of the GCP project containing Cloud TPUs. If omitted or
        empty, we will try to discover the project name of the GCE VM from the
        GCE metadata service.
      job_name: Name of the TensorFlow job the TPUs belong to.
      coordinator_name: The name to use for the coordinator. Set to None if the
        coordinator should not be included in the computed ClusterSpec.
      coordinator_address: The address of the coordinator (typically an ip:port
        pair). If set to None, a TF server will be started. If coordinator_name
        is None, a TF server will not be started even if coordinator_address is
        None.
      credentials: GCE Credentials. If None, then we use default credentials
        from the oauth2client
      service: The GCE API object returned by the googleapiclient.discovery
        function. If you specify a custom service object, then the credentials
        parameter will be ignored.
      discovery_url: A URL template that points to the location of
        the discovery service. It should have two parameters {api} and
        {apiVersion} that when filled in produce an absolute URL to the
        discovery document for that service. The environment variable
        'TPU_API_DISCOVERY_URL' will override this.

    Raises:
      ImportError: If the googleapiclient is not installed.
      ValueError: If no TPUs are specified.
    """
    if isinstance(tpu, list):
      if not tpu:
        raise ValueError('At least one TPU must be specified.')
      if len(tpu) != 1:
        raise NotImplementedError(
            'Using multiple TPUs in a single session is not yet implemented')
      tpu = tpu[0]

    in_gke = self._inGke()
    # When using GKE with Cloud TPUs, the env variable will be set.
    if tpu is None:
      if in_gke:
        tpu = self._gkeMaster()
      else:
        tpu = self._envVarFallback()

    self._tpu = compat.as_bytes(tpu)  # self._tpu is always bytes
    self._job_name = job_name
    self._credentials = credentials

    should_resolve = self._shouldResolve()

    if not project and should_resolve:
      project = compat.as_str(
          self._requestComputeMetadata('project/project-id'))

    if not zone and should_resolve:
      zone_path = compat.as_str(self._requestComputeMetadata('instance/zone'))
      zone = zone_path.split('/')[-1]

    self._project = project
    self._zone = zone

    if credentials == 'default' and should_resolve:
      if _GOOGLE_API_CLIENT_INSTALLED:
        self._credentials = GoogleCredentials.get_application_default()

    if service is None and should_resolve:
      if not _GOOGLE_API_CLIENT_INSTALLED:
        raise ImportError('googleapiclient and oauth2client must be installed '
                          'before using the TPU cluster resolver. Execute: '
                          '`pip install --upgrade google-api-python-client` '
                          'and `pip install --upgrade oauth2client` to '
                          'install with pip.')

      final_discovery_url = self._discoveryUrl() or discovery_url
      if final_discovery_url:
        self._service = discovery.build(
            'tpu', 'v1alpha1',
            credentials=self._credentials,
            discoveryServiceUrl=final_discovery_url)
      else:
        self._service = discovery.build(
            'tpu', 'v1alpha1',
            credentials=self._credentials)
    else:
      self._service = service

    self._coordinator_name = coordinator_name
    if coordinator_name and not coordinator_address and (should_resolve or
                                                         in_gke):
      self._start_local_server()
    else:
      self._coordinator_address = coordinator_address

  def master(self):
    """Get the Master string to be used for the session.

    In the normal case, this returns the grpc path (grpc://1.2.3.4:8470) of
    first instance in the ClusterSpec returned by the cluster_spec function.

    If a non-TPU name is used when constructing a TPUClusterResolver, that will
    be returned instead (e.g. If the tpus argument's value when constructing
    this TPUClusterResolver was 'grpc://10.240.1.2:8470',
    'grpc://10.240.1.2:8470' will be returned).

    Returns:
      string, the connection string to use when creating a session.

    Raises:
      ValueError: If none of the TPUs specified exists.
    """
    if not self._shouldResolve():
      return self._tpu

    job_tasks = self.cluster_spec().job_tasks(self._job_name)
    if not job_tasks:
      raise ValueError('No TPUs exists with the specified names exist.')

    return 'grpc://' + job_tasks[0]

  def get_master(self):
    return self.master()

  def cluster_spec(self):
    """Returns a ClusterSpec object based on the latest TPU information.

    We retrieve the information from the GCE APIs every time this method is
    called.

    Returns:
      A ClusterSpec containing host information returned from Cloud TPUs.

    Raises:
      RuntimeError: If the provided TPU is not healthy.
    """
    ############################################################################
    # There are 5 potential cases this code must handle:
    #  1. [Normal case.] We should resolve the TPU name to a set of tasks, and
    #      a. Create a ClusterSpec that includes the coordinator job
    #      b. Create a ClusterSpec without the coordinator job.
    #  2. [GKE / No API Access.] We should not resolve the TPU name to a set of
    #     tasks and
    #      a. Create a ClusterSpec with the coordinator
    #      b. Create a ClusterSpec without the coordinator
    #  3. [Other (legacy non-gRPC).] We should return an empty ClusterSpec.
    ############################################################################

    if self._shouldResolve():
      # Case 1.
      full_name = 'projects/%s/locations/%s/nodes/%s' % (
          self._project, self._zone, compat.as_text(self._tpu))
      request = self._service.projects().locations().nodes().get(name=full_name)
      response = request.execute()

      if 'health' in response and response['health'] != 'HEALTHY':
        raise RuntimeError('TPU "%s" is unhealthy: "%s"' % (self._tpu,
                                                            response['health']))

      if 'networkEndpoints' in response:
        worker_list = [
            '%s:%s' % (endpoint['ipAddress'], endpoint['port'])
            for endpoint in response['networkEndpoints']
        ]
      else:
        # Fall back to the deprecated response format
        instance_url = '%s:%s' % (response['ipAddress'], response['port'])
        worker_list = [instance_url]

      cluster_spec = {self._job_name: worker_list}
    else:
      if not self._tpu.startswith(compat.as_bytes('grpc://')):
        # Case 3.
        return None
      # Case 2.
      cluster_spec = {self._job_name: [self._tpu[len(
          compat.as_bytes('grpc://')):]]}

    if self._coordinator_address:
      # {1, 2}.a
      cluster_spec[self._coordinator_name] = [self._coordinator_address]

    return server_lib.ClusterSpec(cluster_spec)

  def _start_local_server(self):
    address = self._requestComputeMetadata('instance/network-interfaces/0/ip')
    self._server = server_lib.Server(
        {
            'local': ['0.0.0.0:0']
        }, protocol='grpc', config=None, start=True)
    # self._server.target is of the form: grpc://ipaddress:port
    target = compat.as_bytes(self._server.target)
    splits = target.split(compat.as_bytes(':'))
    assert len(splits) == 3, self._server.target
    assert splits[0] == compat.as_bytes('grpc'), self._server.target
    self._coordinator_port = compat.as_text(splits[2])
    self._coordinator_address = '%s:%s' % (
        address, compat.as_text(self._coordinator_port))

  def __deepcopy__(self, memo):
    # TODO(b/73668574): Remove this once RunConfig avoids performing deepcopy.
    return self
