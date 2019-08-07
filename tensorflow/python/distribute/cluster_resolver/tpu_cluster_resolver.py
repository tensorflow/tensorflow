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

import collections
import os
import re

from six.moves import urllib
from six.moves.urllib.error import URLError
from six.moves.urllib.request import Request
from six.moves.urllib.request import urlopen

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import get_accelerator_devices
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export

_GOOGLE_API_CLIENT_INSTALLED = True
try:
  from googleapiclient import discovery  # pylint: disable=g-import-not-at-top
  from oauth2client.client import GoogleCredentials  # pylint: disable=g-import-not-at-top
except ImportError:
  _GOOGLE_API_CLIENT_INSTALLED = False

_GKE_ENV_VARIABLE = 'KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS'
_ENDPOINTS_SEPARATOR = ','
_DEFAULT_ENV_VARIABLE = 'TPU_NAME'
_DISCOVERY_SERVICE_URL_ENV_VARIABLE = 'TPU_API_DISCOVERY_URL'

_TPU_DEVICE_REGEX = re.compile(
    r'.*task:(?P<host_id>\d+)/.*device:TPU:(?P<core_id>\d+)$')
_TPU_CONN_RETRIES = 120

_GCE_METADATA_ENDPOINT = 'http://metadata.google.internal'

DeviceDetails = collections.namedtuple(
    'DeviceDetails', ['device_map', 'total_cores'])


def is_running_in_gce():
  """Checks for GCE presence by attempting to query the metadata service."""
  try:
    req = Request(
        '%s/computeMetadata/v1' % _GCE_METADATA_ENDPOINT,
        headers={'Metadata-Flavor': 'Google'})
    resp = urllib.request.urlopen(req, timeout=1)
    info = resp.info()
    if 'Metadata-Flavor' in info and info['Metadata-Flavor'] == 'Google':
      return True
  except URLError:
    pass
  return False


@tf_export('distribute.cluster_resolver.TPUClusterResolver')
class TPUClusterResolver(ClusterResolver):
  """Cluster Resolver for Google Cloud TPUs.

  This is an implementation of cluster resolvers for the Google Cloud TPU
  service. As Cloud TPUs are in alpha, you will need to specify a API definition
  file for this to consume, in addition to a list of Cloud TPUs in your Google
  Cloud Platform project.

  TPUClusterResolver supports the following distinct environments:
  Google Compute Engine
  Google Kubernetes Engine
  Google internal
  """

  def _tpu_service(self):
    """Creates a new Cloud TPU API object.

    This works around an issue where the underlying HTTP connection sometimes
    times out when the script has been running for too long. Other methods in
    this object call this method to get a new API object whenever they need
    to communicate with the Cloud API.

    Returns:
      A Google Cloud TPU API object.
    """
    if self._service:
      return self._service

    credentials = self._credentials
    if credentials is None or credentials == 'default':
      credentials = GoogleCredentials.get_application_default()

    if self._discovery_url:
      return discovery.build(
          'tpu',
          'v1',
          credentials=credentials,
          discoveryServiceUrl=self._discovery_url,
          cache_discovery=False)
    else:
      return discovery.build(
          'tpu', 'v1', credentials=credentials, cache_discovery=False)

  def _request_compute_metadata(self, path):
    req = Request('%s/computeMetadata/v1/%s' % (_GCE_METADATA_ENDPOINT, path),
                  headers={'Metadata-Flavor': 'Google'})
    resp = urlopen(req)
    return compat.as_bytes(resp.read())

  def _is_google_environment(self):
    return (
        self._tpu == compat.as_bytes('') or
        self._tpu == compat.as_bytes('local') or
        self._tpu.startswith(compat.as_bytes('localhost:')) or
        self._tpu.startswith(compat.as_bytes('/bns')) or
        self._tpu.startswith(compat.as_bytes('uptc://')))

  def _should_resolve(self):
    if isinstance(self._should_resolve_override, bool):
      return self._should_resolve_override
    else:
      return not (self._tpu.startswith(compat.as_bytes('grpc://')) or
                  self._is_google_environment())

  @staticmethod
  def _get_device_dict_and_cores(devices):
    """Returns a dict of hosts to cores and total cores given devices names.

    Returns a namedtuple with two attributes:
      device_map: A map of host_ids to a list of core_ids.
      total_cores: The total number of cores within the TPU system.

    Args:
      devices: A list of devices returned by session.list_devices()
    """
    device_map = collections.defaultdict(list)
    num_cores = 0
    for device in devices:
      match = _TPU_DEVICE_REGEX.match(device.name)
      if match:
        host_id = match.group('host_id')
        core_id = match.group('core_id')
        device_map[host_id].append(core_id)
        num_cores += 1
    return DeviceDetails(device_map, num_cores)

  @staticmethod
  def _verify_and_return_same_core_count(device_dict):
    """Verifies that every device in device_dict has the same # of cores."""
    num_cores_per_host_set = (
        {len(core_ids) for core_ids in device_dict.values()})
    if len(num_cores_per_host_set) != 1:
      raise RuntimeError('TPU cores on each device is not the same. This '
                         'should never happen. Devices: {}'.format(device_dict))
    return num_cores_per_host_set.pop()

  @staticmethod
  def _in_gke():
    """When running in GKE, the environment variable will be set."""
    return _GKE_ENV_VARIABLE in os.environ

  @staticmethod
  def _gke_endpoints():
    return os.environ[_GKE_ENV_VARIABLE]

  @staticmethod
  def _env_var_fallback():
    if _DEFAULT_ENV_VARIABLE in os.environ:
      return os.environ[_DEFAULT_ENV_VARIABLE]
    return None

  @staticmethod
  def _environment_discovery_url():
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
      tpu: A string corresponding to the TPU to use. If the string is an empty
        string, the string 'local', or a string that begins with 'grpc://' or
          '/bns', then it is assumed to not correspond with a Cloud TPU and will
          instead be passed as the session master and no ClusterSpec propagation
          will be done. In the future, this may also support a list of strings
          when multiple Cloud TPUs are used.
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
      discovery_url: A URL template that points to the location of the discovery
        service. It should have two parameters {api} and {apiVersion} that when
        filled in produce an absolute URL to the discovery document for that
        service. The environment variable 'TPU_API_DISCOVERY_URL' will override
        this.

    Raises:
      ImportError: If the googleapiclient is not installed.
      ValueError: If no TPUs are specified.
      RuntimeError: If an empty TPU name is specified and this is running in a
        Google Cloud environment.
    """
    if isinstance(tpu, list):
      if not tpu:
        raise ValueError('At least one TPU must be specified.')
      if len(tpu) != 1:
        raise NotImplementedError(
            'Using multiple TPUs in a single session is not yet implemented')
      tpu = tpu[0]

    in_gke = self._in_gke()
    # When using GKE with Cloud TPUs, the env variable will be set.
    if tpu is None:
      if in_gke:
        tpu = self._gke_endpoints()
      else:
        tpu = self._env_var_fallback()

    if tpu is None:
      raise ValueError('Please provide a TPU Name to connect to.')

    self._tpu = compat.as_bytes(tpu)  # self._tpu is always bytes

    # If we are running in Cloud and don't specify a TPU name
    if is_running_in_gce() and not self._tpu:
      raise RuntimeError('You need to specify a TPU Name if you are running in '
                         'the Google Cloud environment.')

    # By default the task_type is 'worker` and the task_id is 0 (which is the
    # first worker in the task).
    self.task_type = job_name
    self.task_id = 0

    if self._is_google_environment():
      self._environment = 'google'
      self.rpc_layer = None
    else:
      self._environment = ''
      self.rpc_layer = 'grpc'

    # Setting this overrides the return value of self._should_resolve()
    self._should_resolve_override = None

    # We strip out the protocol if it is included, and override the
    # shouldResolve function to never resolve. We are adding the protocol back
    # in later in self.master().
    if self.rpc_layer is not None and tpu.startswith(self.rpc_layer + '://'):
      tpu = tpu[len(self.rpc_layer + '://'):]
      self._tpu = compat.as_bytes(tpu)  # self._tpu is always bytes
      self._should_resolve_override = False

    # Whether we should actually attempt to contact Cloud APIs
    should_resolve = self._should_resolve()

    # We error out if we are in a non-Cloud environment which cannot talk to the
    # Cloud APIs using the standard class and a special object is not passed in.
    self._service = service
    if (self._service is None and should_resolve and
        not _GOOGLE_API_CLIENT_INSTALLED):
      raise ImportError('googleapiclient and oauth2client must be installed '
                        'before using the TPU cluster resolver. Execute: '
                        '`pip install --upgrade google-api-python-client` '
                        'and `pip install --upgrade oauth2client` to '
                        'install with pip.')

    # We save user-passed credentials, unless the user didn't pass in anything.
    self._credentials = credentials
    if (credentials == 'default' and should_resolve and
        _GOOGLE_API_CLIENT_INSTALLED):
      self._credentials = None

    # Automatically detect project and zone if unspecified.
    if not project and should_resolve:
      project = compat.as_str(
          self._request_compute_metadata('project/project-id'))
    if not zone and should_resolve:
      zone_path = compat.as_str(self._request_compute_metadata('instance/zone'))
      zone = zone_path.split('/')[-1]
    self._project = project
    self._zone = zone

    self._discovery_url = self._environment_discovery_url() or discovery_url

    self._coordinator_name = coordinator_name
    if (coordinator_name and not coordinator_address and
        (should_resolve or in_gke)):
      self._start_local_server()
    else:
      self._coordinator_address = coordinator_address

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    """Get the Master string to be used for the session.

    In the normal case, this returns the grpc path (grpc://1.2.3.4:8470) of
    first instance in the ClusterSpec returned by the cluster_spec function.

    If a non-TPU name is used when constructing a TPUClusterResolver, that will
    be returned instead (e.g. If the tpus argument's value when constructing
    this TPUClusterResolver was 'grpc://10.240.1.2:8470',
    'grpc://10.240.1.2:8470' will be returned).

    Args:
      task_type: (Optional, string) The type of the TensorFlow task of the
        master.
      task_id: (Optional, integer) The index of the TensorFlow task of the
        master.
      rpc_layer: (Optional, string) The RPC protocol TensorFlow should use to
        communicate with TPUs.

    Returns:
      string, the connection string to use when creating a session.

    Raises:
      ValueError: If none of the TPUs specified exists.
    """
    if self._should_resolve():
      # We are going to communicate with the Cloud TPU APIs to get a Cluster.
      cluster_spec = self.cluster_spec()
      if task_type is not None and task_id is not None:
        # task_type and task_id is from the function parameter
        master = cluster_spec.task_address(task_type, task_id)
      elif self.task_type is not None and self.task_id is not None:
        # task_type and task_id is from the object
        master = cluster_spec.task_address(self.task_type, self.task_id)
      else:
        # by default we take the first item in the cluster with the right name
        job_tasks = cluster_spec.job_tasks(self.task_type)
        if not job_tasks:
          raise ValueError('No TPUs with the specified names exist.')
        master = job_tasks[0]
    else:
      if isinstance(self._tpu, (bytes, bytearray)):
        master = compat.as_text(self._tpu).split(_ENDPOINTS_SEPARATOR)[0]
      else:
        master = self._tpu.split(_ENDPOINTS_SEPARATOR)[0]
    return format_master_url(master, rpc_layer or self.rpc_layer)

  def get_master(self):
    return self.master()

  def get_job_name(self):
    if ops.executing_eagerly_outside_functions() or self._should_resolve(
    ) or is_running_in_gce():
      return self.task_type

  def cluster_spec(self):
    """Returns a ClusterSpec object based on the latest TPU information.

    We retrieve the information from the GCE APIs every time this method is
    called.

    Returns:
      A ClusterSpec containing host information returned from Cloud TPUs,
      or None.

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
    #  3. [Other (legacy non-gRPC).] We should return None.
    ############################################################################

    if self._should_resolve():
      # Case 1.
      response = self._fetch_cloud_tpu_metadata()  # pylint: disable=protected-access

      if 'state' in response and response['state'] != 'READY':
        raise RuntimeError('TPU "%s" is not yet ready; state: "%s"' %
                           (compat.as_text(self._tpu), response['state']))

      if 'networkEndpoints' in response:
        worker_list = [
            '%s:%s' % (endpoint['ipAddress'], endpoint['port'])
            for endpoint in response['networkEndpoints']
        ]
      else:
        # Fall back to the deprecated response format
        instance_url = '%s:%s' % (response['ipAddress'], response['port'])
        worker_list = [instance_url]

      cluster_spec = {self.task_type: worker_list}
    else:
      is_eager = ops.executing_eagerly_outside_functions()
      if self.rpc_layer is None and not is_eager:
        # Case 3.
        return None
      # Case 2.
      tpus = []
      for tpu in compat.as_text(self._tpu).split(_ENDPOINTS_SEPARATOR):
        # We are working around the fact that GKE environment variable that is
        # supplied to us has the protocol string embedded in it, but we want
        # to strip it out for the ClusterSpec.
        if (self.rpc_layer is not None and
            tpu.startswith(self.rpc_layer + '://')):
          tpus.append(tpu[len(self.rpc_layer + '://'):])
        else:
          tpus.append(tpu)
      cluster_spec = {self.task_type: tpus}

    if self._coordinator_address:
      # {1, 2}.a
      cluster_spec[self._coordinator_name] = [self._coordinator_address]

    return server_lib.ClusterSpec(cluster_spec)

  def _fetch_cloud_tpu_metadata(self):
    """Returns the TPU metadata object from the TPU Get API call."""
    res = []
    try:
      full_name = 'projects/%s/locations/%s/nodes/%s' % (
          self._project, self._zone, compat.as_text(self._tpu))
      service = self._tpu_service()
      request = service.projects().locations().nodes().get(name=full_name)
      res = request.execute()
    except:  # pylint: disable=bare-except
      pass
    finally:
      return res  # pylint: disable=lost-exception

  def num_accelerators(self,
                       task_type=None,
                       task_id=None,
                       config_proto=None):
    """Returns the number of TPU cores per worker.

    Connects to the master and list all the devices present in the master,
    and counts them up. Also verifies that the device counts per host in the
    cluster is the same before returning the number of TPU cores per host.

    Args:
      task_type: Unused.
      task_id: Unused.
      config_proto: Used to create a connection to a TPU master in order to
        retrieve the system metadata.

    Raises:
      RuntimeError: If we cannot talk to a TPU worker after retrying or if the
        number of TPU devices per host is different.
    """
    retry_count = 1
    # TODO(b/120564445): Replace with standard library for retries.
    while True:
      try:
        device_details = TPUClusterResolver._get_device_dict_and_cores(
            get_accelerator_devices(self.master(), config_proto=config_proto))
        break
      except errors.DeadlineExceededError:
        error_message = ('Failed to connect to master. The TPU might not be '
                         'ready (e.g. still scheduling) or the master '
                         'address is incorrect: got (%s)' % self.master())
        if retry_count <= _TPU_CONN_RETRIES:
          logging.warning(error_message)
          logging.warning('Retrying (%d/%d)...', retry_count, _TPU_CONN_RETRIES)
          retry_count += 1
        else:
          raise RuntimeError(error_message)

    if device_details.total_cores:
      return {'TPU': TPUClusterResolver._verify_and_return_same_core_count(
          device_details.device_map)}
    return {'TPU': 0}

  @property
  def environment(self):
    """Returns the current environment which TensorFlow is running in."""
    return self._environment

  def _start_local_server(self):
    address = compat.as_text(
        self._request_compute_metadata('instance/network-interfaces/0/ip'))
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
