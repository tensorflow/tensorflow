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

import collections
import re

from tensorflow.python.distribute.cluster_resolver import cluster_resolver
from tensorflow.python.framework import config as framework_config
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

try:
  from cloud_tpu_client import client  # pylint: disable=g-import-not-at-top
except ImportError:
  logging.debug(
      'Falling back to TensorFlow client; we recommended you install the Cloud '
      'TPU client directly with pip install cloud-tpu-client.')
  from tensorflow.python.tpu.client import client  # pylint: disable=g-import-not-at-top


def is_running_in_gce():
  return True


class _LocalCloudTpuClient(object):
  """Dummy local Cloud TPU client."""

  def api_available(self):
    return False


_TPU_DEVICE_REGEX = re.compile(
    r'.*task:(?P<host_id>\d+)/.*device:TPU:(?P<core_id>\d+)$')
_TPU_CONN_RETRIES = 120
DeviceDetails = collections.namedtuple(
    'DeviceDetails', ['device_map', 'total_cores'])


class TPUClusterResolver(cluster_resolver.ClusterResolver):
  """Cluster Resolver for Google Cloud TPUs.

  This is an implementation of cluster resolvers for the Google Cloud TPU
  service.

  TPUClusterResolver supports the following distinct environments:
  Google Compute Engine
  Google Kubernetes Engine
  Google internal

  It can be passed into `tf.distribute.TPUStrategy` to support TF2 training on
  Cloud TPUs.
  """

  @staticmethod
  def connect(tpu=None,
              zone=None,
              project=None):
    """Initializes TPU and returns a TPUClusterResolver.

    This API will connect to remote TPU cluster and initialize the TPU
    hardwares. Example usage:

    >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
    ...     tpu='')

    It can be viewed as convenient wrapper of the following code:

    >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    >>> tf.config.experimental_connect_to_cluster(resolver)
    >>> tf.tpu.experimental.initialize_tpu_system(resolver)

    Args:
      tpu: A string corresponding to the TPU to use. It can be the TPU name or
        TPU worker gRPC address. If not set, it will try automatically resolve
        the TPU address on Cloud TPUs.
      zone: Zone where the TPUs are located. If omitted or empty, we will assume
        that the zone of the TPU is the same as the zone of the GCE VM, which we
        will try to discover from the GCE metadata service.
      project: Name of the GCP project containing Cloud TPUs. If omitted or
        empty, we will try to discover the project name of the GCE VM from the
        GCE metadata service.

    Returns:
      An instance of TPUClusterResolver object.

    Raises:
      NotFoundError: If no TPU devices found in eager mode.
    """
    resolver = TPUClusterResolver(tpu, zone, project)
    from tensorflow.python.eager import remote  # pylint: disable=g-import-not-at-top
    remote.connect_to_cluster(resolver)
    from tensorflow.python.tpu import tpu_strategy_util  # pylint: disable=g-import-not-at-top
    tpu_strategy_util.initialize_tpu_system(resolver)
    return resolver

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
      tpu: A string corresponding to the TPU to use. It can be the TPU name or
        TPU worker gRPC address. If not set, it will try automatically resolve
        the TPU address on Cloud TPUs. If set to "local", it will assume that
        the TPU is directly connected to the VM instead of over the network.
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

    if tpu != 'local':
      # Default Cloud environment
      self._cloud_tpu_client = client.Client(
          tpu=tpu,
          zone=zone,
          project=project,
          credentials=credentials,
          service=service,
          discovery_url=discovery_url)
      self._tpu = self._cloud_tpu_client.name()
    else:
      # Directly connected TPU environment
      self._cloud_tpu_client = _LocalCloudTpuClient()
      self._tpu = 'local'

    # By default the task_type is 'worker` and the task_id is 0 (which is the
    # first worker in the task).
    self.task_type = job_name
    self.task_id = 0
    self._coordinator_name = coordinator_name
    if (coordinator_name and not coordinator_address):
      self._start_local_server()
    else:
      self._coordinator_address = coordinator_address

  def __enter__(self):
    self._cloud_tpu_client.enter()

  def __exit__(self, type, value, traceback):  # pylint: disable=redefined-builtin
    self._cloud_tpu_client.exit(type, value, traceback)

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

    if self._tpu != 'local':
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
      return cluster_resolver.format_master_url(master, 'grpc')
    else:
      return ''

  def get_master(self):
    return self.master()

  def get_job_name(self):
    return self.task_type

  def get_tpu_system_metadata(self):
    """Returns the metadata of the TPU system.

    Users can call this method to get some facts of the TPU system, like
    total number of cores, number of TPU workers and the devices. E.g.
    ```python

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tpu_system_metadata = resolver.get_tpu_system_metadata()
    num_hosts = tpu_system_metadata.num_hosts
    ```

    Returns:
      A `tf.tpu.experimental.TPUSystemMetadata` object.
    """
    cluster_spec = self.cluster_spec()
    cluster_def = cluster_spec.as_cluster_def() if cluster_spec else None
    tpu_system_metadata = (
        tpu_system_metadata_lib._query_tpu_system_metadata(  # pylint: disable=protected-access
            self.master(),
            cluster_def=cluster_def,
            query_topology=False))

    return tpu_system_metadata

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
    # There are 6 potential cases this code must handle:
    #  0. [Local case.] When a TPU is connected directly to the VM.
    #  1. [Normal case.] We should resolve the TPU name to a set of tasks, and
    #      a. Create a ClusterSpec that includes the coordinator job
    #      b. Create a ClusterSpec without the coordinator job.
    #  2. [GKE / No API Access.] We should not resolve the TPU name to a set of
    #     tasks and
    #      a. Create a ClusterSpec with the coordinator
    #      b. Create a ClusterSpec without the coordinator
    ############################################################################

    if self._tpu != 'local':
      network_endpoints = self._cloud_tpu_client.network_endpoints()
      worker_list = [
          '%s:%s' % (endpoint['ipAddress'], endpoint['port'])
          for endpoint in network_endpoints
      ]
      cluster_spec = {self.task_type: worker_list}
      if self._coordinator_address:
        # {1, 2}.a
        cluster_spec[self._coordinator_name] = [self._coordinator_address]
      return server_lib.ClusterSpec(cluster_spec)
    else:
      return server_lib.ClusterSpec({})

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
    if self._tpu == 'local':
      return {
          'TPU':
              len([
                  d for d in framework_config.list_logical_devices()
                  if d.device_type == 'TPU'
              ])
      }

    retry_count = 1
    # TODO(b/120564445): Replace with standard library for retries.
    while True:
      try:
        device_details = TPUClusterResolver._get_device_dict_and_cores(
            cluster_resolver.get_accelerator_devices(
                self.master(), config_proto=config_proto))
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
      return {
          'TPU':
              TPUClusterResolver._verify_and_return_same_core_count(
                  device_details.device_map)
      }
    return {'TPU': 0}

  @property
  def environment(self):
    """Returns the current environment which TensorFlow is running in."""
    return self._environment

  def _start_local_server(self):
    address = compat.as_text(self._cloud_tpu_client.get_local_ip())
    self._server = server_lib.Server({'local': ['0.0.0.0:0']},
                                     protocol='grpc',
                                     config=None,
                                     start=True)
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
