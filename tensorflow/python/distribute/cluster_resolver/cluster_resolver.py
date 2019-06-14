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
"""Cluster Resolvers are used for dynamic cluster IP/hostname resolution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import collections
import re
import six

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export


DEVICE_TYPE_REGEX = re.compile('.*device:([^:]+).*')


def format_master_url(master, rpc_layer=None):
  if rpc_layer:
    return '%s://%s' % (rpc_layer, master)
  else:
    return master


def get_accelerator_devices(master, config_proto):
  """Returns accelerator devices given a master and a configuration."""
  if context.executing_eagerly():
    device_names = context.list_devices()  # list_devices returns list(string)
    devices = []
    for name in device_names:
      device_type = 'GPU'  # default device type is GPU
      device_match = DEVICE_TYPE_REGEX.match(name)
      if device_match:
        device_type = device_match.group(1)
      if device_type == 'CPU' or device_type == 'XLA_CPU':  # Filter CPUs
        continue
      devices.append(session._DeviceAttributes(name, device_type, 0, 0))  # pylint: disable=protected-access
    return devices
  else:
    with ops.Graph().as_default():
      with session.Session(master, config=config_proto) as s:
        devices = s.list_devices()
    return devices


@tf_export('distribute.cluster_resolver.ClusterResolver')
@six.add_metaclass(abc.ABCMeta)
class ClusterResolver(object):
  """Abstract class for all implementations of ClusterResolvers.

  This defines the skeleton for all implementations of ClusterResolvers.
  ClusterResolvers are a way for TensorFlow to communicate with various cluster
  management systems (e.g. GCE, AWS, etc...).

  By letting TensorFlow communicate with these systems, we will be able to
  automatically discover and resolve IP addresses for various TensorFlow
  workers. This will eventually allow us to automatically recover from
  underlying machine failures and scale TensorFlow worker clusters up and down.

  Note to Implementors: In addition to these abstract methods, you must also
  implement the task_type, task_id, and rpc_layer attributes. You may choose
  to implement them either as properties with getters or setters or directly
  set the attributes.

  - task_type is the name of the server's current named job (e.g. 'worker',
     'ps' in a distributed parameterized training job).
  - task_id is the ordinal index of the server within the task type.
  - rpc_layer is the protocol used by TensorFlow to communicate with other
      TensorFlow servers in a distributed environment.
  """

  @abc.abstractmethod
  def cluster_spec(self):
    """Retrieve the current state of the cluster and returns a ClusterSpec.

    Returns:
      A ClusterSpec representing the state of the cluster at the moment this
      function is called.

    Implementors of this function must take care in ensuring that the
    ClusterSpec returned is up-to-date at the time of calling this function.
    This usually means retrieving the information from the underlying cluster
    management system every time this function is invoked and reconstructing
    a cluster_spec, rather than attempting to cache anything.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def master(self, task_type=None, task_id=None, rpc_layer=None):
    """Retrieves the name or URL of the session master.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC protocol for the given cluster.

    Returns:
      The name or URL of the session master.

    Implementors of this function must take care in ensuring that the master
    returned is up-to-date at the time to calling this function. This usually
    means retrieving the master every time this function is invoked.
    """
    raise NotImplementedError()

  def num_accelerators(self,
                       task_type=None,
                       task_id=None,
                       config_proto=None):
    """Returns the number of accelerator cores per worker.

    This returns the number of accelerator cores (such as GPUs and TPUs)
    available per worker.

    Optionally, we allow callers to specify the task_type, and task_id, for
    if they want to target a specific TensorFlow process to query
    the number of accelerators. This is to support heterogenous environments,
    where the number of accelerators cores per host is different.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the machine we
        want to query.
      task_id: (Optional) The index of the TensorFlow task of the machine we
        want to query.
      config_proto: (Optional) Configuration for starting a new session to
        query how many accelerator cores it has.

    Returns:
      A map of accelerator types to number of cores.
    """
    master = self.master(task_type, task_id)
    devices = get_accelerator_devices(master, config_proto)
    mapping = collections.defaultdict(int)
    for device in devices:
      if task_type is not None and task_id is not None:
        job_path = '/job:%s' % task_type
        task_path = '/task:%s' % task_id
        if job_path not in device.name or task_path not in device.name:
          continue
      mapping[device.device_type] += 1
    return mapping

  @property
  def environment(self):
    """Returns the current environment which TensorFlow is running in.

    There are two possible return values, "google" (when TensorFlow is running
    in a Google-internal environment) or an empty string (when TensorFlow is
    running elsewhere).

    If you are implementing a ClusterResolver that works in both the Google
    environment and the open-source world (for instance, a TPU ClusterResolver
    or similar), you will have to return the appropriate string depending on the
    environment, which you will have to detect.

    Otherwise, if you are implementing a ClusterResolver that will only work
    in open-source TensorFlow, you do not need to implement this property.
    """
    return ''


@tf_export('distribute.cluster_resolver.SimpleClusterResolver')
class SimpleClusterResolver(ClusterResolver):
  """Simple implementation of ClusterResolver that accepts a ClusterSpec."""

  def __init__(self, cluster_spec, master='', task_type=None, task_id=None,
               environment='', num_accelerators=None,
               rpc_layer=None):
    """Creates a SimpleClusterResolver from a ClusterSpec."""
    super(SimpleClusterResolver, self).__init__()

    self._task_type = task_type
    self._task_id = task_id
    self._environment = environment

    self._num_accelerators = num_accelerators
    self._rpc_layer = rpc_layer

    if not isinstance(cluster_spec, ClusterSpec):
      raise TypeError('cluster_spec must be a ClusterSpec.')
    self._cluster_spec = cluster_spec

    if not isinstance(master, str):
      raise TypeError('master must be a string.')
    self._master = master

  def cluster_spec(self):
    """Returns the ClusterSpec passed into the constructor."""
    return self._cluster_spec

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    """Returns the master address to use when creating a session.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC used by distributed TensorFlow.

    Returns:
      The name or URL of the session master.

    If a task_type and task_id is given, this will override the `master`
    string passed into the initialization function.
    """
    if task_type is not None and task_id is not None:
      master = self.cluster_spec().task_address(task_type, task_id)
    else:
      master = self._master

    return format_master_url(master, rpc_layer=rpc_layer or self._rpc_layer)

  @property
  def task_type(self):
    return self._task_type

  @property
  def task_id(self):
    return self._task_id

  @task_type.setter
  def task_type(self, task_type):
    self._task_type = task_type

  @task_id.setter
  def task_id(self, task_id):
    self._task_id = task_id

  @property
  def environment(self):
    return self._environment

  def num_accelerators(self,
                       task_type=None,
                       task_id=None,
                       config_proto=None):
    """Returns the number of accelerator cores per worker.

    The SimpleClusterResolver does not do automatic detection of accelerators,
    so a TensorFlow session will never be created, and thus all arguments are
    unused and we simply assume that the type of accelerator is a GPU and return
    the value in provided to us in the constructor.

    Args:
      task_type: Unused.
      task_id: Unused.
      config_proto: Unused.
    """
    # Unused
    del task_type, task_id, config_proto
    if self._num_accelerators is None:
      return {}
    return self._num_accelerators

  @property
  def rpc_layer(self):
    return self._rpc_layer

  @rpc_layer.setter
  def rpc_layer(self, rpc_layer):
    self._rpc_layer = rpc_layer


@tf_export('distribute.cluster_resolver.UnionResolver')
class UnionClusterResolver(ClusterResolver):
  """Performs a union on underlying ClusterResolvers.

  This class performs a union given two or more existing ClusterResolvers. It
  merges the underlying ClusterResolvers, and returns one unified ClusterSpec
  when cluster_spec is called. The details of the merge function is
  documented in the cluster_spec function.

  For additional Cluster Resolver properties such as task type, task index,
  rpc layer, environment, etc..., we will return the value from the first
  ClusterResolver in the union.
  """

  def __init__(self, *args, **kwargs):
    """Initializes a UnionClusterResolver with other ClusterResolvers.

    Args:
      *args: `ClusterResolver` objects to be unionized.
      **kwargs:
        rpc_layer - (Optional) Override value for the RPC layer used by
          TensorFlow.
        task_type - (Optional) Override value for the current task type.
        task_id - (Optional) Override value for the current task index.

    Raises:
      TypeError: If any argument is not a subclass of `ClusterResolvers`.
      ValueError: If there are no arguments passed.
    """
    super(UnionClusterResolver, self).__init__()

    self._rpc_layer = kwargs.pop('rpc_layer', None)
    self._task_type = kwargs.pop('task_type', None)
    self._task_id = kwargs.pop('task_id', None)

    if kwargs:
      raise ValueError('Unexpected kwargs provided {!r}'.format(kwargs))

    if not args:
      raise ValueError('At least one ClusterResolver is required.')

    for cluster_resolver in args:
      if not isinstance(cluster_resolver, ClusterResolver):
        raise TypeError('All arguments must be a sub-class of '
                        '`ClusterResolver.`')
    self._cluster_resolvers = args

  def cluster_spec(self):
    """Returns a union of all the ClusterSpecs from the ClusterResolvers.

    Returns:
      A ClusterSpec containing host information merged from all the underlying
      ClusterResolvers.

    Raises:
      KeyError: If there are conflicting keys detected when merging two or
      more dictionaries, this exception is raised.

    Note: If there are multiple ClusterResolvers exposing ClusterSpecs with the
    same job name, we will merge the list/dict of workers.

    If *all* underlying ClusterSpecs expose the set of workers as lists, we will
    concatenate the lists of workers, starting with the list of workers from
    the first ClusterResolver passed into the constructor.

    If *any* of the ClusterSpecs expose the set of workers as a dict, we will
    treat all the sets of workers as dicts (even if they are returned as lists)
    and will only merge them into a dict if there is no conflicting keys. If
    there is a conflicting key, we will raise a `KeyError`.
    """

    merged_cluster = {}

    # We figure out whether it is all lists for a particular job, or whether
    # there are dicts inside.
    for cluster_resolver in self._cluster_resolvers:
      cluster_spec = cluster_resolver.cluster_spec()
      cluster_dict = cluster_spec.as_dict()

      for job_name, tasks in cluster_dict.items():
        if job_name in merged_cluster:
          # If we see a dict, then we write a dict out regardless.
          if isinstance(tasks, dict):
            merged_cluster[job_name] = {}
        else:
          # We take whichever type is present.
          if isinstance(tasks, list):
            merged_cluster[job_name] = []
          else:
            merged_cluster[job_name] = {}

    # We then do the merge as appropriate in merged_cluster[job].
    for cluster_resolver in self._cluster_resolvers:
      cluster_spec = cluster_resolver.cluster_spec()
      cluster_dict = cluster_spec.as_dict()

      for job_name, tasks in cluster_dict.items():
        if isinstance(merged_cluster[job_name], list):
          # We all have lists, we can just concatenate and be done.
          merged_cluster[job_name].extend(tasks)
        else:
          if isinstance(tasks, list):
            # We convert to a dictionary if the type is a list.
            task_dict = dict(zip(range(0, len(tasks)), tasks))
          else:
            # We can simply make a copy (for update) and be done.
            task_dict = tasks.copy()

          # We detect if there are duplicates, and raise an error if so.
          task_keys = set(task_dict)
          merged_keys = set(merged_cluster[job_name].keys())
          intersected_keys = task_keys.intersection(merged_keys)
          if intersected_keys:
            raise KeyError('Duplicate keys detected when merging two '
                           'ClusterSpecs: %s' % repr(intersected_keys))

          # We do the merge after all the processing.
          merged_cluster[job_name].update(task_dict)

    return ClusterSpec(merged_cluster)

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    """Returns the master address to use when creating a session.

    This usually returns the master from the first ClusterResolver passed in,
    but you can override this by specifying the task_type and task_id.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC protocol for the given cluster.

    Returns:
      The name or URL of the session master.
    """
    if task_type is not None and task_id is not None:
      master = self.cluster_spec().task_address(task_type, task_id)
      return format_master_url(master, rpc_layer or self._rpc_layer)

    return self._cluster_resolvers[0].master(rpc_layer=rpc_layer)

  @property
  def task_type(self):
    return self._task_type or self._cluster_resolvers[0].task_type

  @property
  def task_id(self):
    return self._task_id or self._cluster_resolvers[0].task_id

  @task_type.setter
  def task_type(self, task_type):
    self._task_type = task_type

  @task_id.setter
  def task_id(self, task_id):
    self._task_id = task_id

  @property
  def environment(self):
    return self._cluster_resolvers[0].environment

  def num_accelerators(self,
                       task_type=None,
                       task_id=None,
                       config_proto=None):
    return self._cluster_resolvers[0].num_accelerators(
        task_type, task_id, config_proto)

  @property
  def rpc_layer(self):
    return self._rpc_layer or self._cluster_resolvers[0].rpc_layer

  @rpc_layer.setter
  def rpc_layer(self, rpc_layer):
    self._rpc_layer = rpc_layer
