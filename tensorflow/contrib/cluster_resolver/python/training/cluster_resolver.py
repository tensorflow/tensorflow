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

from tensorflow.python.training.server_lib import ClusterSpec


class ClusterResolver(object):
  """Abstract class for all implementations of ClusterResolvers.

  This defines the skeleton for all implementations of ClusterResolvers.
  ClusterResolvers are a way for TensorFlow to communicate with various cluster
  management systems (e.g. GCE, AWS, etc...).

  By letting TensorFlow communicate with these systems, we will be able to
  automatically discover and resolve IP addresses for various TensorFlow
  workers. This will eventually allow us to automatically recover from
  underlying machine failures and scale TensorFlow worker clusters up and down.
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
    raise NotImplementedError(
        'cluster_spec is not implemented for {}.'.format(self))

  @abc.abstractmethod
  def master(self):
    """..."""
    raise NotImplementedError('master is not implemented for {}.'.format(self))


class SimpleClusterResolver(ClusterResolver):
  """Simple implementation of ClusterResolver that accepts a ClusterSpec."""

  def __init__(self, cluster_spec, master=''):
    """Creates a SimpleClusterResolver from a ClusterSpec."""
    super(SimpleClusterResolver, self).__init__()

    if not isinstance(cluster_spec, ClusterSpec):
      raise TypeError('cluster_spec must be a ClusterSpec.')
    self._cluster_spec = cluster_spec

    if not isinstance(master, str):
      raise TypeError('master must be a string.')
    self._master = master

  def cluster_spec(self):
    """Returns the ClusterSpec passed into the constructor."""
    return self._cluster_spec

  def master(self):
    """Returns the master address to use when creating a session."""
    return self._master


class UnionClusterResolver(ClusterResolver):
  """Performs a union on underlying ClusterResolvers.

  This class performs a union given two or more existing ClusterResolvers. It
  merges the underlying ClusterResolvers, and returns one unified ClusterSpec
  when cluster_spec is called. The details of the merge function is
  documented in the cluster_spec function.
  """

  def __init__(self, *args):
    """Initializes a UnionClusterResolver with other ClusterResolvers.

    Args:
      *args: `ClusterResolver` objects to be unionized.

    Raises:
      TypeError: If any argument is not a subclass of `ClusterResolvers`.
      ValueError: If there are no arguments passed.
    """
    super(UnionClusterResolver, self).__init__()

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

  def master(self):
    """master returns the master address from the first cluster resolver."""
    return self._cluster_resolvers[0].master()
