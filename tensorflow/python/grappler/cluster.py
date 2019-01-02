# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A python interface for Grappler clusters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.grappler.costs import op_performance_data_pb2
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python import pywrap_tensorflow as tf_cluster
from tensorflow.python.framework import errors


class Cluster(object):
  """Grappler Clusters."""

  def __init__(self,
               allow_soft_placement=True,
               disable_detailed_stats=True,
               disable_timeline=True,
               devices=None):
    """Creates a Cluster.

    Args:
      allow_soft_placement: If True, TF will automatically fix illegal
        placements instead of erroring out if the placement isn't legal.
      disable_detailed_stats: If True, detailed statistics will not be
        available.
      disable_timeline: If True, the timeline information will not be reported.
      devices: A list of devices of type device_properties_pb2.NamedDevice.
        If None, a device list will be created based on the spec of
        the local machine.
    """
    self._tf_cluster = None
    self._generate_timeline = not disable_timeline
    with errors.raise_exception_on_not_ok_status() as status:
      if devices is None:
        self._tf_cluster = tf_cluster.TF_NewCluster(
            allow_soft_placement, disable_detailed_stats, status)
      else:
        devices_serialized = [device.SerializeToString() for device in devices]
        self._tf_cluster = tf_cluster.TF_NewVirtualCluster(
            devices_serialized, status)

  def Shutdown(self):
    if self._tf_cluster is not None:
      tf_cluster.TF_ShutdownCluster(self._tf_cluster)
      self._tf_cluster = None

  def __del__(self):
    self.Shutdown()

  @property
  def tf_cluster(self):
    return self._tf_cluster

  def ListDevices(self):
    """Returns a list of available hardware devices."""
    if self._tf_cluster is None:
      return []
    return [device_properties_pb2.NamedDevice.FromString(device)
            for device in tf_cluster.TF_ListDevices(self._tf_cluster)]

  def ListAvailableOps(self):
    """Returns a list of all available operations (sorted alphabetically)."""
    return tf_cluster.TF_ListAvailableOps()

  def GetSupportedDevices(self, item):
    return tf_cluster.TF_GetSupportedDevices(self._tf_cluster, item.tf_item)

  def EstimatePerformance(self, device):
    return tf_cluster.TF_EstimatePerformance(device.SerializeToString())

  def MeasureCosts(self, item):
    """Returns the cost of running the specified item.

    Args:
      item: The item for which to measure the costs.
    Returns: The triplet op_perfs, runtime, step_stats.
    """
    with errors.raise_exception_on_not_ok_status() as status:
      ret_from_swig = tf_cluster.TF_MeasureCosts(
          item.tf_item, self._tf_cluster, self._generate_timeline, status)

    if ret_from_swig is None:
      return None

    op_perf_bytes_list, run_time, step_stats_bytes = ret_from_swig
    op_perfs = [op_performance_data_pb2.OpPerformance.FromString(op_perf_bytes)
                for op_perf_bytes in op_perf_bytes_list]
    return (op_perfs, run_time,
            step_stats_pb2.StepStats.FromString(step_stats_bytes))

  def DeterminePeakMemoryUsage(self, item):
    """Returns a snapshot of the peak memory usage.

    Args:
      item: The item for which to measure the costs.
    Returns: A hashtable indexed by device name.
    """
    with errors.raise_exception_on_not_ok_status() as status:
      return tf_cluster.TF_DeterminePeakMemoryUsage(
          item.tf_item, self._tf_cluster, status)


@contextlib.contextmanager
def Provision(allow_soft_placement=True,
              disable_detailed_stats=True,
              disable_timeline=True,
              devices=None):
  cluster = Cluster(allow_soft_placement, disable_detailed_stats,
                    disable_timeline, devices)
  yield cluster
  cluster.Shutdown()
