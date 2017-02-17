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
"""Library for getting system information during TensorFlow tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import multiprocessing
import platform
import re
import socket

# pylint: disable=g-bad-import-order
# Note: cpuinfo and psutil are not installed for you in the TensorFlow
# OSS tree.  They are installable via pip.
import cpuinfo
import psutil
# pylint: enable=g-bad-import-order

from tensorflow.core.util import test_log_pb2
from tensorflow.python.client import device_lib
from tensorflow.python.framework import errors
from tensorflow.python.platform import gfile
from tensorflow.tools.test import gpu_info_lib


def gather_machine_configuration():
  """Gather Machine Configuration.  This is the top level fn of this library."""
  config = test_log_pb2.MachineConfiguration()

  config.cpu_info.CopyFrom(gather_cpu_info())
  config.platform_info.CopyFrom(gather_platform_info())

  # gather_available_device_info must come before gather_gpu_devices
  # because the latter may access libcudart directly, which confuses
  # TensorFlow StreamExecutor.
  for d in gather_available_device_info():
    config.available_device_info.add().CopyFrom(d)
  for gpu in gpu_info_lib.gather_gpu_devices():
    config.device_info.add().Pack(gpu)

  config.memory_info.CopyFrom(gather_memory_info())

  config.hostname = gather_hostname()

  return config


def gather_hostname():
  return socket.gethostname()


def gather_memory_info():
  """Gather memory info."""
  mem_info = test_log_pb2.MemoryInfo()
  vmem = psutil.virtual_memory()
  mem_info.total = vmem.total
  mem_info.available = vmem.available
  return mem_info


def gather_cpu_info():
  """Gather CPU Information.  Assumes all CPUs are the same."""
  cpu_info = test_log_pb2.CPUInfo()
  cpu_info.num_cores = multiprocessing.cpu_count()

  # Gather num_cores_allowed
  try:
    with gfile.GFile('/proc/self/status', 'rb') as fh:
      nc = re.search(r'(?m)^Cpus_allowed:\s*(.*)$', fh.read())
    if nc:  # e.g. 'ff' => 8, 'fff' => 12
      cpu_info.num_cores_allowed = (
          bin(int(nc.group(1).replace(',', ''), 16)).count('1'))
  except errors.OpError:
    pass
  finally:
    if cpu_info.num_cores_allowed == 0:
      cpu_info.num_cores_allowed = cpu_info.num_cores

  # Gather the rest
  info = cpuinfo.get_cpu_info()
  cpu_info.cpu_info = info['brand']
  cpu_info.num_cores = info['count']
  cpu_info.mhz_per_cpu = info['hz_advertised_raw'][0] / 1.0e6
  l2_cache_size = re.match(r'(\d+)', str(info['l2_cache_size']))
  if l2_cache_size:
    # If a value is returned, it's in KB
    cpu_info.cache_size['L2'] = int(l2_cache_size.group(0)) * 1024

  # Try to get the CPU governor
  try:
    cpu_governors = set([
        gfile.GFile(f, 'r').readline().rstrip()
        for f in glob.glob(
            '/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')
    ])
    if cpu_governors:
      if len(cpu_governors) > 1:
        cpu_info.cpu_governor = 'mixed'
      else:
        cpu_info.cpu_governor = list(cpu_governors)[0]
  except errors.OpError:
    pass

  return cpu_info


def gather_available_device_info():
  """Gather list of devices available to TensorFlow.

  Returns:
    A list of test_log_pb2.AvailableDeviceInfo messages.
  """
  device_info_list = []
  devices = device_lib.list_local_devices()

  for d in devices:
    device_info = test_log_pb2.AvailableDeviceInfo()
    device_info.name = d.name
    device_info.type = d.device_type
    device_info.memory_limit = d.memory_limit
    device_info.physical_description = d.physical_device_desc
    device_info_list.append(device_info)

  return device_info_list


def gather_platform_info():
  """Gather platform info."""
  platform_info = test_log_pb2.PlatformInfo()
  (platform_info.bits, platform_info.linkage) = platform.architecture()
  platform_info.machine = platform.machine()
  platform_info.release = platform.release()
  platform_info.system = platform.system()
  platform_info.version = platform.version()
  return platform_info
