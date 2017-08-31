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

import ctypes as ct
import platform

from tensorflow.core.util import test_log_pb2
from tensorflow.python.framework import errors
from tensorflow.python.platform import gfile


def _gather_gpu_devices_proc():
  """Try to gather NVidia GPU device information via /proc/driver."""
  dev_info = []
  for f in gfile.Glob("/proc/driver/nvidia/gpus/*/information"):
    bus_id = f.split("/")[5]
    key_values = dict(line.rstrip().replace("\t", "").split(":", 1)
                      for line in gfile.GFile(f, "r"))
    key_values = dict((k.lower(), v.strip(" ").rstrip(" "))
                      for (k, v) in key_values.items())
    info = test_log_pb2.GPUInfo()
    info.model = key_values.get("model", "Unknown")
    info.uuid = key_values.get("gpu uuid", "Unknown")
    info.bus_id = bus_id
    dev_info.append(info)
  return dev_info


class CUDADeviceProperties(ct.Structure):
  # See $CUDA_HOME/include/cuda_runtime_api.h for the definition of
  # the cudaDeviceProp struct.
  _fields_ = [
      ("name", ct.c_char * 256),
      ("totalGlobalMem", ct.c_size_t),
      ("sharedMemPerBlock", ct.c_size_t),
      ("regsPerBlock", ct.c_int),
      ("warpSize", ct.c_int),
      ("memPitch", ct.c_size_t),
      ("maxThreadsPerBlock", ct.c_int),
      ("maxThreadsDim", ct.c_int * 3),
      ("maxGridSize", ct.c_int * 3),
      ("clockRate", ct.c_int),
      ("totalConstMem", ct.c_size_t),
      ("major", ct.c_int),
      ("minor", ct.c_int),
      ("textureAlignment", ct.c_size_t),
      ("texturePitchAlignment", ct.c_size_t),
      ("deviceOverlap", ct.c_int),
      ("multiProcessorCount", ct.c_int),
      ("kernelExecTimeoutEnabled", ct.c_int),
      ("integrated", ct.c_int),
      ("canMapHostMemory", ct.c_int),
      ("computeMode", ct.c_int),
      ("maxTexture1D", ct.c_int),
      ("maxTexture1DMipmap", ct.c_int),
      ("maxTexture1DLinear", ct.c_int),
      ("maxTexture2D", ct.c_int * 2),
      ("maxTexture2DMipmap", ct.c_int * 2),
      ("maxTexture2DLinear", ct.c_int * 3),
      ("maxTexture2DGather", ct.c_int * 2),
      ("maxTexture3D", ct.c_int * 3),
      ("maxTexture3DAlt", ct.c_int * 3),
      ("maxTextureCubemap", ct.c_int),
      ("maxTexture1DLayered", ct.c_int * 2),
      ("maxTexture2DLayered", ct.c_int * 3),
      ("maxTextureCubemapLayered", ct.c_int * 2),
      ("maxSurface1D", ct.c_int),
      ("maxSurface2D", ct.c_int * 2),
      ("maxSurface3D", ct.c_int * 3),
      ("maxSurface1DLayered", ct.c_int * 2),
      ("maxSurface2DLayered", ct.c_int * 3),
      ("maxSurfaceCubemap", ct.c_int),
      ("maxSurfaceCubemapLayered", ct.c_int * 2),
      ("surfaceAlignment", ct.c_size_t),
      ("concurrentKernels", ct.c_int),
      ("ECCEnabled", ct.c_int),
      ("pciBusID", ct.c_int),
      ("pciDeviceID", ct.c_int),
      ("pciDomainID", ct.c_int),
      ("tccDriver", ct.c_int),
      ("asyncEngineCount", ct.c_int),
      ("unifiedAddressing", ct.c_int),
      ("memoryClockRate", ct.c_int),
      ("memoryBusWidth", ct.c_int),
      ("l2CacheSize", ct.c_int),
      ("maxThreadsPerMultiProcessor", ct.c_int),
      ("streamPrioritiesSupported", ct.c_int),
      ("globalL1CacheSupported", ct.c_int),
      ("localL1CacheSupported", ct.c_int),
      ("sharedMemPerMultiprocessor", ct.c_size_t),
      ("regsPerMultiprocessor", ct.c_int),
      ("managedMemSupported", ct.c_int),
      ("isMultiGpuBoard", ct.c_int),
      ("multiGpuBoardGroupID", ct.c_int),
      # Pad with extra space to avoid dereference crashes if future
      # versions of CUDA extend the size of this struct.
      ("__future_buffer", ct.c_char * 4096)
  ]


def _gather_gpu_devices_cudart():
  """Try to gather NVidia GPU device information via libcudart."""
  dev_info = []

  system = platform.system()
  if system == "Linux":
    libcudart = ct.cdll.LoadLibrary("libcudart.so")
  elif system == "Darwin":
    libcudart = ct.cdll.LoadLibrary("libcudart.dylib")
  elif system == "Windows":
    libcudart = ct.windll.LoadLibrary("libcudart.dll")
  else:
    raise NotImplementedError("Cannot identify system.")

  version = ct.c_int()
  rc = libcudart.cudaRuntimeGetVersion(ct.byref(version))
  if rc != 0:
    raise ValueError("Could not get version")
  if version.value < 6050:
    raise NotImplementedError("CUDA version must be between >= 6.5")

  device_count = ct.c_int()
  libcudart.cudaGetDeviceCount(ct.byref(device_count))

  for i in range(device_count.value):
    properties = CUDADeviceProperties()
    rc = libcudart.cudaGetDeviceProperties(ct.byref(properties), i)
    if rc != 0:
      raise ValueError("Could not get device properties")
    pci_bus_id = " " * 13
    rc = libcudart.cudaDeviceGetPCIBusId(ct.c_char_p(pci_bus_id), 13, i)
    if rc != 0:
      raise ValueError("Could not get device PCI bus id")

    info = test_log_pb2.GPUInfo()  # No UUID available
    info.model = properties.name
    info.bus_id = pci_bus_id
    dev_info.append(info)

    del properties

  return dev_info


def gather_gpu_devices():
  """Gather gpu device info.

  Returns:
    A list of test_log_pb2.GPUInfo messages.
  """
  try:
    # Prefer using /proc if possible, it provides the UUID.
    dev_info = _gather_gpu_devices_proc()
    if not dev_info:
      raise ValueError("No devices found")
    return dev_info
  except (IOError, ValueError, errors.OpError):
    pass

  try:
    # Fall back on using libcudart
    return _gather_gpu_devices_cudart()
  except (OSError, ValueError, NotImplementedError, errors.OpError):
    return []
