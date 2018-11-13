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

"""Platform-specific code for checking the integrity of the TensorFlow build."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


try:
  from tensorflow.python.platform import build_info
except ImportError:
  raise ImportError("Could not import tensorflow. Do not import tensorflow "
                    "from its source directory; change directory to outside "
                    "the TensorFlow source tree, and relaunch your Python "
                    "interpreter from there.")


def preload_check():
  """Raises an exception if the environment is not correctly configured.

  Raises:
    ImportError: If the check detects that the environment is not correctly
      configured, and attempting to load the TensorFlow runtime will fail.
  """
  if os.name == "nt":
    # Attempt to load any DLLs that the Python extension depends on before
    # we load the Python extension, so that we can raise an actionable error
    # message if they are not found.
    import ctypes  # pylint: disable=g-import-not-at-top
    if hasattr(build_info, "msvcp_dll_name"):
      try:
        ctypes.WinDLL(build_info.msvcp_dll_name)
      except OSError:
        raise ImportError(
            "Could not find %r. TensorFlow requires that this DLL be "
            "installed in a directory that is named in your %%PATH%% "
            "environment variable. You may install this DLL by downloading "
            "Visual C++ 2015 Redistributable Update 3 from this URL: "
            "https://www.microsoft.com/en-us/download/details.aspx?id=53587"
            % build_info.msvcp_dll_name)

    if build_info.is_cuda_build:
      # Attempt to check that the necessary CUDA DLLs are loadable.

      if hasattr(build_info, "nvcuda_dll_name"):
        try:
          ctypes.WinDLL(build_info.nvcuda_dll_name)
        except OSError:
          raise ImportError(
              "Could not find %r. TensorFlow requires that this DLL "
              "be installed in a directory that is named in your %%PATH%% "
              "environment variable. Typically it is installed in "
              "'C:\\Windows\\System32'. If it is not present, ensure that you "
              "have a CUDA-capable GPU with the correct driver installed."
              % build_info.nvcuda_dll_name)

      if hasattr(build_info, "cudart_dll_name") and hasattr(
          build_info, "cuda_version_number"):
        try:
          ctypes.WinDLL(build_info.cudart_dll_name)
        except OSError:
          raise ImportError(
              "Could not find %r. TensorFlow requires that this DLL be "
              "installed in a directory that is named in your %%PATH%% "
              "environment variable. Download and install CUDA %s from "
              "this URL: https://developer.nvidia.com/cuda-90-download-archive"
              % (build_info.cudart_dll_name, build_info.cuda_version_number))

      if hasattr(build_info, "cudnn_dll_name") and hasattr(
          build_info, "cudnn_version_number"):
        try:
          ctypes.WinDLL(build_info.cudnn_dll_name)
        except OSError:
          raise ImportError(
              "Could not find %r. TensorFlow requires that this DLL be "
              "installed in a directory that is named in your %%PATH%% "
              "environment variable. Note that installing cuDNN is a separate "
              "step from installing CUDA, and this DLL is often found in a "
              "different directory from the CUDA DLLs. You may install the "
              "necessary DLL by downloading cuDNN %s from this URL: "
              "https://developer.nvidia.com/cudnn"
              % (build_info.cudnn_dll_name, build_info.cudnn_version_number))

  else:
    # TODO(mrry): Consider adding checks for the Linux and Mac OS X builds.
    pass
