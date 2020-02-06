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
import json


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
          link_dict = {
              "1.0": "https://developer.nvidia.com/content/cuda-10",
              "1.1": "https://developer.nvidia.com/content/cuda-toolkit-11-june-2007",
              "2.0": "https://developer.nvidia.com/cuda-toolkit-20-august-2008",
              "2.1": "https://developer.nvidia.com/cuda-toolkit-21-january-2009",
              "2.2": "https://developer.nvidia.com/content/cuda-toolkit-22-may-2009",
              "2.3": "https://developer.nvidia.com/cuda-toolkit-23-downloads",
              "3.0": "https://developer.nvidia.com/cuda-toolkit-30-downloads",
              "3.1": "https://developer.nvidia.com/cuda-toolkit-31-downloads",
              "3.2": "https://developer.nvidia.com/cuda-toolkit-32-downloads",
              "4.0": "https://developer.nvidia.com/cuda-toolkit-40",
              "4.1": "https://developer.nvidia.com/cuda-toolkit-41-archive",
              "4.2": "https://developer.nvidia.com/cuda-toolkit-42-archive",
              "5.0": "https://developer.nvidia.com/cuda-toolkit-50-archive",
              "5.5": "https://developer.nvidia.com/cuda-toolkit-55-archive",
              "6.0": "https://developer.nvidia.com/cuda-toolkit-60",
              "6.5": "https://developer.nvidia.com/cuda-toolkit-65",
              "7.0": "https://developer.nvidia.com/cuda-toolkit-70",
              "7.5": "https://developer.nvidia.com/cuda-75-downloads-archive",
              "8.0": "https://developer.nvidia.com/cuda-80-ga2-download-archive",
              "9.0": "https://developer.nvidia.com/cuda-90-download-archive",
              "9.1": "https://developer.nvidia.com/cuda-91-download-archive-new",
              "9.2": "https://developer.nvidia.com/cuda-92-download-archive",
              "10.0": "https://developer.nvidia.com/cuda-10.0-download-archive",
              "10.1": "https://developer.nvidia.com/cuda-10.1-download-archive-update2",
              "10.2": "https://developer.nvidia.com/cuda-downloads"
          }
          raise ImportError(
              "Could not find %r. TensorFlow requires that this DLL be "
              "installed in a directory that is named in your %%PATH%% "
              "environment variable. Download and install CUDA %s from "
              "this URL: %s"
              % (
                  build_info.cudart_dll_name,
                  build_info.cuda_version_number,
                  link_dict.get(
                      str(build_info.cuda_version_number),
                      link_dict["10.2"]
                  )
              )
          )

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
