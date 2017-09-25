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
    try:
      ctypes.WinDLL("msvcp140.dll")
    except OSError:
      raise ImportError(
          "Could not find 'msvcp140.dll'. TensorFlow requires that this DLL be "
          "installed in a directory that is named in your %PATH% environment "
          "variable. You may install this DLL by downloading Visual C++ 2015 "
          "Redistributable Update 3 from this URL: "
          "https://www.microsoft.com/en-us/download/details.aspx?id=53587")
    # TODO(mrry): Add specific checks for GPU DLLs if build_info indicates
    # that this is a GPU build.
  else:
    # TODO(mrry): Consider adding checks for the Linux and Mac OS X builds.
    pass
