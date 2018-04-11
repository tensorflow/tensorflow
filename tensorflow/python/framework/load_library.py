# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Function for loading TensorFlow plugins."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import imp
import sys
import threading  # pylint: disable=unused-import

from tensorflow.core.framework import op_def_pb2
from tensorflow.core.lib.core import error_codes_pb2  # pylint: disable=unused-import
from tensorflow.python import pywrap_tensorflow as py_tf
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export


@tf_export('load_op_library')
def load_op_library(library_filename):
  """Loads a TensorFlow plugin, containing custom ops and kernels.

  Pass "library_filename" to a platform-specific mechanism for dynamically
  loading a library. The rules for determining the exact location of the
  library are platform-specific and are not documented here. When the
  library is loaded, ops and kernels registered in the library via the
  `REGISTER_*` macros are made available in the TensorFlow process. Note
  that ops with the same name as an existing op are rejected and not
  registered with the process.

  Args:
    library_filename: Path to the plugin.
      Relative or absolute filesystem path to a dynamic library file.

  Returns:
    A python module containing the Python wrappers for Ops defined in
    the plugin.

  Raises:
    RuntimeError: when unable to load the library or get the python wrappers.
  """
  lib_handle = py_tf.TF_LoadLibrary(library_filename)

  op_list_str = py_tf.TF_GetOpList(lib_handle)
  op_list = op_def_pb2.OpList()
  op_list.ParseFromString(compat.as_bytes(op_list_str))
  wrappers = py_tf.GetPythonWrappers(op_list_str)

  # Delete the library handle to release any memory held in C
  # that are no longer needed.
  py_tf.TF_DeleteLibraryHandle(lib_handle)

  # Get a unique name for the module.
  module_name = hashlib.md5(wrappers).hexdigest()
  if module_name in sys.modules:
    return sys.modules[module_name]
  module = imp.new_module(module_name)
  # pylint: disable=exec-used
  exec(wrappers, module.__dict__)
  # Stash away the library handle for making calls into the dynamic library.
  module.LIB_HANDLE = lib_handle
  # OpDefs of the list of ops defined in the library.
  module.OP_LIST = op_list
  sys.modules[module_name] = module
  return module


@tf_export('load_file_system_library')
def load_file_system_library(library_filename):
  """Loads a TensorFlow plugin, containing file system implementation.

  Pass `library_filename` to a platform-specific mechanism for dynamically
  loading a library. The rules for determining the exact location of the
  library are platform-specific and are not documented here.

  Args:
    library_filename: Path to the plugin.
      Relative or absolute filesystem path to a dynamic library file.

  Returns:
    None.

  Raises:
    RuntimeError: when unable to load the library.
  """
  py_tf.TF_LoadLibrary(library_filename)
