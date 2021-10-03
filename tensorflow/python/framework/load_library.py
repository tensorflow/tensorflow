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
import errno
import hashlib
import importlib
import os
import platform
import sys

from tensorflow.python.client import pywrap_tf_session as py_tf
from tensorflow.python.eager import context
from tensorflow.python.framework import _pywrap_python_op_gen
from tensorflow.python.util import deprecation
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
  try:
    wrappers = _pywrap_python_op_gen.GetPythonWrappers(
        py_tf.TF_GetOpList(lib_handle))
  finally:
    # Delete the library handle to release any memory held in C
    # that are no longer needed.
    py_tf.TF_DeleteLibraryHandle(lib_handle)

  # Get a unique name for the module.
  module_name = hashlib.sha1(wrappers).hexdigest()
  if module_name in sys.modules:
    return sys.modules[module_name]
  module_spec = importlib.machinery.ModuleSpec(module_name, None)
  module = importlib.util.module_from_spec(module_spec)
  # pylint: disable=exec-used
  exec(wrappers, module.__dict__)
  # Allow this to be recognized by AutoGraph.
  setattr(module, '_IS_TENSORFLOW_PLUGIN', True)
  sys.modules[module_name] = module
  return module


@deprecation.deprecated(date=None,
                        instructions='Use `tf.load_library` instead.')
@tf_export(v1=['load_file_system_library'])
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


def _is_shared_object(filename):
  """Check the file to see if it is a shared object, only using extension."""
  if platform.system() == 'Linux':
    if filename.endswith('.so'):
      return True
    else:
      index = filename.rfind('.so.')
      if index == -1:
        return False
      else:
        # A shared object with the API version in filename
        return filename[index + 4].isdecimal()
  elif platform.system() == 'Darwin':
    return filename.endswith('.dylib')
  elif platform.system() == 'Windows':
    return filename.endswith('.dll')
  else:
    return False


@tf_export('load_library')
def load_library(library_location):
  """Loads a TensorFlow plugin.

  "library_location" can be a path to a specific shared object, or a folder.
  If it is a folder, all shared objects that are named "libtfkernel*" will be
  loaded. When the library is loaded, kernels registered in the library via the
  `REGISTER_*` macros are made available in the TensorFlow process.

  Args:
    library_location: Path to the plugin or the folder of plugins.
      Relative or absolute filesystem path to a dynamic library file or folder.

  Returns:
    None

  Raises:
    OSError: When the file to be loaded is not found.
    RuntimeError: when unable to load the library.
  """
  if os.path.exists(library_location):
    if os.path.isdir(library_location):
      directory_contents = os.listdir(library_location)

      kernel_libraries = [
          os.path.join(library_location, f) for f in directory_contents
          if _is_shared_object(f)]
    else:
      kernel_libraries = [library_location]

    for lib in kernel_libraries:
      py_tf.TF_LoadLibrary(lib)

  else:
    raise OSError(
        errno.ENOENT,
        'The file or folder to load kernel libraries from does not exist.',
        library_location)


def load_pluggable_device_library(library_location):
  """Loads a TensorFlow PluggableDevice plugin.

  "library_location" can be a path to a specific shared object, or a folder.
  If it is a folder, all shared objects will be loaded. when the library is
  loaded, devices/kernels registered in the library via StreamExecutor C API
  and Kernel/Op Registration C API are made available in TensorFlow process.

  Args:
    library_location: Path to the plugin or folder of plugins. Relative or
      absolute filesystem path to a dynamic library file or folder.

  Raises:
    OSError: When the file to be loaded is not found.
    RuntimeError: when unable to load the library.
  """
  if os.path.exists(library_location):
    if os.path.isdir(library_location):
      directory_contents = os.listdir(library_location)

      pluggable_device_libraries = [
          os.path.join(library_location, f)
          for f in directory_contents
          if _is_shared_object(f)
      ]
    else:
      pluggable_device_libraries = [library_location]

    for lib in pluggable_device_libraries:
      py_tf.TF_LoadPluggableDeviceLibrary(lib)
    # Reinitialized physical devices list after plugin registration.
    context.context().reinitialize_physical_devices()
  else:
    raise OSError(
        errno.ENOENT,
        'The file or folder to load pluggable device libraries from does not '
        'exist.', library_location)


@tf_export('experimental.register_filesystem_plugin')
def register_filesystem_plugin(plugin_location):
  """Loads a TensorFlow FileSystem plugin.

  Args:
    plugin_location: Path to the plugin. Relative or absolute filesystem plugin
      path to a dynamic library file.

  Returns:
    None

  Raises:
    OSError: When the file to be loaded is not found.
    RuntimeError: when unable to load the library.
  """
  if os.path.exists(plugin_location):
    py_tf.TF_RegisterFilesystemPlugin(plugin_location)

  else:
    raise OSError(errno.ENOENT,
                  'The file to load file system plugin from does not exist.',
                  plugin_location)
