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
"""
Top-level module of TensorFlow. By convention, we refer to this module as
`tf` instead of `tensorflow`, following the common practice of importing
TensorFlow via the command `import tensorflow as tf`.

The primary function of this module is to import all of the public TensorFlow
interfaces into a single place. The interfaces themselves are located in
sub-modules, as described below.

Note that the file `__init__.py` in the TensorFlow source code tree is actually
only a placeholder to enable test cases to run. The TensorFlow build replaces
this file with a file generated from [`api_template.__init__.py`](https://www.github.com/tensorflow/tensorflow/blob/master/tensorflow/api_template.__init__.py)
"""
# pylint: disable=g-bad-import-order,protected-access,g-import-not-at-top

import distutils as _distutils
import importlib
import inspect as _inspect
import os as _os
import site as _site
import sys as _sys

# Do not remove this line; See https://github.com/tensorflow/tensorflow/issues/42596
from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.tools import module_util as _module_util
from tensorflow.python.util.lazy_loader import KerasLazyLoader as _KerasLazyLoader

# Make sure code inside the TensorFlow codebase can use tf2.enabled() at import.
_os.environ["TF2_BEHAVIOR"] = "1"
from tensorflow.python import tf2 as _tf2
_tf2.enable()

# API IMPORTS PLACEHOLDER

# WRAPPER_PLACEHOLDER

# Make sure directory containing top level submodules is in
# the __path__ so that "from tensorflow.foo import bar" works.
# We're using bitwise, but there's nothing special about that.
_API_MODULE = _sys.modules[__name__].bitwise
_tf_api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
_current_module = _sys.modules[__name__]

if not hasattr(_current_module, "__path__"):
  __path__ = [_tf_api_dir]
elif _tf_api_dir not in __path__:
  __path__.append(_tf_api_dir)

# Hook external TensorFlow modules.

# Load tensorflow-io-gcs-filesystem if enabled
if (_os.getenv("TF_USE_MODULAR_FILESYSTEM", "0") == "true" or
    _os.getenv("TF_USE_MODULAR_FILESYSTEM", "0") == "1"):
  import tensorflow_io_gcs_filesystem as _tensorflow_io_gcs_filesystem

# Lazy-load Keras v2/3.
_tf_uses_legacy_keras = (
    _os.environ.get("TF_USE_LEGACY_KERAS", None) in ("true", "True", "1"))
setattr(_current_module, "keras", _KerasLazyLoader(globals()))
_module_dir = _module_util.get_parent_dir_for_name("keras._tf_keras.keras")
_current_module.__path__ = [_module_dir] + _current_module.__path__
if _tf_uses_legacy_keras:
  _module_dir = _module_util.get_parent_dir_for_name("tf_keras.api._v2.keras")
else:
  _module_dir = _module_util.get_parent_dir_for_name("keras.api._v2.keras")
_current_module.__path__ = [_module_dir] + _current_module.__path__


# Enable TF2 behaviors
from tensorflow.python.compat import v2_compat as _compat
_compat.enable_v2_behavior()
_major_api_version = 2


# Load all plugin libraries from site-packages/tensorflow-plugins if we are
# running under pip.
# TODO(gunan): Find a better location for this code snippet.
from tensorflow.python.framework import load_library as _ll
from tensorflow.python.lib.io import file_io as _fi

# Get sitepackages directories for the python installation.
_site_packages_dirs = []
if _site.ENABLE_USER_SITE and _site.USER_SITE is not None:
  _site_packages_dirs += [_site.USER_SITE]
_site_packages_dirs += [p for p in _sys.path if "site-packages" in p]
if "getsitepackages" in dir(_site):
  _site_packages_dirs += _site.getsitepackages()

if "sysconfig" in dir(_distutils):
  _site_packages_dirs += [_distutils.sysconfig.get_python_lib()]

_site_packages_dirs = list(set(_site_packages_dirs))

# Find the location of this exact file.
_current_file_location = _inspect.getfile(_inspect.currentframe())

def _running_from_pip_package():
  return any(
      _current_file_location.startswith(dir_) for dir_ in _site_packages_dirs)

if _running_from_pip_package():
  # TODO(gunan): Add sanity checks to loaded modules here.

  # Load first party dynamic kernels.
  _tf_dir = _os.path.dirname(_current_file_location)
  _kernel_dir = _os.path.join(_tf_dir, "core", "kernels")
  if _os.path.exists(_kernel_dir):
    _ll.load_library(_kernel_dir)

  # Load third party dynamic kernels.
  for _s in _site_packages_dirs:
    _plugin_dir = _os.path.join(_s, "tensorflow-plugins")
    if _os.path.exists(_plugin_dir):
      _ll.load_library(_plugin_dir)
      # Load Pluggable Device Library
      _ll.load_pluggable_device_library(_plugin_dir)

if _os.getenv("TF_PLUGGABLE_DEVICE_LIBRARY_PATH", ""):
  _ll.load_pluggable_device_library(
      _os.getenv("TF_PLUGGABLE_DEVICE_LIBRARY_PATH")
  )

# Add Keras module aliases
_losses = _KerasLazyLoader(globals(), submodule="losses", name="losses")
_metrics = _KerasLazyLoader(globals(), submodule="metrics", name="metrics")
_optimizers = _KerasLazyLoader(
    globals(), submodule="optimizers", name="optimizers")
_initializers = _KerasLazyLoader(
    globals(), submodule="initializers", name="initializers")
setattr(_current_module, "losses", _losses)
setattr(_current_module, "metrics", _metrics)
setattr(_current_module, "optimizers", _optimizers)
setattr(_current_module, "initializers", _initializers)


# Do an eager load for Keras' code so that any function/method that needs to
# happen at load time will trigger, eg registration of optimizers in the
# SavedModel registry.
# See b/196254385 for more details.
try:
  if _tf_uses_legacy_keras:
    importlib.import_module("tf_keras.src.optimizers")
  else:
    importlib.import_module("keras.src.optimizers")
except (ImportError, AttributeError):
  pass

del importlib

# Delete modules that should be hidden from dir().
# Don't fail if these modules are not available.
# For e.g. this file will be originally placed under tensorflow/_api/v1 which
# does not have "python", "core" directories. Then, it will be copied
# to tensorflow/ which does have these two directories.
try:
  del python
except NameError:
  pass
try:
  del core
except NameError:
  pass
try:
  del compiler
except NameError:
  pass

# __all__ PLACEHOLDER
