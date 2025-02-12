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
"""Bring in all of the public TensorFlow interface into this module."""

import importlib
import inspect as _inspect
import os as _os
import site as _site
import sys as _sys
import sysconfig

# pylint: disable=g-bad-import-order,protected-access,g-import-not-at-top
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.tools import module_util as _module_util
from tensorflow.python.platform import tf_logging as _logging
from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader
from tensorflow.python.util.lazy_loader import KerasLazyLoader as _KerasLazyLoader

# API IMPORTS PLACEHOLDER

# WRAPPER_PLACEHOLDER

if "dev" in __version__:   # pylint: disable=undefined-variable
  _logging.warning("""

  TensorFlow's `tf-nightly` package will soon be updated to TensorFlow 2.0.

  Please upgrade your code to TensorFlow 2.0:
    * https://www.tensorflow.org/guide/migrate

  Or install the latest stable TensorFlow 1.X release:
    * `pip install -U "tensorflow==1.*"`

  Otherwise your code may be broken by the change.

  """)

# Make sure directory containing top level submodules is in
# the __path__ so that "from tensorflow.foo import bar" works.
# We're using bitwise, but there's nothing special about that.
_API_MODULE = _sys.modules[__name__].bitwise  # pylint: disable=undefined-variable
_current_module = _sys.modules[__name__]
_tf_api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
if not hasattr(_current_module, "__path__"):
  __path__ = [_tf_api_dir]
elif _tf_api_dir not in __path__:
  __path__.append(_tf_api_dir)

# Hook external TensorFlow modules.
# Import compat before trying to import summary from tensorboard, so that
# reexport_tf_summary can get compat from sys.modules. Only needed if using
# lazy loading.
_current_module.compat.v2  # pylint: disable=pointless-statement

# Load tensorflow-io-gcs-filesystem if enabled
if (_os.getenv("TF_USE_MODULAR_FILESYSTEM", "0") == "true" or
    _os.getenv("TF_USE_MODULAR_FILESYSTEM", "0") == "1"):
  import tensorflow_io_gcs_filesystem as _tensorflow_io_gcs_filesystem

# Lazy-load Keras v1.
_tf_uses_legacy_keras = (
    _os.environ.get("TF_USE_LEGACY_KERAS", None) in ("true", "True", "1"))
setattr(_current_module, "keras", _KerasLazyLoader(globals(), mode="v1"))
_module_dir = _module_util.get_parent_dir_for_name("keras._tf_keras.keras")
_current_module.__path__ = [_module_dir] + _current_module.__path__
if _tf_uses_legacy_keras:
  _module_dir = _module_util.get_parent_dir_for_name("tf_keras.api._v1.keras")
else:
  _module_dir = _module_util.get_parent_dir_for_name("keras.api._v1.keras")
_current_module.__path__ = [_module_dir] + _current_module.__path__

_CONTRIB_WARNING = """
The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
  * https://github.com/tensorflow/io (for I/O related ops)
If you depend on functionality not listed there, please file an issue.
"""
contrib = _LazyLoader("contrib", globals(), "tensorflow.contrib",
                      _CONTRIB_WARNING)
# The templated code that replaces the placeholder above sometimes
# sets the __all__ variable. If it does, we have to be sure to add
# "contrib".
if "__all__" in vars():
  vars()["__all__"].append("contrib")

from tensorflow.python.platform import flags
# The "app" module will be imported as part of the placeholder section above.
_current_module.app.flags = flags  # pylint: disable=undefined-variable
setattr(_current_module, "flags", flags)

_major_api_version = 1

# Add module aliases from Keras to TF.
# Some tf endpoints actually lives under Keras.
_current_module.layers = _KerasLazyLoader(
    globals(),
    submodule="__internal__.legacy.layers",
    name="layers",
    mode="v1")
if _tf_uses_legacy_keras:
  _module_dir = _module_util.get_parent_dir_for_name(
      "tf_keras.api._v1.keras.__internal__.legacy.layers")
else:
  _module_dir = _module_util.get_parent_dir_for_name(
      "keras.api._v1.keras.__internal__.legacy.layers")
_current_module.__path__ = [_module_dir] + _current_module.__path__

_current_module.nn.rnn_cell = _KerasLazyLoader(
    globals(),
    submodule="__internal__.legacy.rnn_cell",
    name="rnn_cell",
    mode="v1")
if _tf_uses_legacy_keras:
  _module_dir = _module_util.get_parent_dir_for_name(
      "tf_keras.api._v1.keras.__internal__.legacy.rnn_cell")
else:
  _module_dir = _module_util.get_parent_dir_for_name(
      "keras.api._v1.keras.__internal__.legacy.rnn_cell")
_current_module.nn.__path__ = [_module_dir] + _current_module.nn.__path__

del importlib

# Load all plugin libraries from site-packages/tensorflow-plugins if we are
# running under pip.
# TODO(gunan): Find a better location for this code snippet.
from tensorflow.python.framework import load_library as _ll
from tensorflow.python.lib.io import file_io as _fi

# Get sitepackages directories for the python installation.
_site_packages_dirs = []
_site_packages_dirs += [] if _site.USER_SITE is None else [_site.USER_SITE]
_site_packages_dirs += [p for p in _sys.path if "site-packages" in p]
if "getsitepackages" in dir(_site):
  _site_packages_dirs += _site.getsitepackages()

for _scheme in sysconfig.get_scheme_names():
  for _name in ["purelib", "platlib"]:
    _site_packages_dirs += [sysconfig.get_path(_name, _scheme)]

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
