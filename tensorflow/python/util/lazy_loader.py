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

"""A LazyLoader class."""

import importlib
import os
import types
from tensorflow.python.platform import tf_logging as logging

_TENSORFLOW_LAZY_LOADER_PREFIX = "_tfll"


class LazyLoader(types.ModuleType):
  """Lazily import a module, mainly to avoid pulling in large dependencies.

  `contrib`, and `ffmpeg` are examples of modules that are large and not always
  needed, and this allows them to only be loaded when they are used.
  """

  # The lint error here is incorrect.
  def __init__(self, local_name, parent_module_globals, name, warning=None):
    self._tfll_local_name = local_name
    self._tfll_parent_module_globals = parent_module_globals
    self._tfll_warning = warning

    # These members allows doctest correctly process this module member without
    # triggering self._load(). self._load() mutates parent_module_globals and
    # triggers a dict mutated during iteration error from doctest.py.
    # - for from_module()
    super().__setattr__("__module__", name.rsplit(".", 1)[0])
    # - for is_routine()
    super().__setattr__("__wrapped__", None)

    super().__init__(name)

  def _load(self):
    """Load the module and insert it into the parent's globals."""
    # Import the target module and insert it into the parent's namespace
    module = importlib.import_module(self.__name__)
    self._tfll_parent_module_globals[self._tfll_local_name] = module

    # Emit a warning if one was specified
    if self._tfll_warning:
      logging.warning(self._tfll_warning)
      # Make sure to only warn once.
      self._tfll_warning = None

    # Update this object's dict so that if someone keeps a reference to the
    #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
    #   that fail).
    self.__dict__.update(module.__dict__)

    return module

  def __getattr__(self, name):
    module = self._load()
    return getattr(module, name)

  def __setattr__(self, name, value):
    if name.startswith(_TENSORFLOW_LAZY_LOADER_PREFIX):
      super().__setattr__(name, value)
    else:
      module = self._load()
      setattr(module, name, value)
      self.__dict__[name] = value
      try:
        # check if the module has __all__
        if name not in self.__all__ and name != "__all__":
          self.__all__.append(name)
      except AttributeError:
        pass

  def __delattr__(self, name):
    if name.startswith(_TENSORFLOW_LAZY_LOADER_PREFIX):
      super().__delattr__(name)
    else:
      module = self._load()
      delattr(module, name)
      self.__dict__.pop(name)
      try:
        # check if the module has __all__
        if name in self.__all__:
          self.__all__.remove(name)
      except AttributeError:
        pass

  def __repr__(self):
    # Carefully to not trigger _load, since repr may be called in very
    # sensitive places.
    return f"<LazyLoader {self.__name__} as {self._tfll_local_name}>"

  def __dir__(self):
    module = self._load()
    return dir(module)

  def __reduce__(self):
    return importlib.import_module, (self.__name__,)


class KerasLazyLoader(LazyLoader):
  """LazyLoader that handles routing to different Keras version."""

  def __init__(  # pylint: disable=super-init-not-called
      self, parent_module_globals, mode=None, submodule=None, name="keras"):
    self._tfll_parent_module_globals = parent_module_globals
    self._tfll_mode = mode
    self._tfll_submodule = submodule
    self._tfll_name = name
    self._tfll_initialized = False

  def _initialize(self):
    """Resolve the Keras version to use and initialize the loader."""
    self._tfll_initialized = True
    package_name = None
    keras_version = None
    if os.environ.get("TF_USE_LEGACY_KERAS", None) in ("true", "True", "1"):
      try:
        import tf_keras  # pylint: disable=g-import-not-at-top,unused-import

        keras_version = "tf_keras"
        if self._tfll_mode == "v1":
          package_name = "tf_keras.api._v1.keras"
        else:
          package_name = "tf_keras.api._v2.keras"
      except ImportError:
        logging.warning(
            "Your environment has TF_USE_LEGACY_KERAS set to True, but you "
            "do not have the tf_keras package installed. You must install it "
            "in order to use the legacy tf.keras. Install it via: "
            "`pip install tf_keras`"
        )
    else:
      try:
        import keras  # pylint: disable=g-import-not-at-top

        if keras.__version__.startswith("3."):
          # This is the Keras 3.x case.
          keras_version = "keras_3"
          package_name = "keras._tf_keras.keras"
        else:
          # This is the Keras 2.x case.
          keras_version = "keras_2"
          if self._tfll_mode == "v1":
            package_name = "keras.api._v1.keras"
          else:
            package_name = "keras.api._v2.keras"
      except ImportError:
        raise ImportError(  # pylint: disable=raise-missing-from
            "Keras cannot be imported. Check that it is installed."
        )

    self._tfll_keras_version = keras_version
    if keras_version is not None:
      if self._tfll_submodule is not None:
        package_name += "." + self._tfll_submodule
      super().__init__(
          self._tfll_name, self._tfll_parent_module_globals, package_name
      )
    else:
      raise ImportError(  # pylint: disable=raise-missing-from
          "Keras cannot be imported. Check that it is installed."
      )

  def __getattr__(self, item):
    if item in ("_tfll_mode", "_tfll_initialized", "_tfll_name"):
      return super(types.ModuleType, self).__getattribute__(item)
    if not self._tfll_initialized:
      self._initialize()
    if self._tfll_keras_version == "keras_3":
      if (
          self._tfll_mode == "v1"
          and not self._tfll_submodule
          and item.startswith("compat.v1.")
      ):
        raise AttributeError(
            "`tf.compat.v1.keras` is not available with Keras 3. Keras 3 has "
            "no support for TF 1 APIs. You can install the `tf_keras` package "
            "as an alternative, and set the environment variable "
            "`TF_USE_LEGACY_KERAS=True` to configure TensorFlow to route "
            "`tf.compat.v1.keras` to `tf_keras`."
        )
      elif (
          self._tfll_mode == "v2"
          and not self._tfll_submodule
          and item.startswith("compat.v2.")
      ):
        raise AttributeError(
            "`tf.compat.v2.keras` is not available with Keras 3. Just use "
            "`import keras` instead."
        )
      elif self._tfll_submodule and self._tfll_submodule.startswith(
          "__internal__.legacy."
      ):
        raise AttributeError(
            f"`{item}` is not available with Keras 3."
        )
    module = self._load()
    return getattr(module, item)

  def __repr__(self):
    if self._tfll_initialized:
      return (
          f"<KerasLazyLoader ({self._tfll_keras_version}) "
          f"{self.__name__} as {self._tfll_local_name} mode={self._tfll_mode}>"
      )
    return "<KerasLazyLoader>"

  def __dir__(self):
    if not self._tfll_initialized:
      self._initialize()
    return super().__dir__()
