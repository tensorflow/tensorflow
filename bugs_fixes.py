import os as _os
import sys as _sys

from tensorflow.python.tools import module_util as _module_util
from tensorflow.python.util.lazy_loader import KerasLazyLoader as _KerasLazyLoader

# API IMPORTS PLACEHOLDER

# WRAPPER_PLACEHOLDER

# Hook external TensorFlow modules.
_current_module = _sys.modules[__name__]

# Lazy load Keras v1 with environment variable handling
_tf_uses_legacy_keras = (
    _os.environ.get("TF_USE_LEGACY_KERAS", "").lower() in ("true", "1"))
setattr(_current_module, "keras", _KerasLazyLoader(globals(), mode="v1"))

try:
    if _tf_uses_legacy_keras:
        _module_dir = _module_util.get_parent_dir_for_name("tf_keras.api._v1.keras")
    else:
        _module_dir = _module_util.get_parent_dir_for_name("keras.api._v1.keras")
    _current_module.__path__ = [_module_dir] + _current_module.__path__
except ImportError as e:
    raise ImportError(f"Failed to load Keras modules. Error: {e}")

from tensorflow.python.platform import flags  # pylint: disable=g-import-not-at-top
_current_module.app.flags = flags  # pylint: disable=undefined-variable
setattr(_current_module, "flags", flags)

# Add module aliases from Keras to TF.
# Some tf endpoints actually live under Keras.
_current_module.layers = _KerasLazyLoader(
    globals(),
    submodule="__internal__.legacy.layers",
    name="layers",
    mode="v1"
)

try:
    if _tf_uses_legacy_keras:
        _module_dir = _module_util.get_parent_dir_for_name(
            "tf_keras.api._v1.keras.__internal__.legacy.layers")
    else:
        _module_dir = _module_util.get_parent_dir_for_name(
            "keras.api._v1.keras.__internal__.legacy.layers")
    _current_module.__path__ = [_module_dir] + _current_module.__path__
except ImportError as e:
    raise ImportError(f"Failed to load Keras layers. Error: {e}")

# Lazy load Keras RNN cell under TensorFlow's nn module
_current_module.nn.rnn_cell = _KerasLazyLoader(
    globals(),
    submodule="__internal__.legacy.rnn_cell",
    name="rnn_cell",
    mode="v1"
)

try:
    if _tf_uses_legacy_keras:
        _module_dir = _module_util.get_parent_dir_for_name(
            "tf_keras.api._v1.keras.__internal__.legacy.rnn_cell")
    else:
        _module_dir = _module_util.get_parent_dir_for_name(
            "keras.api._v1.keras.__internal__.legacy.rnn_cell")
    _current_module.nn.__path__ = [_module_dir] + _current_module.nn.__path__
except ImportError as e:
    raise ImportError(f"Failed to load Keras RNN cell. Error: {e}")

# Additional enhancements can be added here as needed

