# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

import copy
import sys

from dataclasses import dataclass
from functools import total_ordering

from tensorflow.python.compiler.tensorrt import _pywrap_py_utils
from tensorflow.python.compiler.tensorrt.lazy_utils import LazyObj
from tensorflow.python.compiler.tensorrt.types import ExtendedEnum
from tensorflow.python.compiler.tensorrt.types import TrtVersion
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


TRT_MINIMUM_LINKED_VERSION = TrtVersion("7.0.0")
TRT_MINIMUM_LOADED_VERSION = TrtVersion("7.0.0")


TRT_ENGINE_OP_NAME = "TRTEngineOp"


@dataclass(order=False)
@total_ordering
class TrtVersionEnv(object):

  linked: TrtVersion = LazyObj(TrtVersion)(
      _pywrap_py_utils.get_linked_tensorrt_version)

  loaded: TrtVersion = LazyObj(TrtVersion)(
      _pywrap_py_utils.get_loaded_tensorrt_version)

  # Singleton
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls, *args, **kwargs)
    return cls._instance

  @classmethod
  def load(cls):
    return cls()

  def __init__(self):
    if self.__class__._instance is not None: return
    logging.info(f"Linked TensorRT version: {self.linked}")
    logging.info(f"Loaded TensorRT version: {self.loaded}")
    super().__init__()
    self.__validate__()

  @staticmethod
  def _raise_trt_version_deprecated(version_type, trt_version):
    assert \
      version_type in ["linked", "loaded"], \
      (f"Incorrect value received for version_type: {version_type}. Accepted: "
      "['linked', 'loaded']")

    if not isinstance(trt_version, TrtVersion):
      raise ValueError("`_raise_trt_version_deprecated()` only accepts "
                       "instances of `TrtVersion`, received: "
                       f"`{type(trt_version)}`.")

    logging.error(
      f"The {version_type} version of TensorRT: "
      f"`{trt_version}` has now been removed. Please upgrade to TensorRT 7 "
      "or more recent."
    )

    raise RuntimeError(f"Incompatible {version_type} TensorRT versions")

  def __validate__(self):
    if self.linked < TRT_MINIMUM_LINKED_VERSION:
      TrtVersionEnv._raise_trt_version_deprecated("linked", self.linked)

    if self.loaded < TRT_MINIMUM_LOADED_VERSION:
      TrtVersionEnv._raise_trt_version_deprecated("loaded", self.loaded)

    if (self.loaded.major != self.linked.major or self.loaded < self.linked):
      logging.error(
          f"Loaded TensorRT {self.loaded} but linked TensorFlow against "
          f"TensorRT {self.linked}. A few requirements must be met:\n"
          "\t-It is required to use the same major version of TensorRT during "
          "compilation and runtime.\n"
          "\t-TensorRT does not support forward compatibility. The loaded "
          "version has to be equal or more recent than the linked version.")

      raise RuntimeError("Incompatible TensorRT major version")

    elif self.loaded != self.linked:
      logging.info(
          f"Loaded TensorRT {self.loaded} and linked TensorFlow against "
          f"TensorRT {self.linked}. This is supported because TensorRT "
          "minor/patch upgrades are backward compatible."
      )

  def __eq__(self, other):
    if not isinstance(other, TrtVersion):
      other = TrtVersion(other)
    return self.loaded == other and self.linked == other

  def __lt__(self, other):
    if not isinstance(other, TrtVersion):
      other = TrtVersion(other)
    return self.loaded < other and self.linked < other


@tf_export("experimental.tensorrt.TrtPrecisionMode", v1=[])
class TrtPrecisionMode(ExtendedEnum):
  FP32 = "FP32"
  FP16 = "FP16"
  INT8 = "INT8"


@tf_export("experimental.tensorrt.TrtProfileStrategy", v1=[])
class TrtProfileStrategy(ExtendedEnum):
  RANGE = "Range"
  OPTIMAL = "Optimal"
  RANGE_OPTIMAL = "Range+Optimal"
  IMPLICIT_BATCH_MODE_COMPATIBLE = "ImplicitBatchModeCompatible"

  def __validate__(self):
    if self == TrtProfileStrategy.IMPLICIT_BATCH_MODE_COMPATIBLE:
      logging.warn(
          "ImplicitBatchModeCompatible strategy is deprecated, and"
          " using it may result in errors during engine building. Please"
          " consider using a different profile strategy."
      )


def _get_default_max_trt_workspace_size():
  # Use a large enough number as the default max_workspace_size for TRT engines,
  # so it can produce reasonable performance results with the default.
  # For TRT >= 8.4, the recommendation is MAX_INT.
  if TrtVersionEnv() >= TrtVersion("8.4.0"):
    # We must use `sys.maxsize - 512` to avoid overflow during casting.
    return sys.maxsize - 512
  else:
    return 1 << 30  # 1,073,741,824


DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES = LazyObj(int)(
    _get_default_max_trt_workspace_size)


@tf_export("experimental.tensorrt.ConversionParams", v1=[])
@dataclass(order=False)
class TrtConversionParams(object):
  max_workspace_size_bytes: int = DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES

  precision_mode: TrtPrecisionMode = TrtPrecisionMode.FP32

  minimum_segment_size: int = 3

  maximum_cached_engines: int = 1

  use_calibration: bool = True

  allow_build_at_runtime: bool = True

  def __new__(cls, *args, **kwargs):
    obj = super().__new__(cls)
    obj.validate()
    return obj

  @deprecation.deprecated_args(None,
                               "`is_v2` is deprecated and ignored. Please "
                               "remove from your code", "is_v2")
  def validate(self, is_v2=True):
    """Validate the provided TrtConversionParams.

    Args:
      conversion_params: a TrtConversionParams instance.
      is_v2: whether we're getting a RewriterConfig for TF 2.0.

    Raises:
      TypeError: if any of the parameters are of unexpected type.
      ValueError: if any of the parameters are of unexpected value.
    """

    if (self.minimum_segment_size <= 0 and self.minimum_segment_size != -1):
      raise ValueError("minimum segment size should be positive or -1 "
                       "(to disable main graph conversion).")

  # Exists for backward compatibility
  def _asdict(self):
    return self.__dict__

  # Exists for backward compatibility
  def _replace(self, **kwargs):
    cpy_obj = copy.deepcopy(self)
    for key, value in kwargs.items():
      cpy_obj.__setattr__(key, value)
    return cpy_obj


DEFAULT_TRT_CONVERSION_PARAMS = TrtConversionParams()
