# Copyright 2024 The OpenXLA Authors.
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
"""tensorflow.compiler.xla.python.tools.types.

This module provides Python bindings for various functions in
'tensorflow/compiler/xla/python/types.h'.  It is primarily intended
to assist internal users in debugging things; and is not considered
part of the public API for OpenXLA.

NOTE: This module *does* depend on Python protocol buffers; so beware!
The XLA Python bindings are currently packaged both as part of jaxlib and
as part of TensorFlow.  Therefore, since we use protocol buffers here,
importing both jaxlib and TensorFlow may fail with duplicate protocol
buffer message definitions.
"""

from typing import Union
# NOTE: `ml_dtypes` is implicitly required by `xla::LiteralToPython`.
# The entire goal of this wrapper library is to capture this dependency,
# so that client code need not be aware of it.
import ml_dtypes  # pylint: disable=unused-import
import numpy
# NOTE: These protos are not part of TensorFlow's public API, therefore
# we cannot abide by [g-direct-tensorflow-import].
# pylint: disable=g-direct-tensorflow-import,unused-import
from local_xla.xla import xla_data_pb2
# pylint: enable=g-direct-tensorflow-import,unused-import

# NOTE: `import <name> as <name>` is required for names to be exported.
# See PEP 484 & <https://github.com/google/jax/issues/7570>
# pylint: disable=g-importing-member,useless-import-alias,unused-import,g-multiple-import
# LINT.IfChange
from ._types import (
    make_ndarray as make_ndarray,
    dtype_to_etype as dtype_to_etype,
    etype_to_dtype as etype_to_dtype,
)
# TODO(wrengr): We can't import the `NdarrayTree` defined in the pyi file.
# So re-defining it here for now.
NdarrayTree = Union[numpy.ndarray, tuple['NdarrayTree', ...]]
# LINT.ThenChange(_types.pyi)
