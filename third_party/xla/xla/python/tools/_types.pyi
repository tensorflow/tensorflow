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

from typing import Union
import numpy as np
from xla import xla_data_pb2

# LINT.IfChange
NdarrayTree = Union[np.ndarray, tuple['NdarrayTree', ...]]
def make_ndarray(proto: xla_data_pb2.LiteralProto, /) -> NdarrayTree: ...
def dtype_to_etype(dtype: np.dtype, /) -> xla_data_pb2.PrimitiveType: ...
def etype_to_dtype(ptype: xla_data_pb2.PrimitiveType, /) -> np.dtype: ...
# LINT.ThenChange(types.py, _types.cc)
