# Copyright 2021 The OpenXLA Authors
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

from typing import Any, Sequence

from jax.jaxlib._jax import OpSharding_Type
from jax.jaxlib._jax import ProgramShape
from jax.jaxlib._jax import Shape
from jax.jaxlib._jax import XlaComputation

_XlaOpMetadata = Any

class FrontendAttributes:
  def __init__(self) -> None: ...
  def __setitem__(self, key: str, value: str) -> None: ...

class XlaOp: ...

class XlaBuilder:
  def __init__(self, name: str) -> None: ...
  def Build(self, root: XlaOp | None = ...) -> XlaComputation: ...
  def GetShape(self, __op: XlaOp) -> Shape: ...
  build = Build
  def clear_op_metadata(self) -> None: ...
  get_shape = GetShape
  def get_program_shape(self, root: XlaOp | None = ...) -> ProgramShape: ...
  def is_constant(self, __op: XlaOp) -> bool: ...
  def set_op_metadata(self, metadata: _XlaOpMetadata) -> None: ...
  def set_sharding(self, sharding: OpSharding_Type) -> None: ...
  def clear_sharding(self) -> None: ...
  def setup_alias(
      self,
      __output_index: Sequence[int],
      __param_number: int,
      __param_index: Sequence[int],
  ) -> None: ...
