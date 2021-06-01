# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

import inspect
from typing import Any, Callable, List, Sequence

_AvalDimSharding = Any
_MeshDimAssignment = Any

class NoSharding:
  def __init__(self) -> None: ...
  def __repr__(self) -> str: ...
  def __eq__(self, __other: Any) -> bool: ...

class Chunked:
  def __init__(self, __chunks: Sequence[int]) -> None: ...
  def __repr__(self) -> str: ...
  def __eq__(self, __other: Any) -> bool: ...

class Unstacked:
  def __init__(self, __sz: int) -> None: ...
  def __repr__(self) -> str: ...
  def __eq__(self, __other: Any) -> bool: ...

class ShardedAxis:
  def __init__(self, __axis: int) -> None: ...
  def __repr__(self) -> str: ...
  def __eq__(self, __other: ShardedAxis) -> bool: ...

class Replicated:
  def __init__(self, __replicas: int) -> None: ...
  def __repr__(self) -> str: ...
  def __eq__(self, __other: Replicated) -> bool: ...

class ShardingSpec:
  def __init__(self,
               sharding: Sequence[_AvalDimSharding],
               mesh_mapping: Sequence[_MeshDimAssignment]) -> None: ...
  sharding: Sequence[_AvalDimSharding]
  mesh_mapping: Sequence[_MeshDimAssignment]
  def __eq__(self, __other: ShardingSpec) -> bool: ...

class ShardedDeviceArray:
  def __init__(self,
               __aval: Any,
               __sharding_spec: ShardingSpec,
               __device_buffers: List[Any]) -> None: ...
  aval: Any
  sharding_spec: ShardingSpec
  device_buffers: List[Any]

class PmapFunction:
   def __call__(self, *args, **kwargs) -> Any: ...
   __signature__: inspect.Signature

def pmap(__fun: Callable[..., Any],
         __cache_miss: Callable[..., Any],
         __static_argnums: Sequence[int]) -> PmapFunction: ...
