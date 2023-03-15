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

from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
from tensorflow.compiler.xla.python import xla_extension

Client = xla_extension.Client
Device = xla_extension.Device

CompiledFunction = xla_extension.CompiledFunction

class CompiledFunctionCache:
  def __init__(self, capacity: int = ...): ...
  def __getstate__(self) -> Any: ...
  def __setstate__(self, Any): ...
  def size(self) -> int: ...
  def capacity(self) -> int: ...
  def clear(self): ...
  @staticmethod
  def clear_all(): ...

class JitState:
  disable_jit: Optional[bool]
  enable_x64: Optional[bool]
  extra_jit_context: Any
  post_hook: Optional[Callable]

def global_state() -> JitState: ...
def thread_local_state() -> JitState: ...

def jit_is_disabled() -> bool: ...
def get_enable_x64() -> bool: ...
def set_thread_local_state_initialization_callback(
    function: Callable[[], None]): ...

def jit(fun: Callable[..., Any],
        cache_miss: Callable[..., Any],
        get_device: Callable[..., Any],
        static_argnums: Sequence[int],
        static_argnames: Sequence[str] = ...,
        donate_argnums: Sequence[int] = ...,
        jit_device: Optional[Device] = ...,
        cache: Optional[CompiledFunctionCache] = ...) -> CompiledFunction: ...

class ArgSignature:
  dtype: np.dtype
  shape: Tuple[int, ...]
  weak_type: bool

def _ArgSignatureOfValue(
    __arg: Any,
    __jax_enable_x64: bool) -> ArgSignature: ...

def _is_float0(__arg: Any) -> bool: ...
