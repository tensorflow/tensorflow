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
from typing import Any, Callable, Sequence, Tuple

import numpy as np
from tensorflow.compiler.xla.python import xla_extension

Client = xla_extension.Client

class CompiledFunctionCache:
  def __init__(self, capacity): ...
  def size(self) -> int: ...
  def capacity(self) -> int: ...
  def clear(self): ...

class CompiledFunction:
  def __call__(self, *args, **kwargs) -> Any: ...
  __signature__: inspect.Signature
  def _cache_size(self) -> int: ...
  def _clear_cache(self) -> None: ...

class GlobalJitState:
  disable_jit: bool
  enable_x64: bool
  extra_jit_context: Any

class ThreadLocalJitState:
  disable_jit: bool
  enable_x64: bool
  extra_jit_context: Any

def global_state() -> GlobalJitState: ...
def thread_local_state() -> ThreadLocalJitState: ...

def jit_is_enabled() -> bool: ...
def get_enable_x64() -> bool: ...

def set_disable_jit_cpp_flag(__arg: bool) -> None: ...
def get_disable_jit_cpp_flag() -> bool: ...
def set_disable_jit_thread_local(__arg: bool) -> None: ...
def get_disable_jit_thread_local() -> bool: ...
def set_disable_jit(__arg: bool) -> None: ...
def get_disable_jit() -> bool: ...

def set_disable_x64_cpp_flag(__arg: bool) -> None: ...
def get_disable_x64_cpp_flag() -> bool: ...
def set_disable_x64_thread_local(__arg: bool) -> None: ...
def get_disable_x64_thread_local() -> bool: ...

def jit(fun: Callable[..., Any],
        cache_miss: Callable[..., Any],
        get_device: Callable[..., Any],
        static_argnums: Sequence[int],
        static_argnames: Sequence[str] = ...,
        donate_argnums: Sequence[int] = ...,
        cache: Optional[CompiledFunctionCache] = ...) -> CompiledFunction: ...

def device_put(
    __obj: Any,
    __jax_enable_x64: bool,
    __to_device: Client) -> Any: ...

class ArgSignature:
  dtype: np.dtype
  shape: Tuple[int, ...]
  weak_type: bool

def _ArgSignatureOfValue(
    __arg: Any,
    __jax_enable_x64: bool) -> ArgSignature: ...

def _is_float0(__arg: Any) -> bool: ...
