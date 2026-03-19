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

from jax.jaxlib import _jax

class HloPassInterface:
  @property
  def name(self) -> str: ...
  def is_pass_pipeline(self) -> bool: ...
  def run(self, module: _jax.HloModule) -> bool: ...

class HloDCE(HloPassInterface):
  def __init__(self) -> None: ...

class CallInliner(HloPassInterface):
  def __init__(self) -> None: ...

class FlattenCallGraph(HloPassInterface):
  def __init__(self) -> None: ...

class TupleSimplifer(HloPassInterface):
  def __init__(self) -> None: ...
