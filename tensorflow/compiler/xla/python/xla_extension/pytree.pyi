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

from typing import Any, Callable, Hashable, Iterable, List, Optional, Sequence, Tuple, Type, TypeVar

_T = TypeVar("_T")

def flatten(
    tree: Any,
    leaf_predicate: Optional[Callable[[Any], bool]] = ...,
) -> Tuple[Sequence[Any], PyTreeDef]: ...
def tuple(arg0: Sequence[PyTreeDef]) -> PyTreeDef: ...
def all_leaves(arg0: Iterable[Any]) -> bool: ...

class PyTreeDef:
  def unflatten(self, __leaves: Iterable[Any]) -> PyTreeDef: ...
  def flatten_up_to(self, __xs: Any) -> List[Any]: ...
  def compose(self, __inner: PyTreeDef) -> PyTreeDef: ...
  def walk(self,
           __f_node: Callable[[Any], Any],
           __f_leaf: Optional[Callable[[_T], Any]],
           leaves: Iterable[Any]) -> Any: ...
  def from_iterable_tree(self, __xs: Any): ...
  def children(self) -> Sequence[PyTreeDef]: ...
  num_leaves: int
  num_nodes: int
  def __repr__(self) -> str: ...
  def __eq__(self, __other: PyTreeDef) -> bool: ...
  def __ne__(self, __other: PyTreeDef) -> bool: ...
  def __hash__(self) -> int: ...

_Children = TypeVar("_Children", bound=Iterable[Any])
_AuxData = TypeVar("_AuxData", bound=Hashable)

def register_node(
    __type: Type[_T],
    to_iterable: Callable[[_T], Tuple[_Children, _AuxData]],
    from_iterable: Callable[[_AuxData, _Children], _T]) -> Any: ...
