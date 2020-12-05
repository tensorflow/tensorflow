# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""Typing annotation utils."""

import abc
import os
import typing
from typing import Any, AnyStr, Dict, Iterator, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  from typing import Protocol
except ImportError:
  import typing_extensions
  Protocol = typing_extensions.Protocol
# pylint: enable=g-import-not-at-top

# Accept both `str` and `pathlib.Path`-like
PathLike = Union[str, os.PathLike]
PathLikeCls = (str, os.PathLike)  # Used in `isinstance`

T = TypeVar('T')

# Note: `TupleOrList` avoid abiguity from `Sequence` (`str` is `Sequence[str]`,
# `bytes` is `Sequence[int]`).
TupleOrList = Union[Tuple[T, ...], List[T]]

TreeDict = Union[T, Dict[str, 'TreeDict']]  # pytype: disable=not-supported-yet
Tree = Union[T, TupleOrList['Tree'], Dict[str, 'Tree']]  # pytype: disable=not-supported-yet


Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]

Dim = Optional[int]
Shape = TupleOrList[Dim]

JsonValue = Union[
    str, bool, int, float, None, List['JsonValue'], Dict[str, 'JsonValue'],  # pytype: disable=not-supported-yet
]
Json = Dict[str, JsonValue]


# pytype: disable=ignored-abstractmethod


class PurePath(Protocol):
  """Protocol for pathlib.PurePath-like API."""
  parts: Tuple[str, ...]
  drive: str
  root: str
  anchor: str
  name: str
  suffix: str
  suffixes: List[str]
  stem: str

  # pylint: disable=multiple-statements,line-too-long

  def __new__(cls: Type[T], *args: PathLike) -> T: raise NotImplementedError
  def __fspath__(self) -> str: raise NotImplementedError
  def __hash__(self) -> int: raise NotImplementedError
  def __lt__(self, other: 'PurePath') -> bool: raise NotImplementedError
  def __le__(self, other: 'PurePath') -> bool: raise NotImplementedError
  def __gt__(self, other: 'PurePath') -> bool: raise NotImplementedError
  def __ge__(self, other: 'PurePath') -> bool: raise NotImplementedError
  def __truediv__(self: T, key: PathLike) -> T: raise NotImplementedError
  def __rtruediv__(self: T, key: PathLike) -> T: raise NotImplementedError
  def __bytes__(self) -> bytes: raise NotImplementedError
  def as_posix(self) -> str: raise NotImplementedError
  def as_uri(self) -> str: raise NotImplementedError
  def is_absolute(self) -> bool: raise NotImplementedError
  def is_reserved(self) -> bool: raise NotImplementedError
  def match(self, path_pattern: str) -> bool: raise NotImplementedError
  def relative_to(self: T, *other: PathLike) -> T: raise NotImplementedError
  def with_name(self: T, name: str) -> T: raise NotImplementedError
  def with_suffix(self: T, suffix: str) -> T: raise NotImplementedError
  def joinpath(self: T, *other: PathLike) -> T: raise NotImplementedError

  @property
  def parents(self: T) -> Sequence[T]: raise NotImplementedError
  @property
  def parent(self: T) -> T: raise NotImplementedError

  # py3.9 backport of PurePath.is_relative_to.
  def is_relative_to(self, *other: PathLike) -> bool:
    """Return True if the path is relative to another path or False."""
    try:
      self.relative_to(*other)
      return True
    except ValueError:
      return False

  # pylint: enable=multiple-statements,line-too-long


class ReadOnlyPath(PurePath, Protocol):
  """Protocol for read-only methods of pathlib.Path-like API.

  See [pathlib.Path](https://docs.python.org/3/library/pathlib.html)
  documentation.
  """

  @abc.abstractmethod
  def exists(self) -> bool:
    """Returns True if self exists."""

  @abc.abstractmethod
  def is_dir(self) -> bool:
    """Returns True if self is a dir."""

  def is_file(self) -> bool:
    """Returns True if self is a file."""
    return not self.is_dir()

  @abc.abstractmethod
  def iterdir(self: T) -> Iterator[T]:
    """Iterates over the directory."""

  @abc.abstractmethod
  def glob(self: T, pattern: str) -> Iterator[T]:
    """Yielding all matching files (of any kind)."""
    # Might be able to implement using `iterdir` (recursivelly for `rglob`).

  def rglob(self: T, pattern: str) -> Iterator[T]:
    """Yielding all matching files recursivelly (of any kind)."""
    return self.glob(f'**/{pattern}')

  def expanduser(self: T) -> T:
    """Returns a new path with expanded `~` and `~user` constructs."""
    if '~' not in self.parts:  # pytype: disable=attribute-error
      return self
    raise NotImplementedError

  @abc.abstractmethod
  def resolve(self: T, strict: bool = False) -> T:
    """Returns the absolute path."""

  @abc.abstractmethod
  def open(
      self,
      mode: str = 'r',
      encoding: Optional[str] = None,
      errors: Optional[str] = None,
      **kwargs: Any,
  ) -> typing.IO[AnyStr]:
    """Opens the file."""

  def read_bytes(self) -> bytes:
    """Reads contents of self as bytes."""
    with self.open('rb') as f:
      return f.read()

  def read_text(self, encoding: Optional[str] = None) -> str:
    """Reads contents of self as bytes."""
    with self.open('r', encoding=encoding) as f:
      return f.read()


class ReadWritePath(ReadOnlyPath, Protocol):
  """Protocol for pathlib.Path-like API.

  See [pathlib.Path](https://docs.python.org/3/library/pathlib.html)
  documentation.
  """

  @abc.abstractmethod
  def mkdir(
      self,
      mode: int = 0o777,
      parents: bool = False,
      exist_ok: bool = False,
  ) -> None:
    """Create a new directory at this given path."""

  @abc.abstractmethod
  def rmdir(self) -> None:
    """Remove the empty directory at this given path."""

  @abc.abstractmethod
  def rmtree(self) -> None:
    """Remove the directory, including all sub-files."""

  def write_bytes(self, data: bytes) -> None:
    """Writes content as bytes."""
    with self.open('wb') as f:
      return f.write(data)

  def write_text(
      self,
      data: str,
      encoding: Optional[str] = None,
      errors: Optional[str] = None,
  ) -> None:
    """Writes content as str."""
    with self.open('w') as f:
      return f.write(data)

  def touch(self, mode: int = 0o666, exist_ok: bool = True) -> None:
    """Create a file at this given path."""
    del mode  # Unused
    if self.exists() and not exist_ok:
      raise FileExistsError(f'{self} already exists.')
    self.write_text('')

  @abc.abstractmethod
  def rename(self: T, target: PathLike) -> T:
    """Renames the path."""

  @abc.abstractmethod
  def replace(self: T, target: PathLike) -> T:
    """Overwrites the destination path."""
