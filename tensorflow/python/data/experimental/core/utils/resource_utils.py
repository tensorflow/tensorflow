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

"""Utils to handle resources."""

import itertools
import os
import pathlib
import sys
import types
import typing
from typing import Union

from tensorflow.data.experimental.core.utils import generic_path
from tensorflow.data.experimental.core.utils import type_utils

# pylint: disable=g-import-not-at-top
if sys.version_info >= (3, 9):  # `importlib.resources.files` was added in 3.9
  import importlib.resources as importlib_resources
else:
  import importlib_resources

if sys.version_info >= (3, 8):  # `zipfiles.Path` was added in 3.8
  import zipfile
else:
  import zipp as zipfile  # Before 3.8, importlib_resources uses backport
# pylint: enable=g-import-not-at-top

PathLike = type_utils.PathLike
ReadOnlyPath = type_utils.ReadOnlyPath
ReadWritePath = type_utils.ReadWritePath


@generic_path.register_pathlike_cls
class ResourcePath(zipfile.Path):
  """Wrapper around `zipfile.Path` compatible with `os.PathLike`.

  Note: Calling `os.fspath` on the path will extract the file so should be
  discouraged.

  """

  def __fspath__(self) -> str:
    """Path string for `os.path.join`, `open`,... compatibility.

    Note: Calling `os.fspath` on the path extract the file, so should be
    discouraged. Prefer using `read_bytes`,... This only works for files,
    not directories.

    Returns:
      the extracted path string.
    """
    raise NotImplementedError('zipapp not supported. Please send us a PR.')

  # Required due to: https://bugs.python.org/issue42043
  def _next(self, at) -> 'ResourcePath':
    return type(self)(self.root, at)

  def joinpath(self, *parts: PathLike) -> 'ResourcePath':
    """Overwrite `joinpath` to be consistent with `pathlib.Path`."""
    if not parts:
      return self
    else:
      return super().joinpath(os.path.join(*parts))  # pylint: disable=no-value-for-parameter


def resource_path(package: Union[str, types.ModuleType]) -> ReadOnlyPath:
  """Returns `importlib.resources.files`."""
  path = importlib_resources.files(package)  # pytype: disable=module-attr
  if isinstance(path, pathlib.Path):
    # TODO(tfds): To ensure compatibility with zipfile.Path, we should ensure
    # that the returned `pathlib.Path` isn't missused. More specifically:
    # * `os.fspath` should only be called on files (not directories)
    # * `str(path)` should be forbidden (only `__format__` allowed).
    # In practice, it is trickier to do as `__fspath__` and `__str__` are
    # called internally.
    # Convert to `GPath` for consistency and compatibility with `MockFs`.
    return generic_path.as_path(path)
  elif isinstance(path, zipfile.Path):
    path = ResourcePath(path.root, path.at)
    return typing.cast(ReadOnlyPath, path)
  else:
    raise TypeError(f'Unknown resource path: {type(path)}: {path}')


def to_write_path(path: ReadOnlyPath) -> ReadWritePath:
  """Cast the path to a read-write Path."""
  path = typing.cast(ReadWritePath, path)
  return path


def tfds_path(*relative_path: PathLike) -> ReadOnlyPath:
  """Path to `tensorflow_datasets/` root dir.

  The following examples are equivalent:

  ```py
  path = tfds.core.tfds_path() / 'path/to/data.txt'
  path = tfds.core.tfds_path('path/to/data.txt')
  path = tfds.core.tfds_path('path', 'to', 'data.txt')
  ```

  Note: Even if `/` is used, those examples are compatible with Windows, as
  pathlib will automatically normalize the paths.

  Args:
    *relative_path: Relative path, eventually to concatenate.

  Returns:
    path: The root TFDS path.
  """
  return resource_path('tensorflow_datasets').joinpath(*relative_path)


def tfds_write_path(*relative_path: PathLike) -> ReadWritePath:
  """Path to `tensorflow_datasets/` root dir (read-write).

  Contrary to `tfds.core.tfds_path`, path returned here support write
  operations. Used in scripts to update the TFDS repository.

  Args:
    *relative_path: Relative path, eventually to concatenate.

  Returns:
    path: The root TFDS path.
  """
  return to_write_path(tfds_path(*relative_path))
