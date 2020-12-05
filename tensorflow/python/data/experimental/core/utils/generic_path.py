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

"""Pathlib-like generic abstraction."""

import os
from typing import List, Type, Union, TypeVar

from tensorflow.data.experimental.core.utils import gpath
from tensorflow.data.experimental.core.utils import type_utils

PathLike = type_utils.PathLike
ReadOnlyPath = type_utils.ReadOnlyPath
ReadWritePath = type_utils.ReadWritePath

PathLikeCls = Union[Type[ReadOnlyPath], Type[ReadWritePath]]
T = TypeVar('T')


# Could eventually expose some `tfds.core.register_path_cls` API to unlock
# additional file system supports (e.g. `s3path.S3Path('s3://bucket/data')`)
_PATHLIKE_CLS: List[PathLikeCls] = [
    gpath.PosixGPath,
    gpath.WindowsGPath,
]


def register_pathlike_cls(path_cls: T) -> T:
  """Register the class to be forwarded as-is in `as_path`."""
  _PATHLIKE_CLS.append(path_cls)
  return path_cls


def as_path(path: PathLike) -> ReadWritePath:
  """Create a generic `pathlib.Path`-like abstraction.

  Depending on the input (e.g. `gs://` url, `ResourcePath`,...), the
  system (Windows, Linux,...), the function will create the right pathlib-like
  abstraction.

  Args:
    path: Pathlike object.

  Returns:
    path: The `pathlib.Path`-like abstraction.
  """
  is_windows = os.name == 'nt'
  if isinstance(path, str):
    if is_windows and not path.startswith(gpath.URI_PREFIXES):
      return gpath.WindowsGPath(path)
    else:
      return gpath.PosixGPath(path)  # On linux, or for `gs://`, uses `GPath`
  elif isinstance(path, tuple(_PATHLIKE_CLS)):
    return path  # Forward resource path, gpath,... as-is  # pytype: disable=bad-return-type
  elif isinstance(path, os.PathLike):  # Other `os.fspath` compatible objects
    path_cls = gpath.WindowsGPath if is_windows else gpath.PosixGPath
    return path_cls(path)
  else:
    raise TypeError(f'Invalid path type: {path!r}')
