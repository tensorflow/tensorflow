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

"""Defines convenience constants/functions for converting various units."""

# The constants below are used for conveniently defining memory quantities.
# pylint: disable=invalid-name
KiB = 2**10
MiB = 2**20
GiB = 2**30
TiB = 2**40
PiB = 2**50

_NAME_LIST = [("PiB", PiB), ("TiB", TiB), ("GiB", GiB), ("MiB", MiB),
              ("KiB", KiB)]


def size_str(size_in_bytes):
  """Returns a human readable size string.

  If size_in_bytes is None, then returns "Unknown size".

  For example `size_str(1.5 * tfds.units.GiB) == "1.50 GiB"`.

  Args:
    size_in_bytes: `int` or `None`, the size, in bytes, that we want to
      format as a human-readable size string.
  """
  if not size_in_bytes:
    return "Unknown size"

  size_in_bytes = float(size_in_bytes)
  for (name, size_bytes) in _NAME_LIST:
    value = size_in_bytes / size_bytes
    if value >= 1.0:
      return "{:.2f} {}".format(value, name)
  return "{} {}".format(int(size_in_bytes), "bytes")


# pylint: enable=invalid-name
