# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Exposes utility classes for TF-TRT."""

from packaging import version


class TrtVersion(version.Version):
  def __init__(self, version):
    if isinstance(version, tuple):
      if len(version) != 3:
        raise ValueError(f"A tuple of size 3 was expected, received: {version}")
      version = ".".join([str(s) for s in version])

    if not isinstance(version, str):
      raise ValueError(f"Expected tuple of size 3 or str, received: {version}")

    super().__init__(version)
