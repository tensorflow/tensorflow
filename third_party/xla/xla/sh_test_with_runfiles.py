# Copyright 2025 The OpenXLA Authors.
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
"""Lit runner configuration."""

import os
import pathlib

import lit.formats


class ShTestWithRunfiles(lit.formats.ShTest):
  """Used to symlink bazels runfiles subdirs into the lit tmp test dir."""

  def execute(self, test, lit_config):
    runfiles = os.environ.get("RUNFILES_DIR")
    if not runfiles:
      return super().execute(test, lit_config)

    runfiles_dir = pathlib.Path(runfiles) / "xla"
    if not runfiles_dir.is_dir():
      return super().execute(test, lit_config)

    dst = pathlib.Path(test.getExecPath()).parent
    dst.mkdir(parents=True, exist_ok=True)
    runfiles = []
    for item in runfiles_dir.iterdir():
      target = dst / item.name
      try:
        target.symlink_to(item, target_is_directory=item.is_dir())
        runfiles.append(target)
      except FileExistsError:
        pass

    result = super().execute(test, lit_config)
    for target in runfiles:
      target.unlink()
    return result
