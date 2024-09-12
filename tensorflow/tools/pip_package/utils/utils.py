# Copyright 2023 The Tensorflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for pip package builder."""
import os
import shutil
import sys


def is_windows() -> bool:
  return sys.platform.startswith("win32")


def is_macos() -> bool:
  return sys.platform.startswith("darwin")


def create_init_files(dst_dir: str) -> None:
  """Create __init__.py files."""
  for root, _, files in os.walk(dst_dir):
    if any(file.endswith(".py") or file.endswith(".so") for file in files):
      curr_dir = root
      while curr_dir != dst_dir:
        init_path = os.path.join(curr_dir, "__init__.py")
        if not os.path.exists(init_path):
          open(init_path, "w").close()
        curr_dir = os.path.dirname(curr_dir)


def replace_inplace(directory, search, to_replace) -> None:
  """Traverse the directory and replace search phrase in each file."""
  for root, _, files in os.walk(directory):
    for file_name in files:
      if file_name.endswith(".py"):
        file_path = os.path.join(root, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
          filedata = file.read()
        if search in filedata:
          filedata = filedata.replace(search, to_replace)
          with open(file_path, "w") as file:
            file.write(filedata)


def copy_file(
    src_file: str,
    dst_dir: str,
    strip: str = None,
) -> None:
  """Copy a file to the destination directory.

  Args:
    src_file: file to be copied
    dst_dir: destination directory
    strip: prefix to strip before copying to destination
  """

  # strip `bazel-out/.../bin/` for generated files.
  if src_file.startswith("bazel-out"):
    dest = src_file[src_file.index("bin") + 4:]
  else:
    dest = src_file

  if strip:
    dest = dest.removeprefix(strip)
  dest_dir_path = os.path.join(dst_dir, os.path.dirname(dest))
  os.makedirs(dest_dir_path, exist_ok=True)
  shutil.copy(src_file, dest_dir_path)
  os.chmod(os.path.join(dst_dir, dest), 0o644)
