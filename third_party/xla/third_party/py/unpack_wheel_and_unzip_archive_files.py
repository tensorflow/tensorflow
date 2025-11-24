# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

import argparse
import os
import pathlib
import shutil
import zipfile

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument(
    "--wheel",
    help="Wheel file to be unpacked",
    required=True,
)
parser.add_argument(
    "--zip_files",
    help="Additional zip files to extract in the same folder as unpacked wheel",
    action="append",
)
parser.add_argument(
    "--wheel_files",
    help="Additional wheel files to copy in the same folder as unpacked wheel",
    action="append",
)
parser.add_argument(
    "--output_dir",
    default=None,
    required=True,
    help="Path to which the output wheel should be written. Required.",
)
args = parser.parse_args()


def copy_file(
    src_file: str,
    dst_dir: str,
) -> None:
  """Copy a file to the destination directory.

  Args:
    src_file: file to be copied
    dst_dir: destination directory
  """

  src_dirname = os.path.dirname(src_file)
  site_packages = "site-packages"
  site_packages_ind = (
      src_dirname.rfind(site_packages) + len(site_packages) + 1
  )
  dest_dir_path = os.path.join(dst_dir, src_dirname[site_packages_ind:])
  os.makedirs(dest_dir_path, exist_ok=True)
  shutil.copy(src_file, dest_dir_path)


def prepare_outputs(
    wheel: str,
    zip_files: list[str],
    wheel_files: list[str],
    output_dir: str,
) -> None:
  """Create the output directory and copy/extract the sources to it.

  Args:
    wheel: path to the wheel file to extract
    wheel_files: paths to additional wheel files to copy to the output directory
    zip_files: paths to additional zip files to extract to the output directory
    output_dir: target directory where files are copied to.
  """
  with zipfile.ZipFile(wheel, "r") as zip_ref:
    zip_ref.extractall(output_dir)
  if zip_files:
    for archive in zip_files:
      with zipfile.ZipFile(archive, "r") as zip_ref:
        zip_ref.extractall(output_dir)

  if wheel_files:
    for f in wheel_files:
      copy_file(f, output_dir)


os.makedirs(args.output_dir, exist_ok=True)
prepare_outputs(
    args.wheel,
    args.zip_files,
    args.wheel_files,
    pathlib.Path(args.output_dir),
)
