#!/usr/bin/python
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#
# Automatically copy TensorFlow binaries
#
# Usage:
#           ./tensorflow/tools/ci_build/copy_binary.py --filename
# tf_nightly/tf_nightly_gpu-1.4.0.dev20170914-cp35-cp35m-manylinux1_x86_64.whl
# --new_py_ver 36
#
"""Copy binaries of TensorFlow for different python versions."""

# pylint: disable=superfluous-parens

import argparse
import os
import re
import shutil
import tempfile
import zipfile

TF_NIGHTLY_REGEX = (r"(.+)tf_nightly(|_gpu)-(\d\.[\d]{1,2}"
                    "\.\d.dev[\d]{0,8})-(.+)\.whl")
BINARY_STRING_TEMPLATE = "%s-%s-%s.whl"


def check_existence(filename):
  """Check the existence of file or dir."""
  if not os.path.exists(filename):
    raise RuntimeError("%s not found." % filename)


def copy_binary(directory, origin_tag, new_tag, version, gpu=False):
  """Rename and copy binaries for different python versions.

  Arguments:
    directory: string of directory
    origin_tag: str of the old python version tag
    new_tag: str of the new tag
    version: the version of the package
    gpu: bool if its a gpu build or not

  """
  print("Rename and copy binaries with %s to %s." % (origin_tag, new_tag))
  if gpu:
    package = "tf_nightly_gpu"
  else:
    package = "tf_nightly"
  origin_binary = BINARY_STRING_TEMPLATE % (package, version, origin_tag)
  new_binary = BINARY_STRING_TEMPLATE % (package, version, new_tag)
  zip_ref = zipfile.ZipFile(os.path.join(directory, origin_binary), "r")

  try:
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)

    zip_ref.extractall()
    zip_ref.close()
    old_py_ver = re.search(r"(cp\d\d-cp\d\d)", origin_tag).group(1)
    new_py_ver = re.search(r"(cp\d\d-cp\d\d)", new_tag).group(1)

    wheel_file = os.path.join(
        tmpdir, "%s-%s.dist-info" % (package, version), "WHEEL")
    with open(wheel_file, "r") as f:
      content = f.read()
    with open(wheel_file, "w") as f:
      f.write(content.replace(old_py_ver, new_py_ver))

    zout = zipfile.ZipFile(directory + new_binary, "w", zipfile.ZIP_DEFLATED)
    zip_these_files = [
        "%s-%s.dist-info" % (package, version),
        "%s-%s.data" % (package, version),
    ]
    for dirname in zip_these_files:
      for root, _, files in os.walk(dirname):
        for filename in files:
          zout.write(os.path.join(root, filename))
    zout.close()
  finally:
    shutil.rmtree(tmpdir)


def main():
  """This script copies binaries.

  Requirements:
    filename: The path to the whl file
    AND
    new_py_ver: Create a nightly tag with current date

  Raises:
    RuntimeError: If the whl file was not found
  """

  parser = argparse.ArgumentParser(description="Cherry picking automation.")

  # Arg information
  parser.add_argument(
      "--filename", help="path to whl file we are copying", required=True)
  parser.add_argument(
      "--new_py_ver", help="two digit py version eg. 27 or 33", required=True)

  args = parser.parse_args()

  # Argument checking
  args.filename = os.path.abspath(args.filename)
  check_existence(args.filename)
  regex_groups = re.search(TF_NIGHTLY_REGEX, args.filename)
  directory = regex_groups.group(1)
  gpu = regex_groups.group(2)
  version = regex_groups.group(3)
  origin_tag = regex_groups.group(4)
  old_py_ver = re.search(r"(cp\d\d)", origin_tag).group(1)

  # Create new tags
  new_tag = origin_tag.replace(old_py_ver, "cp" + args.new_py_ver)

  # Copy the binary with the info we have
  copy_binary(directory, origin_tag, new_tag, version, gpu)


if __name__ == "__main__":
  main()
