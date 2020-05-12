# Lint as: python3
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
"""Generates a Python module containing information about the build."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import platform
import sys

import six

# CUDA library gathering is only valid in OSS
try:
  from third_party.gpus import find_cuda_config  # pylint: disable=g-import-not-at-top
except ImportError:
  find_cuda_config = None


def write_build_info(filename, key_value_list):
  """Writes a Python that describes the build.

  Args:
    filename: filename to write to.
    key_value_list: A list of "key=value" strings that will be added to the
      module's "build_info" dictionary as additional entries.
  """

  build_info = {}
  for arg in key_value_list:
    key, value = six.ensure_str(arg).split("=")
    if value.lower() == "true":
      build_info[key] = True
    elif value.lower() == "false":
      build_info[key] = False
    else:
      build_info[key] = value

  # Generate cuda_build_info, a dict describing the CUDA component versions
  # used to build TensorFlow.
  if find_cuda_config and build_info.get("is_cuda_build", False):
    libs = ["_", "cuda", "cudnn"]
    if platform.system() == "Linux":
      if os.environ.get("TF_NEED_TENSORRT", "0") == "1":
        libs.append("tensorrt")
      if "TF_NCCL_VERSION" in os.environ:
        libs.append("nccl")
    # find_cuda_config accepts libraries to inspect as argv from the command
    # line. We can work around this restriction by setting argv manually
    # before calling find_cuda_config.
    backup_argv = sys.argv
    sys.argv = libs
    cuda = find_cuda_config.find_cuda_config()

    build_info["cuda_version"] = cuda["cuda_version"]
    build_info["cudnn_version"] = cuda["cudnn_version"]
    build_info["tensorrt_version"] = cuda.get("tensorrt_version", None)
    build_info["nccl_version"] = cuda.get("nccl_version", None)
    sys.argv = backup_argv

  contents = """
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
\"\"\"Auto-generated module providing information about the build.\"\"\"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

build_info = {build_info}
""".format(build_info=build_info)
  open(filename, "w").write(contents)


parser = argparse.ArgumentParser(
    description="""Build info injection into the PIP package.""")

parser.add_argument("--raw_generate", type=str, help="Generate build_info.py")

parser.add_argument(
    "--key_value", type=str, nargs="*", help="List of key=value pairs.")

args = parser.parse_args()

if args.raw_generate:
  write_build_info(args.raw_generate, args.key_value)
else:
  raise RuntimeError("--raw_generate must be used.")
