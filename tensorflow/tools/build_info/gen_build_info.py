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

import six

# cuda.cuda is only valid in OSS
try:
  from cuda.cuda import cuda_config  # pylint: disable=g-import-not-at-top
except ImportError:
  cuda_config = None

# tensorrt.tensorrt is only valid in OSS
try:
  from tensorrt.tensorrt import tensorrt_config  # pylint: disable=g-import-not-at-top
except ImportError:
  tensorrt_config = None


def write_build_info(filename, key_value_list):
  """Writes a Python that describes the build.

  Args:
    filename: filename to write to.
    key_value_list: A list of "key=value" strings that will be added to the
      module's "build_info" dictionary as additional entries.
  """

  build_info = {}

  if cuda_config:
    build_info.update(cuda_config.config)

  if tensorrt_config:
    build_info.update(tensorrt_config.config)

  for arg in key_value_list:
    key, value = six.ensure_str(arg).split("=")
    if value.lower() == "true":
      build_info[key] = True
    elif value.lower() == "false":
      build_info[key] = False
    else:
      build_info[key] = value.format(**build_info)

  # Sort the build info to ensure deterministic output.
  sorted_build_info_pairs = sorted(build_info.items())

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

import collections

build_info = collections.OrderedDict(%s)
""" % sorted_build_info_pairs
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
