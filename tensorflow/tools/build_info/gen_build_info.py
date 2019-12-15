# Lint as: python2, python3
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


def write_build_info(filename, is_config_cuda, is_config_rocm, key_value_list):
  """Writes a Python that describes the build.

  Args:
    filename: filename to write to.
    is_config_cuda: Whether this build is using CUDA.
    is_config_rocm: Whether this build is using ROCm.
    key_value_list: A list of "key=value" strings that will be added to the
      module as additional fields.

  Raises:
    ValueError: If `key_value_list` includes the key "is_cuda_build", which
      would clash with one of the default fields.
  """
  module_docstring = "\"\"\"Generates a Python module containing information "
  module_docstring += "about the build.\"\"\""

  build_config_rocm_bool = "False"
  build_config_cuda_bool = "False"

  if is_config_rocm == "True":
    build_config_rocm_bool = "True"
  elif is_config_cuda == "True":
    build_config_cuda_bool = "True"

  key_value_pair_stmts = []
  if key_value_list:
    for arg in key_value_list:
      key, value = six.ensure_str(arg).split("=")
      if key == "is_cuda_build":
        raise ValueError("The key \"is_cuda_build\" cannot be passed as one of "
                         "the --key_value arguments.")
      if key == "is_rocm_build":
        raise ValueError("The key \"is_rocm_build\" cannot be passed as one of "
                         "the --key_value arguments.")
      key_value_pair_stmts.append("%s = %r" % (key, value))
  key_value_pair_content = "\n".join(key_value_pair_stmts)

  contents = """
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
%s
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

is_rocm_build = %s
is_cuda_build = %s

%s
""" % (module_docstring, build_config_rocm_bool, build_config_cuda_bool,
       key_value_pair_content)
  open(filename, "w").write(contents)


parser = argparse.ArgumentParser(
    description="""Build info injection into the PIP package.""")

parser.add_argument(
    "--is_config_cuda",
    type=str,
    help="'True' for CUDA GPU builds, 'False' otherwise.")

parser.add_argument(
    "--is_config_rocm",
    type=str,
    help="'True' for ROCm GPU builds, 'False' otherwise.")

parser.add_argument("--raw_generate", type=str, help="Generate build_info.py")

parser.add_argument(
    "--key_value", type=str, nargs="*", help="List of key=value pairs.")

args = parser.parse_args()

if (args.raw_generate is not None) and (args.is_config_cuda is not None) and (
    args.is_config_rocm is not None):
  write_build_info(args.raw_generate, args.is_config_cuda, args.is_config_rocm,
                   args.key_value)
else:
  raise RuntimeError(
      "--raw_generate, --is_config_cuda and --is_config_rocm must be used")
