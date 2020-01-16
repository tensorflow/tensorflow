# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Configure build environment for certain Intel platforms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import subprocess

NEHALEM_CPU_INSTRUCTIONS = [
    "MMX", "SSE", "SSE2", "SSE3", "SSSE3", "SSE4.1", "SSE4.2", "POPCNT"
]

SANDYBRIDGE_CPU_INSTRUCTIONS = NEHALEM_CPU_INSTRUCTIONS[:]
SANDYBRIDGE_CPU_INSTRUCTIONS.extend(["AVX", "AES", "PCLMUL"])

HASWELL_CPU_INSTRUCTIONS = SANDYBRIDGE_CPU_INSTRUCTIONS[:]
HASWELL_CPU_INSTRUCTIONS.extend(
    ["FSGSBASE", "RDRND", "FMA", "BMI", "BMI2", "F16C", "MOVBE", "AVX2"])

SKYLAKE_CPU_INSTRUCTIONS = HASWELL_CPU_INSTRUCTIONS[:]
SKYLAKE_CPU_INSTRUCTIONS.extend([
    "PKU", "RDSEED", "ADCX", "PREFETCHW", "CLFLUSHOPT", "XSAVEC", "XSAVES",
    "AVX512F", "CLWB", "AVX512VL", "AVX512BW", "AVX512DQ", "AVX512CD"
])

ICELAKE_CPU_INSTRUCTIONS = SKYLAKE_CPU_INSTRUCTIONS[:]
ICELAKE_CPU_INSTRUCTIONS.extend([
    "AVX512VBMI", "AVX512IFMA", "SHA", "CLWB", "UMIP", "RDPID", "GFNI",
    "AVX512VBMI2", "AVX512VPOPCNTDQ", "AVX512BITALG", "AVX512VNNI",
    "VPCLMULQDQ", "VAES"
])

BASIC_BUILD_OPTS = ["--cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0", "--copt=-O3"]

SECURE_BUILD_OPTS = [
    "--copt=-Wformat", "--copt=-Wformat-security", "--copt=-fstack-protector",
    "--copt=-fPIC", "--copt=-fpic", "--linkopt=-znoexecstack",
    "--linkopt=-zrelro", "--linkopt=-znow", "--linkopt=-fstack-protector"
]


class BuildEnvSetter(object):
  """Prepares the proper environment settings for various Intel platforms."""
  default_platform_ = "haswell"
  PLATFORMS = {
      "nehalem": {
          "min_gcc_major_version": "4",
          "min_gcc_minor_version": "8",
          "flags": NEHALEM_CPU_INSTRUCTIONS
      },
      "sandybridge": {
          "min_gcc_major_version": "4",
          "min_gcc_minor_version": "8",
          "flags": SANDYBRIDGE_CPU_INSTRUCTIONS
      },
      "haswell": {
          "min_gcc_major_version": "4",
          "min_gcc_minor_version": "8",
          "flags": HASWELL_CPU_INSTRUCTIONS
      },
      "skylake": {
          "min_gcc_major_version": "6",
          "min_gcc_minor_version": "0",
          "flags": SKYLAKE_CPU_INSTRUCTIONS
      },
      "icelake": {
          "min_gcc_major_version": "8",
          "min_gcc_minor_version": "0",
          "flags": ICELAKE_CPU_INSTRUCTIONS
      }
  }

  def __init__(self):
    self.args = None
    self.bazel_flags_ = "build "
    self.go()

  def gcc_version_ok(self, min_gcc_major_version, min_gcc_minor_version):
    """Make sure the GCC version installed on the machine is acceptable."""
    # check to see if gcc is present
    gcc_path = ""
    gcc_path_cmd = "command -v gcc"
    try:
      print("gcc_path_cmd = {}".format(gcc_path_cmd))
      gcc_path = subprocess.check_output(gcc_path_cmd, shell=True,
                                         stderr=subprocess.STDOUT).\
      strip()
      print("gcc located here: {}".format(gcc_path))
      if not os.access(gcc_path, os.F_OK | os.X_OK):
        raise ValueError(
            "{} does not exist or is not executable.".format(gcc_path))

      gcc_output = subprocess.check_output(
          [gcc_path, "-dumpfullversion", "-dumpversion"],
          stderr=subprocess.STDOUT).strip()
      # handle python2 vs 3 (bytes vs str type)
      if isinstance(gcc_output, bytes):
        gcc_output = gcc_output.decode("utf-8")
      print("gcc version: {}".format(gcc_output))
      gcc_info = gcc_output.split(".")
      if gcc_info[0] < min_gcc_major_version:
        print("Your MAJOR version of GCC is too old: {}; "
              "it must be at least {}.{}".format(gcc_info[0],
                                                 min_gcc_major_version,
                                                 min_gcc_minor_version))
        return False

      elif gcc_info[0] == min_gcc_major_version:
        if gcc_info[1] < min_gcc_minor_version:
          print("Your MINOR version of GCC is too old: {}; "
                "it must be at least {}.{}".format(gcc_info[1],
                                                   min_gcc_major_version,
                                                   min_gcc_minor_version))
          return False
        return True
      else:
        self._debug("gcc version OK: {}.{}".format(gcc_info[0], gcc_info[1]))
        return True
    except subprocess.CalledProcessException as e:
      print("Problem getting gcc info: {}".format(e))
      return False

  def parse_args(self):
    """Set up argument parser, and parse CLI args."""
    arg_parser = argparse.ArgumentParser(
        description="Parse the arguments for the "
        "TensorFlow build environment "
        " setter")
    arg_parser.add_argument(
        "--disable-mkl",
        dest="disable_mkl",
        help="Turn off MKL. By default the compiler flag "
        "--config=mkl is enabled.",
        action="store_true")
    arg_parser.add_argument(
        "--disable-v2",
        dest="disable_v2",
        help="Don't build TensorFlow v2. By default the "
        " compiler flag --config=v2 is enabled.",
        action="store_true")
    arg_parser.add_argument(
        "--enable-bfloat16",
        dest="enable_bfloat16",
        help="Enable bfloat16 build. By default it is "
        " disabled if no parameter is passed.",
        action="store_true")
    arg_parser.add_argument(
        "-s",
        "--secure-build",
        dest="secure_build",
        help="Enable secure build flags.",
        action="store_true")
    arg_parser.add_argument(
        "-p",
        "--platform",
        choices=self.PLATFORMS.keys(),
        help="The target platform.",
        dest="target_platform",
        default=self.default_platform_)
    arg_parser.add_argument(
        "-f",
        "--bazelrc-file",
        dest="bazelrc_file",
        help="The full path to the bazelrc file into which "
        "the build command will be written. The path "
        "will be relative to the container "
        " environment.",
        required=True)

    self.args = arg_parser.parse_args()

  def validate_args(self):
    if os.path.exists(self.args.bazelrc_file):
      if os.path.isfile(self.args.bazelrc_file):
        self._debug("The file {} exists and will be deleted.".format(
            self.args.bazelrc_file))
      elif os.path.isdir(self.args.bazelrc_file):
        raise ValueError("{} is not a valid file name".format(
            self.args.bazelrc_file))
    return True

  def set_build_args(self):
    """Generate Bazel build flags."""
    for flag in BASIC_BUILD_OPTS:
      self.bazel_flags_ += "{} ".format(flag)
    if self.args.secure_build:
      for flag in SECURE_BUILD_OPTS:
        self.bazel_flags_ += "{} ".format(flag)
    for flag in self.PLATFORMS.get(self.args.target_platform)["flags"]:
      self.bazel_flags_ += "--copt=-m{} ".format(flag.lower())
    if not self.args.disable_mkl:
      self.bazel_flags_ += "--config=mkl "
    if not self.args.disable_v2:
      self.bazel_flags_ += "--config=v2 "
    if self.args.enable_bfloat16:
      self.bazel_flags_ += "--copt=-DENABLE_INTEL_MKL_BFLOAT16 "

  def write_build_args(self):
    self._debug("Writing build flags: {}".format(self.bazel_flags_))
    with open(self.args.bazelrc_file, "w") as f:
      f.write(self.bazel_flags_)

  def _debug(self, msg):
    print(msg)

  def go(self):
    self.parse_args()
    target_platform = self.PLATFORMS.get(self.args.target_platform)
    if self.validate_args() and \
      self.gcc_version_ok(target_platform["min_gcc_major_version"],
                          target_platform["min_gcc_minor_version"]):
      self.set_build_args()
      self.write_build_args()
    else:
      print("Error.")


env_setter = BuildEnvSetter()
