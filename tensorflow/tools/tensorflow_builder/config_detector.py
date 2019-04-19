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
#
# ==============================================================================
"""Auto-detects machine configurations and outputs the results to shell or file.

Supports linux only currently.

Usage:
  python config_detector.py [--save_output] [--filename] [--debug]

Example command:
  python config_detector.py --save_output=True --filename=configs.json
  --debug=False

Flag option(s):
  save_output  (True | False)       Save output to a file.
                                    (Default: True)
  filename     <file_name>.json     Filename(.json) for storing configs.
                                    (Default: `configs.json`)
  debug        (True | False)       View debug and stderr messages.
                                    (Default: False)

The following machine configuration will be detected:
  Platform              Operating system (linux | macos | windows)
  CPU                   CPU type (e.g. `GenuineIntel`)
  CPU architecture      Processor type (32-bit | 64-bit)
  CPU ISA               CPU instruction set (e.g. `sse4`, `sse4_1`, `avx`)
  Distribution          Operating system distribution (e.g. Ubuntu)
  Distribution version  Operating system distribution version (e.g. 14.04)
  GPU                   GPU type (e.g. `Tesla K80`)
  GPU count             Number of GPU's available
  CUDA version          CUDA version by default (e.g. `10.1`)
  CUDA version all      CUDA version(s) all available
  cuDNN version         cuDNN version (e.g. `7.5.0`)
  GCC version           GCC version (e.g. `7.3.0`)
  GLIBC version         GLIBC version (e.g. `2.24`)
  libstdc++ version     libstdc++ version (e.g. `3.4.25`)

Output:
  Shell output (print)
      A table containing status and info on all configurations will be
      printed out to shell.

  Configuration file (.json):
      Depending on `--save_output` option, this script outputs a .json file
      (in the same directory) containing all user machine configurations
      that were detected.
"""
# pylint: disable=broad-except
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import re
import subprocess
import sys
from absl import app
from absl import flags

from tensorflow.tools.tensorflow_builder.data import cuda_compute_capability

FLAGS = flags.FLAGS
# Define all flags
flags.DEFINE_boolean("save_output", True, "Save output to a file. [True/False]")
flags.DEFINE_string("filename", "configs.json", "Output filename.")
flags.DEFINE_boolean("debug", False, "View debug messages. [True/False]")

# For linux: commands for retrieving user machine configs.
cmds_linux = {
    "cpu_type": (
        "cat /proc/cpuinfo 2>&1 | grep 'vendor' | uniq"),
    "cpu_arch": (
        "uname -m"),
    "distrib": (
        "cat /etc/*-release | grep DISTRIB_ID* | sed 's/^.*=//'"),
    "distrib_ver": (
        "cat /etc/*-release | grep DISTRIB_RELEASE* | sed 's/^.*=//'"),
    "gpu_type": (
        "sudo lshw -C display | grep product:* | sed 's/^.*: //'"),
    "gpu_type_no_sudo":
        r"lspci | grep 'VGA compatible\|3D controller' | cut -d' ' -f 1 | "
        r"xargs -i lspci -v -s {} | head -n 2 | tail -1 | "
        r"awk '{print $(NF-2), $(NF-1), $NF}'",
    "gpu_count": (
        "sudo lshw -C display | grep *-display:* | wc -l"),
    "gpu_count_no_sudo": (
        r"lspci | grep 'VGA compatible\|3D controller' | wc -l"),
    "cuda_ver_all": (
        "ls -d /usr/local/cuda* 2> /dev/null"),
    "cuda_ver_dflt": (
        ["nvcc --version 2> /dev/null",
         "cat /usr/local/cuda/version.txt 2> /dev/null | awk '{print $NF}'"]),
    "cudnn_ver": (
        ["whereis cudnn.h",
         "cat `awk '{print $2}'` | grep CUDNN_MAJOR -A 2 | echo "
         "`awk '{print $NF}'` | awk '{print $1, $2, $3}' | sed 's/ /./g'"]),
    "gcc_ver": (
        "gcc --version | awk '{print $NF}' | head -n 1"),
    "glibc_ver": (
        "ldd --version | tail -n+1 | head -n 1 | awk '{print $NF}'"),
    "libstdcpp_ver":
        "strings $(/sbin/ldconfig -p | grep libstdc++ | head -n 1 | "
        "awk '{print $NF}') | grep LIBCXX | tail -2 | head -n 1",
    "cpu_isa": (
        "cat /proc/cpuinfo | grep flags | head -n 1"),
}

cmds_all = {
    "linux": cmds_linux,
}

# Global variable(s).
PLATFORM = None
GPU_TYPE = None
PATH_TO_DIR = "tensorflow/tools/tensorflow_builder"


def run_shell_cmd(args):
  """Executes shell commands and returns output.

  Args:
    args: String of shell commands to run.

  Returns:
    Tuple output (stdoutdata, stderrdata) from running the shell commands.
  """
  proc = subprocess.Popen(
      args,
      shell=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT
  )
  return proc.communicate()


def get_platform():
  """Retrieves platform information.

  Currently the script only support linux. If other platoforms such as Windows
  or MacOS is detected, it throws an error and terminates.

  Returns:
    String that is platform type.
      e.g. 'linux'
  """
  global PLATFORM
  cmd = "uname"
  out, err = run_shell_cmd(cmd)
  platform_detected = out.strip().lower()
  if platform_detected != "linux":
    if err and FLAGS.debug:
      print("Error in detecting platform:\n %s" % str(err))

    print("Error: Detected unsupported operating system.\nStopping...")
    sys.exit(1)
  else:
    PLATFORM = platform_detected

  return PLATFORM


def get_cpu_type():
  """Retrieves CPU (type) information.

  Returns:
    String that is name of the CPU.
      e.g. 'GenuineIntel'
  """
  key = "cpu_type"
  out, err = run_shell_cmd(cmds_all[PLATFORM][key])
  cpu_detected = out.split(":")[1].strip()
  if err and FLAGS.debug:
    print("Error in detecting CPU type:\n %s" % str(err))

  return cpu_detected


def get_cpu_arch():
  """Retrieves processor architecture type (32-bit or 64-bit).

  Returns:
    String that is CPU architecture.
      e.g. 'x86_64'
  """
  key = "cpu_arch"
  out, err = run_shell_cmd(cmds_all[PLATFORM][key])
  if err and FLAGS.debug:
    print("Error in detecting CPU arch:\n %s" % str(err))

  return out.strip("\n")


def get_distrib():
  """Retrieves distribution name of the operating system.

  Returns:
    String that is the name of distribution.
      e.g. 'Ubuntu'
  """
  key = "distrib"
  out, err = run_shell_cmd(cmds_all[PLATFORM][key])
  if err and FLAGS.debug:
    print("Error in detecting distribution:\n %s" % str(err))

  return out.strip("\n")


def get_distrib_version():
  """Retrieves distribution version of the operating system.

  Returns:
    String that is the distribution version.
      e.g. '14.04'
  """
  key = "distrib_ver"
  out, err = run_shell_cmd(cmds_all[PLATFORM][key])
  if err and FLAGS.debug:
    print(
        "Error in detecting distribution version:\n %s" % str(err)
    )

  return out.strip("\n")


def get_gpu_type():
  """Retrieves GPU type.

  Returns:
    String that is the name of the detected NVIDIA GPU.
      e.g. 'Tesla K80'

    'unknown' will be returned if detected GPU type is an unknown name.
      Unknown name refers to any GPU name that is not specified in this page:
      https://developer.nvidia.com/cuda-gpus
  """
  global GPU_TYPE
  key = "gpu_type_no_sudo"
  gpu_dict = cuda_compute_capability.retrieve_from_golden()
  out, err = run_shell_cmd(cmds_all[PLATFORM][key])
  ret_val = out.split(" ")
  gpu_id = ret_val[0]
  if err and FLAGS.debug:
    print("Error in detecting GPU type:\n %s" % str(err))

  if not isinstance(ret_val, list):
    GPU_TYPE = "unknown"
    return gpu_id, GPU_TYPE
  else:
    if "[" or "]" in ret_val[1]:
      gpu_release = ret_val[1].replace("[", "") + " "
      gpu_release += ret_val[2].replace("]", "").strip("\n")
    else:
      gpu_release = ret_val[1].replace("\n", " ")

    if gpu_release not in gpu_dict:
      GPU_TYPE = "unknown"
    else:
      GPU_TYPE = gpu_release

    return gpu_id, GPU_TYPE


def get_gpu_count():
  """Retrieves total number of GPU's available in the system.

  Returns:
    Integer that is the total # of GPU's found.
  """
  key = "gpu_count_no_sudo"
  out, err = run_shell_cmd(cmds_all[PLATFORM][key])
  if err and FLAGS.debug:
    print("Error in detecting GPU count:\n %s" % str(err))

  return out.strip("\n")


def get_cuda_version_all():
  """Retrieves all additional CUDA versions available (other than default).

  For retrieving default CUDA version, use `get_cuda_version` function.

  stderr is silenced by default. Setting FLAGS.debug mode will not enable it.
  Remove `2> /dev/null` command from `cmds_linux['cuda_ver_dflt']` to enable
  stderr.

  Returns:
    List of all CUDA versions found (except default version).
      e.g. ['10.1', '10.2']
  """
  key = "cuda_ver_all"
  out, err = run_shell_cmd(cmds_all[PLATFORM.lower()][key])
  ret_val = out.split("\n")
  filtered = []
  for item in ret_val:
    if item not in ["\n", ""]:
      filtered.append(item)

  all_vers = []
  for item in filtered:
    ver_re = re.search(r".*/cuda(\-[\d]+\.[\d]+)?", item)
    if ver_re.group(1):
      all_vers.append(ver_re.group(1).strip("-"))

  if err and FLAGS.debug:
    print("Error in detecting CUDA version:\n %s" % str(err))

  return all_vers


def get_cuda_version_default():
  """Retrieves default CUDA version.

  Default verion is the version found in `/usr/local/cuda/` installation.

  stderr is silenced by default. Setting FLAGS.debug mode will not enable it.
  Remove `2> /dev/null` command from `cmds_linux['cuda_ver_dflt']` to enable
  stderr.

  It iterates through two types of version retrieval method:
    1) Using `nvcc`: If `nvcc` is not available, then it uses next method.
    2) Read version file (`version.txt`) found in CUDA install directory.

  Returns:
    String that is the default CUDA version.
      e.g. '10.1'
  """
  key = "cuda_ver_dflt"
  out = ""
  cmd_list = cmds_all[PLATFORM.lower()][key]
  for i, cmd in enumerate(cmd_list):
    try:
      out, err = run_shell_cmd(cmd)
      if not out:
        raise Exception(err)

    except Exception as e:
      if FLAGS.debug:
        print("\nWarning: Encountered issue while retrieving default CUDA "
              "version. (%s) Trying a different method...\n" % e)

      if i == len(cmd_list) - 1:
        if FLAGS.debug:
          print("Error: Cannot retrieve CUDA default version.\nStopping...")

      else:
        pass

  return out.strip("\n")


def get_cuda_compute_capability(source_from_url=False):
  """Retrieves CUDA compute capability based on the detected GPU type.

  This function uses the `cuda_compute_capability` module to retrieve the
  corresponding CUDA compute capability for the given GPU type.

  Args:
    source_from_url: Boolean deciding whether to source compute capability
                     from NVIDIA website or from a local golden file.

  Returns:
    List of all supported CUDA compute capabilities for the given GPU type.
      e.g. ['3.5', '3.7']
  """
  if not GPU_TYPE:
    if FLAGS.debug:
      print("Warning: GPU_TYPE is empty. "
            "Make sure to call `get_gpu_type()` first.")

  elif GPU_TYPE == "unknown":
    if FLAGS.debug:
      print("Warning: Unknown GPU is detected. "
            "Skipping CUDA compute capability retrieval.")

  else:
    if source_from_url:
      cuda_compute_capa = cuda_compute_capability.retrieve_from_web()
    else:
      cuda_compute_capa = cuda_compute_capability.retrieve_from_golden()

    return cuda_compute_capa[GPU_TYPE]
  return


def get_cudnn_version():
  """Retrieves the version of cuDNN library detected.

  Returns:
    String that is the version of cuDNN library detected.
      e.g. '7.5.0'
  """
  key = "cudnn_ver"
  cmds = cmds_all[PLATFORM.lower()][key]
  out, err = run_shell_cmd(cmds[0])
  if err and FLAGS.debug:
    print("Error in finding `cudnn.h`:\n %s" % str(err))

  if len(out.split(" ")) > 1:
    cmd = cmds[0] + " | " + cmds[1]
    out_re, err_re = run_shell_cmd(cmd)
    if err_re and FLAGS.debug:
      print("Error in detecting cuDNN version:\n %s" % str(err_re))

    return out_re.strip("\n")
  else:
    return


def get_gcc_version():
  """Retrieves version of GCC detected.

  Returns:
    String that is the version of GCC.
      e.g. '7.3.0'
  """
  key = "gcc_ver"
  out, err = run_shell_cmd(cmds_all[PLATFORM.lower()][key])
  if err and FLAGS.debug:
    print("Error in detecting GCC version:\n %s" % str(err))

  return out.strip("\n")


def get_glibc_version():
  """Retrieves version of GLIBC detected.

  Returns:
    String that is the version of GLIBC.
      e.g. '2.24'
  """
  key = "glibc_ver"
  out, err = run_shell_cmd(cmds_all[PLATFORM.lower()][key])
  if err and FLAGS.debug:
    print("Error in detecting GCC version:\n %s" % str(err))

  return out.strip("\n")


def get_libstdcpp_version():
  """Retrieves version of libstdc++ detected.

  Returns:
    String that is the version of libstdc++.
      e.g. '3.4.25'
  """
  key = "libstdcpp_ver"
  out, err = run_shell_cmd(cmds_all[PLATFORM.lower()][key])
  if err and FLAGS.debug:
    print("Error in detecting libstdc++ version:\n %s" % str(err))

  ver = out.split("_")[-1].replace("\n", "")
  return ver


def get_cpu_isa_version():
  """Retrieves all Instruction Set Architecture(ISA) available.

  Required ISA(s): 'avx', 'avx2', 'avx512f', 'sse4', 'sse4_1'

  Returns:
    Tuple
      (list of available ISA, list of missing ISA)
  """
  key = "cpu_isa"
  out, err = run_shell_cmd(cmds_all[PLATFORM.lower()][key])
  if err and FLAGS.debug:
    print("Error in detecting supported ISA:\n %s" % str(err))

  ret_val = out
  required_isa = ["avx", "avx2", "avx512f", "sse4", "sse4_1"]
  found = []
  missing = []
  for isa in required_isa:
    for sys_isa in ret_val.split(" "):
      if isa == sys_isa:
        if isa not in found:
          found.append(isa)

  missing = list(set(required_isa) - set(found))
  return found, missing


def get_python_version():
  """Retrieves default Python version.

  Returns:
    String that is the version of default Python.
      e.g. '2.7.4'
  """
  ver = str(sys.version_info)
  mmm = re.search(r".*major=([\d]), minor=([\d]), micro=([\d]+),.*", ver)
  return mmm.group(1) + "." + mmm.group(2) + "." + mmm.group(3)


def get_all_configs():
  """Runs all functions for detecting user machine configurations.

  Returns:
    Tuple
      (List of all configurations found,
       List of all missing configurations,
       List of all configurations found with warnings,
       Dict of all configurations)
  """
  all_functions = collections.OrderedDict(
      [("Platform", get_platform()),
       ("CPU", get_cpu_type()),
       ("CPU arch", get_cpu_arch()),
       ("Distribution", get_distrib()),
       ("Distribution version", get_distrib_version()),
       ("GPU", get_gpu_type()[1]),
       ("GPU count", get_gpu_count()),
       ("CUDA version (default)", get_cuda_version_default()),
       ("CUDA versions (all)", get_cuda_version_all()),
       ("CUDA compute capability",
        get_cuda_compute_capability(get_gpu_type()[1])),
       ("cuDNN version", get_cudnn_version()),
       ("GCC version", get_gcc_version()),
       ("Python version (default)", get_python_version()),
       ("GNU C Lib (glibc) version", get_glibc_version()),
       ("libstdc++ version", get_libstdcpp_version()),
       ("CPU ISA (min requirement)", get_cpu_isa_version())]
  )
  configs_found = []
  json_data = {}
  missing = []
  warning = []
  for config, call_func in all_functions.iteritems():
    ret_val = call_func
    if not ret_val:
      configs_found.append([config, "\033[91m\033[1mMissing\033[0m"])
      missing.append([config])
      json_data[config] = ""
    elif ret_val == "unknown":
      configs_found.append([config, "\033[93m\033[1mUnknown type\033[0m"])
      warning.append([config, ret_val])
      json_data[config] = "unknown"

    else:
      if "ISA" in config:
        if not ret_val[1]:
          # Not missing any required ISA
          configs_found.append([config, ret_val[0]])
          json_data[config] = ret_val[0]
        else:
          configs_found.append(
              [config,
               "\033[91m\033[1mMissing " + str(ret_val[1])[1:-1] + "\033[0m"]
          )
          missing.append(
              [config,
               "\n\t=> Found %s but missing %s"
               % (str(ret_val[0]), str(ret_val[1]))]
          )
          json_data[config] = ret_val[0]

      else:
        configs_found.append([config, ret_val])
        json_data[config] = ret_val

  return (configs_found, missing, warning, json_data)


def print_all_configs(configs, missing, warning):
  """Prints the status and info on all configurations in a table format.

  Args:
    configs: List of all configurations found.
    missing: List of all configurations that are missing.
    warning: List of all configurations found with warnings.
  """
  print_text = ""
  llen = 65  # line length
  for i, row in enumerate(configs):
    if i != 0:
      print_text += "-"*llen + "\n"

    if isinstance(row[1], list):
      val = ", ".join(row[1])
    else:
      val = row[1]

    print_text += " {: <28}".format(row[0]) + "    {: <25}".format(val) + "\n"

  print_text += "="*llen
  print("\n\n {: ^32}    {: ^25}".format("Configuration(s)",
                                         "Detected value(s)"))
  print("="*llen)
  print(print_text)

  if missing:
    print("\n * ERROR: The following configurations are missing:")
    for m in missing:
      print("   ", *m)

  if warning:
    print("\n * WARNING: The following configurations could cause issues:")
    for w in warning:
      print("   ", *w)

  if not missing and not warning:
    print("\n * INFO: Successfully found all configurations.")

  print("\n")


def save_to_file(json_data, filename):
  """Saves all detected configuration(s) into a JSON file.

  Args:
    json_data: Dict of all configurations found.
    filename: String that is the name of the output JSON file.
  """
  if filename[-5:] != ".json":
    print("filename: %s" % filename)
    filename += ".json"

  with open(PATH_TO_DIR + "/" + filename, "w") as f:
    json.dump(json_data, f, sort_keys=True, indent=4)

  print(" Successfully wrote configs to file `%s`.\n" % (filename))


def manage_all_configs(save_results, filename):
  """Manages configuration detection and retrieval based on user input.

  Args:
    save_results: Boolean indicating whether to save the results to a file.
    filename: String that is the name of the output JSON file.
  """
  # Get all configs
  all_configs = get_all_configs()
  # Print all configs based on user input
  print_all_configs(all_configs[0], all_configs[1], all_configs[2])
  # Save all configs to a file based on user request
  if save_results:
    save_to_file(all_configs[3], filename)


def main(argv):
  if len(argv) > 3:
    raise app.UsageError("Too many command-line arguments.")

  manage_all_configs(
      save_results=FLAGS.save_output,
      filename=FLAGS.filename,
  )


if __name__ == "__main__":
  app.run(main)
