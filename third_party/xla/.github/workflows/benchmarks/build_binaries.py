# Copyright 2025 The OpenXLA Authors. All Rights Reserved.
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
# ============================================================================
"""Builds HLO runner and stats computation binaries using build.py.

This script expects the 'HARDWARE_CATEGORY' (e.g., CPU_X86, GPU_L4)
and 'OUTPUT_DIR' environment variables to be set.

It outputs the paths to the runner and stats binaries, and a device type flag
to the file specified by the GITHUB_OUTPUT environment variable.
"""

import os
import pathlib
import subprocess
import sys

Path = pathlib.Path

# --- Configuration ---
BAZEL_BIN_DIR = "bazel-bin"
HARDWARE_CONFIG = {
    "CPU_X86": {
        "build_type": "XLA_LINUX_X86_CPU_128_VCPU_PRESUBMIT_GITHUB_ACTIONS",
        "runner_binary": (
            f"./{BAZEL_BIN_DIR}/xla/tools/multihost_hlo_runner/hlo_runner_main"
        ),
        "stats_binary": (
            f"./{BAZEL_BIN_DIR}/xla/tools/compute_xspace_stats_main"
        ),
        "device_type_flag": "host",
        "configure_backend": "CPU",
    },
    "CPU_ARM64": {
        "build_type": "XLA_LINUX_ARM64_CPU_48_VCPU_PRESUBMIT_GITHUB_ACTIONS",
        "runner_binary": (
            f"./{BAZEL_BIN_DIR}/xla/tools/multihost_hlo_runner/hlo_runner_main"
        ),
        "stats_binary": (
            f"./{BAZEL_BIN_DIR}/xla/tools/compute_xspace_stats_main"
        ),
        "device_type_flag": "host",
        "configure_backend": "CPU",
    },
    "GPU_L4": {
        "build_type": "XLA_LINUX_X86_GPU_L4_16_VCPU_PRESUBMIT_GITHUB_ACTIONS",
        "runner_binary": (
            f"./{BAZEL_BIN_DIR}/xla/tools/multihost_hlo_runner/hlo_runner_main_gpu"
        ),
        "stats_binary": (
            f"./{BAZEL_BIN_DIR}/xla/tools/compute_xspace_stats_main_gpu"
        ),
        "device_type_flag": "gpu",
        "configure_backend": "CUDA",
    },
    "GPU_B200": {
        "build_type": "XLA_LINUX_X86_GPU_A4_224_VCPU_PRESUBMIT_GITHUB_ACTIONS",
        "runner_binary": (
            f"./{BAZEL_BIN_DIR}/xla/tools/multihost_hlo_runner/hlo_runner_main_gpu"
        ),
        "stats_binary": (
            f"./{BAZEL_BIN_DIR}/xla/tools/compute_xspace_stats_main_gpu"
        ),
        "device_type_flag": "gpu",
        "configure_backend": "CUDA",
    },
}


def run_command(command, check=True):
  """Executes a shell command and returns its output."""
  print(f"Running command: {' '.join(command)}")
  try:
    process = subprocess.run(
        command, check=check, text=True, capture_output=True
    )
    if process.stdout:
      print(process.stdout)
    if process.stderr:
      print(process.stderr, file=sys.stderr)
    return process
  except subprocess.CalledProcessError as e:
    print(
        f"::error::Command failed with exit code {e.returncode}",
        file=sys.stderr,
    )
    raise


def configure_backend(config):
  """Runs the ./configure.py script for the specified backend."""
  print("--- Configuring Backend ---")
  configure_script = Path("./configure.py")
  print(f"configure_script: {configure_script}")
  if not configure_script.exists():
    print(f"INFO: {configure_script} not found. Skipping configuration.")
    return

  backend = config.get("configure_backend")
  if not backend:
    print(f"INFO: No backend configuration for {config['build_type']}")
    return

  command = [str(configure_script), f"--backend={backend}"]
  if backend == "CUDA":
    command.extend(["--cuda_compiler=nvcc"])

  # Change the current directory to the root of the xla repository to run the
  # configure script.
  os.chdir("../..")
  print(f"os.getcwd(): {os.getcwd()}")
  try:
    run_command(command)
  except subprocess.CalledProcessError:
    print(f"INFO: {backend} configure script failed or is not applicable.")
  finally:
    # Change back to the original directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


def build_binaries(build_type):
  """Executes the main build process using build.py."""
  print(f"--- Building Binaries for {build_type} ---")
  command = ["python3", "build_tools/ci/build.py", f"--build={build_type}"]
  try:
    run_command(command)
  except subprocess.CalledProcessError:
    print(f"::error::build.py failed for {build_type}!")
    sys.exit(1)


def verify_binaries(config):
  """Verifies the existence of the generated binary files."""
  print("--- Verifying Binaries ---")
  runner_binary = Path(config["runner_binary"])
  stats_binary = Path(config["stats_binary"])

  if not runner_binary.is_file():
    print(f"::error::Runner binary not found: {runner_binary}")
    sys.exit(1)
  if not stats_binary.is_file():
    print(f"::error::Stats binary not found: {stats_binary}")
    sys.exit(1)

  print(f"  Runner binary: {runner_binary}")
  print(f"  Stats binary: {stats_binary}")


def set_github_outputs(config):
  """Writes the output variables to the GITHUB_OUTPUT file."""
  github_output = os.getenv("GITHUB_OUTPUT")
  if not github_output:
    print("GITHUB_OUTPUT environment variable not set. Skipping output.")
    return

  print("--- Setting GitHub Outputs ---")
  with open(github_output, "a") as f:
    f.write(f"runner_binary={config['runner_binary']}\n")
    f.write(f"stats_binary={config['stats_binary']}\n")
    f.write(f"device_type_flag={config['device_type_flag']}\n")

  print(f"  Device type flag: {config['device_type_flag']}")


def main():
  """Main function to configure, build, and verify binaries."""
  hardware_category = os.getenv("HARDWARE_CATEGORY")
  output_dir = os.getenv("OUTPUT_DIR")

  if not hardware_category:
    print("::error::HARDWARE_CATEGORY environment variable not set.")
    sys.exit(1)

  print(f"Workspace: {os.getcwd()}")
  print(f"Hardware Category: {hardware_category}")
  print(f"Output Directory: {output_dir}")

  config = HARDWARE_CONFIG.get(hardware_category)
  if not config:
    print(
        f"::error::Unsupported HARDWARE_CATEGORY: '{hardware_category}'. "
        f"Supported values are: {', '.join(HARDWARE_CONFIG.keys())}"
    )
    sys.exit(1)

  configure_backend(config)
  build_binaries(config["build_type"])
  verify_binaries(config)
  set_github_outputs(config)

  print("--- Build Script Finished Successfully ---")


if __name__ == "__main__":
  main()

