# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Builds the tensorflow_text pip package."""

import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile


def is_windows():
  # Matches the bash logic for msys/mingw/cygwin/windows
  return (
      platform.system()
      .lower()
      .startswith(("windows", "msys", "mingw", "cygwin"))
  )


def is_macos():
  return platform.system().lower() == "darwin"


def is_nightly():
  return os.environ.get("IS_NIGHTLY") == "nightly"


def get_platform_name():
  if not is_macos():
    return []

  # Check architecture for macOS
  arch = platform.machine().lower()
  if arch == "arm64":
    return ["--plat-name", "macosx_11_0_arm64"]
  else:
    return ["--plat-name", "macosx-10.9-x86_64"]


def main():
  # Handle Bazel workspace directory
  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
  if workspace_dir:
    os.chdir(workspace_dir)

  # Output directory logic
  output_dir_arg = (
      sys.argv[1] if len(sys.argv) > 1 else "/tmp/tensorflow_text_pkg"
  )
  output_path = pathlib.Path(output_dir_arg).resolve()
  output_path.mkdir(parents=True, exist_ok=True)

  print(f"=== Destination directory: {output_path}")
  print(f"=== Current directory: {pathlib.Path.cwd()}")

  # Verify bazel-bin exists
  if not pathlib.Path("bazel-bin/tensorflow_text").exists():
    print(
        "ERROR: Could not find bazel-bin. Did you run from the root of the"
        " build tree?",
        file=sys.stderr,
    )
    sys.exit(1)

  # Determine runfiles path
  if is_windows():
    runfiles_base = pathlib.Path(
        "bazel-bin/oss_scripts/pip_package/build_pip_package.exe.runfiles"
    )
  else:
    runfiles_base = pathlib.Path(
        "bazel-bin/oss_scripts/pip_package/build_pip_package.runfiles"
    )

  source_root = runfiles_base / "org_tensorflow_text"

  # Create temp directory
  with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = pathlib.Path(temp_dir)
    print(f"=== Using tmpdir {temp_path}")

    # Copy source files (Equivalent to cp -LR)
    shutil.copytree(
        source_root / "tensorflow_text",
        temp_path / "tensorflow_text",
        dirs_exist_ok=True,
    )

    # Setup.py selection
    setup_file = "setup.nightly.py" if is_nightly() else "setup.py"
    shutil.copy2(
        source_root / "oss_scripts/pip_package" / setup_file,
        temp_path / setup_file,
    )

    # Manifest and License
    shutil.copy2(
        source_root / "oss_scripts/pip_package/MANIFEST.in",
        temp_path / "MANIFEST.in",
    )
    shutil.copy2(
        source_root / "oss_scripts/pip_package/LICENSE", temp_path / "LICENSE"
    )

    # Prepare Python execution
    python_bin = sys.executable  # Uses the currently running python

    # Build pip package
    build_cmd = [
        python_bin,
        setup_file,
        "bdist_wheel",
        "--universal",
    ] + get_platform_name()

    print(f"=== Running: {' '.join(build_cmd)}")
    subprocess.run(build_cmd, cwd=temp_path, check=True)

    # Copy wheels to output directory
    dist_path = temp_path / "dist"
    for wheel in dist_path.glob("*.whl"):
      shutil.copy2(wheel, output_path)
      print(f"=== Built: {output_path / wheel.name}")


if __name__ == "__main__":
  main()
