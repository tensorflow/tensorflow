# Copyright 2025 The Tensorflow Authors.
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
"""This script is used to build a python wheel from a list of source files.

It takes a list of source files, a setup.py file, and a version string as input.
It then uses a python script, wheel_builder.py, to generate the wheel file. The
wheel builder binary is responsible for preparing the build environment and
calling the setuptools command to generate the wheel file.

args:
  --setup_py: Path to the setup.py file.
  --output: Output directory for the wheel.
  --version: Version of the wheel.
  --src: List of source files for the wheel.
  --platform: Platform name to be passed to build module.

output:
  A python wheel file is created in the output directory. The name of the wheel
  file is based on various factors, including the version and platform.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tomllib
from typing import Optional


def parse_args() -> argparse.Namespace:
  """Arguments parser."""
  parser = argparse.ArgumentParser(
      description="Helper for building python wheel from pyproject.toml",
      fromfile_prefix_chars="@",
  )
  parser.add_argument("--pyproject", help="location of pyproject.toml file")
  parser.add_argument("--setup_py", help="location of setup.py file")
  parser.add_argument("--output", help="output directory")
  parser.add_argument("--version", help="version of the wheel")
  parser.add_argument(
      "--src", help="single source file for the wheel", action="append"
  )
  parser.add_argument(
      "--platform",
      required=True,
      help="Platform name to be passed to build module",
  )
  return parser.parse_args()


def get_project_name(pyproject_path: str) -> str:
  with open(pyproject_path, "rb") as f:
    pyproject = tomllib.load(f)
    try:
      return pyproject["project"]["name"]
    except KeyError as e:
      raise ValueError(
          "Invalid pyproject.toml file. Please check the project name."
          " Dynamically generated project names are not supported."
      ) from e


def create_empty_init_files(dst_dir: str) -> None:
  """Create __init__.py files."""
  dir_list = [f for f in os.scandir(dst_dir) if f.is_dir()]
  for dir_name in dir_list:
    with open(os.path.join(dir_name, "__init__.py"), "w"):
      pass


def create_init_files(dst_dir: str, meta_dict: Optional[dict[str, str]] = None):
  create_empty_init_files(dst_dir)

  if meta_dict:
    with open(os.path.join(dst_dir, "__init__.py"), "w") as f:
      for key, value in meta_dict.items():
        f.write(f"{key} = \"{value}\"\n")


def construct_meta_dict(args) -> dict[str, str]:
  return {
      "__version__": args.version,
  }


def prepare_build_tree(tree_path, args, project_name: str):
  """Prepares the build tree for the wheel build.

  Args:
    tree_path: Path to the build tree.
    args: Command line arguments.
    project_name: Name of the project.
  """
  src_dir = os.path.join(tree_path, project_name.replace("-", "_"))
  os.makedirs(src_dir)

  shutil.copyfile(args.setup_py, os.path.join(tree_path, "setup.py"))

  for src in args.src:
    shutil.copyfile(src, os.path.join(src_dir, os.path.basename(src)))

  meta_dict = construct_meta_dict(args)

  create_init_files(src_dir, meta_dict)


def build_pyproject_wheel(
    buildtree_path: str, platform_name: Optional[str] = None
):
  """Builds a python wheel from a pyproject.toml file.

  Args:
    buildtree_path: Path to the build tree.
    platform_name: Platform name to be passed to build module.
  """
  env = os.environ.copy()

  command = [
      sys.executable,
      "-m",
      "build",
      "-w",
      "-o",
      os.getcwd(),
  ]

  if platform_name:
    command.append(
        # This is due to setuptools not making it possible to pass the
        # platform name as a dynamic pyproject.toml property.
        f"--config-setting=--build-option=--plat-name={platform_name}"
    )

  subprocess.run(
      command,
      check=True,
      cwd=buildtree_path,
      env=env,
  )


def build_setup_py_wheel(
    buildtree_path: str,
    output_dir: str,
    version: str,
    platform_name: Optional[str] = None,
):
  """Builds a python wheel from a setup.py file.

  Args:
    buildtree_path: Path to the build tree.
    output_dir: Output directory for the wheel.
    version: Version of the wheel.
    platform_name: Platform name to be passed to build module.
  """
  env = os.environ.copy()

  env["PROJECT_NAME"] = "ai_edge_litert"
  env["PACKAGE_VERSION"] = version

  command = [
      sys.executable,
      f"{buildtree_path}/setup.py",
      "bdist_wheel",
      f"--plat-name={platform_name}",
  ]

  subprocess.run(
      command,
      check=True,
      cwd=buildtree_path,
      env=env,
  )

  for filename in glob.glob(os.path.join(buildtree_path, "dist/*.whl")):
    shutil.copy(filename, output_dir)


if __name__ == "__main__":
  build_dir = os.path.join(os.getcwd(), "wheel_build")
  arg_data = parse_args()

  prepare_build_tree(build_dir, arg_data, "ai_edge_litert")
  build_setup_py_wheel(
      build_dir, arg_data.output, arg_data.version, arg_data.platform
  )
