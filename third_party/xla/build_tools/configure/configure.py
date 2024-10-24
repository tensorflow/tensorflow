#!/usr/bin/env python3
# Copyright 2024 The OpenXLA Authors.
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
"""Configure script to get build parameters from user.

This script populates a bazelrc file that tells Bazel where to look for
cuda versions and compilers. Note: that a configuration is possible to request,
does not mean that it is supported (e.g. building with gcc). That being said,
if this stops working for you on an unsupported build and you have a fix, please
send a PR!

Example usage:
  `./configure.py --backend=cpu --host_compiler=clang`
  Will write a bazelrc to the root of the repo with the lines required to find
  the clang in your path. If that isn't the correct clang, you can override like
  `./configure.py --backend=cpu --clang_path=<PATH_TO_YOUR_CLANG>`.

TODO(ddunleavy): add more thorough validation.
"""
import argparse
import dataclasses
import enum
import logging
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from typing import Optional

_DEFAULT_BUILD_AND_TEST_TAG_FILTERS = ("-no_oss",)
# Assume we are being invoked from the symlink at the root of the repo
_XLA_SRC_ROOT = pathlib.Path(__file__).absolute().parent
_XLA_BAZELRC_NAME = "xla_configure.bazelrc"
_KW_ONLY_IF_PYTHON310 = {"kw_only": True} if sys.version_info >= (3, 10) else {}


def _find_executable(executable: str) -> Optional[str]:
  logging.info("Trying to find path to %s...", executable)
  # Resolving the symlink is necessary for finding system headers.
  if unresolved_path := shutil.which(executable):
    return str(pathlib.Path(unresolved_path).resolve())
  return None


def _find_executable_or_die(
    executable_name: str, executable_path: Optional[str] = None
) -> str:
  """Finds executable and resolves symlinks or raises RuntimeError.

  Resolving symlinks is sometimes necessary for finding system headers.

  Args:
    executable_name: The name of the executable that we want to find.
    executable_path: If not None, the path to the executable.

  Returns:
    The path to the executable we are looking for, after symlinks are resolved.
  Raises:
    RuntimeError: if path to the executable cannot be found.
  """
  if executable_path:
    return str(pathlib.Path(executable_path).resolve(strict=True))
  resolved_path_to_exe = _find_executable(executable_name)
  if resolved_path_to_exe is None:
    raise RuntimeError(
        f"Could not find executable `{executable_name}`! "
        "Please change your $PATH or pass the path directly like"
        f"`--{executable_name}_path=path/to/executable."
    )
  logging.info("Found path to %s at %s", executable_name, resolved_path_to_exe)

  return resolved_path_to_exe


def _get_cuda_compute_capabilities_or_die() -> list[str]:
  """Finds compute capabilities via nvidia-smi or rasies exception.

  Returns:
    list of unique, sorted strings representing compute capabilities:
  Raises:
    RuntimeError: if path to nvidia-smi couldn't be found.
    subprocess.CalledProcessError: if nvidia-smi process failed.
  """
  try:
    nvidia_smi = _find_executable_or_die("nvidia-smi")
    nvidia_smi_proc = subprocess.run(
        [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        check=True,
        text=True,
    )
    # Command above returns a newline separated list of compute capabilities
    # with possible repeats. So we should unique them and sort the final result.
    capabilities = sorted(set(nvidia_smi_proc.stdout.strip().split("\n")))
    logging.info("Found CUDA compute capabilities: %s", capabilities)
    return capabilities
  except (RuntimeError, subprocess.CalledProcessError) as e:
    logging.info(
        "Could not find nvidia-smi, or nvidia-smi command failed. Please pass"
        " capabilities directly using --cuda_compute_capabilities."
    )
    raise e


def _get_clang_major_version(path_to_clang: str) -> int:
  """Gets the major version of the clang at `path_to_clang`.

  Args:
    path_to_clang: Path to a clang executable

  Returns:
    The major version.
  """
  logging.info("Running echo __clang_major__ | %s -E -P -", path_to_clang)
  clang_version_proc = subprocess.run(
      [path_to_clang, "-E", "-P", "-"],
      input="__clang_major__",
      check=True,
      capture_output=True,
      text=True,
  )
  major_version = int(clang_version_proc.stdout)
  logging.info("%s reports major version %s.", path_to_clang, major_version)

  return major_version


class ArgparseableEnum(enum.Enum):
  """Enum base class with helper methods for working with argparse.

  Example usage:
  ```
  class Fruit(ArgparseableEnum):
    APPLE = enum.auto()

  # argparse setup
  parser.add_argument("--fruit", type=Fruit.from_str, choices=list(Fruit))
  ```
  Users can pass strings like `--fruit=apple` with nice error messages and the
  parser will get the corresponding enum value.

  NOTE: PyType gets confused when this class is used to create Enums in the
  functional style like `ArgparseableEnum("Fruit", ["APPLE", "BANANA"])`.
  """

  def __str__(self):
    return self.name

  @classmethod
  def from_str(cls, s):
    s = s.upper()
    try:
      return cls[s]
    except KeyError:
      # Sloppy looking exception handling, but argparse will catch ValueError
      # and give a pleasant error message. KeyError would not work here.
      raise ValueError  # pylint: disable=raise-missing-from


class Backend(ArgparseableEnum):
  CPU = enum.auto()
  CUDA = enum.auto()
  ROCM = enum.auto()
  SYCL = enum.auto()


class HostCompiler(ArgparseableEnum):
  CLANG = enum.auto()
  GCC = enum.auto()


class CudaCompiler(ArgparseableEnum):
  CLANG = enum.auto()
  NVCC = enum.auto()


class RocmCompiler(ArgparseableEnum):
  HIPCC = enum.auto()


class OS(ArgparseableEnum):
  """Modeled after the values returned by `platform.system()`."""
  LINUX = enum.auto()
  DARWIN = enum.auto()
  WINDOWS = enum.auto()


@dataclasses.dataclass(**_KW_ONLY_IF_PYTHON310)
class DiscoverablePathsAndVersions:
  """Paths to various tools and libraries needed to build XLA.

  This class is where all 'stateful' activity should happen, like trying to read
  environment variables or looking for things in the $PATH. An instance that has
  all fields set should not try to do any of these things though, so that this
  file can remain unit testable.
  """

  clang_path: Optional[str] = None
  clang_major_version: Optional[int] = None
  gcc_path: Optional[str] = None
  lld_path: Optional[str] = None
  ld_library_path: Optional[str] = None

  # CUDA specific
  cuda_version: Optional[str] = None
  cuda_compute_capabilities: Optional[list[str]] = None
  cudnn_version: Optional[str] = None
  local_cuda_path: Optional[str] = None
  local_cudnn_path: Optional[str] = None
  local_nccl_path: Optional[str] = None

  def get_relevant_paths_and_versions(self, config: "XLAConfigOptions"):
    """Gets paths and versions as needed by the config.

    Args:
      config: XLAConfigOptions instance that determines what paths and versions
        to try to autoconfigure.
    """
    if self.ld_library_path is None:
      self.ld_library_path = os.environ.get("LD_LIBRARY_PATH", None)

    if config.host_compiler == HostCompiler.CLANG:
      self.clang_path = _find_executable_or_die("clang", self.clang_path)
      self.clang_major_version = (
          self.clang_major_version or _get_clang_major_version(self.clang_path)
      )

      # Notably, we don't use `_find_executable_or_die` for lld, as it changes
      # which commands it accepts based on its name! ld.lld is symlinked to a
      # different executable just called lld, which should not be invoked
      # directly.
      self.lld_path = self.lld_path or shutil.which("ld.lld")
    elif config.host_compiler == HostCompiler.GCC:
      self.gcc_path = _find_executable_or_die("gcc", self.gcc_path)

    if config.backend == Backend.CUDA:
      if config.cuda_compiler == CudaCompiler.CLANG:
        self.clang_path = _find_executable_or_die("clang", self.clang_path)

      if not self.cuda_compute_capabilities:
        self.cuda_compute_capabilities = _get_cuda_compute_capabilities_or_die()


@dataclasses.dataclass(frozen=True, **_KW_ONLY_IF_PYTHON310)
class XLAConfigOptions:
  """Represents XLA configuration options."""

  backend: Backend
  os: OS
  python_bin_path: str
  host_compiler: HostCompiler
  compiler_options: list[str]

  # CUDA specific
  cuda_compiler: CudaCompiler
  using_nccl: bool

  # ROCM specific
  rocm_compiler: RocmCompiler

  def to_bazelrc_lines(
      self,
      dpav: DiscoverablePathsAndVersions,
  ) -> list[str]:
    """Creates a bazelrc given an XLAConfigOptions.

    Necessary paths are provided by the user, or retrieved via
    `self._get_relevant_paths`.

    Args:
      dpav: DiscoverablePathsAndVersions that may hold user-specified paths and
        versions. The dpav will then read from `self` to determine what to try
        to auto-configure.

    Returns:
      The lines of a bazelrc.
    """
    dpav.get_relevant_paths_and_versions(self)
    rc = []
    build_and_test_tag_filters = list(_DEFAULT_BUILD_AND_TEST_TAG_FILTERS)

    if self.os == OS.DARWIN:
      build_and_test_tag_filters.append("-no_mac")

    # Platform independent options based on host compiler
    if self.host_compiler == HostCompiler.GCC:
      rc.append(f"build --action_env GCC_HOST_COMPILER_PATH={dpav.gcc_path}")
    elif self.host_compiler == HostCompiler.CLANG:
      rc.append(f"build --action_env CLANG_COMPILER_PATH={dpav.clang_path}")
      rc.append(f"build --repo_env CC={dpav.clang_path}")
      rc.append(f"build --repo_env BAZEL_COMPILER={dpav.clang_path}")
      self.compiler_options.append("-Wno-error=unused-command-line-argument")
      if dpav.lld_path:
        rc.append(f"build --linkopt --ld-path={dpav.lld_path}")

    if self.backend == Backend.CPU:
      build_and_test_tag_filters.append("-gpu")

    elif self.backend == Backend.CUDA:
      build_and_test_tag_filters.append("-rocm-only")
      build_and_test_tag_filters.append("-sycl-only")

      compiler_pair = self.cuda_compiler, self.host_compiler

      if compiler_pair == (CudaCompiler.CLANG, HostCompiler.CLANG):
        rc.append("build --config cuda_clang")
        rc.append(
            f"build --action_env CLANG_CUDA_COMPILER_PATH={dpav.clang_path}"
        )
      elif compiler_pair == (CudaCompiler.NVCC, HostCompiler.CLANG):
        rc.append("build --config cuda_nvcc")
        # This is demanded by cuda_configure.bzl
        rc.append(
            f"build --action_env CLANG_CUDA_COMPILER_PATH={dpav.clang_path}"
        )
      elif compiler_pair == (CudaCompiler.NVCC, HostCompiler.GCC):
        rc.append("build --config cuda")
      else:
        raise NotImplementedError(
            "CUDA clang with host compiler gcc not supported"
        )

      # Lines needed for CUDA backend regardless of CUDA/host compiler
      if dpav.cuda_version:
        rc.append(
            f"build:cuda --repo_env HERMETIC_CUDA_VERSION={dpav.cuda_version}"
        )
      rc.append(
          "build:cuda --repo_env HERMETIC_CUDA_COMPUTE_CAPABILITIES="
          f"{','.join(dpav.cuda_compute_capabilities)}"
      )
      if dpav.cudnn_version:
        rc.append(
            f"build:cuda --repo_env HERMETIC_CUDNN_VERSION={dpav.cudnn_version}"
        )
      if dpav.local_cuda_path:
        rc.append(
            f"build:cuda --repo_env LOCAL_CUDA_PATH={dpav.local_cuda_path}"
        )
      if dpav.local_cudnn_path:
        rc.append(
            f"build:cuda --repo_env LOCAL_CUDNN_PATH={dpav.local_cudnn_path}"
        )
      if dpav.local_nccl_path:
        rc.append(
            f"build:cuda --repo_env LOCAL_NCCL_PATH={dpav.local_nccl_path}"
        )
      if not self.using_nccl:
        rc.append("build --config nonccl")
    elif self.backend == Backend.ROCM:
      build_and_test_tag_filters.append("-cuda-only")
      build_and_test_tag_filters.append("-sycl-only")

      compiler_pair = self.rocm_compiler, self.host_compiler

      if compiler_pair == (RocmCompiler.HIPCC, HostCompiler.CLANG):
        rc.append("build --config rocm")
        # This is demanded by rocm_configure.bzl.
        rc.append(f"build --action_env CLANG_COMPILER_PATH={dpav.clang_path}")
      elif compiler_pair == (RocmCompiler.HIPCC, HostCompiler.GCC):
        rc.append("build --config rocm")
      else:
        raise NotImplementedError("ROCm clang with host compiler not supported")
    elif self.backend == Backend.SYCL:
      build_and_test_tag_filters.append("-cuda-only")
      build_and_test_tag_filters.append("-rocm-only")

      rc.append("build --config sycl")

    # Lines that are added for every backend
    if dpav.ld_library_path:
      rc.append(f"build --action_env LD_LIBRARY_PATH={dpav.ld_library_path}")

    if dpav.clang_major_version in (16, 17, 18):
      self.compiler_options.append("-Wno-gnu-offsetof-extensions")

    rc.append(f"build --action_env PYTHON_BIN_PATH={self.python_bin_path}")
    rc.append(f"build --python_path {self.python_bin_path}")
    rc.append("test --test_env LD_LIBRARY_PATH")
    rc.append("test --test_size_filters small,medium")

    rc.extend([
        f"build --copt {compiler_option}"
        for compiler_option in self.compiler_options
    ])

    # Add build and test tag filters
    build_and_test_tag_filters = ",".join(build_and_test_tag_filters)
    rc.append(f"build --build_tag_filters {build_and_test_tag_filters}")
    rc.append(f"build --test_tag_filters {build_and_test_tag_filters}")
    rc.append(f"test --build_tag_filters {build_and_test_tag_filters}")
    rc.append(f"test --test_tag_filters {build_and_test_tag_filters}")

    return rc


def _parse_args():
  """Creates an argparse.ArgumentParser and parses arguments."""
  comma_separated_list = lambda l: [s.strip() for s in l.split(",")]

  parser = argparse.ArgumentParser(allow_abbrev=False)
  parser.add_argument(
      "--backend",
      type=Backend.from_str,
      choices=list(Backend),
      required=True,
  )
  parser.add_argument(
      "--os", type=OS.from_str, choices=list(OS), default=platform.system()
  )
  parser.add_argument(
      "--host_compiler",
      type=HostCompiler.from_str,
      choices=list(HostCompiler),
      default="clang",
  )
  parser.add_argument(
      "--cuda_compiler",
      type=CudaCompiler.from_str,
      choices=list(CudaCompiler),
      default="nvcc",
  )
  parser.add_argument(
      "--rocm_compiler",
      type=RocmCompiler.from_str,
      choices=list(RocmCompiler),
      default="hipcc",
  )
  parser.add_argument(
      "--cuda_compute_capabilities",
      type=comma_separated_list,
      default=None,
  )
  parser.add_argument("--python_bin_path", default=sys.executable)
  parser.add_argument(
      "--compiler_options",
      type=comma_separated_list,
      default="-Wno-sign-compare",
  )
  parser.add_argument("--nccl", action="store_true")

  # Path and version overrides
  path_help = "Optional: will be found on PATH if possible."
  parser.add_argument("--clang_path", help=path_help)
  parser.add_argument("--gcc_path", help=path_help)
  parser.add_argument(
      "--ld_library_path",
      help=(
          "Optional: will be automatically taken from the current environment"
          " if flag is not set"
      ),
  )
  parser.add_argument("--lld_path", help=path_help)

  # CUDA specific
  parser.add_argument(
      "--cuda_version",
      help="Optional: CUDA will be downloaded by Bazel if the flag is set",
  )
  parser.add_argument(
      "--cudnn_version",
      help="Optional: CUDNN will be downloaded by Bazel if the flag is set",
  )
  parser.add_argument(
      "--local_cuda_path",
      help=(
          "Optional: Local CUDA dir will be used in dependencies if the flag"
          " is set"
      ),
  )
  parser.add_argument(
      "--local_cudnn_path",
      help=(
          "Optional: Local CUDNN dir will be used in dependencies if the flag"
          " is set"
      ),
  )
  parser.add_argument(
      "--local_nccl_path",
      help=(
          "Optional: Local NCCL dir will be used in dependencies if the flag"
          " is set"
      ),
  )

  return parser.parse_args()


def main():
  # Setup logging
  logging.basicConfig()
  logging.getLogger().setLevel(logging.INFO)

  args = _parse_args()

  config = XLAConfigOptions(
      backend=args.backend,
      os=args.os,
      host_compiler=args.host_compiler,
      cuda_compiler=args.cuda_compiler,
      python_bin_path=args.python_bin_path,
      compiler_options=args.compiler_options,
      using_nccl=args.nccl,
      rocm_compiler=args.rocm_compiler,
  )

  bazelrc_lines = config.to_bazelrc_lines(
      DiscoverablePathsAndVersions(
          clang_path=args.clang_path,
          gcc_path=args.gcc_path,
          lld_path=args.lld_path,
          ld_library_path=args.ld_library_path,
          cuda_version=args.cuda_version,
          cudnn_version=args.cudnn_version,
          cuda_compute_capabilities=args.cuda_compute_capabilities,
          local_cuda_path=args.local_cuda_path,
          local_cudnn_path=args.local_cudnn_path,
          local_nccl_path=args.local_nccl_path,
      )
  )

  bazelrc_path = _XLA_SRC_ROOT / _XLA_BAZELRC_NAME
  bazelrc_contents = "\n".join(bazelrc_lines) + "\n"

  with (bazelrc_path).open("w") as f:
    logging.info("Writing bazelrc to %s...", bazelrc_path)
    f.write(bazelrc_contents)

if __name__ == "__main__":
  raise SystemExit(main())
