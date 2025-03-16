#!/usr/bin/python3
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
r"""XLA build script for use in CI.

This build script aims to be completely agnostic to the specifics of the VM, the
exceptions are uses of `KOKORO_ARTIFACTS_DIR` and `GITHUB_WORKSPACE` to know
where JAX or TensorFlow lives depending on which build is being executed.

To update the goldens associated with this file, run:
  ```PYTHONDONTWRITEBYTECODE=1 python3 build.py \
      --dump_commands > golden_commands.txt```
"""
import argparse
import dataclasses
import enum
import logging
import os
import subprocess
import sys
from typing import Any, ClassVar, Dict, List, Tuple


# TODO(ddunleavy): move this to the bazelrc
_DEFAULT_BAZEL_OPTIONS = dict(
    color="yes",
    test_output="errors",
    verbose_failures=True,
    keep_going=True,
    nobuild_tests_only=True,
    profile="profile.json.gz",
    flaky_test_attempts=3,
    jobs=150,
    bes_upload_mode="fully_async",
)

_KW_ONLY_IF_PYTHON310 = {"kw_only": True} if sys.version_info >= (3, 10) else {}
_XLA_DEFAULT_TARGET_PATTERNS = (
    "//xla/...",
    "//build_tools/...",
    "@local_tsl//tsl/...",
)
_XLA_CPU_PRESUBMIT_BENCHMARKS_DEFAULT_TARGET_PATTERNS = (
    "//xla/tools:run_hlo_module",
)
_XLA_GPU_PRESUBMIT_BENCHMARKS_DEFAULT_TARGET_PATTERNS = (
    "//xla/tools/multihost_hlo_runner:hlo_runner_main_gpu",
)
_KOKORO_ARTIFACTS_DIR = os.environ.get(
    "KOKORO_ARTIFACTS_DIR", "$KOKORO_ARTIFACTS_DIR"
)
_GITHUB_WORKSPACE = os.environ.get("GITHUB_WORKSPACE", "$GITHUB_WORKSPACE")


def retry(
    args: List[str], delay_seconds: int = 15, retries: int = 3
) -> List[str]:
  # Possibly a slight abuse of `parallel` as nothing happens in parallel, just
  # retries with delay if the command fails.
  # pyformat:disable
  return [
      "parallel", "--ungroup",
      "--retries", str(retries),
      "--delay", str(delay_seconds),
      "--nonall",
      "--", *args,
  ]


def sh(args, check=True, **kwargs):
  logging.info("Starting process: %s", " ".join(args))
  return subprocess.run(args, check=check, **kwargs)


def _dict_to_cli_options(d: Dict[str, Any]) -> List[str]:
  # pylint: disable=g-bool-id-comparison
  return [f"--{k}" if v is True else f"--{k}={v}" for k, v in d.items()]


def _write_to_sponge_config(key, value) -> None:
  with open("custom_sponge_config.csv", "a") as f:
    f.write(f"{key},{value}\n")


class BuildType(enum.Enum):
  """Enum representing all types of builds.

  Should be named as `REPO,OS,HOST_TYPE,BACKEND,GPU_TYPE,CI_TYPE`.
  """
  XLA_LINUX_X86_CPU_GITHUB_ACTIONS = enum.auto()
  XLA_LINUX_ARM64_CPU_GITHUB_ACTIONS = enum.auto()
  XLA_LINUX_X86_GPU_T4_GITHUB_ACTIONS = enum.auto()

  # Presubmit builds for regression testing.
  XLA_LINUX_X86_CPU_16_VCPU_PRESUBMIT_GITHUB_ACTIONS = enum.auto()
  XLA_LINUX_ARM64_CPU_16_VCPU_PRESUBMIT_GITHUB_ACTIONS = enum.auto()
  XLA_LINUX_X86_CPU_128_VCPU_PRESUBMIT_GITHUB_ACTIONS = enum.auto()
  XLA_LINUX_X86_GPU_T4_16_VCPU_PRESUBMIT_GITHUB_ACTIONS = enum.auto()

  XLA_MACOS_X86_CPU_KOKORO = enum.auto()
  XLA_MACOS_ARM64_CPU_KOKORO = enum.auto()

  JAX_LINUX_X86_CPU_GITHUB_ACTIONS = enum.auto()
  JAX_LINUX_X86_GPU_T4_GITHUB_ACTIONS = enum.auto()

  TENSORFLOW_LINUX_X86_CPU_GITHUB_ACTIONS = enum.auto()
  TENSORFLOW_LINUX_X86_GPU_T4_GITHUB_ACTIONS = enum.auto()

  @classmethod
  def from_str(cls, s):
    try:
      return cls[s.replace(" ", "_").upper()]
    except KeyError:
      # Sloppy looking exception handling, but argparse will catch ValueError
      # and give a pleasant error message. KeyError would not work here.
      raise ValueError  # pylint: disable=raise-missing-from


@dataclasses.dataclass(frozen=True, **_KW_ONLY_IF_PYTHON310)
class Build:
  """Class representing a build of XLA."""
  _builds: ClassVar[Dict[BuildType, "Build"]] = {}

  type_: BuildType
  repo: str
  target_patterns: Tuple[str, ...]
  configs: Tuple[str, ...] = ()
  build_tag_filters: Tuple[str, ...] = ()
  test_tag_filters: Tuple[str, ...] = ()
  action_env: Dict[str, Any] = dataclasses.field(default_factory=dict)
  test_env: Dict[str, Any] = dataclasses.field(default_factory=dict)
  options: Dict[str, Any] = dataclasses.field(default_factory=dict)
  extra_setup_commands: Tuple[List[str], ...] = ()
  subcommand: str = "test"

  def __post_init__(self):
    # pylint: disable=protected-access
    assert self.type_ not in self.__class__._builds
    self.__class__._builds[self.type_] = self

  @classmethod
  def all_builds(cls):
    return cls._builds

  def bazel_command(
      self, subcommand: str = "test", extra_options: Tuple[str, ...] = ()
  ) -> List[str]:
    """Returns a bazel test command for this build.

    Args:
      subcommand: The subcommand to give to bazel. `test` by default.
      extra_options: Extra options. For now just used to pass in `--nobuild`.

    Returns: List of command line arguments
    """
    options = _dict_to_cli_options(self.options)
    configs = [f"--config={config}" for config in self.configs]
    build_tag_filters = (
        f"--build_tag_filters={','.join(self.build_tag_filters)}"
    )
    test_tag_filters = f"--test_tag_filters={','.join(self.test_tag_filters)}"
    action_env = [f"--action_env={k}={v}" for k, v in self.action_env.items()]
    test_env = [f"--test_env={k}={v}" for k, v in self.test_env.items()]

    tag_filters = [build_tag_filters, test_tag_filters]
    all_options = (
        tag_filters
        + configs
        + action_env
        + test_env
        + options
        + list(extra_options)
    )
    return ["bazel", subcommand, *all_options, "--", *self.target_patterns]

  def commands(self) -> List[List[str]]:
    """Returns list of commands for a build."""
    cmds = []

    cmds.extend(self.extra_setup_commands)

    # We really want `bazel fetch` here, but it uses `bazel query` and not
    # `cquery`, which means that it fails due to config issues that aren't
    # problems in practice.
    # TODO(ddunleavy): Remove the condition here. Need to get parallel on the
    # MacOS VM.
    macos_build = (
        self.type_ == BuildType.XLA_MACOS_X86_CPU_KOKORO
        or self.type_ == BuildType.XLA_MACOS_ARM64_CPU_KOKORO
    )
    if not macos_build:
      cmds.append(
          retry(
              self.bazel_command(
                  subcommand="build", extra_options=("--nobuild",)
              )
          )
      )
    cmds.append(self.bazel_command(subcommand=self.subcommand))
    cmds.append(["bazel", "analyze-profile", "profile.json.gz"])

    return cmds


def _tag_filters_for_compute_capability(
    compute_capability: int,
) -> Tuple[str, ...]:
  tag_filters = (f"requires-gpu-sm{compute_capability}-only",)
  for cc in (60, 70, 80, 90, 100):
    if compute_capability >= cc:
      tag_filters += (f"requires-gpu-sm{cc}",)
    else:
      tag_filters += (f"-requires-gpu-sm{cc}",)
      tag_filters += (f"-requires-gpu-sm{cc}-only",)
  tag_filters += ("-requires-gpu-amd",)
  return tag_filters


def nvidia_gpu_build_with_compute_capability(
    *,
    type_: BuildType,
    configs: Tuple[str, ...],
    compute_capability: int,
) -> Build:
  extra_gpu_tags = _tag_filters_for_compute_capability(compute_capability)
  return Build(
      type_=type_,
      repo="openxla/xla",
      target_patterns=_XLA_DEFAULT_TARGET_PATTERNS,
      configs=configs,
      test_tag_filters=("-no_oss", "requires-gpu-nvidia", "gpu", "-rocm-only")
      + extra_gpu_tags,
      build_tag_filters=("-no_oss", "requires-gpu-nvidia", "gpu", "-rocm-only"),
      options={
          "run_under": "//build_tools/ci:parallel_gpu_execute",
          "repo_env": f"TF_CUDA_COMPUTE_CAPABILITIES={compute_capability/10}",
          "@cuda_driver//:enable_forward_compatibility": "true",
          **_DEFAULT_BAZEL_OPTIONS,
      },
      extra_setup_commands=(["nvidia-smi"],),
  )


cpu_x86_tag_filter = (
    "-no_oss",
    "-gpu",
    "-requires-gpu-nvidia",
    "-requires-gpu-amd",
)
_XLA_LINUX_X86_CPU_GITHUB_ACTIONS_BUILD = Build(
    type_=BuildType.XLA_LINUX_X86_CPU_GITHUB_ACTIONS,
    repo="openxla/xla",
    configs=("warnings", "nonccl", "rbe_linux_cpu"),
    target_patterns=_XLA_DEFAULT_TARGET_PATTERNS,
    build_tag_filters=cpu_x86_tag_filter,
    test_tag_filters=cpu_x86_tag_filter,
    options=_DEFAULT_BAZEL_OPTIONS,
)

cpu_arm_tag_filter = (
    "-no_oss",
    "-gpu",
    "-requires-gpu-nvidia",
    "-requires-gpu-amd",
    "-not_run:arm",
)
_XLA_LINUX_ARM64_CPU_GITHUB_ACTIONS_BUILD = Build(
    type_=BuildType.XLA_LINUX_ARM64_CPU_GITHUB_ACTIONS,
    repo="openxla/xla",
    configs=("warnings", "rbe_cross_compile_linux_arm64", "nonccl"),
    target_patterns=_XLA_DEFAULT_TARGET_PATTERNS,
    options={**_DEFAULT_BAZEL_OPTIONS, "build_tests_only": True},
    build_tag_filters=cpu_arm_tag_filter,
    test_tag_filters=cpu_arm_tag_filter,
)

_XLA_LINUX_X86_GPU_T4_GITHUB_ACTIONS_BUILD = (
    nvidia_gpu_build_with_compute_capability(
        type_=BuildType.XLA_LINUX_X86_GPU_T4_GITHUB_ACTIONS,
        configs=("warnings", "rbe_linux_cuda_nvcc"),
        compute_capability=75,
    )
)

_XLA_LINUX_X86_CPU_16_VCPU_PRESUBMIT_GITHUB_ACTIONS_BUILD = Build(
    type_=BuildType.XLA_LINUX_X86_CPU_16_VCPU_PRESUBMIT_GITHUB_ACTIONS,
    repo="openxla/xla",
    configs=("warnings", "nonccl", "rbe_linux_cpu"),
    target_patterns=_XLA_CPU_PRESUBMIT_BENCHMARKS_DEFAULT_TARGET_PATTERNS,
    build_tag_filters=cpu_x86_tag_filter,
    test_tag_filters=cpu_x86_tag_filter,
    options=_DEFAULT_BAZEL_OPTIONS,
    subcommand="build",
)

_XLA_LINUX_X86_CPU_128_VCPU_PRESUBMIT_GITHUB_ACTIONS_BUILD = Build(
    type_=BuildType.XLA_LINUX_X86_CPU_128_VCPU_PRESUBMIT_GITHUB_ACTIONS,
    repo="openxla/xla",
    configs=("warnings", "nonccl", "rbe_linux_cpu"),
    target_patterns=_XLA_CPU_PRESUBMIT_BENCHMARKS_DEFAULT_TARGET_PATTERNS,
    build_tag_filters=cpu_x86_tag_filter,
    test_tag_filters=cpu_x86_tag_filter,
    options=_DEFAULT_BAZEL_OPTIONS,
    subcommand="build",
)

_XLA_LINUX_ARM64_CPU_16_VCPU_PRESUBMIT_GITHUB_ACTIONS_BUILD = Build(
    type_=BuildType.XLA_LINUX_ARM64_CPU_16_VCPU_PRESUBMIT_GITHUB_ACTIONS,
    repo="openxla/xla",
    configs=("warnings", "rbe_cross_compile_linux_arm64", "nonccl"),
    target_patterns=_XLA_CPU_PRESUBMIT_BENCHMARKS_DEFAULT_TARGET_PATTERNS,
    options={**_DEFAULT_BAZEL_OPTIONS, "build_tests_only": False},
    build_tag_filters=cpu_arm_tag_filter,
    test_tag_filters=cpu_arm_tag_filter,
    subcommand="build",
)

_XLA_LINUX_X86_GPU_T4_16_VCPU_PRESUBMIT_GITHUB_ACTIONS_BUILD = Build(
    type_=BuildType.XLA_LINUX_X86_GPU_T4_16_VCPU_PRESUBMIT_GITHUB_ACTIONS,
    repo="openxla/xla",
    target_patterns=_XLA_GPU_PRESUBMIT_BENCHMARKS_DEFAULT_TARGET_PATTERNS,
    configs=("warnings", "rbe_linux_cuda_nvcc"),
    test_tag_filters=("-no_oss", "requires-gpu-nvidia", "gpu", "-rocm-only")
    + _tag_filters_for_compute_capability(compute_capability=75),
    build_tag_filters=("-no_oss", "requires-gpu-nvidia", "gpu", "-rocm-only"),
    options={
        "run_under": "//build_tools/ci:parallel_gpu_execute",
        "repo_env": f"TF_CUDA_COMPUTE_CAPABILITIES={7.5}",
        # Use User Mode and Kernel Mode Drivers pre-installed on the system.
        "@cuda_driver//:enable_forward_compatibility": "false",
        **_DEFAULT_BAZEL_OPTIONS,
    },
    extra_setup_commands=(["nvidia-smi"],),
    subcommand="build",
)

macos_tag_filter = (
    "-no_oss",
    "-gpu",
    "-no_mac",
    "-mac_excluded",
    "-requires-gpu-nvidia",
    "-requires-gpu-amd",
)

_XLA_MACOS_X86_CPU_KOKORO_BUILD = Build(
    type_=BuildType.XLA_MACOS_X86_CPU_KOKORO,
    repo="openxla/xla",
    configs=("nonccl",),
    target_patterns=(
        "//xla/...",
        "-//xla/hlo/experimental/...",
        "-//xla/python_api/...",
        "-//xla/python/...",
        "-//xla/service/gpu/...",
    ),
    options=dict(
        **_DEFAULT_BAZEL_OPTIONS,
        macos_minimum_os="10.15",
        test_tmpdir="/Volumes/BuildData/bazel_output",
        define="xnn_enable_avxvnniint8=false",
    ),
    build_tag_filters=macos_tag_filter,
    test_tag_filters=macos_tag_filter,
    extra_setup_commands=(
        [
            "sudo",
            "wget",
            "--no-verbose",
            "-O",
            "/usr/local/bin/bazel",
            "https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-darwin-amd64",
        ],
        ["chmod", "+x", "/usr/local/bin/bazel"],
        ["bazel", "--version"],  # Sanity check due to strange failures
        ["mkdir", "-p", "/Volumes/BuildData/bazel_output"],
    ),
)

_XLA_MACOS_ARM64_CPU_KOKORO_BUILD = Build(
    type_=BuildType.XLA_MACOS_ARM64_CPU_KOKORO,
    repo="openxla/xla",
    configs=("nonccl",),
    target_patterns=(
        "//xla/...",
        "-//xla/hlo/experimental/...",
        "-//xla/python_api/...",
        "-//xla/python/...",
        "-//xla/service/gpu/...",
    ),
    options=dict(
        **_DEFAULT_BAZEL_OPTIONS,
        macos_minimum_os="10.15",
        test_tmpdir="/tmpfs/bazel_output",
        test_size_filters="small,medium",
        define="xnn_enable_avxvnniint8=false",
    ),
    build_tag_filters=macos_tag_filter,
    test_tag_filters=macos_tag_filter,
    extra_setup_commands=(
        ["df", "-h"],  # Debug "No space left on device" error: b/396611909.
        ["bazel", "--version"],  # Sanity check due to strange failures
        ["mkdir", "-p", "/tmpfs/bazel_output"],
    ),
)

_JAX_LINUX_X86_CPU_GITHUB_ACTIONS_BUILD = Build(
    type_=BuildType.JAX_LINUX_X86_CPU_GITHUB_ACTIONS,
    repo="google/jax",
    configs=("rbe_linux_x86_64",),
    target_patterns=("//tests:cpu_tests", "//tests:backend_independent_tests"),
    test_env=dict(
        JAX_NUM_GENERATED_CASES=25,
        JAX_SKIP_SLOW_TESTS=1,
    ),
    options=dict(
        **_DEFAULT_BAZEL_OPTIONS,
        override_repository=f"xla={_GITHUB_WORKSPACE}/openxla/xla",
        repo_env="HERMETIC_PYTHON_VERSION=3.12",
    ),
)

_JAX_LINUX_X86_GPU_T4_GITHUB_ACTIONS_BUILD = Build(
    type_=BuildType.JAX_LINUX_X86_GPU_T4_GITHUB_ACTIONS,
    repo="google/jax",
    configs=("rbe_linux_x86_64_cuda",),
    target_patterns=("//tests:gpu_tests", "//tests:backend_independent_tests"),
    build_tag_filters=("-multiaccelerator",),
    test_tag_filters=("-multiaccelerator",),
    test_env=dict(
        JAX_SKIP_SLOW_TESTS=1,
        TF_CPP_MIN_LOG_LEVEL=0,
        JAX_EXCLUDE_TEST_TARGETS="PmapTest.testSizeOverflow",
    ),
    options=dict(
        **_DEFAULT_BAZEL_OPTIONS,
        override_repository=f"xla={_GITHUB_WORKSPACE}/openxla/xla",
        repo_env="HERMETIC_PYTHON_VERSION=3.10",
    ),
)

tensorflow_tag_filters = (
    "-no_oss",
    "-tf_tosa",
    "-oss_excluded",
    "-oss_serial",
    "-tpu",
    "-benchmark-test",
    "-v1only",
)

tensorflow_cpu_tag_filters = tensorflow_tag_filters + ("-gpu",)
tensorflow_gpu_tag_filters = tensorflow_tag_filters + (
    "-no_gpu",
    "-no_gpu_presubmit",
    "-no_cuda11",
    "+gpu",
)

_TENSORFLOW_LINUX_X86_CPU_GITHUB_ACTIONS_BUILD = Build(
    type_=BuildType.TENSORFLOW_LINUX_X86_CPU_GITHUB_ACTIONS,
    repo="tensorflow/tensorflow",
    configs=(
        "release_cpu_linux",
        "rbe_linux_cpu",
    ),
    target_patterns=(
        "//tensorflow/compiler/...",
        "-//tensorflow/compiler/tf2tensorrt/...",
        "//tensorflow/python/...",
        "-//tensorflow/python/distribute/...",
        "-//tensorflow/python/kernel_tests/...",
        "-//tensorflow/python/compiler/tensorrt/...",
    ),
    build_tag_filters=tensorflow_cpu_tag_filters,
    test_tag_filters=tensorflow_cpu_tag_filters,
    options=dict(
        verbose_failures=True,
        test_output="errors",
        override_repository=f"xla={_GITHUB_WORKSPACE}/openxla/xla",
        profile="profile.json.gz",
        test_lang_filters="cc,py",
        color="yes",
    ),
)

_TENSORFLOW_LINUX_X86_GPU_T4_GITHUB_ACTIONS_BUILD = Build(
    type_=BuildType.TENSORFLOW_LINUX_X86_GPU_T4_GITHUB_ACTIONS,
    repo="tensorflow/tensorflow",
    configs=(
        "release_gpu_linux",
        "rbe_linux_cuda",
    ),
    target_patterns=(
        "//tensorflow/compiler/...",
        "-//tensorflow/compiler/tf2tensorrt/...",
        "//tensorflow/python/...",
        "-//tensorflow/python/distribute/...",
        "-//tensorflow/python/kernel_tests/...",
        "-//tensorflow/python/compiler/tensorrt/...",
    ),
    build_tag_filters=tensorflow_gpu_tag_filters,
    test_tag_filters=tensorflow_gpu_tag_filters,
    options=dict(
        verbose_failures=True,
        test_output="errors",
        override_repository=f"xla={_GITHUB_WORKSPACE}/openxla/xla",
        profile="profile.json.gz",
        test_lang_filters="cc,py",
        color="yes",
    ),
)


def dump_all_build_commands():
  """Used to generate what commands are run for each build."""
  # Awkward workaround b/c Build instances are not hashable
  for build in sorted(Build.all_builds().values(), key=lambda b: str(b.type_)):
    sys.stdout.write(f"# BEGIN {build.type_}\n")
    for cmd in build.commands():
      sys.stdout.write(" ".join(cmd) + "\n")
    sys.stdout.write(f"# END {build.type_}\n")


def _parse_args():
  """Defines flags and parses args."""
  parser = argparse.ArgumentParser(allow_abbrev=False)
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument(
      "--build",
      type=BuildType.from_str,
      choices=list(BuildType),
  )
  group.add_argument(
      "--dump_commands",
      action="store_true",
  )

  return parser.parse_args()


def main():
  logging.basicConfig()
  logging.getLogger().setLevel(logging.INFO)

  args = _parse_args()

  if args.dump_commands:
    dump_all_build_commands()
    return
  else:
    for cmd in Build.all_builds()[args.build].commands():
      sh(cmd)

if __name__ == "__main__":
  main()
