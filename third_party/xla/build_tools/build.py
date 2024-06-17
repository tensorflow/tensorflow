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
"""XLA build script for use in CI.

This script is used for the Kokoro builds of XLA, but aims to be as agnostic to
the specifics of the VM as possible. The only Kokoro-specific things that are
assumed are:
  * that `KOKORO_JOB_NAME` is set, which is used to decide what build to run.
  * and all code ends up in `$PWD/github/$REPO_NAME`.
The script also assumes that the working directory never changes modulo `cd`ing
into the repo that should be built (mostly `github/xla`, but also JAX and TF).
"""
import contextlib
import dataclasses
import enum
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

_KW_ONLY_IF_PYTHON310 = {"kw_only": True} if sys.version_info >= (3, 10) else {}

# TODO(ddunleavy): move this to the bazelrc
_DEFAULT_BAZEL_OPTIONS = dict(
    test_output="errors",
    verbose_failures=True,
    keep_going=True,
    nobuild_tests_only=True,
    profile="profile.json.gz",
    flaky_test_attempts=3,
    jobs=150,
    bes_upload_mode="fully_async",
)

_DEFAULT_DOCKER_OPTIONS = dict(
    rm=True,
    interactive=True,
    detach=True,
    tty=True,
    volume="./github:/github",
)

_XLA_DEFAULT_TARGET_PATTERNS = (
    "//xla/...",
    "//build_tools/...",
    "@local_tsl//tsl/...",
)


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
  CPU_X86 = enum.auto()
  CPU_ARM64 = enum.auto()
  GPU = enum.auto()
  GPU_CONTINUOUS = enum.auto()

  JAX_CPU = enum.auto()
  JAX_GPU = enum.auto()


@dataclasses.dataclass(frozen=True, **_KW_ONLY_IF_PYTHON310)
class DockerImage:
  """Class representing a docker image."""

  image_url: str

  def _pull_docker_image_with_retries(self, retries=3) -> None:
    """Pulls docker image with retries to avoid transient rate limit errors."""
    for _ in range(retries):
      pull_proc = sh(["docker", "pull", self.image_url], check=False)
      if pull_proc.returncode != 0:
        time.sleep(15)

    # write SHA of image to the sponge config
    _write_to_sponge_config("TF_INFO_DOCKER_IMAGE", self.image_url)

    _ = sh(["docker", "pull", self.image_url])
    # TODO(ddunleavy): get sha
    # _write_to_sponge_config("TF_INFO_DOCKER_SHA", sha)

  @contextlib.contextmanager
  def pull_and_run(
      self,
      name: str = "xla_ci",
      command: Tuple[str, ...] = ("bash",),
      **kwargs: Any,
  ):
    """Context manager for the container that yields `docker exec` lambda.

    Args:
      name: The name of the docker container.
      command: Command given to `docker run`, e.g. `bash`
      **kwargs: Extra options passed to `docker run`.

    Yields:
      A function that accepts a command as a list of args, and runs those on the
      corresponding docker container. It shouldn't be used outside the `with`
      block, as the container will be stopped after the end of the block.

    This manages pulling, starting, and stopping the container. Example usage:
    ```
    with image.pull_and_run() as docker_exec:
      docker_exec(["command", "--with", "--flags"])
    ```
    """
    try:
      self._pull_docker_image_with_retries()
      options = _dict_to_cli_options(kwargs)
      sh([
          "docker",
          "run",
          "--name",
          name,
          *options,
          self.image_url,
          *command,
      ])
      docker_exec = lambda args: sh(["docker", "exec", name, *args])
      yield docker_exec
    finally:
      sh(["docker", "stop", name])


@dataclasses.dataclass(frozen=True, **_KW_ONLY_IF_PYTHON310)
class Build:
  """Class representing a build of XLA."""

  type_: BuildType
  repo: str
  docker_image: DockerImage
  target_patterns: Tuple[str, ...]
  configs: Tuple[str, ...] = ()
  tag_filters: Tuple[str, ...] = ()
  action_env: Dict[str, Any] = dataclasses.field(default_factory=dict)
  test_env: Dict[str, Any] = dataclasses.field(default_factory=dict)
  options: Dict[str, Any] = dataclasses.field(default_factory=dict)

  def bazel_test_command(self) -> List[str]:
    options = _dict_to_cli_options(self.options)
    configs = [f"--config={config}" for config in self.configs]
    build_tag_filters = f"--build_tag_filters={','.join(self.tag_filters)}"
    test_tag_filters = f"--test_tag_filters={','.join(self.tag_filters)}"
    action_env = [f"--action_env={k}={v}" for k, v in self.action_env.items()]
    test_env = [f"--test_env={k}={v}" for k, v in self.test_env.items()]

    tag_filters = [build_tag_filters, test_tag_filters]
    all_options = tag_filters + configs + action_env + test_env + options
    return ["bazel", "test", *all_options, "--", *self.target_patterns]


def _tag_filters_for_compute_capability(
    compute_capability: int,
) -> Tuple[str, ...]:
  tag_filters = (f"requires-gpu-sm{compute_capability}-only",)
  for cc in (60, 70, 80, 90):
    if compute_capability >= cc:
      tag_filters += (f"requires-gpu-sm{cc}",)
    else:
      tag_filters += (f"-requires-gpu-sm{cc}",)
      tag_filters += (f"-requires-gpu-sm{cc}-only",)
  return tag_filters


_DEFAULT_IMAGE = DockerImage(
    image_url="gcr.io/tensorflow-sigs/build:latest-python3.11",
)

# TODO(b/338885148): Remove this once the TF containers have cuDNN 9
_CUDNN_9_IMAGE = DockerImage(
    image_url="gcr.io/tensorflow-sigs/build@sha256:dddcaf30321e9007103dce75c51b83fea3c06de462fcf41e7c6ae93f37fc3545",
)

_ARM64_JAX_MULTI_PYTHON_IMAGE = DockerImage(
    image_url="us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/build-arm64:jax-latest-multi-python",
)


def nvidia_gpu_build_with_compute_capability(
    *, type_: BuildType, configs: Tuple[str, ...], compute_capability: int
) -> Build:
  extra_gpu_tags = _tag_filters_for_compute_capability(compute_capability)
  return Build(
      type_=type_,
      repo="openxla/xla",
      docker_image=_CUDNN_9_IMAGE,
      target_patterns=_XLA_DEFAULT_TARGET_PATTERNS,
      configs=configs,
      tag_filters=("-no_oss", "requires-gpu-nvidia") + extra_gpu_tags,
      options=dict(
          run_under="//tools/ci_build/gpu_build:parallel_gpu_execute",
          repo_env=f"TF_CUDA_COMPUTE_CAPABILITIES={compute_capability/10}",
          **_DEFAULT_BAZEL_OPTIONS,
      ),
  )


_CPU_X86_BUILD = Build(
    type_=BuildType.CPU_X86,
    repo="openxla/xla",
    docker_image=_DEFAULT_IMAGE,
    configs=("warnings", "nonccl", "rbe_linux_cpu"),
    target_patterns=_XLA_DEFAULT_TARGET_PATTERNS + ("-//xla/service/gpu/...",),
    tag_filters=(
        "-no_oss",
        "-gpu",
        "-requires-gpu-nvidia",
        "-requires-gpu-amd",
    ),
    options=_DEFAULT_BAZEL_OPTIONS,
)
_CPU_ARM64_BUILD = Build(
    type_=BuildType.CPU_ARM64,
    repo="openxla/xla",
    docker_image=_ARM64_JAX_MULTI_PYTHON_IMAGE,
    configs=("warnings", "rbe_cross_compile_linux_arm64_xla", "nonccl"),
    target_patterns=_XLA_DEFAULT_TARGET_PATTERNS + ("-//xla/service/gpu/...",),
    tag_filters=(
        "-no_oss",
        "-gpu",
        "-requires-gpu-nvidia",
        "-not_run:arm",
        "-requires-gpu-amd",
    ),
    options={**_DEFAULT_BAZEL_OPTIONS, "build_tests_only": True},
)
# TODO(ddunleavy): Setup additional build for a100 tests once L4 RBE is ready.
_GPU_BUILD = nvidia_gpu_build_with_compute_capability(
    type_=BuildType.GPU,
    configs=("warnings", "rbe_linux_cuda_nvcc"),
    compute_capability=75,
)

_JAX_CPU_BUILD = Build(
    type_=BuildType.JAX_CPU,
    repo="google/jax",
    docker_image=_DEFAULT_IMAGE,
    configs=(
        "avx_posix",
        "mkl_open_source_only",
        "rbe_cpu_linux_py3.12",
        "tensorflow_testing_rbe_linux",
    ),
    target_patterns=("//tests:cpu_tests", "//tests:backend_independent_tests"),
    test_env=dict(
        JAX_NUM_GENERATED_CASES=25,
        JAX_SKIP_SLOW_TESTS=1,
    ),
    options=_DEFAULT_BAZEL_OPTIONS,
)

_JAX_GPU_BUILD = Build(
    type_=BuildType.JAX_GPU,
    repo="google/jax",
    docker_image=_DEFAULT_IMAGE,
    configs=(
        "avx_posix",
        "mkl_open_source_only",
        "rbe_linux_cuda12.3_nvcc_py3.9",
        "tensorflow_testing_rbe_linux",
    ),
    target_patterns=("//tests:gpu_tests", "//tests:backend_independent_tests"),
    tag_filters=("-multiaccelerator",),
    test_env=dict(
        JAX_SKIP_SLOW_TESTS=1,
        TF_CPP_MIN_LOG_LEVEL=0,
        JAX_EXCLUDE_TEST_TARGETS="PmapTest.testSizeOverflow",
    ),
    options=_DEFAULT_BAZEL_OPTIONS,
)

_KOKORO_JOB_NAME_TO_BUILD_MAP = {
    "tensorflow/xla/linux/arm64/build_cpu": _CPU_ARM64_BUILD,
    "tensorflow/xla/linux/cpu/build_cpu": _CPU_X86_BUILD,
    "tensorflow/xla/linux/gpu/build_gpu": _GPU_BUILD,
    "tensorflow/xla/linux/github_continuous/arm64/build_cpu": _CPU_ARM64_BUILD,
    "tensorflow/xla/linux/github_continuous/build_gpu": _GPU_BUILD,
    "tensorflow/xla/linux/github_continuous/build_cpu": _CPU_X86_BUILD,
    "tensorflow/xla/jax/cpu/build_cpu": _JAX_CPU_BUILD,
    "tensorflow/xla/jax/gpu/build_gpu": _JAX_GPU_BUILD,
}


def main():
  logging.basicConfig()
  logging.getLogger().setLevel(logging.INFO)
  kokoro_job_name = os.getenv("KOKORO_JOB_NAME")
  build = _KOKORO_JOB_NAME_TO_BUILD_MAP[kokoro_job_name]

  sh(["./github/xla/.kokoro/generate_index_html.sh", "index.html"])

  _, repo_name = build.repo.split("/")
  if build.repo != "openxla/xla":
    sh([
        "git",
        "clone",
        "--depth=1",
        f"https://github.com/{build.repo}",
        f"./github/{repo_name}",
    ])

  # TODO(b/338885148): Remove this block after TF was updated to cuDNN 9
  if build.type_ in (BuildType.GPU, BuildType.GPU_CONTINUOUS):
    sh(
        [
            "sed",
            "-i",
            r"s/@sigbuild-r2\.17-clang_/@sigbuild-r2.17-clang-cudnn9_/g",
            "github/xla/.bazelrc",
        ],
    )
    sh(["nvidia-smi"])

  with build.docker_image.pull_and_run(
      workdir=f"/github/{repo_name}", **_DEFAULT_DOCKER_OPTIONS
  ) as docker_exec:
    docker_exec(build.bazel_test_command())
    docker_exec(["bazel", "analyze-profile", "profile.json.gz"])


if __name__ == "__main__":
  main()
