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
import dataclasses
import enum
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_KW_ONLY_IF_PYTHON310 = {"kw_only": True} if sys.version_info >= (3, 10) else {}

# TODO(ddunleavy): move this to the bazelrc
_DEFAULT_BAZEL_OPTIONS = dict(
    test_output="errors",
    keep_going=True,
    nobuild_tests_only=True,
    features="layering_check",
    profile="profile.json.gz",
    flaky_test_attempts=3,
    jobs=150,
)

_DEFAULT_TARGET_PATTERNS = ("//xla/...", "//build_tools/...", "@local_tsl//tsl/...")


class BuildType(enum.Enum):
  CPU_X86 = enum.auto()
  CPU_ARM64 = enum.auto()
  GPU = enum.auto()
  GPU_CONTINUOUS = enum.auto()


@dataclasses.dataclass(frozen=True, **_KW_ONLY_IF_PYTHON310)
class Build:
  """Class representing a build of XLA."""

  type_: BuildType
  repo: str
  configs: Tuple[str, ...] = ()
  target_patterns: Tuple[str, ...] = _DEFAULT_TARGET_PATTERNS
  tag_filters: Tuple[str, ...] = ()
  options: Dict[str, Any] = dataclasses.field(default_factory=dict)
  docker_image: Optional[str] = None

  def bazel_test_command(self) -> List[str]:
    # pylint: disable=g-bool-id-comparison
    options = [
        f"--{k}" if v is True else f"--{k}={v}" for k, v in self.options.items()
    ]
    configs = [f"--config={config}" for config in self.configs]
    build_tag_filters = f"--build_tag_filters={','.join(self.tag_filters)}"
    test_tag_filters = f"--test_tag_filters={','.join(self.tag_filters)}"
    all_options = [build_tag_filters, test_tag_filters] + configs + options
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


def nvidia_gpu_build_with_compute_capability(
    *, type_: BuildType, compute_capability: int
) -> Build:
  extra_gpu_tags = _tag_filters_for_compute_capability(compute_capability)
  return Build(
      type_=type_,
      repo="xla",
      configs=("warnings", "rbe_linux_cuda_nvcc"),
      tag_filters=("-no_oss", "requires-gpu-nvidia") + extra_gpu_tags,
      options=dict(
          run_under="//tools/ci_build/gpu_build:parallel_gpu_execute",
          **_DEFAULT_BAZEL_OPTIONS,
      ),
      # TODO(b/338885148): Remove this once the TF containers have cuDNN 9
      docker_image="gcr.io/tensorflow-sigs/build@sha256:dddcaf30321e9007103dce75c51b83fea3c06de462fcf41e7c6ae93f37fc3545",
  )


_CPU_X86_BUILD = Build(
    type_=BuildType.CPU_X86,
    repo="xla",
    configs=("warnings", "nonccl", "rbe_linux_cpu"),
    target_patterns=_DEFAULT_TARGET_PATTERNS + ("-//xla/service/gpu/...",),
    tag_filters=(
        "-no_oss",
        "-gpu",
        "-requires-gpu-nvidia",
        "-requires-gpu-amd",
    ),
    options=_DEFAULT_BAZEL_OPTIONS,
    docker_image="gcr.io/tensorflow-sigs/build:latest-python3.11",
)
_CPU_ARM64_BUILD = Build(
    type_=BuildType.CPU_ARM64,
    repo="xla",
    configs=("warnings", "rbe_cross_compile_linux_arm64_xla", "nonccl"),
    target_patterns=_DEFAULT_TARGET_PATTERNS + ("-//xla/service/gpu/...",),
    tag_filters=(
        "-no_oss",
        "-gpu",
        "-requires-gpu-nvidia",
        "-not_run:arm",
        "-requires-gpu-amd",
    ),
    options={**_DEFAULT_BAZEL_OPTIONS, "build_tests_only": True},
    docker_image="us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/build-arm64:jax-latest-multi-python",
)
_GPU_BUILD = nvidia_gpu_build_with_compute_capability(
    type_=BuildType.GPU, compute_capability=75
)
_GPU_CONTINUOUS_BUILD = nvidia_gpu_build_with_compute_capability(
    type_=BuildType.GPU_CONTINUOUS, compute_capability=80
)

_KOKORO_JOB_NAME_TO_BUILD_MAP = {
    "tensorflow/xla/linux/arm64/build_cpu": _CPU_ARM64_BUILD,
    "tensorflow/xla/linux/cpu/build_cpu": _CPU_X86_BUILD,
    "tensorflow/xla/linux/gpu/build_gpu": _GPU_BUILD,
    "tensorflow/xla/linux/github_continuous/build_gpu": _GPU_CONTINUOUS_BUILD,
    "tensorflow/xla/linux/github_continuous/build_cpu": _CPU_X86_BUILD,
}


def _write_to_sponge_config(key, value) -> None:
  with open("custom_sponge_config.csv", "a") as f:
    f.write(f"{key},{value}\n")


def sh(args, check=True, **kwargs):
  logging.info("Starting process: %s", " ".join(args))
  return subprocess.run(args, check=check, **kwargs)


def _pull_docker_image_with_retries(image_url: str, retries=3) -> None:
  """Pulls docker image with retries to avoid transient rate limit errors."""
  for _ in range(retries):
    pull_proc = sh(["docker", "pull", image_url], check=False)
    if pull_proc.returncode != 0:
      time.sleep(15)

  # write SHA of image to the sponge config
  _write_to_sponge_config("TF_INFO_DOCKER_IMAGE", image_url)

  pull_proc = sh(["docker", "pull", image_url], capture_output=True, text=True)
  # TODO(ddunleavy): get sha
  # _write_to_sponge_config("TF_INFO_DOCKER_SHA", sha)


def main():
  logging.basicConfig()
  logging.getLogger().setLevel(logging.INFO)
  kokoro_job_name = os.getenv("KOKORO_JOB_NAME")
  build = _KOKORO_JOB_NAME_TO_BUILD_MAP[kokoro_job_name]

  sh(["./github/xla/.kokoro/generate_index_html.sh", "index.html"])

  _pull_docker_image_with_retries(build.docker_image)

  sh(
      # pyformat: disable
      [
          "docker", "run",
          "--name", "xla",
          "-w", f"/github/{build.repo}",
          "-itd",
          "--rm",
          "-v", "./github:/github",
          build.docker_image,
          "bash",
      ],
      # pyformat: enable
  )

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

  sh(["docker", "exec", "xla", *build.bazel_test_command()])

  sh(["docker", "exec", "xla", "bazel", "analyze-profile", "profile.json.gz"])

  sh(["docker", "stop", "xla"])


if __name__ == "__main__":
  main()
