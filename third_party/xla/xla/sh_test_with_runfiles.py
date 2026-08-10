# Copyright 2025 The OpenXLA Authors.
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
"""Lit runner configuration."""

import os
import pathlib

import lit.formats


class ShTestWithRunfiles(lit.formats.ShTest):
  """Used to symlink bazels runfiles subdirs into the lit tmp test dir."""

  def execute(self, test, lit_config):
    runfiles_env = os.environ.get("RUNFILES_DIR")
    created_symlinks = []
    if runfiles_env:
      rf_path = pathlib.Path(runfiles_env)
      runfiles_dir = rf_path / "xla"
      if runfiles_dir.is_dir():
        dst = pathlib.Path(test.getExecPath()).parent
        dst.mkdir(parents=True, exist_ok=True)
        for item in runfiles_dir.iterdir():
          target = dst / item.name
          try:
            target.symlink_to(item, target_is_directory=item.is_dir())
            created_symlinks.append(target)
          except FileExistsError:
            pass

      # Dynamically resolve hermetic cuda_nvcc in runfiles if present.
      # Static relative paths (e.g. %S/../../..) break for deeply nested targets
      # or under Bazel 8 Bzlmod runfiles layouts. Instead, we dynamically locate
      # the runfiles directory containing `bin/ptxas`.
      cuda_dir = None
      # Fast-path check for standard runfiles repository locations
      for candidate in [
          rf_path / "cuda_nvcc",
          rf_path / "xla" / "cuda_nvcc",
          rf_path.parent / "cuda_nvcc",
      ]:
        if (candidate / "bin" / "ptxas").is_file():
          cuda_dir = candidate
          break
      # Fallback search for Bzlmod mangled repository names
      # (e.g. rules_cuda~...~cuda_nvcc)
      if not cuda_dir and rf_path.is_dir():
        for p in rf_path.rglob("*cuda_nvcc*"):
          if (p / "bin" / "ptxas").is_file():
            cuda_dir = p
            break

      # Inject --xla_gpu_cuda_data_dir into XLA_FLAGS if resolved
      if cuda_dir:
        existing_flags = test.config.environment.get("XLA_FLAGS", "")
        if "--xla_gpu_cuda_data_dir" not in existing_flags:
          flag = f"--xla_gpu_cuda_data_dir={cuda_dir}"
          test.config.environment["XLA_FLAGS"] = (
              f"{existing_flags} {flag}".strip()
          )

    result = super().execute(test, lit_config)
    for target in created_symlinks:
      target.unlink()
    return result
