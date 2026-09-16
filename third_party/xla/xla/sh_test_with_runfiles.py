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

      test_exec_dir = pathlib.Path(test.getExecPath()).parent
      test_exec_dir.mkdir(parents=True, exist_ok=True)

      # Symlink the entire runfiles structure to match what RUNPATH expects
      # Binaries have RUNPATH like $ORIGIN/../../../../../_solib_x86_64/...
      # They execute from lit_bin which is at runfiles/_main/xla/.../lit_bin/
      # So we need _main accessible from there
      # Symlink to test_exec_dir.parent to match RUNPATH patterns
      for item in rf_path.iterdir():
        if item.is_dir():
          test_exec_symlink = test_exec_dir.parent / item.name
          if not test_exec_symlink.exists():
            try:
              # Use relative symlinks for portability between local and RBE
              relative_target = os.path.relpath(item, test_exec_dir.parent)
              test_exec_symlink.symlink_to(
                  relative_target, target_is_directory=True
              )
              created_symlinks.append(test_exec_symlink)
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

      # For --dynamic_mode=fully, set LD_LIBRARY_PATH to point to
      # library directories.
      # Support both bzlmod and workspace mode layouts
      lib_dirs = []

      for candidate_root in [rf_path / "_main", rf_path / "xla", rf_path]:
        if candidate_root.is_dir():
          for solib in candidate_root.glob("_solib_*"):
            if solib.is_dir():
              lib_dirs.append(str(solib))

      if lib_dirs:
        existing_ld_path = test.config.environment.get("LD_LIBRARY_PATH", "")
        new_ld_path = ":".join(lib_dirs)
        if existing_ld_path:
          new_ld_path = f"{new_ld_path}:{existing_ld_path}"
        test.config.environment["LD_LIBRARY_PATH"] = new_ld_path

    result = super().execute(test, lit_config)

    # Clean up created symlinks
    for target in created_symlinks:
      try:
        target.unlink()
      except FileNotFoundError:
        pass

    return result
