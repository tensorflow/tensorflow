# Copyright 2019 The OpenXLA Authors.
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
import sys
import tempfile

import lit.formats

# copybara:uncomment_begin(google-only)
# from xla.lit_google_cfg import ENV_FLAGS as google_env_flags
# copybara:uncomment_end

# pylint: disable=undefined-variable

extra_env_flags = []

# copybara:uncomment_begin(google-only)
# extra_env_flags += google_env_flags
# copybara:uncomment_end

config.name = "XLA"
config.suffixes = [".cc", ".hlo", ".json", ".mlir", ".pbtxt", ".py"]

config.test_format = lit.formats.ShTest(execute_external=True)


for env in [
    # Passthrough XLA_FLAGS.
    "XLA_FLAGS",
    # Propagate environment variables used by 'bazel coverage'.
    # These are exported by tools/coverage/collect_coverage.sh
    "BULK_COVERAGE_RUN",
    "COVERAGE",
    "COVERAGE_DIR",
    "COVERAGE_MANIFEST",
    "LLVM_PROFILE_FILE",
    "LLVM_COVERAGE_FILE",
    "GCOV_PREFIX",
    "GCOV_PREFIX_STRIP",
] + extra_env_flags:
  value = os.environ.get(env)
  if value:
    config.environment[env] = value


# Use the most preferred temp directory.
config.test_exec_root = (
    os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    or os.environ.get("TEST_TMPDIR")
    or os.path.join(tempfile.gettempdir(), "lit")
)

config.substitutions.extend([
    ("%PYTHON", os.getenv("PYTHON", sys.executable) or ""),
])

if lit_config.params.get("PTX") == "GCN":
  config.available_features.add("IS_ROCM")


# Include additional substitutions that may be defined via params
config.substitutions.extend(
    ("%%{%s}" % key, val) for key, val in lit_config.params.items()
)
