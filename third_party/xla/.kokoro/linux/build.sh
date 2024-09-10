#!/bin/bash
# Copyright 2022 Google LLC All Rights Reserved.
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

# -e: abort script if one command fails
# -u: error if undefined variable used
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history

set -euox pipefail -o history

capture_test_logs() {
  mkdir -p "$KOKORO_ARTIFACTS_DIR"
  pwd; ls -l github/xla/
  # copy all test.log and test.xml files to the kokoro artifacts directory
  sudo find -L github/xla/bazel-testlogs \( -name "test.log" -o -name "test.xml" \) -exec cp --parents {} "$KOKORO_ARTIFACTS_DIR" \;
  # Rename the copied test.log and test.xml files to sponge_log.log and sponge_log.xml
  sudo find -L "$KOKORO_ARTIFACTS_DIR" -name "test.log" -exec rename 's/test.log/sponge_log.log/' {} \;
  sudo find -L "$KOKORO_ARTIFACTS_DIR" -name "test.xml" -exec rename 's/test.xml/sponge_log.xml/' {} \;

  sudo find -L github/xla/bazel-testlogs -type f -printf "%T@ %p\n"
}

# Run capture_test_logs when the script exits
trap capture_test_logs EXIT

"$KOKORO_ARTIFACTS_DIR"/github/xla/build_tools/ci/build.py
