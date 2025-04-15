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
# Note: set -x <code> +x around anything you want to have logged.
set -euox pipefail

# Generate a templated results file to make output accessible to everyone
"$KOKORO_ARTIFACTS_DIR"/github/xla/.kokoro/generate_index_html.sh "$KOKORO_ARTIFACTS_DIR"/index.html

cd "${KOKORO_ARTIFACTS_DIR}/github/xla"

export PATH="$PATH:/c/Python38"

TARGET_FILTER="-//xla/hlo/experimental/... -//xla/python_api/... -//xla/python/..."
TAGS_FILTER="-no_oss,-oss_excluded,-gpu,-no_windows,-windows_excluded"
/c/tools/bazel.exe test \
  --keep_going \
  --build_tag_filters=$TAGS_FILTER  --test_tag_filters=$TAGS_FILTER \
  -- //xla/... $TARGET_FILTER |& grep -v "violates visibility of" \
  || { exit 1; }

exit 0
