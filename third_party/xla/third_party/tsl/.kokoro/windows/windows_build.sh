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
set -euo pipefail

cd "${KOKORO_ARTIFACTS_DIR}/github/tsl"

# Generate a templated results file to make output accessible to everyone
"$KOKORO_ARTIFACTS_DIR"/github/tsl/.kokoro/generate_index_html.sh "$KOKORO_ARTIFACTS_DIR"/index.html

function is_continuous_job() {
  [[ "$KOKORO_JOB_NAME" =~ tensorflow/tsl/.*continuous.* ]]
}

ADDITIONAL_FLAGS=""
TAGS_FILTER="-no_oss,-oss_excluded,-gpu,-no_windows,-windows_excluded"

if is_continuous_job ; then
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --google_default_credentials"
else
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --remote_upload_local_results=false"
fi

export PATH="$PATH:/c/Python38"

# Build TSL
/c/tools/bazel.exe build \
  --output_filter="" \
  --keep_going \
  --build_tag_filters=$TAGS_FILTER  \
  --test_tag_filters=$TAGS_FILTER \
  --remote_cache="https://storage.googleapis.com/tensorflow-devinfra-bazel-cache/tsl/windows" \
    $ADDITIONAL_FLAGS \
  -- //tsl/... \
  || { echo "Bazel Build Failed" && exit 1; }

# Test TSL TODO(ddunleavy) enable all tests
/c/tools/bazel.exe test \
  --output_filter="" \
  --flaky_test_attempts=3 \
  --test_output=errors \
  --build_tests_only \
  --verbose_failures=true \
  --build_tag_filters=$TAGS_FILTER  \
  --test_tag_filters=$TAGS_FILTER \
  --keep_going \
  -- //tsl/... -//tsl/platform:subprocess_test -//tsl/platform/cloud:google_auth_provider_test -//tsl/platform/cloud:oauth_client_test \
  || { echo "Bazel Test Failed" && exit 1; }

exit 0
