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
set -euo pipefail -o history

# Generate a templated results file to make output accessible to everyone
"$KOKORO_ARTIFACTS_DIR"/github/tsl/.kokoro/generate_index_html.sh "$KOKORO_ARTIFACTS_DIR"/index.html

function is_continuous_job() {
  [[ "$KOKORO_JOB_NAME" =~ tensorflow/tsl/.*continuous.* ]]
}

ADDITIONAL_FLAGS=""
TAGS_FILTER="-no_oss,-oss_excluded,-oss_serial,-gpu,-requires-gpu-nvidia"

if is_continuous_job ; then
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --google_default_credentials"
else
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --remote_upload_local_results=false"
fi

# Pull the container (in case it was updated since the instance started) and
# store its SHA in the Sponge log.
docker pull "$DOCKER_IMAGE"
echo "TF_INFO_DOCKER_IMAGE,$DOCKER_IMAGE" >> "$KOKORO_ARTIFACTS_DIR/custom_sponge_config.csv"
echo "TF_INFO_DOCKER_SHA,$(docker pull "$DOCKER_IMAGE" | sed -n '/Digest:/s/Digest: //g p')" >> "$KOKORO_ARTIFACTS_DIR/custom_sponge_config.csv"

# Start a container in the background
docker run --name tsl -w /tf/tsl -itd --rm \
    -v "$KOKORO_ARTIFACTS_DIR/github/tsl:/tf/tsl" \
    "$DOCKER_IMAGE" \
    bash

# Build TSL
docker exec tsl bazel --bazelrc=/usertools/cpu.bazelrc build \
    --output_filter="" \
    --keep_going \
    --build_tag_filters=$TAGS_FILTER  \
    --test_tag_filters=$TAGS_FILTER \
    --remote_cache="https://storage.googleapis.com/tensorflow-devinfra-bazel-cache/tsl/linux" \
    $ADDITIONAL_FLAGS \
    -- //tsl/...

# Test TSL
docker exec tsl bazel --bazelrc=/usertools/cpu.bazelrc test \
    --output_filter="" \
    --keep_going \
    --flaky_test_attempts=3 \
    --test_output=errors \
    --build_tests_only \
    --build_tag_filters=$TAGS_FILTER  \
    --test_tag_filters=$TAGS_FILTER \
    --verbose_failures=true \
    -- //tsl/...

# Stop container
docker stop tsl
