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

# Generate a templated results file to make output accessible to everyone
"$KOKORO_ARTIFACTS_DIR"/github/xla/.kokoro/generate_index_html.sh "$KOKORO_ARTIFACTS_DIR"/index.html

function is_linux_gpu_job() {
  [[ "$KOKORO_JOB_NAME" =~ tensorflow/xla/linux/.*gpu.* ]]
}

function is_linux_cpu_arm64_job() {
  [[ "$KOKORO_JOB_NAME" =~ tensorflow/xla/linux/.*arm64.*/.*cpu.* ]]
}

function pull_docker_image_with_retries() {
  # Pull the container (in case it was updated since the instance started) and
  # store its SHA in the Sponge log.
  docker pull "$DOCKER_IMAGE" || sleep 15
  docker pull "$DOCKER_IMAGE" || sleep 15
  docker pull "$DOCKER_IMAGE"
  echo "TF_INFO_DOCKER_IMAGE,$DOCKER_IMAGE" >> "$KOKORO_ARTIFACTS_DIR/custom_sponge_config.csv"
  echo "TF_INFO_DOCKER_SHA,$(docker pull "$DOCKER_IMAGE" | sed -n '/Digest:/s/Digest: //g p')" >> "$KOKORO_ARTIFACTS_DIR/custom_sponge_config.csv"
}

pull_docker_image_with_retries
# Start a container in the background
docker run --name xla -w /tf/xla -itd --rm \
    -v "$KOKORO_ARTIFACTS_DIR/github/xla:/tf/xla" \
    -v "$KOKORO_ARTIFACTS_DIR/pkg:/tf/pkg" \
    "$DOCKER_IMAGE" \
    bash

TAGS_FILTER="-no_oss,-oss_excluded,-oss_serial"
ADDITIONAL_FLAGS=""
RBE_FLAGS=""
TARGET_FILTERS="-@local_tsl//tsl/platform:subprocess_test -@local_tsl//tsl/platform/cloud:google_auth_provider_test -@local_tsl//tsl/platform/cloud:oauth_client_test"

if is_linux_gpu_job ; then
    TAGS_FILTER="$TAGS_FILTER,gpu,requires-gpu-nvidia,-no_gpu"

    # We are currently running XLA presubmits on machines with NVIDIA T4 GPUs,
    # which have a compute compatibility of 7.5. Se we filter out all the tests
    # that need a newer GPU:
    UNSUPPORTED_GPU_TAGS="$(echo -requires-gpu-sm{80,86,89,90}{,-only})"
    TAGS_FILTER="${TAGS_FILTER},${UNSUPPORTED_GPU_TAGS// /,}"

    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --nobuild_tests_only --run_under=//tools/ci_build/gpu_build:parallel_gpu_execute"
    RBE_FLAGS="--config=rbe_linux_cuda_nvcc --jobs=150"
    echo "***NOTE: nvidia-smi lists the highest CUDA version the driver supports, which may be different than the version of CUDA actually used!!***"
    nvidia-smi
else
    TAGS_FILTER="$TAGS_FILTER,-gpu,-requires-gpu-nvidia"
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --config=nonccl"

    if is_linux_cpu_arm64_job ; then
        TAGS_FILTER="$TAGS_FILTER,-no_aarch64"
        ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --action_env PYTHON_BIN_PATH=/usr/bin/python3.11 --python_path=/usr/bin/python3.11"
        # Some cross-compile tests are not working for XLA Linux Aarch64.
        # TODO(ddunleavy): Revisit these when hermetic python is available.
        TARGET_FILTERS="$TARGET_FILTERS -//xla/python_api:xla_shape_test -//xla/python_api:xla_literal_test -//xla/service:xla_aot_compile_stablehlo_cpu_test -//xla/tests:local_client_aot_test"
        RBE_FLAGS="--config=rbe_cross_compile_linux_arm64_xla --jobs=150"
    else
        RBE_FLAGS="--config=rbe_linux_cpu --jobs=150"
        ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --nobuild_tests_only"
    fi
fi

# Build & test XLA
docker exec xla bazel \
        test \
        --build_tag_filters=$TAGS_FILTER  \
        --test_tag_filters=$TAGS_FILTER \
        --test_output=errors \
        --keep_going \
        --features=layering_check \
        --profile=/tf/pkg/profile.json.gz \
        --flaky_test_attempts=3 \
        --config=warnings \
        $RBE_FLAGS \
        $ADDITIONAL_FLAGS \
        -- //xla/... //build_tools/... @local_tsl//tsl/... $TARGET_FILTERS


# Print build time statistics, including critical path.
docker exec xla bazel analyze-profile "/tf/pkg/profile.json.gz"

# Stop container
docker stop xla


