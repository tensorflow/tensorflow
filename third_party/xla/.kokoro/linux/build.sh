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

# TODO(b/338885148): Remove this once the TF containers have cuDNN 9
if is_linux_gpu_job ; then
  DOCKER_IMAGE="gcr.io/tensorflow-sigs/build@sha256:dddcaf30321e9007103dce75c51b83fea3c06de462fcf41e7c6ae93f37fc3545"
fi

pull_docker_image_with_retries


# Start a container in the background
docker run --name xla -w /github/xla -itd --rm \
    -v "./github:/github" \
    "$DOCKER_IMAGE" \
    bash

TAGS_FILTER="-no_oss"
ADDITIONAL_FLAGS=""
RBE_FLAGS=""
TARGET_FILTERS=""

if is_linux_gpu_job ; then
    TAGS_FILTER="$TAGS_FILTER,requires-gpu-nvidia,-requires-gpu-amd"

    # We are currently running XLA presubmits on machines with NVIDIA T4 GPUs,
    # which have a compute compatibility of 7.5. Se we filter out all the tests
    # that need a newer GPU:
    UNSUPPORTED_GPU_TAGS="$(echo -requires-gpu-sm{80,86,89,90}{,-only})"
    TAGS_FILTER="${TAGS_FILTER},${UNSUPPORTED_GPU_TAGS// /,}"

    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --run_under=//tools/ci_build/gpu_build:parallel_gpu_execute"
    RBE_FLAGS="--config=rbe_linux_cuda_nvcc --jobs=150"
    (
      #TODO(b/338885148): Remove this block after TF was updated to cuDNN 9
      pushd github/xla
      sed -i 's/@sigbuild-r2\.17-clang_/@sigbuild-r2.17-clang-cudnn9_/g' .bazelrc
      echo "The following changes were made:"
      git diff -- .bazelrc || true
      popd
    )
    echo "***NOTE: nvidia-smi lists the highest CUDA version the driver supports, which may be different than the version of CUDA actually used!!***"
    nvidia-smi
else
    TAGS_FILTER="$TAGS_FILTER,-gpu,-requires-gpu-nvidia,-requires-gpu-amd"
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --config=nonccl"
    TARGET_FILTERS="$TARGET_FILTERS -//xla/service/gpu/..."

    if is_linux_cpu_arm64_job ; then
        TAGS_FILTER="$TAGS_FILTER,-not_run:arm"
        RBE_FLAGS="--config=rbe_cross_compile_linux_arm64_xla --jobs=150"
    else
        RBE_FLAGS="--config=rbe_linux_cpu --jobs=150"
    fi
fi

# Build & test XLA
docker exec xla bazel \
        test \
        --build_tag_filters=$TAGS_FILTER \
        --test_tag_filters=$TAGS_FILTER \
        --test_output=errors \
        --keep_going \
        --nobuild_tests_only \
        --features=layering_check \
        --profile=profile.json.gz \
        --flaky_test_attempts=3 \
        --config=warnings \
        $RBE_FLAGS \
        $ADDITIONAL_FLAGS \
        -- //xla/... //build_tools/... @local_tsl//tsl/... $TARGET_FILTERS


# Print build time statistics, including critical path.
docker exec xla bazel analyze-profile profile.json.gz

# Stop container
docker stop xla


