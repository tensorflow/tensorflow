#!/bin/bash
source "${BASH_SOURCE%/*}/utilities/setup.sh"

# Record GPU count and CUDA version status
if [[ "$TFCI_NVIDIA_SMI_ENABLE" == 1 ]]; then
  tfrun nvidia-smi
fi

# Update the version numbers for Nightly only
if [[ "$TFCI_NIGHTLY_UPDATE_VERSION_ENABLE" == 1 ]]; then
  tfrun python3 tensorflow/tools/ci_build/update_version.py --nightly
fi

tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" test "${TFCI_BAZEL_COMMON_ARGS[@]}" --config=libtensorflow_test
tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" build "${TFCI_BAZEL_COMMON_ARGS[@]}" --config=libtensorflow_build

tfrun ./ci/official/utilities/repack_libtensorflow.sh build "$TFCI_LIB_SUFFIX"

if [[ "$TFCI_UPLOAD_LIB_ENABLE" == 1 ]]; then
  gsutil cp build/*.tar.gz "$TFCI_UPLOAD_LIB_GCS_URI"
  if [[ "$TFCI_UPLOAD_LIB_LATEST_ENABLE" == 1 ]]; then
    gsutil cp build/*.tar.gz "$TFCI_UPLOAD_LIB_LATEST_GCS_URI"
  fi
fi
