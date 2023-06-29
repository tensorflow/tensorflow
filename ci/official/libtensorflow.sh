#!/bin/bash
# -e: abort script if one command fails
# -u: error if undefined variable used
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
set -euxo pipefail -o history
set -o allexport && source "$TFCI" && set +o allexport

cd "$TFCI_GIT_DIR" && mkdir -p build
tfrun() { "$@"; }
[[ "$TFCI_COPYBARA_ENABLE" = 1 ]] && source ./ci/official/utilities/copybara.sh
[[ "$TFCI_DOCKER_ENABLE" = 1 ]] && source ./ci/official/utilities/docker.sh
./ci/official/utilities/generate_index_html.sh build/index.html

# Record GPU count and CUDA version status
[[ "$TFCI_NVIDIA_SMI_ENABLE" = 1 ]] && tfrun nvidia-smi

# Update the version numbers for Nightly only
[[ "$TFCI_NIGHTLY_UPDATE_VERSION_ENABLE" = 1 ]] && tfrun python3 tensorflow/tools/ci_build/update_version.py --nightly

tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" test "${TFCI_BAZEL_CACHE_ARGS[@]}" --config=libtensorflow_test
tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" build "${TFCI_BAZEL_CACHE_ARGS[@]}" --config=libtensorflow_build

tfrun ./ci/official/utilities/repack_libtensorflow.sh build "$TFCI_LIB_SUFFIX"

if [[ "$TFCI_UPLOAD_LIB_ENABLE" = 1 ]]; then
  gsutil cp build/*.tar.gz "$TFCI_UPLOAD_LIB_GCS_URI"
  if [[ "$TFCI_UPLOAD_LIB_LATEST_ENABLE" = 1 ]]; then
    gsutil cp build/*.tar.gz "$TFCI_UPLOAD_LIB_LATEST_GCS_URI"
  fi
fi
