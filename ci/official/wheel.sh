#!/bin/bash
# -e: abort script if one command fails
# -u: error if undefined variable used
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
set -euxo pipefail -o history
set -o allexport && source "$TFCI" && set +o allexport

cd "$TFCI_GIT_DIR" && mkdir -p build
tfrun() { "$@"; }
[[ "$TFCI_COPYBARA_ENABLE" == 1 ]] && source ./ci/official/utilities/copybara.sh
[[ "$TFCI_DOCKER_ENABLE" == 1 ]] && source ./ci/official/utilities/docker.sh
./ci/official/utilities/generate_index_html.sh build/index.html

# Record GPU count and CUDA version status
[[ "$TFCI_NVIDIA_SMI_ENABLE" == 1 ]] && tfrun nvidia-smi

# Update the version numbers for Nightly only
[[ "$TFCI_NIGHTLY_UPDATE_VERSION_ENABLE" == 1 ]] && tfrun python3 tensorflow/tools/ci_build/update_version.py --nightly

tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" build "${TFCI_BAZEL_CACHE_ARGS[@]}" //tensorflow/tools/pip_package:build_pip_package
tfrun ./bazel-bin/tensorflow/tools/pip_package/build_pip_package build "${TFCI_BUILD_PIP_PACKAGE_ARGS[@]}"
tfrun ./ci/official/utilities/rename_and_verify_wheels.sh build

if [[ "$TFCI_UPLOAD_ENABLE" == 1 ]]; then
  twine upload "${TFCI_UPLOAD_PYPI_ARGS[@]}" build/*.whl
  gsutil cp build/*.whl "$TFCI_UPLOAD_GCS_DESTINATION"
fi

tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" test "${TFCI_BAZEL_CACHE_ARGS[@]}" --config=nonpip
