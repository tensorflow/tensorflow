#!/bin/bash
# -e: abort script if one command fails
# -u: error if undefined variable used
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
set -euxo pipefail -o history
set -o allexport && source "$TFCI" && set +o allexport

# If this is a CL presubmit, then run Copybara on the Piper code and place it
# in the same directory as the GitHub source code would normally be. This lets
# the rest of the script proceed as normal.
_TFCI_HOST_ARTIFACTS_DIR="$TFCI_RUNTIME_ARTIFACTS_DIR"
tfrun() { "$@"; }
[[ "$TFCI_COPYBARA_ENABLE" = 1 ]] && source $TFCI_RUNTIME_USERTOOLS_DIR/copybara.sh
[[ "$TFCI_DOCKER_ENABLE" = 1 ]] && source $TFCI_RUNTIME_USERTOOLS_DIR/docker.sh
"$TFCI_RUNTIME_USERTOOLS_DIR/generate_index_html.sh" "$TFCI_RUNTIME_ARTIFACTS_DIR/index.html"

# Record GPU count and CUDA version status
[[ "$TFCI_NVIDIA_SMI_ENABLE" = 1 ]] && tfrun nvidia-smi

# Update the version numbers for Nightly only
[[ "$TFCI_NIGHTLY_UPDATE_VERSION_ENABLE" = 1 ]] && tfrun python3 tensorflow/tools/ci_build/update_version.py --nightly

tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" build "${TFCI_BAZEL_CACHE_ARGS[@]}" tensorflow/tools/pip_package:build_pip_package
tfrun ./bazel-bin/tensorflow/tools/pip_package/build_pip_package "$TFCI_RUNTIME_ARTIFACTS_DIR" "${TFCI_BUILD_PIP_PACKAGE_ARGS[@]}"
tfrun "$TFCI_RUNTIME_USERTOOLS_DIR/rename_and_verify_wheels.sh"

if [[ "$TFCI_UPLOAD_ENABLE" = 1 ]]; then
  twine upload "${TFCI_UPLOAD_PYPI_ARGS[@]}" "$_TFCI_HOST_ARTIFACTS_DIR"/*.whl
  gsutil cp "$_TFCI_HOST_ARTIFACTS_DIR"/*.whl "$TFCI_UPLOAD_GCS_DESTINATION"
fi

tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" test "${TFCI_BAZEL_CACHE_ARGS[@]}" --config=nonpip
