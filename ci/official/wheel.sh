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

tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" build "${TFCI_BAZEL_COMMON_ARGS[@]}" //tensorflow/tools/pip_package:build_pip_package
tfrun ./bazel-bin/tensorflow/tools/pip_package/build_pip_package build "${TFCI_BUILD_PIP_PACKAGE_ARGS[@]}"
tfrun ./ci/official/utilities/rename_and_verify_wheels.sh build

if [[ "$TFCI_UPLOAD_ENABLE" == 1 ]]; then
  twine upload "${TFCI_UPLOAD_PYPI_ARGS[@]}" build/*.whl
  gsutil cp build/*.whl "$TFCI_UPLOAD_GCS_DESTINATION"
fi

tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" test "${TFCI_BAZEL_COMMON_ARGS[@]}" --config=nonpip
