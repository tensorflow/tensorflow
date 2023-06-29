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

# Parse options and build targets into arrays, so that shelllint doesn't yell
# about readability. We can't pipe into 'read -ra' to create an array because
# piped commands run in subshells, which can't store variables outside of the
# subshell environment.
# See https://g3doc.corp.google.com/devtools/staticanalysis/pipeline/analyzers/shell/lint/g3doc/findings/SC2086.md?cl=head
# Ignore grep failures since we're using it for basic filtering
set +e
filtered_build_targets=( $(echo "$BUILD_TARGETS" | tr ' ' '\n' | grep . | tee build_targets.txt) )
nonpip_targets=( $(echo "$TEST_TARGETS" | tr ' ' '\n' | grep -E "^//tensorflow/" | tee nonpip_targets.txt) )
config=( $(echo "$CONFIG_OPTIONS" ) )
test_flags=( $(echo "$TEST_FLAGS" ) )
set -e

[[ "$TFCI_NVIDIA_SMI_ENABLE" == 1 ]] && tfrun nvidia-smi

if [[ -s build_targets.txt ]]; then
  tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" "${config[@]}" "${filtered_build_targets[@]}"
fi

if [[ "${PIP_WHEEL}" -eq "1" ]]; then
  # Update the version numbers to build a "nightly" package
  [[ "$TFCI_NIGHTLY_UPDATE_VERSION_ENABLE" == 1 ]] && tfrun python3 tensorflow/tools/ci_build/update_version.py --nightly

  tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" build "${TFCI_BAZEL_CACHE_ARGS[@]}" tensorflow/tools/pip_package:build_pip_package
  tfrun ./bazel-bin/tensorflow/tools/pip_package/build_pip_package build "${TFCI_BUILD_PIP_PACKAGE_ARGS[@]}"
  tfrun ./ci/official/utilities/rename_and_verify_wheels.sh
fi

if [[ -s nonpip_targets.txt ]]; then
  tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" test "${config[@]}" "${test_flags[@]}" "${nonpip_targets[@]}"
fi
