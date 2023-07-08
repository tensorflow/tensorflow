#!/bin/bash
source "${BASH_SOURCE%/*}/utilities/setup.sh"

tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" test "${TFCI_BAZEL_COMMON_ARGS[@]}" --config=pycpp

tfrun bazel analyze-profile build/profile.json.gz
