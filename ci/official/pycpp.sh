#!/bin/bash
source "${BASH_SOURCE%/*}/utilities/setup.sh"

tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" test "${TFCI_BAZEL_CACHE_ARGS[@]}" --config=rbe --config=pycpp --config=build_event_export

tfrun bazel analyze-profile build/profile.json.gz
