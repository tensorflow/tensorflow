#!/bin/bash
source "${BASH_SOURCE%/*}/utilities/setup.sh"

# TODO(b/284172313) Revert this difference between presubmits and continuous. RBE serverside behavior is causing flakes,
# so we're temporarily allowing flaky tests again for presubmits.
tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" test "${TFCI_BAZEL_CACHE_ARGS[@]}" --config=rbe --config=pycpp --config=build_event_export

tfrun bazel analyze-profile build/profile.json.gz
