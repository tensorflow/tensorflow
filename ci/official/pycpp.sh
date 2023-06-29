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

# TODO(b/284172313) Revert this difference between presubmits and continuous. RBE serverside behavior is causing flakes,
# so we're temporarily allowing flaky tests again for presubmits.
tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" test "${TFCI_BAZEL_CACHE_ARGS[@]}" --config=rbe --config=pycpp --config=build_event_export

tfrun bazel analyze-profile $TFCI_RUNTIME_ART/profile.json.gz
