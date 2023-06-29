#!/bin/bash
# -e: abort script if one command fails
# -u: error if undefined variable used
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
set -euxo pipefail -o history
set -o allexport && source "$TFCI" && set +o allexport

_TFCI_HOST_ARTIFACTS_DIR="$TFCI_RUNTIME_ARTIFACTS_DIR"
tfrun() { "$@"; }
[[ "$TFCI_COPYBARA_ENABLE" = 1 ]] && source $TFCI_RUNTIME_USERTOOLS_DIR/copybara.sh
[[ "$TFCI_DOCKER_ENABLE" = 1 ]] && source $TFCI_RUNTIME_USERTOOLS_DIR/docker.sh
"$TFCI_RUNTIME_USERTOOLS_DIR/generate_index_html.sh" "$TFCI_RUNTIME_ARTIFACTS_DIR/index.html"

tfrun bats "$TFCI_RUNTIME_USERTOOLS_DIR"/code_check_full.bats --timing --output "$TFCI_RUNTIME_ARTIFACTS_DIR"
