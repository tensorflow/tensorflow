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

tfrun bats ./ci/official/utilities/code_check_changed_files.bats --timing --output build
