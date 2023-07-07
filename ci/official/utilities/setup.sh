#!/bin/bash
# -e: abort script if one command fails
# -u: error if undefined variable used
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -euxo pipefail -o history -o allexport

# Import all variables as set in $TFCI, which should be a file like those in
# the envs directory that sets all TFCI_ variables, e.g. /path/to/envs/local_cpu
source "$TFCI"

# Make a "build" directory for outputting all build artifacts (TF's .gitignore
# ignores the "build" directory)
cd "$TFCI_GIT_DIR" && mkdir -p build

# Setup tfrun, a helper function for executing steps that can either be run
# locally or run under Docker. docker.sh, below, redefines it as "docker exec".
tfrun() { "$@"; }

# For Google-internal jobs, run copybara, which will overwrite the source tree.
# Never useful for outside users.
[[ "$TFCI_COPYBARA_ENABLE" == 1 ]] && source ./ci/official/utilities/copybara.sh

# Run all "tfrun" commands under Docker. See docker.sh for details
[[ "$TFCI_DOCKER_ENABLE" == 1 ]] && source ./ci/official/utilities/docker.sh

# Generate an overview page describing the build
[[ "$TFCI_INDEX_HTML_ENABLE" == 1 ]] && ./ci/official/utilities/generate_index_html.sh build/index.html
