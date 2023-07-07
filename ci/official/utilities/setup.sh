#!/bin/bash
# Common setup for all TF scripts.
#
# Make as FEW changes to this file as possible. It should not contain utility
# functions (except for tfrun); use dedicated scripts instead and reference them
# specifically. Use your best judgment to keep the scripts in this directory
# lean and easy to follow. When in doubt, remember that for CI scripts, "keep it
# simple" is MUCH more important than "don't repeat yourself."

# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
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
# Important: "tfrun foo | bar" is "( tfrun foo ) | bar", not tfrun (foo | bar).
# Therefore, "tfrun" commands cannot include pipes -- which is probably for the
# better. If a pipe is necessary for something, it is probably complex. Write a
# well-documented script under utilities/ to encapsulate the functionality
# instead.
tfrun() { "$@"; }

# For Google-internal jobs, run copybara, which will overwrite the source tree.
# Never useful for outside users.
[[ "$TFCI_COPYBARA_ENABLE" == 1 ]] && source ./ci/official/utilities/copybara.sh

# Run all "tfrun" commands under Docker. See docker.sh for details
[[ "$TFCI_DOCKER_ENABLE" == 1 ]] && source ./ci/official/utilities/docker.sh

# Generate an overview page describing the build
[[ "$TFCI_INDEX_HTML_ENABLE" == 1 ]] && ./ci/official/utilities/generate_index_html.sh build/index.html
