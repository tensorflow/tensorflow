#!/bin/bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
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
#               (affects 'source $TFCI')
set -euxo pipefail -o history -o allexport

# Set TFCI_GIT_DIR, the root directory for all commands, to two directories
# above the location of this file (setup.sh). We could also use "git rev-parse
# --show-toplevel", but that wouldn't work for non-git repos (like if someone
# downloaded TF as a zip archive).
export TFCI_GIT_DIR=$(cd $(dirname "$0"); realpath ../../)
cd "$TFCI_GIT_DIR"

# "TFCI" may optionally be set to the name of an env-type file with TFCI
# variables in it, OR may be left empty if the user has already exported the
# relevant variables in their environment. Because of 'set -o allexport' above
# (which is equivalent to "set -a"), every variable in the file is exported
# for other files to use.
if [[ -n "${TFCI:-}" ]]; then
  # Sourcing this twice, the first time with "-u" unset, means that variable
  # order does not matter. i.e. "TFCI_BAR=$TFCI_FOO; TFCI_FOO=true" will work.
  # TFCI_FOO is only valid the second time through.
  set +u
  source "$TFCI"
  set -u
  source "$TFCI"
else
  echo '==TFCI==: The $TFCI variable is not set. This is fine as long as you'
  echo 'already sourced a TFCI env file with "set -a; source <path>; set +a".'
  echo 'If you have not, you will see a lot of undefined variable errors.'
fi

# Create and expand to the full path of TFCI_OUTPUT_DIR
export TFCI_OUTPUT_DIR=$(realpath "$TFCI_OUTPUT_DIR")
mkdir -p "$TFCI_OUTPUT_DIR"

# In addition to dumping all script output to the terminal, place it into
# $TFCI_OUTPUT_DIR/script.log
exec > >(tee "$TFCI_OUTPUT_DIR/script.log") 2>&1

# Setup tfrun, a helper function for executing steps that can either be run
# locally or run under Docker. docker.sh, below, redefines it as "docker exec".
# Important: "tfrun foo | bar" is "( tfrun foo ) | bar", not tfrun (foo | bar).
# Therefore, "tfrun" commands cannot include pipes -- which is probably for the
# better. If a pipe is necessary for something, it is probably complex. Write a
# well-documented script under utilities/ to encapsulate the functionality
# instead.
tfrun() { "$@"; }

# Run all "tfrun" commands under Docker. See docker.sh for details
if [[ "$TFCI_DOCKER_ENABLE" == 1 ]]; then
  source ./ci/official/utilities/docker.sh
fi

# Generate an overview page describing the build
if [[ "$TFCI_INDEX_HTML_ENABLE" == 1 ]]; then
  ./ci/official/utilities/generate_index_html.sh "$TFCI_OUTPUT_DIR/index.html"
fi

# Single handler for all cleanup actions, triggered on an EXIT trap.
# TODO(angerson) Making this use different scripts may be overkill.
cleanup() {
  if [[ "$TFCI_DOCKER_ENABLE" == 1 ]]; then
    ./ci/official/utilities/cleanup_docker.sh
  fi
  ./ci/official/utilities/cleanup_summary.sh
}
trap cleanup EXIT
