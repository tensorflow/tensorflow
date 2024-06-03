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
#
# Separately, if TFCI is set *and* there are also additional TFCI_ variables
# set in the shell environment, those variables will be restored after the
# TFCI env has been loaded. This is useful for e.g. on-demand "generic" jobs
# where the user may wish to change just one option.
if [[ -z "${TFCI:-}" ]]; then
  echo '==TFCI==: The $TFCI variable is not set. This is fine as long as you'
  echo 'already sourced a TFCI env file with "set -a; source <path>; set +a".'
  echo 'If you have not, you will see a lot of undefined variable errors.'
else
  FROM_ENV=$(mktemp)
  # "export -p" prints a list of environment values in a safe-to-source format,
  # e.g. `declare -x TFCI_BAZEL_COMMON_ARGS="list of args"` for bash.
  export -p | grep TFCI > "$FROM_ENV"

  # Source the default ci values
  source ./ci/official/envs/ci_default

  # TODO(angerson) write this documentation
  # Sources every env, in order, from the comma-separated list "TFCI"
  # Assumes variables will resolve themselves correctly.
  set +u
  for env_file in ${TFCI//,/ }; do
    source "./ci/official/envs/$env_file"
  done
  set -u
  echo '==TFCI==: Evaluated the following TFCI variables from $TFCI:'
  export -p | grep TFCI

  # Load those stored pre-existing TFCI_ vars, if any
  if [[ -s "$FROM_ENV" ]]; then
    echo '==TFCI==: NOTE: Loading the following env parameters, which were'
    echo 'already set in the shell environment. If you want to disable this'
    echo 'behavior, create a new shell.'
    cat "$FROM_ENV"
    source "$FROM_ENV"
    rm "$FROM_ENV"
  fi
fi

# Mac builds have some specific setup needs. See setup_macos.sh for details
if [[ "${OSTYPE}" =~ darwin* ]]; then
  source ./ci/official/utilities/setup_macos.sh
fi

# Create and expand to the full path of TFCI_OUTPUT_DIR
export TFCI_OUTPUT_DIR=$(realpath "$TFCI_OUTPUT_DIR")
mkdir -p "$TFCI_OUTPUT_DIR"

# In addition to dumping all script output to the terminal, place it into
# $TFCI_OUTPUT_DIR/script.log
exec > >(tee "$TFCI_OUTPUT_DIR/script.log") 2>&1

# Setup tfrun, a helper function for executing steps that can either be run
# locally or run under Docker. setup_docker.sh, below, redefines it as "docker
# exec".
# Important: "tfrun foo | bar" is "( tfrun foo ) | bar", not "tfrun (foo | bar)".
# Therefore, "tfrun" commands cannot include pipes -- which is
# probably for the better. If a pipe is necessary for something, it is probably
# complex. Write a well-documented script under utilities/ to encapsulate the
# functionality instead.
tfrun() { "$@"; }

# Run all "tfrun" commands under Docker. See setup_docker.sh for details
if [[ "$TFCI_DOCKER_ENABLE" == 1 ]]; then
  source ./ci/official/utilities/setup_docker.sh
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
