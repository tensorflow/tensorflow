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

# "TFCI" may optionally be set to the name of an env-type file with TFCI
# variables in it, OR may be left empty if the user has already exported the
# relevant variables in their environment. Because of 'set -o allexport' above
# (which is equivalent to "set -a"), every variable in the file is exported
# for other files to use.
if [[ -n "${TFCI:-}" ]]; then
  source "$TFCI"
else
  echo '==TFCI==: The $TFCI variable is not set. This is fine as long as you'
  echo 'already sourced a TFCI env file with "set -a; source <path>; set +a".'
  echo 'If you have not, you will see a lot of undefined variable errors.'
fi

# Make a "build" directory for outputting all build artifacts (TF's .gitignore
# ignores the "build" directory)
cd "$TFCI_GIT_DIR"
mkdir -p build

# Setup tfrun, a helper function for executing steps that can either be run
# locally or run under Docker. docker.sh, below, redefines it as "docker exec".
# Important: "tfrun foo | bar" is "( tfrun foo ) | bar", not tfrun (foo | bar).
# Therefore, "tfrun" commands cannot include pipes -- which is probably for the
# better. If a pipe is necessary for something, it is probably complex. Write a
# well-documented script under utilities/ to encapsulate the functionality
# instead.
tfrun() { "$@"; }

# For Google-internal jobs, run copybara, which will overwrite the source tree.
# Never useful for outside users. Requires that the Kokoro job define a gfile
# resource pointing to copybara.sh, which is then loaded into the GFILE_DIR.
# See: cs/official/copybara.sh
if [[ "$TFCI_COPYBARA_ENABLE" == 1 ]]; then
  if [[ -e "$KOKORO_GFILE_DIR/copybara.sh" ]]; then
    source "$KOKORO_GFILE_DIR/copybara.sh"
  else
    echo "TF_CI_COPYBARA_ENABLE is 1, but \$KOKORO_GFILE_DIR/copybara.sh"
    echo "could not be found. If you are an internal user, make sure your"
    echo "Kokoro job has a gfile_resources item pointing to the right file."
    echo "If you are an external user, Copybara is useless for you, and you"
    echo "should set TFCI_COPYBARA_ENABLE=0"
    exit 1
  fi
fi

# Run all "tfrun" commands under Docker. See docker.sh for details
if [[ "$TFCI_DOCKER_ENABLE" == 1 ]]; then
  source ./ci/official/utilities/docker.sh
fi

# Generate an overview page describing the build
if [[ "$TFCI_INDEX_HTML_ENABLE" == 1 ]]; then
  ./ci/official/utilities/generate_index_html.sh build/index.html
fi
