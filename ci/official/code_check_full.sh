#!/bin/bash
source "${BASH_SOURCE%/*}/utilities/setup.sh"

tfrun bats ./ci/official/utilities/code_check_full.bats --timing --output build
