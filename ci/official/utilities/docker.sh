#!/bin/bash
# -e: abort script if one command fails
# -u: error if undefined variable used
# -o pipefail: entire command fails if pipe fails. watch out for yes | ...
# -o history: record shell history
set -euxo pipefail -o history
set -o allexport && source "$TFCI" && set +o allexport

trap "docker rm -f tf" EXIT
if [[ "$TFCI_DOCKER_PULL_ENABLE" == 1 ]]; then
  docker pull "$TFCI_DOCKER_IMAGE"
fi
docker run "${TFCI_DOCKER_GPU_ARGS[@]}" --name tf -w "$TFCI_GIT_DIR" -itd --rm \
    -v "$TFCI_GIT_DIR:$TFCI_GIT_DIR" \
    "$TFCI_DOCKER_IMAGE" \
  bash
tfrun() { docker exec tf "$@"; }
