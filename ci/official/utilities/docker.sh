#!/bin/bash
trap "docker rm -f tf" EXIT
if [[ "$TFCI_DOCKER_PULL_ENABLE" == 1 ]]; then
  docker pull "$TFCI_DOCKER_IMAGE"
fi
docker run "${TFCI_DOCKER_GPU_ARGS[@]}" --name tf -w "$TFCI_GIT_DIR" -itd --rm \
    -v "$TFCI_GIT_DIR:$TFCI_GIT_DIR" \
    "$TFCI_DOCKER_IMAGE" \
  bash
tfrun() { docker exec tf "$@"; }
