#!/bin/bash

set -ex

if [ -z "${TASKCLUSTER_TASK_DIR}" ]; then
    echo "No TASKCLUSTER_TASK_DIR, aborting."
    exit 1
fi

# install patch, unzip (TensorFlow deps)
pacman --noconfirm -R bsdtar
pacman --noconfirm -S patch
pacman --noconfirm -S unzip
pacman --noconfirm -S tar
