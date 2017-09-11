#!/bin/bash

set -ex

if [ -z "${TASKCLUSTER_TASK_DIR}" ]; then
    echo "No TASKCLUSTER_TASK_DIR, aborting."
    exit 1
fi

mkdir "${TASKCLUSTER_TASK_DIR}/homebrew" && curl -L https://github.com/Homebrew/brew/tarball/master | tar xz --strip 1 -C "${TASKCLUSTER_TASK_DIR}/homebrew"

LOCAL_BREW="${TASKCLUSTER_TASK_DIR}/homebrew"
export PATH=${LOCAL_BREW}/bin:$PATH

echo "local brew list (should me empty) ..."
brew list

echo "local brew prefix ..."
local_prefix=$(brew --prefix)
echo "${local_prefix}"

if [ "${LOCAL_BREW}" != "${local_prefix}" ]; then
    echo "Weird state:"
    echo "LOCAL_BREW=${LOCAL_BREW}"
    echo "local_prefix=${local_prefix}"
    exit 1
fi;

# coreutils, pyenv-virtualenv required for build of tensorflow
# node@6 pkg-config sox swig required for later build of deepspeech
# boost required by native_client decoder
all_pkgs="coreutils pyenv-virtualenv node@6 pkg-config sox swig boost"

for pkg in ${all_pkgs};
do
	brew list --versions ${pkg} || brew install ${pkg}
done;
