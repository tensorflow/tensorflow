#!/bin/bash

set -xe

source $(dirname $0)/tc-vars.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

cp ${DS_ROOT_TASK}/DeepSpeech/tf/bazel_*.log ${TASKCLUSTER_ARTIFACTS}

cp ${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/tensorflow/libtensorflow_cc.so ${TASKCLUSTER_ARTIFACTS}
cp ${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/tensorflow/tools/graph_transforms/transform_graph ${TASKCLUSTER_ARTIFACTS}
cp ${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph ${TASKCLUSTER_ARTIFACTS}
cp ${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/tensorflow/tools/benchmark/benchmark_model ${TASKCLUSTER_ARTIFACTS}

# It seems that bsdtar and gnutar are behaving a bit differently on the way
# they deal with --exclude="./public/*" ; this caused ./DeepSpeech/tensorflow/core/public/
# to be ditched when we just wanted to get rid of ./public/ on OSX.
# Switching to gnutar (already needed for the --transform on DeepSpeech tasks)
# does the trick.
TAR=tar
TAR_EXCLUDE="--exclude=./dls/*"
if [ "${OS}" = "Darwin" ]; then
    TAR=gtar
    TAR_EXCLUDE="--exclude=./dls/* --exclude=./public/* --exclude=./generic-worker/* --exclude=./homebrew/* --exclude=./homebrew.cache/* --exclude=./homebrew.logs/*"
fi;

# Make a tar of
#  - /home/build-user/ (linux
#  - /Users/build-user/TaskCluster/HeavyTasks/X/ (OSX)
${TAR} -C ${DS_ROOT_TASK} ${TAR_EXCLUDE} -cf - . | pixz -9 > ${TASKCLUSTER_ARTIFACTS}/home.tar.xz
