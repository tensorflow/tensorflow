#!/bin/bash

set -xe

source $(dirname $0)/tc-vars.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

cp ${DS_ROOT_TASK}/DeepSpeech/tf/bazel_*.log ${TASKCLUSTER_ARTIFACTS}

for output_bin in                                                            \
    tensorflow/libtensorflow_cc.so                                           \
    tensorflow/contrib/lite/experimental/c/libtensorflowlite_c.so            \
    tensorflow/tools/graph_transforms/transform_graph                        \
    tensorflow/tools/graph_transforms/summarize_graph                        \
    tensorflow/tools/benchmark/benchmark_model                               \
    tensorflow/contrib/util/convert_graphdef_memmapped_format                \
    tensorflow/contrib/lite/toco/toco;
do
    if [ -f "${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/${output_bin}" ]; then
        cp ${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/${output_bin} ${TASKCLUSTER_ARTIFACTS}/
    fi;
done;

if [ -f "${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/tensorflow/contrib/lite/tools/benchmark/benchmark_model" ]; then
    cp ${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/tensorflow/contrib/lite/tools/benchmark/benchmark_model ${TASKCLUSTER_ARTIFACTS}/lite_benchmark_model
fi;

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
