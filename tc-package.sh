#!/bin/bash

set -xe

source $(dirname $0)/tc-vars.sh

mkdir -p ${TASKCLUSTER_ARTIFACTS} || true

cp ${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/tensorflow/libtensorflow_cc.so ${TASKCLUSTER_ARTIFACTS}
cp ${DS_ROOT_TASK}/DeepSpeech/tf/bazel-bin/tensorflow/tools/graph_transforms/transform_graph ${TASKCLUSTER_ARTIFACTS}

artifact_root_dir=$(dirname "${DS_ROOT_TASK}")
artifact_tar_dir=$(basename "${DS_ROOT_TASK}")
tar -C ${artifact_root_dir} -cf - ${artifact_tar_dir} | pixz -9 > ${TASKCLUSTER_ARTIFACTS}/home.tar.xz
