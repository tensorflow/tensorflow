#!/bin/bash

set -xe

mkdir -p /tmp/artifacts/
cp ${HOME}/DeepSpeech/tf/bazel-bin/tensorflow/libtensorflow_cc.so /tmp/artifacts/
cp ${HOME}/DeepSpeech/tf/bazel-bin/tensorflow/tools/graph_transforms/transform_graph /tmp/artifacts/
tar -C ${HOME} -cf - . | pixz -9 > /tmp/artifacts/home.tar.xz
