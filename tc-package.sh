#!/bin/bash

set -xe

mkdir -p /tmp/artifacts/
cp ${HOME}/DeepSpeech/tf/bazel-bin/tensorflow/libtensorflow.so /tmp/artifacts/
