#!/bin/bash

set -ex

source $(dirname $0)/tc-vars.sh

build_gpu=no
build_arm=no

if [ "$1" = "--gpu" ]; then
    build_gpu=yes
fi

if [ "$1" = "--arm" ]; then
    build_gpu=no
    build_arm=yes
fi

pushd ${DS_ROOT_TASK}/DeepSpeech/tf/
    BAZEL_BUILD="bazel ${BAZEL_OUTPUT_USER_ROOT} build -s --explain bazel_monolithic_tf.log --verbose_explanations --experimental_strict_action_env --config=monolithic"

    # Pure amd64 CPU-only build
    if [ "${build_gpu}" = "no" -a "${build_arm}" = "no" ]; then
        echo "" | TF_NEED_CUDA=0 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_OPT_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LIB_CPP_API} ${BUILD_TARGET_GRAPH_TRANSFORMS} ${BUILD_TARGET_GRAPH_SUMMARIZE} ${BUILD_TARGET_GRAPH_BENCHMARK} ${BUILD_TARGET_CONVERT_MMAP} ${BUILD_TARGET_AOT_DEPS}
    fi

    # Cross RPi3 CPU-only build
    if [ "${build_gpu}" = "no" -a "${build_arm}" = "yes" ]; then
        echo "" | TF_NEED_CUDA=0 ./configure && ${BAZEL_BUILD} -c opt ${BAZEL_ARM_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BUILD_TARGET_LIB_CPP_API} ${BUILD_TARGET_GRAPH_TRANSFORMS} ${BUILD_TARGET_GRAPH_SUMMARIZE} ${BUILD_TARGET_GRAPH_BENCHMARK} ${BUILD_TARGET_AOT_DEPS}
    fi

    # Pure amd64 GPU-enabled build
    if [ "${build_gpu}" = "yes" -a "${build_arm}" = "no" ]; then
        eval "export ${TF_CUDA_FLAGS}" && (echo "" | TF_NEED_CUDA=1 ./configure) && ${BAZEL_BUILD} -c opt ${BAZEL_CUDA_FLAGS} ${BAZEL_EXTRA_FLAGS} ${BAZEL_OPT_FLAGS} ${BUILD_TARGET_LIB_CPP_API} ${BUILD_TARGET_GRAPH_TRANSFORMS} ${BUILD_TARGET_GRAPH_SUMMARIZE} ${BUILD_TARGET_GRAPH_BENCHMARK}
    fi

    if [ $? -ne 0 ]; then
        # There was a failure, just account for it.
        echo "Build failure, please check the output above. Exit code was: $?"
        return 1
    fi
popd
