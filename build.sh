#!/bin/bash

set -ex

HOME_CLEAN=$(/usr/bin/realpath "${HOME}")

PATH=${HOME_CLEAN}/bin/:$PATH
export PATH

LD_LIBRARY_PATH=${HOME_CLEAN}/DeepSpeech/CUDA/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

build_gpu=no
build_arm=no

if [ "$1" = "--gpu" ]; then
    build_gpu=yes
fi

if [ "$1" = "--arm" ]; then
    build_gpu=no
    build_arm=yes
fi

mkdir -p /tmp/artifacts/

pushd ~/DeepSpeech/tf/
    PYTHON_BIN_PATH=${HOME_CLEAN}/DeepSpeech/tf-venv/bin/python
    PYTHONPATH=${HOME_CLEAN}/DeepSpeech/tf-venv/lib/python2.7/site-packages
    TF_NEED_GCP=0
    TF_NEED_HDFS=0
    TF_NEED_OPENCL=0
    TF_NEED_JEMALLOC=1
    TF_ENABLE_XLA=1
    GCC_HOST_COMPILER_PATH=/usr/bin/gcc

    # Enable some SIMD support. Limit ourselves to what Tensorflow needs.
    # Also ensure to not require too recent CPU: AVX2/FMA introduced by:
    #  - Intel with Haswell (2013)
    #  - AMD with Excavator (2015)
    #
    # Build for generic amd64 platforms, no device-specific optimization
    # See https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html for targetting specific CPUs
    CC_OPT_FLAGS="-mtune=generic -march=x86-64 -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma"
    BAZEL_OPT_FLAGS="--copt=-mtune=generic --cxxopt=-mtune=generic --copt=-march=x86-64 --cxxopt=-march=x86-64 --copt=-msse --cxxopt=-msse --copt=-msse2 --cxxopt=-msse2 --copt=-msse3 --cxxopt=-msse3 --copt=-msse4.1 --cxxopt=-msse4.1 --copt=-msse4.2 --cxxopt=-msse4.2 --copt=-mavx --cxxopt=-mavx --copt=-mavx2 --cxxopt=-mavx2 --copt=-mfma --cxxopt=-mfma"

    BUILD_TARGET_PIP="//tensorflow/tools/pip_package:build_pip_package"
    BUILD_TARGET_LIB="//tensorflow:libtensorflow.so"

    export PYTHON_BIN_PATH
    export PYTHONPATH
    export TF_NEED_GCP
    export TF_NEED_HDFS
    export TF_NEED_OPENCL
    export TF_NEED_JEMALLOC
    export TF_ENABLE_XLA
    export GCC_HOST_COMPILER_PATH
    export CC_OPT_FLAGS

    # Pure amd64 CPU-only build
    if [ "${build_gpu}" = "no" -a "${build_arm}" = "no" ]; then
        echo "" | TF_NEED_CUDA=0 ./configure && bazel build -c opt ${BAZEL_OPT_FLAGS} ${BUILD_TARGET_PIP} ${BUILD_TARGET_LIB} && ./tensorflow/tools/pip_package/build_pip_package.sh /tmp/artifacts/
    fi

    # Cross RPi3 CPU-only build
    if [ "${build_gpu}" = "no" -a "${build_arm}" = "yes" ]; then
        echo "" | TF_NEED_CUDA=0 ./configure && bazel build -c opt --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --crosstool_top=//tools/arm_compiler:toolchain --cpu=rpi-armeabi ${BUILD_TARGET_LIB}
    fi

    # Pure amd64 GPU-enabled build
    if [ "${build_gpu}" = "yes" -a "${build_arm}" = "no" ]; then
        echo "" | TF_NEED_CUDA=1 TF_CUDA_VERSION=8.0 TF_CUDNN_VERSION=5 CUDA_TOOLKIT_PATH=${HOME_CLEAN}/DeepSpeech/CUDA CUDNN_INSTALL_PATH=${HOME_CLEAN}/DeepSpeech/CUDA TF_CUDA_COMPUTE_CAPABILITIES="3.0,3.5,3.7,5.2,6.0,6.1" ./configure && bazel build -c opt --config=cuda ${BAZEL_OPT_FLAGS} ${BUILD_TARGET_PIP} ${BUILD_TARGET_LIB} && ./tensorflow/tools/pip_package/build_pip_package.sh /tmp/artifacts/ --gpu
    fi

    if [ $? -eq 0 ]; then
        cp bazel-bin/tensorflow/libtensorflow.so /tmp/artifacts/
    else
        # There was a failure, just account for it.
        echo "Build failure, please check the output above. Exit code was: $?"
        return 1
    fi
popd

ls -halR /tmp/artifacts/
