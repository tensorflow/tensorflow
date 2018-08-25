#!/usr/bin/env bash
# Download and build TensorFlow.
set -euxo pipefail
git clone --branch=master --depth=1 https://github.com/tensorflow/tensorflow.git /tensorflow
cd /tensorflow

ln -s $(which ${PYTHON}) /usr/local/bin/python 

ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
tensorflow/tools/ci_build/builds/configured GPU \
bazel build -c opt --copt=-mavx --config=cuda \
    --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
    tensorflow/tools/pip_package:build_pip_package && \
rm /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
pip --no-cache-dir install --upgrade /tmp/pip/tensorflow-*.whl && \
rm -rf /tmp/pip && \
rm -rf /root/.cache
