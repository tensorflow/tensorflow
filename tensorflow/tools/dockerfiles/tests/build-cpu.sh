#!/usr/bin/env bash

# Download and build TensorFlow.
set -euxo pipefail
git clone --branch=master --depth=1 https://github.com/tensorflow/tensorflow.git /tensorflow
cd /tensorflow

ln -s $(which ${PYTHON}) /usr/local/bin/python 

# For optimized builds appropriate for the hardware platform of your choosing, uncomment below...
# For ivy-bridge or sandy-bridge
# --copt=-march="ivybridge" \
# for haswell, broadwell, or skylake
# --copt=-march="haswell" \
tensorflow/tools/ci_build/builds/configured CPU \
  bazel build -c opt --copt=-mavx --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
      tensorflow/tools/pip_package:build_pip_package && \
  bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
  pip --no-cache-dir install --upgrade /tmp/pip/tensorflow-*.whl && \
  rm -rf /tmp/pip && \
  rm -rf /root/.cache

