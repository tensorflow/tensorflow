# PIP
bazel build  //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package build
pip install build/tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl
# 이미 설치 되었다면,
pip install build/tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl --force-reinstall --no-deps

# .so
bazel build  //tensorflow/tools/lib_package:libtensorflow
tar xvf bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz
