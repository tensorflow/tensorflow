git submodule update --init
bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip2 install --upgrade /tmp/tensorflow_pkg/tensorflow-0.7.1-py2-none-any.whl