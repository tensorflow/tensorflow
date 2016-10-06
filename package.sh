bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package --copt=-DCTC_DISABLE_OMP && bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
