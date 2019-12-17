#!/bin/bash -e
pip install pip numpy wheel
pip install keras_preprocessing --no-deps

bazel build -j 20 --copt=-g --config=cuda --config=opt --config=noaws --config=nogcp --config=nohdfs --config=nonccl --verbose_failures //tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

pip install -U /tmp/tensorflow_pkg/tensorflow-2.4.0rc0-cp38-cp38-linux_x86_64.whl
