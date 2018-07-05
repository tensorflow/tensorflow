#!/bin/bash

bazel build --config=opt --copt="-mtune=generic" //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
cp /tmp/tensorflow_pkg/tensorflow-1.9.0rc0-cp35-cp35m-linux_x86_64.whl /tensorflow/tensorflow-1.9.0rc0-py3-none-linux_x86_64.whl
#sudo /home/ubuntu/.local/bin/pip3 uninstall tensorflow
#sudo /home/ubuntu/.local/bin/pip3 install /tmp/tensorflow_pkg/tensorflow-1.9.0rc0-cp35-cp35m-linux_x86_64.whl
