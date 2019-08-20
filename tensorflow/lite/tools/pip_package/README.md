# Building TensorFlow Lite Standalone Pip

Many users would like to deploy TensorFlow lite interpreter and use it from
Python without requiring the rest of TensorFlow.

## Steps

To build a binary wheel run this script:
```
sudo apt install swig libjpeg-dev zlib1g-dev python3-dev python3-numpy
sh tensorflow/lite/tools/pip_package/build_pip_package.sh
```
That will print out some output and a .whl file. You can then install that
```
pip install --upgrade <wheel>
```

You can also build a wheel inside docker container using make tool. For example
the following command will cross-compile tflite_runtime for python2.7 and
python3.7 (from Debian Buster) on Raspberry Pi:
```
make BASE_IMAGE=debian:buster PYTHON=python TENSORFLOW_TARGET=rpi docker-build
make BASE_IMAGE=debian:buster PYTHON=python3 TENSORFLOW_TARGET=rpi docker-build
```

Another option is to cross-compile for python3.5 (from Debian Stretch) on ARM64
board:
```
make BASE_IMAGE=debian:stretch PYTHON=python3 TENSORFLOW_TARGET=aarch64 docker-build
```

To build for python3.6 (from Ubuntu 18.04) on x86_64 (native to the docker
image) run:
```
make BASE_IMAGE=ubuntu:18.04 PYTHON=python3 TENSORFLOW_TARGET=native docker-build
```

Note, unlike tensorflow this will be installed to a tflite_runtime namespace.
You can then use the Tensorflow Lite interpreter as.
```
from tflite_runtime import interpreter as tflr
interpreter = tflr.Interpreter(model_path="foo.tflite")
```

This currently works to build on Linux machines including Raspberry Pi. In
the future, cross compilation to smaller SOCs like Raspberry Pi from
bigger host will be supported.

## Caveats

* You cannot use TensorFlow Select ops, only TensorFlow Lite builtins.
* Currently custom ops and delegates cannot be registered.

