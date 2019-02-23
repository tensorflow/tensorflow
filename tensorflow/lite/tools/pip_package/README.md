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

Note, unlike tensorflow this will be installed to a tflite_runtime namespace.
You can then use the Tensorflow Lite interpreter as.
```
import tflite_runtime as tflr
interpreter = tflr.lite.Interpreter(model_path="foo.tflite")
```

This currently works to build on Linux machines including Raspberry Pi. In
the future, cross compilation to smaller SOCs like Raspberry Pi from
bigger host will be supported.

## Caveats

* You cannot use TensorFlow Select ops, only TensorFlow Lite builtins.
* Currently custom ops and delegates cannot be registered.

