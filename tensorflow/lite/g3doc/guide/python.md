# Python quickstart

Using TensorFlow Lite with Python is great for embedded devices based on Linux,
such as [Raspberry Pi](https://www.raspberrypi.org/){:.external} and
[Coral devices with Edge TPU](https://coral.withgoogle.com/){:.external},
among many others.

This page shows how you can start running TensorFlow Lite models with Python in
just a few minutes. All you need is a TensorFlow model [converted to TensorFlow
Lite](../convert/). (If you don't have a model converted yet, you can experiment
using the model provided with the example linked below.)

## About the TensorFlow Lite runtime package

To quickly start executing TensorFlow Lite models with Python, you can install
just the TensorFlow Lite interpreter, instead of all TensorFlow packages. We
call this simplified Python package `tflite_runtime`.

The `tflite_runtime` package is a fraction the size of the full `tensorflow`
package and includes the bare minimum code required to run inferences with
TensorFlow Lite—primarily the
[`Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter)
Python class. This small package is ideal when all you want to do is execute
`.tflite` models and avoid wasting disk space with the large TensorFlow library.

Note: If you need access to other Python APIs, such as the
[TensorFlow Lite Converter](../convert/), you must install the
[full TensorFlow package](https://www.tensorflow.org/install/).

## Install TensorFlow Lite for Python

To install the TensorFlow Lite runtime package, run this command:

<pre class="devsite-terminal devsite-click-to-copy">
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
</pre>

If you're on a Raspberry Pi, this command might fail due to a known issue with
the `extra-index-url` option
([#4011](https://github.com/raspberrypi/linux/issues/4011)). So we suggest you
specify one of the
[`tflite_runtime` wheels](https://github.com/google-coral/pycoral/releases/)
that matches your system. For example, if you're running Raspberry Pi OS 10
(which has Python 3.7), instead use this command:

<pre class="devsite-terminal devsite-click-to-copy">
pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl
</pre>

Note: If you're on Debian Linux and using TensorFlow Lite with a Coral ML
accelerator, using pip to install `tflite_runtime` may not be compatible with
other Coral libraries. To ensure all your libraries are compatible, instead
install `tflite_runtime` as a
[Debian package from Coral](https://coral.ai/software/#debian-packages).

## Run an inference using tflite_runtime

Instead of importing `Interpreter` from the `tensorflow` module, you now need to
import it from `tflite_runtime`.

For example, after you install the package above, copy and run the
[`label_image.py`](
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/python/)
file. It will (probably) fail because you don't have the `tensorflow` library
installed. To fix it, edit this line of the file:

```python
import tensorflow as tf
```

So it instead reads:

```python
import tflite_runtime.interpreter as tflite
```

And then change this line:

```python
interpreter = tf.lite.Interpreter(model_path=args.model_file)
```

So it reads:

```python
interpreter = tflite.Interpreter(model_path=args.model_file)
```

Now run `label_image.py` again. That's it! You're now executing TensorFlow Lite
models.

## Learn more

For more details about the `Interpreter` API, read
[Load and run a model in Python](inference.md#load-and-run-a-model-in-python).

If you have a Raspberry Pi, try the
[classify_picamera.py example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi)
to perform image classification with the Pi Camera and TensorFlow Lite.

If you're using a Coral ML accelerator, check out the
[Coral examples on GitHub](https://github.com/google-coral/tflite/tree/master/python/examples).

To convert other TensorFlow models to TensorFlow Lite, read about the
the [TensorFlow Lite Converter](../convert/).
