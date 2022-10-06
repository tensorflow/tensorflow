# Quickstart for Linux-based devices with Python

Using TensorFlow Lite with Python is great for embedded devices based on Linux,
such as [Raspberry Pi](https://www.raspberrypi.org/){:.external} and
[Coral devices with Edge TPU](https://coral.withgoogle.com/){:.external},
among many others.

This page shows how you can start running TensorFlow Lite models with Python in
just a few minutes. All you need is a TensorFlow model [converted to TensorFlow
Lite](../models/convert/). (If you don't have a model converted yet, you can
experiment using the model provided with the example linked below.)

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
[TensorFlow Lite Converter](../models/convert/), you must install the
[full TensorFlow package](https://www.tensorflow.org/install/).
For example, the [Select TF ops]
(https://www.tensorflow.org/lite/guide/ops_select) are not included in the
`tflite_runtime` package. If your models have any dependencies to the Select TF
ops, you need to use the full TensorFlow package instead.

## Install TensorFlow Lite for Python

You can install on Linux with pip:

<pre class="devsite-terminal devsite-click-to-copy">
python3 -m pip install tflite-runtime
</pre>

## Supported platforms

The `tflite-runtime` Python wheels are pre-built and provided for these
platforms:
* Linux armv7l (e.g. Raspberry Pi 2, 3, 4 and Zero 2 running Raspberry Pi OS
  32-bit)
* Linux aarch64 (e.g. Raspberry Pi 3, 4 running Debian ARM64)
* Linux x86_64

If you want to run TensorFlow Lite models on other platforms, you should either
use the [full TensorFlow package](https://www.tensorflow.org/install/), or
[build the tflite-runtime package from source](build_cmake_pip.md).

If you're using TensorFlow with the Coral Edge TPU, you should
instead follow the appropriate [Coral setup documentation](https://coral.ai/docs/setup).

Note: We no longer update the Debian package `python3-tflite-runtime`. The
latest Debian package is for TF version 2.5, which you can install by following
[these older instructions](https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/lite/g3doc/guide/python.md#install-tensorflow-lite-for-python).

Note: We no longer release pre-built `tflite-runtime` wheels for Windows and
macOS. For these platforms, you should use the
[full TensorFlow package](https://www.tensorflow.org/install/), or
[build the tflite-runtime package from source](build_cmake_pip.md).

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

* For more details about the `Interpreter` API, read
  [Load and run a model in Python](inference.md#load-and-run-a-model-in-python).

* If you have a Raspberry Pi, check out a [video series](https://www.youtube.com/watch?v=mNjXEybFn98&list=PLQY2H8rRoyvz_anznBg6y3VhuSMcpN9oe)
  about how to run object detection on Raspberry Pi using TensorFlow Lite.

* If you're using a Coral ML accelerator, check out the
  [Coral examples on GitHub](https://github.com/google-coral/tflite/tree/master/python/examples).

* To convert other TensorFlow models to TensorFlow Lite, read about the
  [TensorFlow Lite Converter](../models/convert/).

* If you want to build `tflite_runtime` wheel, read
  [Build TensorFlow Lite Python Wheel Package](build_cmake_pip.md)
