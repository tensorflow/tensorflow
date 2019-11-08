# Python quickstart

Using TensorFlow Lite with Python is great for embedded devices based on Linux,
such as [Raspberry Pi](https://www.raspberrypi.org/){:.external} and
[Coral devices with Edge TPU](https://coral.withgoogle.com/){:.external},
among many others.

This page shows how you can start running TensorFlow Lite models with Python in
just a few minutes. All you need is a TensorFlow model [converted to TensorFlow
Lite](../convert/). (If you don't have a model converted yet, you can experiment
using the model provided with the example linked below.)

## Install just the TensorFlow Lite interpreter

To quickly start executing TensorFlow Lite models with Python, you can install
just the TensorFlow Lite interpreter, instead of all TensorFlow packages.

This interpreter-only package is a fraction the size of the full TensorFlow
package and includes the bare minimum code required to run inferences with
TensorFlow Lite—it includes only the [`tf.lite.Interpreter`](
https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter) Python class.
This small package is ideal when all you want to do is execute `.tflite` models
and avoid wasting disk space with the large TensorFlow library.

Note: If you need access to other Python APIs, such as the [TensorFlow Lite
Converter](../convert/python_api.md), you must install the [full TensorFlow
package](https://www.tensorflow.org/install/).

To install just the interpreter, download the appropriate Python wheel for your
system from the following table, and then install it with the `pip install`
command.

For example, if you're setting up a Raspberry Pi (using Raspbian Buster, which
has Python 3.7), install the Python wheel as follows (after you click to
download the `.whl` file below):

<pre class="devsite-terminal devsite-click-to-copy">
pip3 install tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl
</pre>

<table>
<tr><th></th><th>ARM 32</th><th>ARM 64</th><th>x86-64</th></tr>
<tr><th style="white-space:nowrap">Python 3.5</th>
  <td><a href="https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp35-cp35m-linux_armv7l.whl"
    >tflite_runtime-1.14.0-cp35-cp35m-linux_armv7l.whl</a></td>
  <td><a href="https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp35-cp35m-linux_aarch64.whl"
    >tflite_runtime-1.14.0-cp35-cp35m-linux_aarch64.whl</a></td>
  <td><a href="https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp35-cp35m-linux_x86_64.whl"
    >tflite_runtime-1.14.0-cp35-cp35m-linux_x86_64.whl</a></td>
</tr>
<tr><th>Python 3.6</th>
  <td>N/A</td>
  <td>N/A</td>
  <td><a href="https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp36-cp36m-linux_x86_64.whl"
    >tflite_runtime-1.14.0-cp36-cp36m-linux_x86_64.whl</a></td>
</tr>
<tr><th>Python 3.7</th>
  <td><a href="https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl"
    >tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl</a></td>
  <td><a href="https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_aarch64.whl"
    >tflite_runtime-1.14.0-cp37-cp37m-linux_aarch64.whl</a></td>
  <td><a href="https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_x86_64.whl"
    >tflite_runtime-1.14.0-cp37-cp37m-linux_x86_64.whl</a></td>
</tr>
</table>


## Run an inference using tflite_runtime

To distinguish this interpreter-only package from the full TensorFlow package
(allowing both to be installed, if you choose), the Python module provided in
the above wheel is named `tflite_runtime`.

So instead of importing `Interpreter` from the `tensorflow` module, you need to
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

If you have a Raspberry Pi, try the
[classify_picamera.py example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi)
to perform image classification with the Pi Camera and TensorFlow Lite.

For more details about the `Interpreter` API, read [Load and run a model
in Python](inference.md#load-and-run-a-model-in-python).

To convert other TensorFlow models to TensorFlow Lite, read about the
the [TensorFlow Lite Converter](../convert/).
