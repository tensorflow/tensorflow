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

To quickly run TensorFlow Lite models with Python, you can install just the
TensorFlow Lite interpreter, instead of all TensorFlow packages.

This interpreter-only package is a fraction the size of the full TensorFlow
package and includes the bare minimum code required to run inferences with
TensorFlow Lite—it includes only the
[`tf.lite.Interpreter`](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter)
Python class. This small package is ideal when all you want to do is execute
`.tflite` models and avoid wasting disk space with the large TensorFlow library.

Note: If you need access to other Python APIs, such as the [TensorFlow Lite
Converter](../convert/python_api.md), you must install the [full TensorFlow
package](https://www.tensorflow.org/install/).

To install, run `pip3 install` and pass it the appropriate Python wheel URL from
the following table.

For example, if you have Raspberry Pi that's running Raspbian Buster (which has
Python 3.7), install the Python wheel as follows:

<pre class="devsite-terminal devsite-click-to-copy">
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
</pre>

<table>
<tr><th>Platform</th><th>Python</th><th>URL</th></tr>
<tr>
  <td style="white-space:nowrap" rowspan="3">Linux (ARM 32)</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_armv7l.whl</td>
</tr>
<tr>
  <!-- ARM 32 -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_armv7l.whl</td>
</tr>
<tr>
  <!-- ARM 32 -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl</td>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="3">Linux (ARM 64)</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_aarch64.whl</td>
</tr>
<tr>
  <!-- ARM 64 -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_aarch64.whl</td>
</tr>
<tr>
  <!-- ARM 64 -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_aarch64.whl</td>
</tr>
<tr>
  <td style="white-space:nowrap" rowspan="3">Linux (x86-64)</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_x86_64.whl</td>
</tr>
<tr>
  <!-- x86-64 -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_x86_64.whl</td>
</tr>
<tr>
  <!-- x86-64 -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl</td>
</tr>

<tr>
  <td style="white-space:nowrap" rowspan="3">macOS 10.14</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-macosx_10_14_x86_64.whl</td>
</tr>
<tr>
  <!-- Mac -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-macosx_10_14_x86_64.whl</td>
</tr>
<tr>
  <!-- Mac -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-macosx_10_14_x86_64.whl</td>
</tr>

<tr>
  <td style="white-space:nowrap" rowspan="3">Windows 10</td>
  <td style="white-space:nowrap">3.5</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-win_amd64.whl</td>
</tr>
<tr>
  <!-- Win -->
  <td style="white-space:nowrap">3.6</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-win_amd64.whl</td>
</tr>
<tr>
  <!-- Win -->
  <td style="white-space:nowrap">3.7</td>
  <td>https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-win_amd64.whl</td>
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

For more details about the `Interpreter` API, read
[Load and run a model in Python](inference.md#load-and-run-a-model-in-python).

If you have a Raspberry Pi, try the
[classify_picamera.py example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi)
to perform image classification with the Pi Camera and TensorFlow Lite.

If you're using a Coral ML accelerator, check out the
[Coral examples on GitHub](https://github.com/google-coral/tflite/tree/master/python/examples).

To convert other TensorFlow models to TensorFlow Lite, read about the
the [TensorFlow Lite Converter](../convert/).
