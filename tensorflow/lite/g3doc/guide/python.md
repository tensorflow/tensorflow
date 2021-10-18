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
For example, the [Select TF ops]
(https://www.tensorflow.org/lite/guide/ops_select) are not included in the
`tflite_runtime` package. If your models have any dependencies to the Select TF
ops, you need to use the full TensorFlow package instead.

## Install TensorFlow Lite for Python

If you're running Debian Linux or a derivative of Debian (including Raspberry Pi
OS), you should install from our Debian package repo. This requires that you add
a new repo list and key to your system and then install as follows:

<pre class="devsite-terminal">
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
<code class="devsite-terminal"
>curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
</code><code class="devsite-terminal"
>sudo apt-get update
</code><code class="devsite-terminal"
>sudo apt-get install python3-tflite-runtime</code>
</pre>

For all other systems, you can install with pip:

<pre class="devsite-terminal devsite-click-to-copy">
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
</pre>

If you'd like to manually install a Python wheel, you can select one from
[all `tflite_runtime` wheels](https://github.com/google-coral/pycoral/releases/).

Note: If you're on Debian Linux and you install the `tflite_runtime` using pip,
it can cause runtime failures when using other software that you installed as
Debian packages and that depends on TF Lite (such as
[Coral libraries](https://coral.ai/software/)). You can fix it if you uninstall
`tflite_runtime` with pip and then reinstall it with the `apt-get` commands
above.

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
[TensorFlow Lite Converter](../convert/).

If you want to build `tflite_runtime` wheel, read
[Build TensorFlow Lite Python Wheel Package](build_cmake_pip.md)
