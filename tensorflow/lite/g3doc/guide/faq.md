# Frequently Asked Questions

If you don't find an answer to your question here, please look through our
detailed documentation for the topic or file a
[GitHub issue](https://github.com/tensorflow/tensorflow/issues).

## Model Conversion

#### What formats are supported for conversion from TensorFlow to TensorFlow Lite?

The supported formats are listed [here](../convert/index#python_api)

#### Why are some operations not implemented in TensorFlow Lite?

In order to keep TFLite lightweight, only certain TF operators (listed in the
[allowlist](op_select_allowlist)) are supported in TFLite.

#### Why doesn't my model convert?

Since the number of TensorFlow Lite operations is smaller than TensorFlow's,
some models may not be able to convert. Some common errors are listed
[here](../convert/index#conversion-errors).

For conversion issues not related to missing operations or control flow ops,
search our
[GitHub issues](https://github.com/tensorflow/tensorflow/issues?q=label%3Acomp%3Alite+)
or file a [new one](https://github.com/tensorflow/tensorflow/issues).

#### How do I test that a TensorFlow Lite model behaves the same as the original TensorFlow model?

The best way to test is to compare the outputs of the TensorFlow and the
TensorFlow Lite models for the same inputs (test data or random inputs) as shown
[here](inference#load-and-run-a-model-in-python).

#### How do I determine the inputs/outputs for GraphDef protocol buffer?

The easiest way to inspect a graph from a `.pb` file is to use
[Netron](https://github.com/lutzroeder/netron), an open-source viewer for
machine learning models.

If Netron cannot open the graph, you can try the
[summarize_graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md#inspecting-graphs)
tool.

If the summarize_graph tool yields an error, you can visualize the GraphDef with
[TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) and
look for the inputs and outputs in the graph. To visualize a `.pb` file, use the
[`import_pb_to_tensorboard.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/import_pb_to_tensorboard.py)
script like below:

```shell
python import_pb_to_tensorboard.py --model_dir <model path> --log_dir <log dir path>
```

#### How do I inspect a `.tflite` file?

[Netron](https://github.com/lutzroeder/netron) is the easiest way to visualize a
TensorFlow Lite model.

If Netron cannot open your TensorFlow Lite model, you can try the
[visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py)
script in our repository.

If you're using TF 2.5 or a later version

```shell
python -m tensorflow.lite.tools.visualize model.tflite visualized_model.html
```

Otherwise, you can run this script with Bazel

*   [Clone the TensorFlow repository](https://www.tensorflow.org/install/source)
*   Run the `visualize.py` script with bazel:

```shell
bazel run //tensorflow/lite/tools:visualize model.tflite visualized_model.html
```

## Optimization

#### How do I reduce the size of my converted TensorFlow Lite model?

[Post-training quantization](../performance/post_training_quantization) can
be used during conversion to TensorFlow Lite to reduce the size of the model.
Post-training quantization quantizes weights to 8-bits of precision from
floating-point and dequantizes them during runtime to perform floating point
computations. However, note that this could have some accuracy implications.

If retraining the model is an option, consider
[Quantization-aware training](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize).
However, note that quantization-aware training is only available for a subset of
convolutional neural network architectures.

For a deeper understanding of different optimization methods, look at
[Model optimization](../performance/model_optimization).

#### How do I optimize TensorFlow Lite performance for my machine learning task?

The high-level process to optimize TensorFlow Lite performance looks something
like this:

*   *Make sure that you have the right model for the task.* For image
    classification, check out the
    [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&module-type=image-classification).
*   *Tweak the number of threads.* Many TensorFlow Lite operators support
    multi-threaded kernels. You can use `SetNumThreads()` in the
    [C++ API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L345)
    to do this. However, increasing threads results in performance variability
    depending on the environment.
*   *Use Hardware Accelerators.* TensorFlow Lite supports model acceleration for
    specific hardware using delegates. See our
    [Delegates](../performance/delegates) guide for information on what
    accelerators are supported and how to use them with your model on-device.
*   *(Advanced) Profile Model.* The Tensorflow Lite
    [benchmarking tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)
    has a built-in profiler that can show per-operator statistics. If you know
    how you can optimize an operator’s performance for your specific platform,
    you can implement a [custom operator](ops_custom).

For a more in-depth discussion on how to optimize performance, take a look at
[Best Practices](../performance/best_practices).
