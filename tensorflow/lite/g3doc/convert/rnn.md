# Convert RNN models

The TensorFlow Lite interpreter currently implements a subset of TensorFlow
operations, meaning some model architectures cannot immediately be converted due
to missing operations.

Some RNN-based architectures are affected by this. The following document
outlines the current state of play and provides strategies for converting RNN
models.

## Currently supported

Currently, RNN models using
[`tf.compat.v1.nn.static_rnn`](https://www.tensorflow.org/api_docs/python/tf/nn/static_rnn)
can be converted successfully as long as no `sequence_length` is specified.

The following `tf.compat.v1.nn.rnn_cell` operations work with
`tf.compat.v1.nn.static_rnn`:

*   [tf.compat.v1.nn.rnn_cell.LSTMCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/LSTMCell)
*   [tf.compat.v1.nn.rnn_cell.RNNCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/RNNCell)
*   [tf.compat.v1.nn.rnn_cell.GRUCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/GRUCell)
*   [tf.compat.v1.nn.rnn_cell.BasicLSTMCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/BasicLSTMCell)
*   [tf.compat.v1.nn.rnn_cell.BasicRNNCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/BasicRNNCell)

In addition, TensorFlow Lite provides some experimental drop-in replacements for
RNN operations that enable dynamic RNN architectures with TensorFlow Lite.

Drop-in replacements are available for the following:

*   [tf.compat.v1.nn.dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
*   [tf.compat.v1.nn.bidirectional_dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn)
*   [tf.compat.v1.nn.rnn_cell.RNNCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/RNNCell)
*   [tf.compat.v1.nn.rnn_cell.LSTMCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/LSTMCell)

## Not currently supported

TensorFlow Lite does not currently support
[Control Flow](https://www.tensorflow.org/api_docs/cc/group/control-flow-ops)
operations. This means that, unless one of the conversion strategies discussed
in the next section are employed, models built with the following TensorFlow
functions will not convert successfully:

*   [tf.compat.v1.nn.static_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/static_rnn)
    where a `sequence_length` is specified
*   [tf.compat.v1.nn.dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
*   [tf.compat.v1.nn.bidirectional_dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn)

Note: TensorFlow Lite plans to implement all required Control Flow operations by
the end of 2019. At this point, all RNN architectures will convert successfully.

## Conversion strategies

To convert an RNN model that uses the functions specified above, you will have
to modify its architecture and retrain it. The following strategies can be used.

### 1. Refactoring

The simplest approach, if possible, is to refactor the model architecture to use
[tf.compat.v1.nn.static_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/static_rnn)
without `sequence_length`.

### 2. Drop-in replacements that use op hints and fused ops

TensorFlow Lite provides the some experimental drop-in replacements for RNN
operations that enable dynamic RNN architectures with TensorFlow Lite. Using
[OpHints](https://www.tensorflow.org/lite/guide/ops_custom#converting_tensorflow_models_to_convert_graphs),
they run normally during training, but are substituted with special fused ops
when run by the Lite interpreter.

The following drop-in replacements are available:

*   [tf.compat.v1.lite.experimental.nn.dynamic_rnn](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/examples/lstm/rnn.py#L41)
    *   replacement for tf.nn.dynamic_rnn
*   [tf.compat.v1.lite.experimental.nn.bidirectional_dynamic_rnn](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/examples/lstm/rnn.py#L279)
    *   replacement for tf.nn.bidirectional_dynamic_rnn
*   [tf.compat.v1.lite.experimental.nn.TfLiteRNNCell](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/examples/lstm/rnn_cell.py#L39)
    *   replacement for tf.nn.rnn_cell.RNNCell
*   [tf.compat.v1.lite.experimental.nn.TfLiteLSTMCell](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/examples/lstm/rnn_cell.py#L159)
    *   replacement for tf.nn.rnn_cell.LSTMCell

Note: These replacements must be used together. For example, if you are using
`tf.compat.v1.lite.experimental.nn.dynamic_rnn`, you must combine it with
`tf.compat.v1.lite.experimental.nn.TfLiteRNNCell` instead of using
`tf.compat.v1.nn.rnn_cell.RNNCell`.

Instead of
[tf.compat.v1.nn.rnn_cell.MultiRNNCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/MultiRNNCell),
you should use
[tf.compat.v1.keras.layers.StackedRNNCells](https://www.tensorflow.org/api_docs/python/tf/keras/layers/StackedRNNCells).

For a tutorial on using these replacements, see
[TensorFlow Lite LSTM ops API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/examples/lstm/g3doc/README.md).

For a Colab demonstrating these classes, refer to
[TensorFlowLite_LSTM_Keras_Tutorial](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/examples/lstm/TensorFlowLite_LSTM_Keras_Tutorial.ipynb).

Note: There is no replacement available for
[tf.compat.v1.nn.rnn_cell.GRUCell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/GRUCell).
