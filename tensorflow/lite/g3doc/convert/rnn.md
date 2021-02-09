# TensorFlow RNN conversion to TensorFlow Lite

## Overview

TensorFlow Lite supports converting TensorFlow RNN models to TensorFlow Lite’s
fused LSTM operations. Fused operations exist to maximize the performance of
their underlying kernel implementations, as well as provide a higher level
interface to define complex transformations like quantizatization.

Since there are many variants of RNN APIs in TensorFlow, our approach has been
two fold:

1.  Provide **native support for standard TensorFlow RNN APIs** like Keras LSTM.
    This is the recommended option.
1.  Provide an **interface** **into the conversion infrastructure for**
    **user-defined** **RNN implementations** to plug in and get converted to
    TensorFlow Lite. We provide a couple of out of box examples of such
    conversion using lingvo’s
    [LSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130)
    and
    [LayerNormalizedLSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137)
    RNN interfaces.

## Converter API

The feature is part of TensorFlow 2.3 release. It is also available through the
[tf-nightly](https://pypi.org/project/tf-nightly/) pip or from head.

This conversion functionality is available when converting to TensorFlow Lite
via a SavedModel or from the Keras model directly. See example usages.

### From saved model

<a id="from_saved_model"></a>

```
# build a saved model. Here concrete_function is the exported function
# corresponding to the TensorFlow model containing one or more
# Keras LSTM layers.
saved_model, saved_model_dir = build_saved_model_lstm(...)
saved_model.save(saved_model_dir, save_format="tf", signatures=concrete_func)

# Convert the model.
converter = TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
```

### From Keras model

```
# build a Keras model
keras_model = build_keras_lstm(...)

# Convert the model.
converter = TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

```

## Example

Keras LSTM to TensorFlow Lite
[Colab](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/experimental_new_converter/Keras_LSTM_fusion_Codelab.ipynb)
illustrates the end to end usage with the TensorFlow Lite interpreter.

## TensorFlow RNNs APIs supported

<a id="rnn_apis"></a>

### Keras LSTM conversion (recommended)

We support out-of-the-box conversion of Keras LSTM to TensorFlow Lite. For
details on how this works please refer to the
[Keras LSTM interface](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/python/keras/layers/recurrent_v2.py#L1238)<span style="text-decoration:space;">
</span>and to the conversion logic
[here](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/utils/lstm_utils.cc#L627).

Also important is to highlight the TensorFlow Lite’s LSTM contract with respect
to the Keras operation definition:

1.  The dimension 0 of the **input** tensor is the batch size.
1.  The dimension 0 of the **recurrent\_weight** tensor is the number of
    outputs.
1.  The **weight** and **recurrent\_kernel** tensors are transposed.
1.  The transposed weight, transposed recurrent\_kernel and **bias** tensors are
    split into 4 equal sized tensors along the dimension 0. These correspond to
    **input gate, forget gate, cell, and output gate**.


#### Keras LSTM Variants

##### Time major

Users may choose time-major or no time-major. Keras LSTM adds a time-major
attribute in the function def attributes. For Unidirectional sequence LSTM, we
can simply map to unidirecional\_sequence\_lstm's
[time major attribute](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/ir/tfl_ops.td#L3902).

##### BiDirectional LSTM

Bidirectional LSTM can be implemented with two Keras LSTM layers, one for
forward and one for backward, see examples
[here](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/python/keras/layers/wrappers.py#L382).
Once we see the go\_backward attribute, we recognize it as backward LSTM, then
we group forward & backward LSTM together. **This is future work.** Currently,
this creates two UnidirectionalSequenceLSTM operations in the TensorFlow Lite
model.

### User-defined LSTM conversion examples

TensorFlow Lite also provides a way to convert user defined LSTM
implementations. Here we use Lingvo’s LSTM as an example of how that can be
implemented. For details please refer to the
[lingvo.LSTMCellSimple interface](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L228)
and the conversion logic
[here](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130).
We also provide an example for another of Lingvo’s LSTM definitions in
[lingvo.LayerNormalizedLSTMCellSimple interface](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L1173)
and its convertion logic
[here](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137).

## “Bring your own TensorFlow RNN” to TensorFlow Lite

If a user's RNN interface is different from the standard supported ones, there
are a couple of options:

**Option 1:** Write adapter code in TensorFlow python to adapt the RNN interface
to the Keras RNN interface. This means a tf.function with
[tf\_implements annotation](https://github.com/tensorflow/community/pull/113) on
the generated RNN interface’s function that is identical to the one generated by
the Keras LSTM layer. After this, the same conversion API used for Keras LSTM
will work.

**Option 2:** If the above is not possible (e.g. the Keras LSTM is missing some
functionality that is currently exposed by TensorFlow Lite’s fused LSTM op like
layer normalization), then extend the TensorFlow Lite converter by writing
custom conversion code and plug it into the prepare-composite-functions
MLIR-pass
[here](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L115).
The function’s interface should be treated like an API contract and should
contain the arguments needed to convert to fused TensorFlow Lite LSTM
operations - i.e. input, bias, weights, projection, layer normalization, etc. It
is preferable for the tensors passed as arguments to this function to have known
rank (i.e. RankedTensorType in MLIR). This makes it much easier to write
conversion code that can assume these tensors as RankedTensorType and helps
transform them to ranked tensors corresponding to the fused TensorFlow Lite
operator’s operands.

A complete example of such conversion flow is Lingvo’s LSTMCellSimple to
TensorFlow Lite conversion.

The LSTMCellSimple in Lingvo is defined
[here](https://github.com/tensorflow/lingvo/blob/91a4609dbc2579748a95110eda59c66d17c594c5/lingvo/core/rnn_cell.py#L228).
Models trained with this LSTM cell can be converted to TensorFlow Lite as
follows:

1.  Wrap all uses of LSTMCellSimple in a tf.function with a tf\_implements
    annotation that is labelled as such (e.g. lingvo.LSTMCellSimple would be a
    good annotation name here). Make sure the tf.function that is generated
    matches the interface of the function expected in the conversion code. This
    is a contract between the model author adding the annotation and the
    conversion code.
1.  Extend the prepare-composite-functions pass to plug in a custom composite op
    to TensorFlow Lite fused LSTM op conversion. See
    [LSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/82abf0dbf316526cd718ae8cd7b11cfcb805805e/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L130)
    conversion code.

    The conversion contract:

1.  **Weight** and **projection** tensors are transposed.

1.  The **{input, recurrent}** to **{cell, input gate, forget gate, output
    gate}** are extracted by slicing the transposed weight tensor.

1.  The **{bias}** to **{cell, input gate, forget gate, output gate}** are
    extracted by slicing the bias tensor.

1.  The **projection** is extracted by slicing the transposed projection tensor.

1.  Similar conversion is written for
    [LayerNormalizedLSTMCellSimple](https://github.com/tensorflow/tensorflow/blob/c11d5d8881fd927165eeb09fd524a80ebaf009f2/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc#L137).

1.  The rest of the TensorFlow Lite conversion infrastructure, including all the
    [MLIR passes](https://github.com/tensorflow/tensorflow/blob/35a3ab91b42503776f428bda574b74b9a99cd110/tensorflow/compiler/mlir/lite/tf_tfl_passes.cc#L57)
    defined as well as the final export to TensorFlow Lite flatbuffer can be
    reused.

## Known issues/limitations

1.  Currently there is support only for converting stateless Keras LSTM (default
    behavior in Keras). Stateful Keras LSTM conversion is future work.
1.  It is still possible to model a stateful Keras LSTM layer using the
    underlying stateless Keras LSTM layer and managing the state explicitly in
    the user program. Such a TensorFlow program can still be converted to
    TensorFlow Lite using the feature being described here.
1.  Bidirectional LSTM is currently modelled as two UnidirectionalSequenceLSTM
    operations in TensorFlow Lite. This will be replaced with a single
    BidirectionalSequenceLSTM op.
