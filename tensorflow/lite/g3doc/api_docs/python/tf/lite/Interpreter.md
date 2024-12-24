page_type: reference
description: Interpreter interface for running TensorFlow Lite models.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lite.Interpreter" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="allocate_tensors"/>
<meta itemprop="property" content="get_input_details"/>
<meta itemprop="property" content="get_output_details"/>
<meta itemprop="property" content="get_signature_list"/>
<meta itemprop="property" content="get_signature_runner"/>
<meta itemprop="property" content="get_tensor"/>
<meta itemprop="property" content="get_tensor_details"/>
<meta itemprop="property" content="invoke"/>
<meta itemprop="property" content="reset_all_variables"/>
<meta itemprop="property" content="resize_tensor_input"/>
<meta itemprop="property" content="set_tensor"/>
<meta itemprop="property" content="tensor"/>
</div>

# tf.lite.Interpreter

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/interpreter.py#L354-L942">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Interpreter interface for running TensorFlow Lite models.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.lite.Interpreter(
    model_path=None,
    model_content=None,
    experimental_delegates=None,
    num_threads=None,
    experimental_op_resolver_type=<a href="../../tf/lite/experimental/OpResolverType#AUTO"><code>tf.lite.experimental.OpResolverType.AUTO</code></a>,
    experimental_preserve_all_tensors=False
)
</code></pre>




<h3>Used in the notebooks</h3>
<table class="vertical-rules">
  <thead>
    <tr>
      <th>Used in the guide</th>
<th>Used in the tutorials</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
  <ul>
    <li><a href="https://www.tensorflow.org/lite/guide/signatures">Signatures in TensorFlow Lite</a></li>
<li><a href="https://www.tensorflow.org/model_optimization/guide/combine/cqat_example">Cluster preserving quantization aware training (CQAT) Keras example</a></li>
<li><a href="https://www.tensorflow.org/model_optimization/guide/combine/pqat_example">Pruning preserving quantization aware training (PQAT) Keras example</a></li>
<li><a href="https://www.tensorflow.org/model_optimization/guide/clustering/clustering_example">Weight clustering in Keras example</a></li>
<li><a href="https://www.tensorflow.org/model_optimization/guide/combine/pcqat_example">Sparsity and cluster preserving quantization aware training (PCQAT) Keras example</a></li>
  </ul>
</td>
<td>
  <ul>
    <li><a href="https://www.tensorflow.org/lite/performance/post_training_integer_quant">Post-training integer quantization</a></li>
<li><a href="https://www.tensorflow.org/lite/examples/jax_conversion/overview">Jax Model Conversion For TFLite</a></li>
<li><a href="https://www.tensorflow.org/lite/examples/on_device_training/overview">On-Device Training with TensorFlow Lite</a></li>
<li><a href="https://www.tensorflow.org/lite/examples/style_transfer/overview">Artistic Style Transfer with TensorFlow Lite</a></li>
<li><a href="https://www.tensorflow.org/lite/performance/post_training_float16_quant">Post-training float16 quantization</a></li>
  </ul>
</td>
    </tr>
  </tbody>
</table>


Models obtained from `TfLiteConverter` can be run in Python with
`Interpreter`.

As an example, lets generate a simple Keras model and convert it to TFLite
(`TfLiteConverter` also supports other input formats with `from_saved_model`
and `from_concrete_function`)

<pre class="devsite-click-to-copy prettyprint lang-py">
<code class="devsite-terminal" data-terminal-prefix="&gt;&gt;&gt;">x = np.array([[1.], [2.]])</code>
<code class="devsite-terminal" data-terminal-prefix="&gt;&gt;&gt;">y = np.array([[2.], [4.]])</code>
<code class="devsite-terminal" data-terminal-prefix="&gt;&gt;&gt;">model = tf.keras.models.Sequential([</code>
<code class="devsite-terminal" data-terminal-prefix="...">          tf.keras.layers.Dropout(0.2),</code>
<code class="devsite-terminal" data-terminal-prefix="...">          tf.keras.layers.Dense(units=1, input_shape=[1])</code>
<code class="devsite-terminal" data-terminal-prefix="...">        ])</code>
<code class="devsite-terminal" data-terminal-prefix="&gt;&gt;&gt;">model.compile(optimizer=&#x27;sgd&#x27;, loss=&#x27;mean_squared_error&#x27;)</code>
<code class="devsite-terminal" data-terminal-prefix="&gt;&gt;&gt;">model.fit(x, y, epochs=1)</code>
<code class="devsite-terminal" data-terminal-prefix="&gt;&gt;&gt;">converter = tf.lite.TFLiteConverter.from_keras_model(model)</code>
<code class="devsite-terminal" data-terminal-prefix="&gt;&gt;&gt;">tflite_model = converter.convert()</code>
</pre>


`tflite_model` can be saved to a file and loaded later, or directly into the
`Interpreter`. Since TensorFlow Lite pre-plans tensor allocations to optimize
inference, the user needs to call `allocate_tensors()` before any inference.

<pre class="devsite-click-to-copy prettyprint lang-py">
<code class="devsite-terminal" data-terminal-prefix="&gt;&gt;&gt;">interpreter = tf.lite.Interpreter(model_content=tflite_model)</code>
<code class="devsite-terminal" data-terminal-prefix="&gt;&gt;&gt;">interpreter.allocate_tensors()  # Needed before execution!</code>
</pre>


#### Sample execution:

<pre class="devsite-click-to-copy prettyprint lang-py">
<code class="devsite-terminal" data-terminal-prefix="&gt;&gt;&gt;">output = interpreter.get_output_details()[0]  # Model has single output.</code>
<code class="devsite-terminal" data-terminal-prefix="&gt;&gt;&gt;">input = interpreter.get_input_details()[0]  # Model has single input.</code>
<code class="devsite-terminal" data-terminal-prefix="&gt;&gt;&gt;">input_data = tf.constant(1., shape=[1, 1])</code>
<code class="devsite-terminal" data-terminal-prefix="&gt;&gt;&gt;">interpreter.set_tensor(input[&#x27;index&#x27;], input_data)</code>
<code class="devsite-terminal" data-terminal-prefix="&gt;&gt;&gt;">interpreter.invoke()</code>
<code class="devsite-terminal" data-terminal-prefix="&gt;&gt;&gt;">interpreter.get_tensor(output[&#x27;index&#x27;]).shape</code>
<code class="no-select nocode">(1, 1)</code>
</pre>


Use `get_signature_runner()` for a more user-friendly inference API.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model_path`<a id="model_path"></a>
</td>
<td>
Path to TF-Lite Flatbuffer file.
</td>
</tr><tr>
<td>
`model_content`<a id="model_content"></a>
</td>
<td>
Content of model.
</td>
</tr><tr>
<td>
`experimental_delegates`<a id="experimental_delegates"></a>
</td>
<td>
Experimental. Subject to change. List of
[TfLiteDelegate](https://www.tensorflow.org/lite/performance/delegates)
  objects returned by lite.load_delegate().
</td>
</tr><tr>
<td>
`num_threads`<a id="num_threads"></a>
</td>
<td>
Sets the number of threads used by the interpreter and
available to CPU kernels. If not set, the interpreter will use an
implementation-dependent default number of threads. Currently, only a
subset of kernels, such as conv, support multi-threading. num_threads
should be >= -1. Setting num_threads to 0 has the effect to disable
multithreading, which is equivalent to setting num_threads to 1. If set
to the value -1, the number of threads used will be
implementation-defined and platform-dependent.
</td>
</tr><tr>
<td>
`experimental_op_resolver_type`<a id="experimental_op_resolver_type"></a>
</td>
<td>
The op resolver used by the interpreter. It
must be an instance of OpResolverType. By default, we use the built-in
op resolver which corresponds to tflite::ops::builtin::BuiltinOpResolver
in C++.
</td>
</tr><tr>
<td>
`experimental_preserve_all_tensors`<a id="experimental_preserve_all_tensors"></a>
</td>
<td>
If true, then intermediate tensors used
during computation are preserved for inspection, and if the passed op
resolver type is AUTO or BUILTIN, the type will be changed to
BUILTIN_WITHOUT_DEFAULT_DELEGATES so that no Tensorflow Lite default
delegates are applied. If false, getting intermediate tensors could
result in undefined values or None, especially when the graph is
successfully modified by the Tensorflow Lite default delegate.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
If the interpreter was unable to create.
</td>
</tr>
</table>



## Methods

<h3 id="allocate_tensors"><code>allocate_tensors</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/interpreter.py#L511-L513">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>allocate_tensors()
</code></pre>




<h3 id="get_input_details"><code>get_input_details</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/interpreter.py#L651-L679">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_input_details()
</code></pre>

Gets model input tensor details.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list in which each item is a dictionary with details about
an input tensor. Each dictionary contains the following fields
that describe the tensor:

+ `name`: The tensor name.
+ `index`: The tensor index in the interpreter.
+ `shape`: The shape of the tensor.
+ `shape_signature`: Same as `shape` for models with known/fixed shapes.
  If any dimension sizes are unknown, they are indicated with `-1`.

+ `dtype`: The numpy data type (such as `np.int32` or `np.uint8`).
+ `quantization`: Deprecated, use `quantization_parameters`. This field
  only works for per-tensor quantization, whereas
  `quantization_parameters` works in all cases.
+ `quantization_parameters`: A dictionary of parameters used to quantize
  the tensor:
  ~ `scales`: List of scales (one if per-tensor quantization).
  ~ `zero_points`: List of zero_points (one if per-tensor quantization).
  ~ `quantized_dimension`: Specifies the dimension of per-axis
  quantization, in the case of multiple scales/zero_points.
+ `sparsity_parameters`: A dictionary of parameters used to encode a
  sparse tensor. This is empty if the tensor is dense.
</td>
</tr>

</table>



<h3 id="get_output_details"><code>get_output_details</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/interpreter.py#L728-L738">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_output_details()
</code></pre>

Gets model output tensor details.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list in which each item is a dictionary with details about
an output tensor. The dictionary contains the same fields as
described for `get_input_details()`.
</td>
</tr>

</table>



<h3 id="get_signature_list"><code>get_signature_list</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/interpreter.py#L740-L765">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_signature_list()
</code></pre>

Gets list of SignatureDefs in the model.

Example,

```
signatures = interpreter.get_signature_list()
print(signatures)

# {
#   'add': {'inputs': ['x', 'y'], 'outputs': ['output_0']}
# }

Then using the names in the signature list you can get a callable from
get_signature_runner().
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of SignatureDef details in a dictionary structure.
It is keyed on the SignatureDef method name, and the value holds
dictionary of inputs and outputs.
</td>
</tr>

</table>



<h3 id="get_signature_runner"><code>get_signature_runner</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/interpreter.py#L790-L835">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_signature_runner(
    signature_key=None
)
</code></pre>

Gets callable for inference of specific SignatureDef.

Example usage,

```
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
fn = interpreter.get_signature_runner('div_with_remainder')
output = fn(x=np.array([3]), y=np.array([2]))
print(output)
# {
#   'quotient': array([1.], dtype=float32)
#   'remainder': array([1.], dtype=float32)
# }
```

None can be passed for signature_key if the model has a single Signature
only.

All names used are this specific SignatureDef names.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`signature_key`
</td>
<td>
Signature key for the SignatureDef, it can be None if and
only if the model has a single SignatureDef. Default value is None.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
This returns a callable that can run inference for SignatureDef defined
by argument 'signature_key'.
The callable will take key arguments corresponding to the arguments of the
SignatureDef, that should have numpy values.
The callable will returns dictionary that maps from output names to numpy
values of the computed results.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If passed signature_key is invalid.
</td>
</tr>
</table>



<h3 id="get_tensor"><code>get_tensor</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/interpreter.py#L837-L852">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_tensor(
    tensor_index, subgraph_index=0
)
</code></pre>

Gets the value of the output tensor (get a copy).

If you wish to avoid the copy, use `tensor()`. This function cannot be used
to read intermediate results.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`tensor_index`
</td>
<td>
Tensor index of tensor to get. This value can be gotten from
the 'index' field in get_output_details.
</td>
</tr><tr>
<td>
`subgraph_index`
</td>
<td>
Index of the subgraph to fetch the tensor. Default value
is 0, which means to fetch from the primary subgraph.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
a numpy array.
</td>
</tr>

</table>



<h3 id="get_tensor_details"><code>get_tensor_details</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/interpreter.py#L634-L649">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_tensor_details()
</code></pre>

Gets tensor details for every tensor with valid tensor details.

Tensors where required information about the tensor is not found are not
added to the list. This includes temporary tensors without a name.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of dictionaries containing tensor information.
</td>
</tr>

</table>



<h3 id="invoke"><code>invoke</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/interpreter.py#L904-L917">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>invoke()
</code></pre>

Invoke the interpreter.

Be sure to set the input sizes, allocate tensors and fill values before
calling this. Also, note that this function releases the GIL so heavy
computation can be done in the background while the Python interpreter
continues. No other function on this object should be called while the
invoke() call has not finished.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
When the underlying interpreter fails raise ValueError.
</td>
</tr>
</table>



<h3 id="reset_all_variables"><code>reset_all_variables</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/interpreter.py#L919-L920">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_all_variables()
</code></pre>




<h3 id="resize_tensor_input"><code>resize_tensor_input</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/interpreter.py#L699-L726">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>resize_tensor_input(
    input_index, tensor_size, strict=False
)
</code></pre>

Resizes an input tensor.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_index`
</td>
<td>
Tensor index of input to set. This value can be gotten from
the 'index' field in get_input_details.
</td>
</tr><tr>
<td>
`tensor_size`
</td>
<td>
The tensor_shape to resize the input to.
</td>
</tr><tr>
<td>
`strict`
</td>
<td>
Only unknown dimensions can be resized when `strict` is True.
Unknown dimensions are indicated as `-1` in the `shape_signature`
attribute of a given tensor. (default False)
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the interpreter could not resize the input tensor.
</td>
</tr>
</table>



#### Usage:


```
interpreter = Interpreter(model_content=tflite_model)
interpreter.resize_tensor_input(0, [num_test_images, 224, 224, 3])
interpreter.allocate_tensors()
interpreter.set_tensor(0, test_images)
interpreter.invoke()
```

<h3 id="set_tensor"><code>set_tensor</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/interpreter.py#L681-L697">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_tensor(
    tensor_index, value
)
</code></pre>

Sets the value of the input tensor.

Note this copies data in `value`.

If you want to avoid copying, you can use the `tensor()` function to get a
numpy buffer pointing to the input buffer in the tflite interpreter.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`tensor_index`
</td>
<td>
Tensor index of tensor to set. This value can be gotten from
the 'index' field in get_input_details.
</td>
</tr><tr>
<td>
`value`
</td>
<td>
Value of tensor to set.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the interpreter could not set the tensor.
</td>
</tr>
</table>



<h3 id="tensor"><code>tensor</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/interpreter.py#L854-L902">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tensor(
    tensor_index
)
</code></pre>

Returns function that gives a numpy view of the current tensor buffer.

This allows reading and writing to this tensors w/o copies. This more
closely mirrors the C++ Interpreter class interface's tensor() member, hence
the name. Be careful to not hold these output references through calls
to `allocate_tensors()` and `invoke()`. This function cannot be used to read
intermediate results.

#### Usage:



```
interpreter.allocate_tensors()
input = interpreter.tensor(interpreter.get_input_details()[0]["index"])
output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
for i in range(10):
  input().fill(3.)
  interpreter.invoke()
  print("inference %s" % output())
```

Notice how this function avoids making a numpy array directly. This is
because it is important to not hold actual numpy views to the data longer
than necessary. If you do, then the interpreter can no longer be invoked,
because it is possible the interpreter would resize and invalidate the
referenced tensors. The NumPy API doesn't allow any mutability of the
the underlying buffers.

#### WRONG:



```
input = interpreter.tensor(interpreter.get_input_details()[0]["index"])()
output = interpreter.tensor(interpreter.get_output_details()[0]["index"])()
interpreter.allocate_tensors()  # This will throw RuntimeError
for i in range(10):
  input.fill(3.)
  interpreter.invoke()  # this will throw RuntimeError since input,output
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`tensor_index`
</td>
<td>
Tensor index of tensor to get. This value can be gotten from
the 'index' field in get_output_details.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A function that can return a new numpy array pointing to the internal
TFLite tensor state at any point. It is safe to hold the function forever,
but it is not safe to hold the numpy array forever.
</td>
</tr>

</table>
