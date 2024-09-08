page_type: reference
description: Configuration for post-training quantization.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.config.QuantizationConfig" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="for_dynamic"/>
<meta itemprop="property" content="for_float16"/>
<meta itemprop="property" content="for_int8"/>
<meta itemprop="property" content="get_converter_with_quantization"/>
</div>

# tflite_model_maker.config.QuantizationConfig

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/configs.py#L53-L183">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Configuration for post-training quantization.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.config.QuantizationConfig(
    optimizations=None,
    representative_data=None,
    quantization_steps=None,
    inference_input_type=None,
    inference_output_type=None,
    supported_ops=None,
    supported_types=None,
    experimental_new_quantizer=None
)
</code></pre>




<h3>Used in the notebooks</h3>
<table class="vertical-rules">
  <thead>
    <tr>
      <th>Used in the tutorials</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
  <ul>
    <li><a href="https://www.tensorflow.org/lite/models/modify/model_maker/image_classification">Image classification with TensorFlow Lite Model Maker</a></li>
  </ul>
</td>
    </tr>
  </tbody>
</table>


Refer to
<a href="https://www.tensorflow.org/lite/performance/post_training_quantization">https://www.tensorflow.org/lite/performance/post_training_quantization</a>
for different post-training quantization options.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`optimizations`<a id="optimizations"></a>
</td>
<td>
A list of optimizations to apply when converting the model.
If not set, use `[Optimize.DEFAULT]` by default.
</td>
</tr><tr>
<td>
`representative_data`<a id="representative_data"></a>
</td>
<td>
A DataLoader holding representative data for
post-training quantization.
</td>
</tr><tr>
<td>
`quantization_steps`<a id="quantization_steps"></a>
</td>
<td>
Number of post-training quantization calibration steps
to run.
</td>
</tr><tr>
<td>
`inference_input_type`<a id="inference_input_type"></a>
</td>
<td>
Target data type of real-number input arrays. Allows
for a different type for input arrays. Defaults to None. If set, must be
be `{tf.float32, tf.uint8, tf.int8}`.
</td>
</tr><tr>
<td>
`inference_output_type`<a id="inference_output_type"></a>
</td>
<td>
Target data type of real-number output arrays.
Allows for a different type for output arrays. Defaults to None. If set,
must be `{tf.float32, tf.uint8, tf.int8}`.
</td>
</tr><tr>
<td>
`supported_ops`<a id="supported_ops"></a>
</td>
<td>
Set of OpsSet options supported by the device. Used to Set
converter.target_spec.supported_ops.
</td>
</tr><tr>
<td>
`supported_types`<a id="supported_types"></a>
</td>
<td>
List of types for constant values on the target device.
Supported values are types exported by lite.constants. Frequently, an
optimization choice is driven by the most compact (i.e. smallest) type
in this list (default [constants.FLOAT]).
</td>
</tr><tr>
<td>
`experimental_new_quantizer`<a id="experimental_new_quantizer"></a>
</td>
<td>
Whether to enable experimental new quantizer.
</td>
</tr>
</table>



## Methods

<h3 id="for_dynamic"><code>for_dynamic</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/configs.py#L121-L124">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>for_dynamic()
</code></pre>

Creates configuration for dynamic range quantization.


<h3 id="for_float16"><code>for_float16</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/configs.py#L157-L160">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>for_float16()
</code></pre>

Creates configuration for float16 quantization.


<h3 id="for_int8"><code>for_int8</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/configs.py#L126-L155">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>for_int8(
    representative_data,
    quantization_steps=DEFAULT_QUANTIZATION_STEPS,
    inference_input_type=tf.uint8,
    inference_output_type=tf.uint8,
    supported_ops=tf.lite.OpsSet.TFLITE_BUILTINS_INT8
)
</code></pre>

Creates configuration for full integer quantization.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`representative_data`
</td>
<td>
Representative data used for post-training
quantization.
</td>
</tr><tr>
<td>
`quantization_steps`
</td>
<td>
Number of post-training quantization calibration steps
to run.
</td>
</tr><tr>
<td>
`inference_input_type`
</td>
<td>
Target data type of real-number input arrays. Used
only when `is_integer_only` is True. Must be in `{tf.uint8, tf.int8}`.
</td>
</tr><tr>
<td>
`inference_output_type`
</td>
<td>
Target data type of real-number output arrays. Used
only when `is_integer_only` is True. Must be in `{tf.uint8, tf.int8}`.
</td>
</tr><tr>
<td>
`supported_ops`
</td>
<td>
 Set of <a href="https://www.tensorflow.org/lite/api_docs/python/tf/lite/OpsSet"><code>tf.lite.OpsSet</code></a> options, where each option
represents a set of operators supported by the target device.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
QuantizationConfig.
</td>
</tr>

</table>



<h3 id="get_converter_with_quantization"><code>get_converter_with_quantization</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/configs.py#L162-L183">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_converter_with_quantization(
    converter, **kwargs
)
</code></pre>

Gets TFLite converter with settings for quantization.
