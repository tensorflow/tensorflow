page_type: reference
description: Debugger for Quantized TensorFlow Lite debug mode models.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lite.experimental.QuantizationDebugger" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_debug_quantized_model"/>
<meta itemprop="property" content="get_nondebug_quantized_model"/>
<meta itemprop="property" content="layer_statistics_dump"/>
<meta itemprop="property" content="run"/>
</div>

# tf.lite.experimental.QuantizationDebugger

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/tools/optimize/debugging/python/debugger.py#L120-L544">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Debugger for Quantized TensorFlow Lite debug mode models.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.lite.experimental.QuantizationDebugger(
    quant_debug_model_path: Optional[str] = None,
    quant_debug_model_content: Optional[bytes] = None,
    float_model_path: Optional[str] = None,
    float_model_content: Optional[bytes] = None,
    debug_dataset: Optional[Callable[[], Iterable[Sequence[np.ndarray]]]] = None,
    debug_options: Optional[<a href="../../../tf/lite/experimental/QuantizationDebugOptions"><code>tf.lite.experimental.QuantizationDebugOptions</code></a>] = None,
    converter: Optional[TFLiteConverter] = None
) -> None
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
    <li><a href="https://www.tensorflow.org/lite/performance/quantization_debugger">Inspecting Quantization Errors with Quantization Debugger</a></li>
  </ul>
</td>
    </tr>
  </tbody>
</table>


This can run the TensorFlow Lite converted models equipped with debug ops and
collect debug information. This debugger calculates statistics from
user-defined post-processing functions as well as default ones.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`quant_debug_model_path`<a id="quant_debug_model_path"></a>
</td>
<td>
Path to the quantized debug TFLite model file.
</td>
</tr><tr>
<td>
`quant_debug_model_content`<a id="quant_debug_model_content"></a>
</td>
<td>
Content of the quantized debug TFLite model.
</td>
</tr><tr>
<td>
`float_model_path`<a id="float_model_path"></a>
</td>
<td>
Path to float TFLite model file.
</td>
</tr><tr>
<td>
`float_model_content`<a id="float_model_content"></a>
</td>
<td>
Content of the float TFLite model.
</td>
</tr><tr>
<td>
`debug_dataset`<a id="debug_dataset"></a>
</td>
<td>
a factory function that returns dataset generator which is
used to generate input samples (list of np.ndarray) for the model. The
generated elements must have same types and shape as inputs to the
model.
</td>
</tr><tr>
<td>
`debug_options`<a id="debug_options"></a>
</td>
<td>
Debug options to debug the given model.
</td>
</tr><tr>
<td>
`converter`<a id="converter"></a>
</td>
<td>
Optional, use converter instead of quantized model.
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
If the debugger was unable to be created.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`options`<a id="options"></a>
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="get_debug_quantized_model"><code>get_debug_quantized_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/tools/optimize/debugging/python/debugger.py#L261-L273">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_debug_quantized_model() -> bytes
</code></pre>

Returns an instrumented quantized model.

Convert the quantized model with the initialized converter and
return bytes for model. The model will be instrumented with numeric
verification operations and should only be used for debugging.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Model bytes corresponding to the model.
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
if converter is not passed to the debugger.
</td>
</tr>
</table>



<h3 id="get_nondebug_quantized_model"><code>get_nondebug_quantized_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/tools/optimize/debugging/python/debugger.py#L247-L259">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_nondebug_quantized_model() -> bytes
</code></pre>

Returns a non-instrumented quantized model.

Convert the quantized model with the initialized converter and
return bytes for nondebug model. The model will not be instrumented with
numeric verification operations.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Model bytes corresponding to the model.
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
if converter is not passed to the debugger.
</td>
</tr>
</table>



<h3 id="layer_statistics_dump"><code>layer_statistics_dump</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/tools/optimize/debugging/python/debugger.py#L521-L544">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>layer_statistics_dump(
    file: IO[str]
) -> None
</code></pre>

Dumps layer statistics into file, in csv format.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`file`
</td>
<td>
file, or file-like object to write.
</td>
</tr>
</table>



<h3 id="run"><code>run</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/tools/optimize/debugging/python/debugger.py#L326-L330">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run() -> None
</code></pre>

Runs models and gets metrics.
