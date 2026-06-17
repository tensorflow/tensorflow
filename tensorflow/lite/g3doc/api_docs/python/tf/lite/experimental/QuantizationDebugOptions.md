page_type: reference
description: Debug options to set up a given QuantizationDebugger.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lite.experimental.QuantizationDebugOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.lite.experimental.QuantizationDebugOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/tools/optimize/debugging/python/debugger.py#L56-L117">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Debug options to set up a given QuantizationDebugger.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.lite.experimental.QuantizationDebugOptions(
    layer_debug_metrics: Optional[Mapping[str, Callable[[np.ndarray], float]]] = None,
    model_debug_metrics: Optional[Mapping[str, Callable[[Sequence[np.ndarray], Sequence[np.ndarray]],
        float]]] = None,
    layer_direct_compare_metrics: Optional[Mapping[str, Callable[[Sequence[np.ndarray], Sequence[np.ndarray],
        float, int], float]]] = None,
    denylisted_ops: Optional[List[str]] = None,
    denylisted_nodes: Optional[List[str]] = None,
    fully_quantize: bool = False
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



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`layer_debug_metrics`<a id="layer_debug_metrics"></a>
</td>
<td>
a dict to specify layer debug functions
{function_name_str: function} where the function accepts result of
  NumericVerify Op, which is value difference between float and
  dequantized op results. The function returns single scalar value.
</td>
</tr><tr>
<td>
`model_debug_metrics`<a id="model_debug_metrics"></a>
</td>
<td>
a dict to specify model debug functions
{function_name_str: function} where the function accepts outputs from
  two models, and returns single scalar value for a metric. (e.g.
  accuracy, IoU)
</td>
</tr><tr>
<td>
`layer_direct_compare_metrics`<a id="layer_direct_compare_metrics"></a>
</td>
<td>
a dict to specify layer debug functions
{function_name_str: function}. The signature is different from that of
  `layer_debug_metrics`, and this one gets passed (original float value,
  original quantized value, scale, zero point). The function's
  implementation is responsible for correctly dequantize the quantized
  value to compare. Use this one when comparing diff is not enough.
  (Note) quantized value is passed as int8, so cast to int32 is needed.
</td>
</tr><tr>
<td>
`denylisted_ops`<a id="denylisted_ops"></a>
</td>
<td>
a list of op names which is expected to be removed from
quantization.
</td>
</tr><tr>
<td>
`denylisted_nodes`<a id="denylisted_nodes"></a>
</td>
<td>
a list of op's output tensor names to be removed from
quantization.
</td>
</tr><tr>
<td>
`fully_quantize`<a id="fully_quantize"></a>
</td>
<td>
Bool indicating whether to fully quantize the model.
Besides model body, the input/output will be quantized as well.
Corresponding to mlir_quantize's fully_quantize parameter.
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
when there are duplicate keys
</td>
</tr>
</table>
