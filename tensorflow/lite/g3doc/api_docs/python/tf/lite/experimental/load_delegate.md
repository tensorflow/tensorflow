page_type: reference
description: Returns loaded Delegate object.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lite.experimental.load_delegate" />
<meta itemprop="path" content="Stable" />
</div>

# tf.lite.experimental.load_delegate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/interpreter.py#L133-L178">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns loaded Delegate object.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.lite.experimental.load_delegate(
    library, options=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Example usage:



```
import tensorflow as tf

try:
  delegate = tf.lite.experimental.load_delegate('delegate.so')
except ValueError:
  // Fallback to CPU

if delegate:
  interpreter = tf.lite.Interpreter(
      model_path='model.tflite',
      experimental_delegates=[delegate])
else:
  interpreter = tf.lite.Interpreter(model_path='model.tflite')
```

This is typically used to leverage EdgeTPU for running TensorFlow Lite models.
For more information see: <a href="https://coral.ai/docs/edgetpu/tflite-python/">https://coral.ai/docs/edgetpu/tflite-python/</a>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`library`<a id="library"></a>
</td>
<td>
Name of shared library containing the
[TfLiteDelegate](https://www.tensorflow.org/lite/performance/delegates).
</td>
</tr><tr>
<td>
`options`<a id="options"></a>
</td>
<td>
Dictionary of options that are required to load the delegate. All
keys and values in the dictionary should be convertible to str. Consult
the documentation of the specific delegate for required and legal options.
(default None)
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Delegate object.
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
Delegate failed to load.
</td>
</tr><tr>
<td>
`RuntimeError`<a id="RuntimeError"></a>
</td>
<td>
If delegate loading is used on unsupported platform.
</td>
</tr>
</table>
