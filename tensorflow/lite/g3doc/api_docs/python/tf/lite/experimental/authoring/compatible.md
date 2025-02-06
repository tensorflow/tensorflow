page_type: reference
description: Wraps tf.function into a callable function with TFLite compatibility checking.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lite.experimental.authoring.compatible" />
<meta itemprop="path" content="Stable" />
</div>

# tf.lite.experimental.authoring.compatible

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/authoring/authoring.py#L268-L306">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Wraps <a href="https://www.tensorflow.org/api_docs/python/tf/function"><code>tf.function</code></a> into a callable function with TFLite compatibility checking.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.lite.experimental.authoring.compatible(
    target=None, converter_target_spec=None, **kwargs
)
</code></pre>




<h3>Used in the notebooks</h3>
<table class="vertical-rules">
  <thead>
    <tr>
      <th>Used in the guide</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
  <ul>
    <li><a href="https://www.tensorflow.org/lite/guide/authoring">TFLite Authoring Tool</a></li>
  </ul>
</td>
    </tr>
  </tbody>
</table>



#### Example:



```python
@tf.lite.experimental.authoring.compatible
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None], dtype=tf.float32)
])
def f(x):
    return tf.cosh(x)

result = f(tf.constant([0.0]))
# COMPATIBILITY WARNING: op 'tf.Cosh' require(s) "Select TF Ops" for model
# conversion for TensorFlow Lite.
# Op: tf.Cosh
#   - tensorflow/python/framework/op_def_library.py:748
#   - tensorflow/python/ops/gen_math_ops.py:2458
#   - <stdin>:6
```

Warning: Experimental interface, subject to change.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`target`<a id="target"></a>
</td>
<td>
A <a href="https://www.tensorflow.org/api_docs/python/tf/function"><code>tf.function</code></a> to decorate.
</td>
</tr><tr>
<td>
`converter_target_spec`<a id="converter_target_spec"></a>
</td>
<td>
target_spec of TFLite converter parameter.
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
The keyword arguments of the decorator class _Compatible.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A callable object of `tf.lite.experimental.authoring._Compatible`.
</td>
</tr>

</table>
