page_type: reference
description: Representative dataset used to optimize the model.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lite.RepresentativeDataset" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.lite.RepresentativeDataset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/lite.py#L158-L179">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Representative dataset used to optimize the model.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.lite.RepresentativeDataset(
    input_gen
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is a generator function that provides a small dataset to calibrate or
estimate the range, i.e, (min, max) of all floating-point arrays in the model
(such as model input, activation outputs of intermediate layers, and model
output) for quantization. Usually, this is a small subset of a few hundred
samples randomly chosen, in no particular order, from the training or
evaluation dataset.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_gen`<a id="input_gen"></a>
</td>
<td>
A generator function that generates input samples for the
model and has the same order, type and shape as the inputs to the model.
Usually, this is a small subset of a few hundred samples randomly
chosen, in no particular order, from the training or evaluation dataset.
</td>
</tr>
</table>
