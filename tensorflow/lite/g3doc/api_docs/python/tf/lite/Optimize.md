page_type: reference
description: Enum defining the optimizations to apply when generating a tflite model.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.lite.Optimize" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="DEFAULT"/>
<meta itemprop="property" content="EXPERIMENTAL_SPARSITY"/>
<meta itemprop="property" content="OPTIMIZE_FOR_LATENCY"/>
<meta itemprop="property" content="OPTIMIZE_FOR_SIZE"/>
</div>

# tf.lite.Optimize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tensorflow/blob/v2.11.0/tensorflow/lite/python/lite.py#L98-L154">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Enum defining the optimizations to apply when generating a tflite model.

<!-- Placeholder for "Used in" -->

DEFAULT The default optimization strategy that enables post-training
quantization. The type of post-training quantization that will be used is
dependent on the other converter options supplied. Refer to the
[documentation](/tensorflow/lite/g3doc/performance/post_training_quantization.md)
for further information on the types available and how to use them.

OPTIMIZE_FOR_SIZE
    Deprecated. Does the same as DEFAULT.

OPTIMIZE_FOR_LATENCY
    Deprecated. Does the same as DEFAULT.

EXPERIMENTAL_SPARSITY
    Experimental flag, subject to change.

    Enable optimization by taking advantage of the sparse model weights
    trained with pruning.

    The converter will inspect the sparsity pattern of the model weights and
    do its best to improve size and latency.
    The flag can be used alone to optimize float32 models with sparse weights.
    It can also be used together with the DEFAULT optimization mode to
    optimize quantized models with sparse weights.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
DEFAULT<a id="DEFAULT"></a>
</td>
<td>
`<Optimize.DEFAULT: 'DEFAULT'>`
</td>
</tr><tr>
<td>
EXPERIMENTAL_SPARSITY<a id="EXPERIMENTAL_SPARSITY"></a>
</td>
<td>
`<Optimize.EXPERIMENTAL_SPARSITY: 'EXPERIMENTAL_SPARSITY'>`
</td>
</tr><tr>
<td>
OPTIMIZE_FOR_LATENCY<a id="OPTIMIZE_FOR_LATENCY"></a>
</td>
<td>
`<Optimize.OPTIMIZE_FOR_LATENCY: 'OPTIMIZE_FOR_LATENCY'>`
</td>
</tr><tr>
<td>
OPTIMIZE_FOR_SIZE<a id="OPTIMIZE_FOR_SIZE"></a>
</td>
<td>
`<Optimize.OPTIMIZE_FOR_SIZE: 'OPTIMIZE_FOR_SIZE'>`
</td>
</tr>
</table>
