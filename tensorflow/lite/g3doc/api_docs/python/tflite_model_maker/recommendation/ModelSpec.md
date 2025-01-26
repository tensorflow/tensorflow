page_type: reference
description: Recommendation model spec.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.recommendation.ModelSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_model"/>
<meta itemprop="property" content="get_default_quantization_config"/>
<meta itemprop="property" content="compat_tf_versions"/>
</div>

# tflite_model_maker.recommendation.ModelSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/recommendation_spec.py#L23-L50">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Recommendation model spec.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.recommendation.ModelSpec(
    input_spec: <a href="../../tflite_model_maker/recommendation/spec/InputSpec"><code>tflite_model_maker.recommendation.spec.InputSpec</code></a>,
    model_hparams: <a href="../../tflite_model_maker/recommendation/spec/ModelHParams"><code>tflite_model_maker.recommendation.spec.ModelHParams</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_spec`<a id="input_spec"></a>
</td>
<td>
InputSpec, specify data format for input and embedding.
</td>
</tr><tr>
<td>
`model_hparams`<a id="model_hparams"></a>
</td>
<td>
ModelHParams, specify hparams for model achitecture.
</td>
</tr>
</table>



## Methods

<h3 id="create_model"><code>create_model</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/recommendation_spec.py#L40-L46">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_model()
</code></pre>

Creates recommendation model based on params.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Keras model.
</td>
</tr>

</table>



<h3 id="get_default_quantization_config"><code>get_default_quantization_config</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/recommendation_spec.py#L48-L50">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_default_quantization_config()
</code></pre>

Gets the default quantization configuration.






<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
compat_tf_versions<a id="compat_tf_versions"></a>
</td>
<td>
`[2]`
</td>
</tr>
</table>
