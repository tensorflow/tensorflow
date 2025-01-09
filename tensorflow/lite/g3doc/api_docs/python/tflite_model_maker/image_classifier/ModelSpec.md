page_type: reference
description: A specification of image model.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.image_classifier.ModelSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_default_quantization_config"/>
<meta itemprop="property" content="mean_rgb"/>
<meta itemprop="property" content="stddev_rgb"/>
</div>

# tflite_model_maker.image_classifier.ModelSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/image_spec.py#L28-L59">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A specification of image model.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.image_classifier.ModelSpec(
    uri, compat_tf_versions=None, input_image_shape=None, name=&#x27;&#x27;
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
<li><a href="https://www.tensorflow.org/hub/tutorials/cropnet_on_device">Fine tuning models for plant disease detection</a></li>
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
`uri`<a id="uri"></a>
</td>
<td>
str, URI to the pretrained model.
</td>
</tr><tr>
<td>
`compat_tf_versions`<a id="compat_tf_versions"></a>
</td>
<td>
list of int, compatible TF versions.
</td>
</tr><tr>
<td>
`input_image_shape`<a id="input_image_shape"></a>
</td>
<td>
list of int, input image shape. Default: [224, 224].
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
str, model spec name.
</td>
</tr>
</table>



## Methods

<h3 id="get_default_quantization_config"><code>get_default_quantization_config</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/model_spec/image_spec.py#L56-L59">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_default_quantization_config(
    representative_data
)
</code></pre>

Gets the default quantization configuration.






<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
mean_rgb<a id="mean_rgb"></a>
</td>
<td>
`[0.0]`
</td>
</tr><tr>
<td>
stddev_rgb<a id="stddev_rgb"></a>
</td>
<td>
`[255.0]`
</td>
</tr>
</table>
