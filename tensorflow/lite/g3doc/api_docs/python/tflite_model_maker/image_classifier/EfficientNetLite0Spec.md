page_type: reference
description: Creates EfficientNet-Lite0 model spec. See also: <a href="../../tflite_model_maker/image_classifier/ModelSpec"><code>tflite_model_maker.image_classifier.ModelSpec</code></a>.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.image_classifier.EfficientNetLite0Spec" />
<meta itemprop="path" content="Stable" />
</div>

# tflite_model_maker.image_classifier.EfficientNetLite0Spec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Creates EfficientNet-Lite0 model spec. See also: <a href="../../tflite_model_maker/image_classifier/ModelSpec"><code>tflite_model_maker.image_classifier.ModelSpec</code></a>.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.image_classifier.EfficientNetLite0Spec(
    *,
    uri=&#x27;https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2&#x27;,
    compat_tf_versions=[1, 2],
    input_image_shape=None,
    name=&#x27;efficientnet_lite0&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->


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
