page_type: reference
description: A container for common metadata information of a model.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.metadata_info.GeneralMd" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_metadata"/>
</div>

# tflite_support.metadata_writers.metadata_info.GeneralMd

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L35-L70">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A container for common metadata information of a model.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p><a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/GeneralMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_info.GeneralMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/GeneralMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.metadata_info.GeneralMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/GeneralMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_info.GeneralMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/GeneralMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.metadata_info.GeneralMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/GeneralMd"><code>tflite_support.metadata_writers.image_classifier.metadata_info.GeneralMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/GeneralMd"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.metadata_info.GeneralMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/GeneralMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_info.GeneralMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/GeneralMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.metadata_info.GeneralMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/GeneralMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_info.GeneralMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/GeneralMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.metadata_info.GeneralMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/GeneralMd"><code>tflite_support.metadata_writers.object_detector.metadata_info.GeneralMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/GeneralMd"><code>tflite_support.metadata_writers.object_detector.metadata_writer.metadata_info.GeneralMd</code></a></p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata_writers.metadata_info.GeneralMd(
    name: Optional[str] = None,
    version: Optional[str] = None,
    description: Optional[str] = None,
    author: Optional[str] = None,
    licenses: Optional[str] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`name`<a id="name"></a>
</td>
<td>
name of the model.
</td>
</tr><tr>
<td>
`version`<a id="version"></a>
</td>
<td>
version of the model.
</td>
</tr><tr>
<td>
`description`<a id="description"></a>
</td>
<td>
description of what the model does.
</td>
</tr><tr>
<td>
`author`<a id="author"></a>
</td>
<td>
author of the model.
</td>
</tr><tr>
<td>
`licenses`<a id="licenses"></a>
</td>
<td>
licenses of the model.
</td>
</tr>
</table>



## Methods

<h3 id="create_metadata"><code>create_metadata</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L58-L70">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_metadata() -> <a href="../../../tflite_support/metadata_schema_py_generated/ModelMetadataT"><code>tflite_support.metadata_schema_py_generated.ModelMetadataT</code></a>
</code></pre>

Creates the model metadata based on the general model information.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Flatbuffers Python object of the model metadata.
</td>
</tr>

</table>
