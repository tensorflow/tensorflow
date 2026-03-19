page_type: reference
description: A container for label file metadata information.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.metadata_info.LabelFileMd" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_metadata"/>
</div>

# tflite_support.metadata_writers.metadata_info.LabelFileMd

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L113-L130">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A container for label file metadata information.

Inherits From: [`AssociatedFileMd`](../../../tflite_support/metadata_writers/metadata_info/AssociatedFileMd)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p><a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/LabelFileMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_info.LabelFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/LabelFileMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.metadata_info.LabelFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/LabelFileMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_info.LabelFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/LabelFileMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.metadata_info.LabelFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/LabelFileMd"><code>tflite_support.metadata_writers.image_classifier.metadata_info.LabelFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/LabelFileMd"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.metadata_info.LabelFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/LabelFileMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_info.LabelFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/LabelFileMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.metadata_info.LabelFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/LabelFileMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_info.LabelFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/LabelFileMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.metadata_info.LabelFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/LabelFileMd"><code>tflite_support.metadata_writers.object_detector.metadata_info.LabelFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/LabelFileMd"><code>tflite_support.metadata_writers.object_detector.metadata_writer.metadata_info.LabelFileMd</code></a></p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata_writers.metadata_info.LabelFileMd(
    file_path: str, locale: Optional[str] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`file_path`<a id="file_path"></a>
</td>
<td>
file_path of the label file.
</td>
</tr><tr>
<td>
`locale`<a id="locale"></a>
</td>
<td>
locale of the label file [1].
[1]:
https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L154
</td>
</tr>
</table>



## Methods

<h3 id="create_metadata"><code>create_metadata</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L99-L110">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_metadata() -> <a href="../../../tflite_support/metadata_schema_py_generated/AssociatedFileT"><code>tflite_support.metadata_schema_py_generated.AssociatedFileT</code></a>
</code></pre>

Creates the associated file metadata.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Flatbuffers Python object of the associated file metadata.
</td>
</tr>

</table>
