page_type: reference
description: A container for common associated file metadata information.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.metadata_info.AssociatedFileMd" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_metadata"/>
</div>

# tflite_support.metadata_writers.metadata_info.AssociatedFileMd

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L73-L110">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A container for common associated file metadata information.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p><a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/AssociatedFileMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_info.AssociatedFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/AssociatedFileMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.metadata_info.AssociatedFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/AssociatedFileMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_info.AssociatedFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/AssociatedFileMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.metadata_info.AssociatedFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/AssociatedFileMd"><code>tflite_support.metadata_writers.image_classifier.metadata_info.AssociatedFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/AssociatedFileMd"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.metadata_info.AssociatedFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/AssociatedFileMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_info.AssociatedFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/AssociatedFileMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.metadata_info.AssociatedFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/AssociatedFileMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_info.AssociatedFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/AssociatedFileMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.metadata_info.AssociatedFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/AssociatedFileMd"><code>tflite_support.metadata_writers.object_detector.metadata_info.AssociatedFileMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/AssociatedFileMd"><code>tflite_support.metadata_writers.object_detector.metadata_writer.metadata_info.AssociatedFileMd</code></a></p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata_writers.metadata_info.AssociatedFileMd(
    file_path: str,
    description: Optional[str] = None,
    file_type: Optional[<a href="../../../tflite_support/metadata_schema_py_generated/AssociatedFileType"><code>tflite_support.metadata_schema_py_generated.AssociatedFileType</code></a>] = _metadata_fb.AssociatedFileType.UNKNOWN,
    locale: Optional[str] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`file_path`<a id="file_path"></a>
</td>
<td>
path to the associated file.
</td>
</tr><tr>
<td>
`description`<a id="description"></a>
</td>
<td>
description of the associated file.
</td>
</tr><tr>
<td>
`file_type`<a id="file_type"></a>
</td>
<td>
file type of the associated file [1].
</td>
</tr><tr>
<td>
`locale`<a id="locale"></a>
</td>
<td>
locale of the associated file [2].
[1]:
  https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L77
[2]:
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
