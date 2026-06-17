page_type: reference
description: Writes the metadata and associated files into a TFLite model.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.audio_classifier.metadata_writer.MetadataWriter" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_from_metadata"/>
<meta itemprop="property" content="create_from_metadata_info"/>
<meta itemprop="property" content="get_metadata_json"/>
<meta itemprop="property" content="get_populated_metadata_json"/>
<meta itemprop="property" content="populate"/>
</div>

# tflite_support.metadata_writers.audio_classifier.metadata_writer.MetadataWriter

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_writer.py#L28-L207">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Writes the metadata and associated files into a TFLite model.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p><a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/audio_classifier/metadata_writer/MetadataWriter"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.MetadataWriter</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/audio_classifier/metadata_writer/MetadataWriter"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.MetadataWriter</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/audio_classifier/metadata_writer/MetadataWriter"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.MetadataWriter</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/audio_classifier/metadata_writer/MetadataWriter"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.MetadataWriter</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/audio_classifier/metadata_writer/MetadataWriter"><code>tflite_support.metadata_writers.object_detector.metadata_writer.MetadataWriter</code></a></p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata_writers.audio_classifier.metadata_writer.MetadataWriter(
    model_buffer: bytearray,
    metadata_buffer: Optional[bytearray] = None,
    associated_files: Optional[List[str]] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model_buffer`<a id="model_buffer"></a>
</td>
<td>
valid buffer of the model file.
</td>
</tr><tr>
<td>
`metadata_buffer`<a id="metadata_buffer"></a>
</td>
<td>
valid buffer of the metadata.
</td>
</tr><tr>
<td>
`associated_files`<a id="associated_files"></a>
</td>
<td>
path to the associated files to be populated.
</td>
</tr>
</table>



## Methods

<h3 id="create_from_metadata"><code>create_from_metadata</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_writer.py#L92-L162">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_metadata(
    model_buffer: bytearray,
    model_metadata: Optional[<a href="../../../../tflite_support/metadata_schema_py_generated/ModelMetadataT"><code>tflite_support.metadata_schema_py_generated.ModelMetadataT</code></a>] = None,
    input_metadata: Optional[List[_metadata_fb.TensorMetadataT]] = None,
    output_metadata: Optional[List[_metadata_fb.TensorMetadataT]] = None,
    associated_files: Optional[List[str]] = None,
    input_process_units: Optional[List[_metadata_fb.ProcessUnitT]] = None,
    output_process_units: Optional[List[_metadata_fb.ProcessUnitT]] = None
)
</code></pre>

Creates MetadataWriter based on the metadata Flatbuffers Python Objects.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model_buffer`
</td>
<td>
valid buffer of the model file.
</td>
</tr><tr>
<td>
`model_metadata`
</td>
<td>
general model metadata [1]. The subgraph_metadata will be
refreshed with input_metadata and output_metadata.
</td>
</tr><tr>
<td>
`input_metadata`
</td>
<td>
a list of metadata of the input tensors [2].
</td>
</tr><tr>
<td>
`output_metadata`
</td>
<td>
a list of metadata of the output tensors [3].
</td>
</tr><tr>
<td>
`associated_files`
</td>
<td>
path to the associated files to be populated.
</td>
</tr><tr>
<td>
`input_process_units`
</td>
<td>
a lits of metadata of the input process units [4].
</td>
</tr><tr>
<td>
`output_process_units`
</td>
<td>
a lits of metadata of the output process units [5].
[1]:
  https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L640-L681
[2]:
  https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L590
[3]:
  https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L599
[4]:
  https://github.com/tensorflow/tflite-support/blob/b5cc57c74f7990d8bc055795dfe8d50267064a57/tensorflow_lite_support/metadata/metadata_schema.fbs#L646
[5]:
  https://github.com/tensorflow/tflite-support/blob/b5cc57c74f7990d8bc055795dfe8d50267064a57/tensorflow_lite_support/metadata/metadata_schema.fbs#L650
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A MetadataWriter Object.
</td>
</tr>

</table>



<h3 id="create_from_metadata_info"><code>create_from_metadata_info</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_writer.py#L47-L90">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_metadata_info(
    model_buffer: bytearray,
    general_md: Optional[<a href="../../../../tflite_support/metadata_writers/metadata_info/GeneralMd"><code>tflite_support.metadata_writers.metadata_info.GeneralMd</code></a>] = None,
    input_md: Optional[List[Type[metadata_info.TensorMd]]] = None,
    output_md: Optional[List[Type[metadata_info.TensorMd]]] = None,
    associated_files: Optional[List[str]] = None
)
</code></pre>

Creates MetadataWriter based on the metadata information.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model_buffer`
</td>
<td>
valid buffer of the model file.
</td>
</tr><tr>
<td>
`general_md`
</td>
<td>
general information about the model.
</td>
</tr><tr>
<td>
`input_md`
</td>
<td>
metadata information of the input tensors.
</td>
</tr><tr>
<td>
`output_md`
</td>
<td>
metadata information of the output tensors.
</td>
</tr><tr>
<td>
`associated_files`
</td>
<td>
path to the associated files to be populated.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A MetadataWriter Object.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if the tensor names from `input_md` and `output_md` do not
match the tensor names read from the model.
</td>
</tr>
</table>



<h3 id="get_metadata_json"><code>get_metadata_json</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_writer.py#L183-L194">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_metadata_json() -> str
</code></pre>

Gets the generated JSON metadata string before populated into model.

This method returns the metadata buffer before populated into the model.
More fields could be filled by MetadataPopulator, such as
min_parser_version. Use get_populated_metadata_json() if you want to get the
final metadata string.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The generated JSON metadata string before populated into model.
</td>
</tr>

</table>



<h3 id="get_populated_metadata_json"><code>get_populated_metadata_json</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_writer.py#L196-L207">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_populated_metadata_json() -> str
</code></pre>

Gets the generated JSON metadata string after populated into model.

More fields could be filled by MetadataPopulator, such as
min_parser_version. Use get_metadata_json() if you want to get the
original metadata string.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The generated JSON metadata string after populated into model.
</td>
</tr>

</table>



<h3 id="populate"><code>populate</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_writer.py#L164-L181">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>populate() -> bytearray
</code></pre>

Populates the metadata and label file to the model file.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A new model buffer with the metadata and associated files.
</td>
</tr>

</table>
