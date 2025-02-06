page_type: reference
description: Packs metadata and associated files into TensorFlow Lite model file.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata.MetadataPopulator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_model_buffer"/>
<meta itemprop="property" content="get_packed_associated_file_list"/>
<meta itemprop="property" content="get_recorded_associated_file_list"/>
<meta itemprop="property" content="load_associated_file_buffers"/>
<meta itemprop="property" content="load_associated_files"/>
<meta itemprop="property" content="load_metadata_and_associated_files"/>
<meta itemprop="property" content="load_metadata_buffer"/>
<meta itemprop="property" content="load_metadata_file"/>
<meta itemprop="property" content="populate"/>
<meta itemprop="property" content="with_model_buffer"/>
<meta itemprop="property" content="with_model_file"/>
<meta itemprop="property" content="METADATA_FIELD_NAME"/>
<meta itemprop="property" content="METADATA_FILE_IDENTIFIER"/>
<meta itemprop="property" content="TFLITE_FILE_IDENTIFIER"/>
</div>

# tflite_support.metadata.MetadataPopulator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata.py#L99-L642">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Packs metadata and associated files into TensorFlow Lite model file.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata.MetadataPopulator(
    model_file
)
</code></pre>



<!-- Placeholder for "Used in" -->

MetadataPopulator can be used to populate metadata and model associated files
into a model file or a model buffer (in bytearray). It can also help to
inspect list of files that have been packed into the model or are supposed to
be packed into the model.

The metadata file (or buffer) should be generated based on the metadata
schema:
third_party/tensorflow/lite/schema/metadata_schema.fbs

#### Example usage:


Populate matadata and label file into an image classifier model.

First, based on metadata_schema.fbs, generate the metadata for this image
classifer model using Flatbuffers API. Attach the label file onto the ouput
tensor (the tensor of probabilities) in the metadata.

Then, pack the metadata and label file into the model as follows.

  ```python
  # Populating a metadata file (or a metadta buffer) and associated files to
  a model file:
  populator = MetadataPopulator.with_model_file(model_file)
  # For metadata buffer (bytearray read from the metadata file), use:
  # populator.load_metadata_buffer(metadata_buf)
  populator.load_metadata_file(metadata_file)
  populator.load_associated_files([label.txt])
  # For associated file buffer (bytearray read from the file), use:
  # populator.load_associated_file_buffers({"label.txt": b"file content"})
  populator.populate()

  # Populating a metadata file (or a metadta buffer) and associated files to
  a model buffer:
  populator = MetadataPopulator.with_model_buffer(model_buf)
  populator.load_metadata_file(metadata_file)
  populator.load_associated_files([label.txt])
  populator.populate()
  # Writing the updated model buffer into a file.
  updated_model_buf = populator.get_model_buffer()
  with open("updated_model.tflite", "wb") as f:
    f.write(updated_model_buf)

  # Transferring metadata and associated files from another TFLite model:
  populator = MetadataPopulator.with_model_buffer(model_buf)
  populator_dst.load_metadata_and_associated_files(src_model_buf)
  populator_dst.populate()
  updated_model_buf = populator.get_model_buffer()
  with open("updated_model.tflite", "wb") as f:
    f.write(updated_model_buf)
  ```

Note that existing metadata buffer (if applied) will be overridden by the new
metadata buffer.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model_file`<a id="model_file"></a>
</td>
<td>
valid path to a TensorFlow Lite model file.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`IOError`<a id="IOError"></a>
</td>
<td>
File not found.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
the model does not have the expected flatbuffer identifer.
</td>
</tr>
</table>



## Methods

<h3 id="get_model_buffer"><code>get_model_buffer</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata.py#L214-L221">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_model_buffer()
</code></pre>

Gets the buffer of the model with packed metadata and associated files.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Model buffer (in bytearray).
</td>
</tr>

</table>



<h3 id="get_packed_associated_file_list"><code>get_packed_associated_file_list</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata.py#L223-L233">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_packed_associated_file_list()
</code></pre>

Gets a list of associated files packed to the model file.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of packed associated files.
</td>
</tr>

</table>



<h3 id="get_recorded_associated_file_list"><code>get_recorded_associated_file_list</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata.py#L235-L254">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_recorded_associated_file_list()
</code></pre>

Gets a list of associated files recorded in metadata of the model file.

Associated files may be attached to a model, a subgraph, or an input/output
tensor.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of recorded associated files.
</td>
</tr>

</table>



<h3 id="load_associated_file_buffers"><code>load_associated_file_buffers</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata.py#L256-L268">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_associated_file_buffers(
    associated_files
)
</code></pre>

Loads the associated file buffers (in bytearray) to be populated.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`associated_files`
</td>
<td>
a dictionary of associated file names and corresponding
file buffers, such as {"file.txt": b"file content"}. If pass in file
  paths for the file name, only the basename will be populated.
</td>
</tr>
</table>



<h3 id="load_associated_files"><code>load_associated_files</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata.py#L270-L283">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_associated_files(
    associated_files
)
</code></pre>

Loads associated files that to be concatenated after the model file.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`associated_files`
</td>
<td>
list of file paths.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`IOError`
</td>
<td>
  File not found.
</td>
</tr>
</table>



<h3 id="load_metadata_and_associated_files"><code>load_metadata_and_associated_files</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata.py#L343-L359">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_metadata_and_associated_files(
    src_model_buf
)
</code></pre>

Loads the metadata and associated files from another model buffer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`src_model_buf`
</td>
<td>
source model buffer (in bytearray) with metadata and
associated files.
</td>
</tr>
</table>



<h3 id="load_metadata_buffer"><code>load_metadata_buffer</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata.py#L285-L321">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_metadata_buffer(
    metadata_buf
)
</code></pre>

Loads the metadata buffer (in bytearray) to be populated.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`metadata_buf`
</td>
<td>
metadata buffer (in bytearray) to be populated.
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
The metadata to be populated is empty.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
The metadata does not have the expected flatbuffer identifer.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
Cannot get minimum metadata parser version.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
The number of SubgraphMetadata is not 1.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
The number of input/output tensors does not match the number
of input/output tensor metadata.
</td>
</tr>
</table>



<h3 id="load_metadata_file"><code>load_metadata_file</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata.py#L323-L341">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_metadata_file(
    metadata_file
)
</code></pre>

Loads the metadata file to be populated.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`metadata_file`
</td>
<td>
path to the metadata file to be populated.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`IOError`
</td>
<td>
File not found.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
The metadata to be populated is empty.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
The metadata does not have the expected flatbuffer identifer.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
Cannot get minimum metadata parser version.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
The number of SubgraphMetadata is not 1.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
The number of input/output tensors does not match the number
of input/output tensor metadata.
</td>
</tr>
</table>



<h3 id="populate"><code>populate</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata.py#L361-L365">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>populate()
</code></pre>

Populates loaded metadata and associated files into the model file.


<h3 id="with_model_buffer"><code>with_model_buffer</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata.py#L199-L212">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>with_model_buffer(
    model_buf
)
</code></pre>

Creates a MetadataPopulator object that populates data to a model buffer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model_buf`
</td>
<td>
TensorFlow Lite model buffer in bytearray.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A MetadataPopulator(_MetadataPopulatorWithBuffer) object.
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
the model does not have the expected flatbuffer identifer.
</td>
</tr>
</table>



<h3 id="with_model_file"><code>with_model_file</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata.py#L181-L195">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>with_model_file(
    model_file
)
</code></pre>

Creates a MetadataPopulator object that populates data to a model file.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model_file`
</td>
<td>
valid path to a TensorFlow Lite model file.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
MetadataPopulator object.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`IOError`
</td>
<td>
File not found.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
the model does not have the expected flatbuffer identifer.
</td>
</tr>
</table>







<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
METADATA_FIELD_NAME<a id="METADATA_FIELD_NAME"></a>
</td>
<td>
`'TFLITE_METADATA'`
</td>
</tr><tr>
<td>
METADATA_FILE_IDENTIFIER<a id="METADATA_FILE_IDENTIFIER"></a>
</td>
<td>
`b'M001'`
</td>
</tr><tr>
<td>
TFLITE_FILE_IDENTIFIER<a id="TFLITE_FILE_IDENTIFIER"></a>
</td>
<td>
`b'TFL3'`
</td>
</tr>
</table>
