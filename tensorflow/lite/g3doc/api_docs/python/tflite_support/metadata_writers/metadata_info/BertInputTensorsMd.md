page_type: reference
description: A container for the input tensor metadata information of Bert models.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.metadata_info.BertInputTensorsMd" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_input_process_unit_metadata"/>
<meta itemprop="property" content="create_input_tesnor_metadata"/>
<meta itemprop="property" content="get_tokenizer_associated_files"/>
</div>

# tflite_support.metadata_writers.metadata_info.BertInputTensorsMd

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L703-L817">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A container for the input tensor metadata information of Bert models.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p><a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/BertInputTensorsMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_info.BertInputTensorsMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/BertInputTensorsMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.metadata_info.BertInputTensorsMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/BertInputTensorsMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_info.BertInputTensorsMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/BertInputTensorsMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.metadata_info.BertInputTensorsMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/BertInputTensorsMd"><code>tflite_support.metadata_writers.image_classifier.metadata_info.BertInputTensorsMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/BertInputTensorsMd"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.metadata_info.BertInputTensorsMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/BertInputTensorsMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_info.BertInputTensorsMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/BertInputTensorsMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.metadata_info.BertInputTensorsMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/BertInputTensorsMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_info.BertInputTensorsMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/BertInputTensorsMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.metadata_info.BertInputTensorsMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/BertInputTensorsMd"><code>tflite_support.metadata_writers.object_detector.metadata_info.BertInputTensorsMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/BertInputTensorsMd"><code>tflite_support.metadata_writers.object_detector.metadata_writer.metadata_info.BertInputTensorsMd</code></a></p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata_writers.metadata_info.BertInputTensorsMd(
    model_buffer: bytearray,
    ids_name: str,
    mask_name: str,
    segment_name: str,
    ids_md: Optional[<a href="../../../tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.metadata_info.TensorMd</code></a>] = None,
    mask_md: Optional[<a href="../../../tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.metadata_info.TensorMd</code></a>] = None,
    segment_ids_md: Optional[<a href="../../../tflite_support/metadata_writers/metadata_info/TensorMd"><code>tflite_support.metadata_writers.metadata_info.TensorMd</code></a>] = None,
    tokenizer_md: Union[None, <a href="../../../tflite_support/metadata_writers/metadata_info/BertTokenizerMd"><code>tflite_support.metadata_writers.metadata_info.BertTokenizerMd</code></a>, <a href="../../../tflite_support/metadata_writers/metadata_info/SentencePieceTokenizerMd"><code>tflite_support.metadata_writers.metadata_info.SentencePieceTokenizerMd</code></a>] = None
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
`ids_name`<a id="ids_name"></a>
</td>
<td>
name of the ids tensor, which represents the tokenized ids of
the input text.
</td>
</tr><tr>
<td>
`mask_name`<a id="mask_name"></a>
</td>
<td>
name of the mask tensor, which represents the mask with 1 for
real tokens and 0 for padding tokens.
</td>
</tr><tr>
<td>
`segment_name`<a id="segment_name"></a>
</td>
<td>
name of the segment ids tensor, where `0` stands for the
first sequence, and `1` stands for the second sequence if exists.
</td>
</tr><tr>
<td>
`ids_md`<a id="ids_md"></a>
</td>
<td>
input ids tensor informaton.
</td>
</tr><tr>
<td>
`mask_md`<a id="mask_md"></a>
</td>
<td>
input mask tensor informaton.
</td>
</tr><tr>
<td>
`segment_ids_md`<a id="segment_ids_md"></a>
</td>
<td>
input segment tensor informaton.
</td>
</tr><tr>
<td>
`tokenizer_md`<a id="tokenizer_md"></a>
</td>
<td>
information of the tokenizer used to process the input
string, if any. Supported tokenziers are: `BertTokenizer` [1] and
  `SentencePieceTokenizer` [2]. If the tokenizer is `RegexTokenizer`
  [3], refer to <a href="../../../tflite_support/metadata_writers/nl_classifier/MetadataWriter"><code>nl_classifier.MetadataWriter</code></a>.
[1]:
https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L436
[2]:
https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L473
[3]:
https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L475
</td>
</tr>
</table>



## Methods

<h3 id="create_input_process_unit_metadata"><code>create_input_process_unit_metadata</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L803-L809">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_input_process_unit_metadata() -> List[<a href="../../../tflite_support/metadata_schema_py_generated/ProcessUnitT"><code>tflite_support.metadata_schema_py_generated.ProcessUnitT</code></a>]
</code></pre>

Creates the input process unit metadata.


<h3 id="create_input_tesnor_metadata"><code>create_input_tesnor_metadata</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L792-L801">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_input_tesnor_metadata() -> List[<a href="../../../tflite_support/metadata_schema_py_generated/TensorMetadataT"><code>tflite_support.metadata_schema_py_generated.TensorMetadataT</code></a>]
</code></pre>

Creates the input metadata for the three input tesnors.


<h3 id="get_tokenizer_associated_files"><code>get_tokenizer_associated_files</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L811-L817">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_tokenizer_associated_files() -> List[str]
</code></pre>

Gets the associated files that are packed in the tokenizer.
