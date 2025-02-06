page_type: reference
description: Gets a list of associated files packed in the tokenzier_options.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.writer_utils.get_tokenizer_associated_files" />
<meta itemprop="path" content="Stable" />
</div>

# tflite_support.metadata_writers.writer_utils.get_tokenizer_associated_files

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/writer_utils.py#L121-L158">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gets a list of associated files packed in the tokenzier_options.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p><a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.audio_classifier.metadata_info.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.metadata_info.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.audio_classifier.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_info.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.metadata_info.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.bert_nl_classifier.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.image_classifier.metadata_info.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.metadata_info.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.image_classifier.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.image_segmenter.metadata_info.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.metadata_info.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.image_segmenter.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.metadata_info.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.nl_classifier.metadata_info.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.metadata_info.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.nl_classifier.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.object_detector.metadata_info.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.object_detector.metadata_writer.metadata_info.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.object_detector.metadata_writer.writer_utils.get_tokenizer_associated_files</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files"><code>tflite_support.metadata_writers.object_detector.writer_utils.get_tokenizer_associated_files</code></a></p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata_writers.writer_utils.get_tokenizer_associated_files(
    tokenizer_options: Union[None, _metadata_fb.BertTokenizerOptionsT, _metadata_fb.
        SentencePieceTokenizerOptionsT, _metadata_fb.RegexTokenizerOptionsT]
) -> List[Optional[str]]
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tokenizer_options`<a id="tokenizer_options"></a>
</td>
<td>
a tokenizer metadata object. Support the following
tokenizer types:

1. BertTokenizerOptions:
  https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L436
2. SentencePieceTokenizerOptions:
  https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L473
3. RegexTokenizerOptions:
  https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L475
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of associated files included in tokenizer_options.
</td>
</tr>

</table>
