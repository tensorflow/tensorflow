page_type: reference
description: Loads file from the file path.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.writer_utils.load_file" />
<meta itemprop="path" content="Stable" />
</div>

# tflite_support.metadata_writers.writer_utils.load_file

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/writer_utils.py#L88-L100">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Loads file from the file path.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p><a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.audio_classifier.metadata_info.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.metadata_info.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.audio_classifier.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_info.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.metadata_info.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.bert_nl_classifier.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.image_classifier.metadata_info.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.metadata_info.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.image_classifier.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.image_segmenter.metadata_info.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.metadata_info.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.image_segmenter.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.metadata_info.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.nl_classifier.metadata_info.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.metadata_info.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.nl_classifier.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.object_detector.metadata_info.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.object_detector.metadata_writer.metadata_info.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.object_detector.metadata_writer.writer_utils.load_file</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/load_file"><code>tflite_support.metadata_writers.object_detector.writer_utils.load_file</code></a></p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata_writers.writer_utils.load_file(
    file_path: str, mode: str = &#x27;rb&#x27;
) -> Union[str, bytes]
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
valid file path string.
</td>
</tr><tr>
<td>
`mode`<a id="mode"></a>
</td>
<td>
a string specifies the model in which the file is opened. Use "rt" for
reading in text mode; use "rb" for reading in binary mode.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The loaded file in str or bytes.
</td>
</tr>

</table>
