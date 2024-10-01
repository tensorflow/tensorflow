page_type: reference
description: Gets a list of the output tensor names.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.writer_utils.get_output_tensor_names" />
<meta itemprop="path" content="Stable" />
</div>

# tflite_support.metadata_writers.writer_utils.get_output_tensor_names

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/writer_utils.py#L49-L56">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gets a list of the output tensor names.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p><a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.audio_classifier.metadata_info.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.metadata_info.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.audio_classifier.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_info.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.metadata_info.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.bert_nl_classifier.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.image_classifier.metadata_info.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.metadata_info.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.image_classifier.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.image_segmenter.metadata_info.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.metadata_info.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.image_segmenter.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.metadata_info.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.nl_classifier.metadata_info.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.metadata_info.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.nl_classifier.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.object_detector.metadata_info.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.object_detector.metadata_writer.metadata_info.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.object_detector.metadata_writer.writer_utils.get_output_tensor_names</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/get_output_tensor_names"><code>tflite_support.metadata_writers.object_detector.writer_utils.get_output_tensor_names</code></a></p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata_writers.writer_utils.get_output_tensor_names(
    model_buffer: bytearray
) -> List[str]
</code></pre>



<!-- Placeholder for "Used in" -->
