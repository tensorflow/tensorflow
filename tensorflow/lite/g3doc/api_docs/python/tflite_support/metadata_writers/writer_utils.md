page_type: reference
description: Helper methods for writing metadata into TFLite models.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.writer_utils" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tflite_support.metadata_writers.writer_utils

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/writer_utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Helper methods for writing metadata into TFLite models.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p><a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.audio_classifier.metadata_info.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.metadata_info.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.audio_classifier.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_info.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.metadata_info.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.bert_nl_classifier.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.image_classifier.metadata_info.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.metadata_info.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.image_classifier.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.image_segmenter.metadata_info.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.metadata_info.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.image_segmenter.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.metadata_info.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.nl_classifier.metadata_info.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.metadata_info.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.nl_classifier.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.object_detector.metadata_info.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.object_detector.metadata_writer.metadata_info.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.object_detector.metadata_writer.writer_utils</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils"><code>tflite_support.metadata_writers.object_detector.writer_utils</code></a></p>
</p>
</section>



## Functions

[`compute_flat_size(...)`](../../tflite_support/metadata_writers/writer_utils/compute_flat_size): Computes the flat size (number of elements) of tensor shape.

[`get_input_tensor_names(...)`](../../tflite_support/metadata_writers/writer_utils/get_input_tensor_names): Gets a list of the input tensor names.

[`get_input_tensor_shape(...)`](../../tflite_support/metadata_writers/writer_utils/get_input_tensor_shape): Gets the shape of the specified input tensor.

[`get_input_tensor_types(...)`](../../tflite_support/metadata_writers/writer_utils/get_input_tensor_types): Gets a list of the input tensor types.

[`get_output_tensor_names(...)`](../../tflite_support/metadata_writers/writer_utils/get_output_tensor_names): Gets a list of the output tensor names.

[`get_output_tensor_types(...)`](../../tflite_support/metadata_writers/writer_utils/get_output_tensor_types): Gets a list of the output tensor types.

[`get_tokenizer_associated_files(...)`](../../tflite_support/metadata_writers/writer_utils/get_tokenizer_associated_files): Gets a list of associated files packed in the tokenzier_options.

[`load_file(...)`](../../tflite_support/metadata_writers/writer_utils/load_file): Loads file from the file path.

[`save_file(...)`](../../tflite_support/metadata_writers/writer_utils/save_file): Loads file from the file path.
