page_type: reference
description: Computes the flat size (number of elements) of tensor shape.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.writer_utils.compute_flat_size" />
<meta itemprop="path" content="Stable" />
</div>

# tflite_support.metadata_writers.writer_utils.compute_flat_size

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/writer_utils.py#L25-L36">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the flat size (number of elements) of tensor shape.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p><a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.audio_classifier.metadata_info.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.metadata_info.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.audio_classifier.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_info.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.metadata_info.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.bert_nl_classifier.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.image_classifier.metadata_info.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.metadata_info.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.image_classifier.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.image_segmenter.metadata_info.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.metadata_info.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.image_segmenter.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.metadata_info.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.nl_classifier.metadata_info.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.metadata_info.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.nl_classifier.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.object_detector.metadata_info.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.object_detector.metadata_writer.metadata_info.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.object_detector.metadata_writer.writer_utils.compute_flat_size</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/writer_utils/compute_flat_size"><code>tflite_support.metadata_writers.object_detector.writer_utils.compute_flat_size</code></a></p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata_writers.writer_utils.compute_flat_size(
    tensor_shape: Optional['array.array[int]']
) -> int
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor_shape`<a id="tensor_shape"></a>
</td>
<td>
an array of the tensor shape values.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The flat size of the tensor shape. Return 0 if tensor_shape is None.
</td>
</tr>

</table>
