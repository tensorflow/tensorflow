page_type: reference
description: A container for score calibration [1] metadata information.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.metadata_info.ScoreCalibrationMd" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_metadata"/>
<meta itemprop="property" content="create_score_calibration_file_md"/>
</div>

# tflite_support.metadata_writers.metadata_info.ScoreCalibrationMd

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L252-L319">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A container for score calibration [1] metadata information.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p><a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ScoreCalibrationMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_info.ScoreCalibrationMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ScoreCalibrationMd"><code>tflite_support.metadata_writers.audio_classifier.metadata_writer.metadata_info.ScoreCalibrationMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ScoreCalibrationMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_info.ScoreCalibrationMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ScoreCalibrationMd"><code>tflite_support.metadata_writers.bert_nl_classifier.metadata_writer.metadata_info.ScoreCalibrationMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ScoreCalibrationMd"><code>tflite_support.metadata_writers.image_classifier.metadata_info.ScoreCalibrationMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ScoreCalibrationMd"><code>tflite_support.metadata_writers.image_classifier.metadata_writer.metadata_info.ScoreCalibrationMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ScoreCalibrationMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_info.ScoreCalibrationMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ScoreCalibrationMd"><code>tflite_support.metadata_writers.image_segmenter.metadata_writer.metadata_info.ScoreCalibrationMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ScoreCalibrationMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_info.ScoreCalibrationMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ScoreCalibrationMd"><code>tflite_support.metadata_writers.nl_classifier.metadata_writer.metadata_info.ScoreCalibrationMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ScoreCalibrationMd"><code>tflite_support.metadata_writers.object_detector.metadata_info.ScoreCalibrationMd</code></a>, <a href="https://www.tensorflow.org/lite/api_docs/python/tflite_support/metadata_writers/metadata_info/ScoreCalibrationMd"><code>tflite_support.metadata_writers.object_detector.metadata_writer.metadata_info.ScoreCalibrationMd</code></a></p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata_writers.metadata_info.ScoreCalibrationMd(
    score_transformation_type: <a href="../../../tflite_support/metadata_schema_py_generated/ScoreTransformationType"><code>tflite_support.metadata_schema_py_generated.ScoreTransformationType</code></a>,
    default_score: float,
    file_path: str
)
</code></pre>



<!-- Placeholder for "Used in" -->

[1]:
  https://github.com/tensorflow/tflite-support/blob/5e0cdf5460788c481f5cd18aab8728ec36cf9733/tensorflow_lite_support/metadata/metadata_schema.fbs#L434

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`score_transformation_type`<a id="score_transformation_type"></a>
</td>
<td>
type of the function used for transforming the
uncalibrated score before applying score calibration.
</td>
</tr><tr>
<td>
`default_score`<a id="default_score"></a>
</td>
<td>
the default calibrated score to apply if the uncalibrated
score is below min_score or if no parameters were specified for a given
index.
</td>
</tr><tr>
<td>
`file_path`<a id="file_path"></a>
</td>
<td>
file_path of the score calibration file [1].
[1]:
  https://github.com/tensorflow/tflite-support/blob/5e0cdf5460788c481f5cd18aab8728ec36cf9733/tensorflow_lite_support/metadata/metadata_schema.fbs#L122
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if the score_calibration file is malformed.
</td>
</tr>
</table>



## Methods

<h3 id="create_metadata"><code>create_metadata</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L301-L314">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_metadata() -> <a href="../../../tflite_support/metadata_schema_py_generated/ProcessUnitT"><code>tflite_support.metadata_schema_py_generated.ProcessUnitT</code></a>
</code></pre>

Creates the score calibration metadata based on the information.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Flatbuffers Python object of the score calibration metadata.
</td>
</tr>

</table>



<h3 id="create_score_calibration_file_md"><code>create_score_calibration_file_md</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/metadata_info.py#L316-L319">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>create_score_calibration_file_md() -> <a href="../../../tflite_support/metadata_writers/metadata_info/AssociatedFileMd"><code>tflite_support.metadata_writers.metadata_info.AssociatedFileMd</code></a>
</code></pre>
