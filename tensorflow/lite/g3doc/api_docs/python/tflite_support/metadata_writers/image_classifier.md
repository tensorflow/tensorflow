page_type: reference
description: Writes metadata and label file to the image classifier models.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers.image_classifier" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="INPUT_DESCRIPTION"/>
<meta itemprop="property" content="INPUT_NAME"/>
<meta itemprop="property" content="MODEL_DESCRIPTION"/>
<meta itemprop="property" content="OUTPUT_DESCRIPTION"/>
<meta itemprop="property" content="OUTPUT_NAME"/>
</div>

# Module: tflite_support.metadata_writers.image_classifier

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/image_classifier.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Writes metadata and label file to the image classifier models.



## Modules

[`metadata_info`](../../tflite_support/metadata_writers/metadata_info) module: Helper classes for common model metadata information.

[`metadata_writer`](../../tflite_support/metadata_writers/audio_classifier/metadata_writer) module: Helper class to write metadata into TFLite models.

[`writer_utils`](../../tflite_support/metadata_writers/writer_utils) module: Helper methods for writing metadata into TFLite models.

## Classes

[`class MetadataWriter`](../../tflite_support/metadata_writers/image_classifier/MetadataWriter): Writes metadata into an image classifier.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
INPUT_DESCRIPTION<a id="INPUT_DESCRIPTION"></a>
</td>
<td>
`'Input image to be classified.'`
</td>
</tr><tr>
<td>
INPUT_NAME<a id="INPUT_NAME"></a>
</td>
<td>
`'image'`
</td>
</tr><tr>
<td>
MODEL_DESCRIPTION<a id="MODEL_DESCRIPTION"></a>
</td>
<td>
`('Identify the most prominent object in the image from a known set of '
 'categories.')`
</td>
</tr><tr>
<td>
OUTPUT_DESCRIPTION<a id="OUTPUT_DESCRIPTION"></a>
</td>
<td>
`'Probabilities of the labels respectively.'`
</td>
</tr><tr>
<td>
OUTPUT_NAME<a id="OUTPUT_NAME"></a>
</td>
<td>
`'probability'`
</td>
</tr>
</table>
