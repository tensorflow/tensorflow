page_type: reference
description: TF Lite Metadata Writer API.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata_writers" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tflite_support.metadata_writers

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata_writers/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



TF Lite Metadata Writer API.


This module provides interfaces for writing metadata for common model types
supported by the task library, such as:

  * Image classification
  * Object detection
  * Image segmentation
  * (Bert) Natural language classification
  * Audio classification

It is provided as part of the `tflite-support` package:

```
pip install tflite-support
```

Learn more about this API in the [metadata writer
tutorial](https://www.tensorflow.org/lite/convert/metadata_writer_tutorial).

## Modules

[`audio_classifier`](../tflite_support/metadata_writers/audio_classifier) module: Writes metadata and label file to the audio classifier models.

[`bert_nl_classifier`](../tflite_support/metadata_writers/bert_nl_classifier) module: Writes metadata and label file to the Bert NL classifier models.

[`image_classifier`](../tflite_support/metadata_writers/image_classifier) module: Writes metadata and label file to the image classifier models.

[`image_segmenter`](../tflite_support/metadata_writers/image_segmenter) module: Writes metadata and label file to the image segmenter models.

[`metadata_info`](../tflite_support/metadata_writers/metadata_info) module: Helper classes for common model metadata information.

[`nl_classifier`](../tflite_support/metadata_writers/nl_classifier) module: Writes metadata and label file to the NL classifier models.

[`object_detector`](../tflite_support/metadata_writers/object_detector) module: Writes metadata and label file to the object detector models.

[`writer_utils`](../tflite_support/metadata_writers/writer_utils) module: Helper methods for writing metadata into TFLite models.
