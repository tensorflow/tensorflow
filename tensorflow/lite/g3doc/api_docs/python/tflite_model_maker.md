page_type: reference
description: Public APIs for TFLite Model Maker, a transfer learning library to train custom TFLite models.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__version__"/>
</div>

# Module: tflite_model_maker

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/tflmm/v0.4.2/tensorflow_examples/lite/model_maker/public/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Public APIs for TFLite Model Maker, a transfer learning library to train custom TFLite models.


You can install the package with

```bash
pip install tflite-model-maker
```

Typical usage of Model Maker is to create a model in a few lines of code, e.g.:

```python
# Load input data specific to an on-device ML app.
data = DataLoader.from_folder('flower_photos/')
train_data, test_data = data.split(0.9)

# Customize the TensorFlow model.
model = image_classifier.create(train_data)

# Evaluate the model.
accuracy = model.evaluate(test_data)

# Export to Tensorflow Lite model and label file in `export_dir`.
model.export(export_dir='/tmp/')
```

For more details, please refer to our guide:
<a href="https://www.tensorflow.org/lite/guide/model_maker">https://www.tensorflow.org/lite/guide/model_maker</a>

## Modules

[`audio_classifier`](./tflite_model_maker/audio_classifier) module: APIs to train an audio classification model.

[`config`](./tflite_model_maker/config) module: APIs for the config of TFLite Model Maker.

[`image_classifier`](./tflite_model_maker/image_classifier) module: APIs to train an image classification model.

[`model_spec`](./tflite_model_maker/model_spec) module: APIs for the model spec of TFLite Model Maker.

[`object_detector`](./tflite_model_maker/object_detector) module: APIs to train an object detection model.

[`question_answer`](./tflite_model_maker/question_answer) module: APIs to train a model that can answer questions based on a predefined text.

[`recommendation`](./tflite_model_maker/recommendation) module: APIs to train an on-device recommendation model.

[`searcher`](./tflite_model_maker/searcher) module: APIs to create the searcher model.

[`text_classifier`](./tflite_model_maker/text_classifier) module: APIs to train a text classification model.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
__version__<a id="__version__"></a>
</td>
<td>
`'0.4.2'`
</td>
</tr>
</table>
