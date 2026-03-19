page_type: reference
description: Loads data and retrains the model.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.audio_classifier.create" />
<meta itemprop="path" content="Stable" />
</div>

# tflite_model_maker.audio_classifier.create

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/audio_classifier.py#L103-L140">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Loads data and retrains the model.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>tflite_model_maker.audio_classifier.create(
    train_data,
    model_spec,
    validation_data=None,
    batch_size=32,
    epochs=5,
    model_dir=None,
    do_train=True,
    train_whole_model=False
)
</code></pre>




<h3>Used in the notebooks</h3>
<table class="vertical-rules">
  <thead>
    <tr>
      <th>Used in the tutorials</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
  <ul>
    <li><a href="https://www.tensorflow.org/lite/models/modify/model_maker/audio_classification">Transfer Learning for the Audio Domain with TensorFlow Lite Model Maker</a></li>
<li><a href="https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition">Retrain a speech recognition model with TensorFlow Lite Model Maker</a></li>
  </ul>
</td>
    </tr>
  </tbody>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`train_data`<a id="train_data"></a>
</td>
<td>
A instance of audio_dataloader.DataLoader class.
</td>
</tr><tr>
<td>
`model_spec`<a id="model_spec"></a>
</td>
<td>
Specification for the model.
</td>
</tr><tr>
<td>
`validation_data`<a id="validation_data"></a>
</td>
<td>
Validation DataLoader. If None, skips validation process.
</td>
</tr><tr>
<td>
`batch_size`<a id="batch_size"></a>
</td>
<td>
Number of samples per training step. If `use_hub_library` is
False, it represents the base learning rate when train batch size is 256
and it's linear to the batch size.
</td>
</tr><tr>
<td>
`epochs`<a id="epochs"></a>
</td>
<td>
Number of epochs for training.
</td>
</tr><tr>
<td>
`model_dir`<a id="model_dir"></a>
</td>
<td>
The location of the model checkpoint files.
</td>
</tr><tr>
<td>
`do_train`<a id="do_train"></a>
</td>
<td>
Whether to run training.
</td>
</tr><tr>
<td>
`train_whole_model`<a id="train_whole_model"></a>
</td>
<td>
Boolean. By default, only the classification head is
trained. When True, the base model is also trained.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An instance based on AudioClassifier.
</td>
</tr>

</table>
