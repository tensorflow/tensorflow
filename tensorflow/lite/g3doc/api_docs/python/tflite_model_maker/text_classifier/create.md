page_type: reference
description: Loads data and train the model for test classification.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.text_classifier.create" />
<meta itemprop="path" content="Stable" />
</div>

# tflite_model_maker.text_classifier.create

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/text_classifier.py#L177-L220">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Loads data and train the model for test classification.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>tflite_model_maker.text_classifier.create(
    train_data,
    model_spec=&#x27;average_word_vec&#x27;,
    validation_data=None,
    batch_size=None,
    epochs=3,
    steps_per_epoch=None,
    shuffle=False,
    do_train=True
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
    <li><a href="https://www.tensorflow.org/lite/models/modify/model_maker/text_classification">Text classification with TensorFlow Lite Model Maker</a></li>
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
Training data.
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
Validation data. If None, skips validation process.
</td>
</tr><tr>
<td>
`batch_size`<a id="batch_size"></a>
</td>
<td>
Batch size for training.
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
`steps_per_epoch`<a id="steps_per_epoch"></a>
</td>
<td>
Integer or None. Total number of steps (batches of
samples) before declaring one epoch finished and starting the next
epoch. If `steps_per_epoch` is None, the epoch will run until the input
dataset is exhausted.
</td>
</tr><tr>
<td>
`shuffle`<a id="shuffle"></a>
</td>
<td>
Whether the data should be shuffled.
</td>
</tr><tr>
<td>
`do_train`<a id="do_train"></a>
</td>
<td>
Whether to run training.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An instance based on TextClassifier.
</td>
</tr>

</table>
