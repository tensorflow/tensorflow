page_type: reference
description: Loads data and retrains the model based on data for image classification.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.image_classifier.create" />
<meta itemprop="path" content="Stable" />
</div>

# tflite_model_maker.image_classifier.create

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/image_classifier.py#L252-L344">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Loads data and retrains the model based on data for image classification.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>tflite_model_maker.image_classifier.create(
    train_data,
    model_spec=&#x27;efficientnet_lite0&#x27;,
    validation_data=None,
    batch_size=None,
    epochs=None,
    steps_per_epoch=None,
    train_whole_model=None,
    dropout_rate=None,
    learning_rate=None,
    momentum=None,
    shuffle=False,
    use_augmentation=False,
    use_hub_library=True,
    warmup_steps=None,
    model_dir=None,
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
    <li><a href="https://www.tensorflow.org/lite/models/modify/model_maker/image_classification">Image classification with TensorFlow Lite Model Maker</a></li>
<li><a href="https://www.tensorflow.org/hub/tutorials/cropnet_on_device">Fine tuning models for plant disease detection</a></li>
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
`train_whole_model`<a id="train_whole_model"></a>
</td>
<td>
If true, the Hub module is trained together with the
classification layer on top. Otherwise, only train the top
classification layer.
</td>
</tr><tr>
<td>
`dropout_rate`<a id="dropout_rate"></a>
</td>
<td>
The rate for dropout.
</td>
</tr><tr>
<td>
`learning_rate`<a id="learning_rate"></a>
</td>
<td>
Base learning rate when train batch size is 256. Linear to
the batch size.
</td>
</tr><tr>
<td>
`momentum`<a id="momentum"></a>
</td>
<td>
a Python float forwarded to the optimizer. Only used when
`use_hub_library` is True.
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
`use_augmentation`<a id="use_augmentation"></a>
</td>
<td>
Use data augmentation for preprocessing.
</td>
</tr><tr>
<td>
`use_hub_library`<a id="use_hub_library"></a>
</td>
<td>
Use `make_image_classifier_lib` from tensorflow hub to
retrain the model.
</td>
</tr><tr>
<td>
`warmup_steps`<a id="warmup_steps"></a>
</td>
<td>
Number of warmup steps for warmup schedule on learning rate.
If None, the default warmup_steps is used which is the total training
steps in two epochs. Only used when `use_hub_library` is False.
</td>
</tr><tr>
<td>
`model_dir`<a id="model_dir"></a>
</td>
<td>
The location of the model checkpoint files. Only used when
`use_hub_library` is False.
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
An instance based on ImageClassifier.
</td>
</tr>

</table>
