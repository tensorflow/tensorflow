page_type: reference
description: Loads data and train the model for recommendation.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.recommendation.create" />
<meta itemprop="path" content="Stable" />
</div>

# tflite_model_maker.recommendation.create

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/recommendation.py#L213-L265">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Loads data and train the model for recommendation.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>tflite_model_maker.recommendation.create(
    train_data,
    model_spec: <a href="../../tflite_model_maker/recommendation/ModelSpec"><code>tflite_model_maker.recommendation.ModelSpec</code></a>,
    model_dir: str = None,
    validation_data=None,
    batch_size: int = 16,
    steps_per_epoch: int = 10000,
    epochs: int = 1,
    learning_rate: float = 0.1,
    gradient_clip_norm: float = 1.0,
    shuffle: bool = True,
    do_train: bool = True
)
</code></pre>



<!-- Placeholder for "Used in" -->


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
ModelSpec, Specification for the model.
</td>
</tr><tr>
<td>
`model_dir`<a id="model_dir"></a>
</td>
<td>
str, path to export model checkpoints and summaries.
</td>
</tr><tr>
<td>
`validation_data`<a id="validation_data"></a>
</td>
<td>
Validation data.
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
`steps_per_epoch`<a id="steps_per_epoch"></a>
</td>
<td>
int, Number of step per epoch.
</td>
</tr><tr>
<td>
`epochs`<a id="epochs"></a>
</td>
<td>
int, Number of epochs for training.
</td>
</tr><tr>
<td>
`learning_rate`<a id="learning_rate"></a>
</td>
<td>
float, learning rate.
</td>
</tr><tr>
<td>
`gradient_clip_norm`<a id="gradient_clip_norm"></a>
</td>
<td>
float, clip threshold (<= 0 meaning no clip).
</td>
</tr><tr>
<td>
`shuffle`<a id="shuffle"></a>
</td>
<td>
boolean, whether the training data should be shuffled.
</td>
</tr><tr>
<td>
`do_train`<a id="do_train"></a>
</td>
<td>
boolean, whether to run training.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An instance based on Recommendation.
</td>
</tr>

</table>
