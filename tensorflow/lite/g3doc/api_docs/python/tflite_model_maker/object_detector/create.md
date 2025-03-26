page_type: reference
description: Loads data and train the model for object detection.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.object_detector.create" />
<meta itemprop="path" content="Stable" />
</div>

# tflite_model_maker.object_detector.create

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/object_detector.py#L219-L264">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Loads data and train the model for object detection.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>tflite_model_maker.object_detector.create(
    train_data: <a href="../../tflite_model_maker/object_detector/DataLoader"><code>tflite_model_maker.object_detector.DataLoader</code></a>,
    model_spec: <a href="../../tflite_model_maker/object_detector/EfficientDetSpec"><code>tflite_model_maker.object_detector.EfficientDetSpec</code></a>,
    validation_data: Optional[<a href="../../tflite_model_maker/object_detector/DataLoader"><code>tflite_model_maker.object_detector.DataLoader</code></a>] = None,
    epochs: Optional[<a href="../../tflite_model_maker/object_detector/DataLoader"><code>tflite_model_maker.object_detector.DataLoader</code></a>] = None,
    batch_size: Optional[int] = None,
    train_whole_model: bool = False,
    do_train: bool = True
) -> T
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
    <li><a href="https://www.tensorflow.org/lite/models/modify/model_maker/object_detection">Object Detection with TensorFlow Lite Model Maker</a></li>
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
`epochs`<a id="epochs"></a>
</td>
<td>
Number of epochs for training.
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
`train_whole_model`<a id="train_whole_model"></a>
</td>
<td>
Boolean, False by default. If true, train the whole
model. Otherwise, only train the layers that are not match
`model_spec.config.var_freeze_expr`.
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
An instance based on ObjectDetector.
</td>
</tr>

</table>
