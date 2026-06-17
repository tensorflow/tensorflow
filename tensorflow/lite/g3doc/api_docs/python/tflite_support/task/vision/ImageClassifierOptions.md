page_type: reference
description: Options for the image classifier task.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.vision.ImageClassifierOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tflite_support.task.vision.ImageClassifierOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/image_classifier.py#L32-L45">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Options for the image classifier task.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.vision.ImageClassifierOptions(
    base_options: <a href="../../../tflite_support/task/core/BaseOptions"><code>tflite_support.task.core.BaseOptions</code></a>,
    classification_options: <a href="../../../tflite_support/task/processor/ClassificationOptions"><code>tflite_support.task.processor.ClassificationOptions</code></a> = dataclasses.field(default_factory=_ClassificationOptions)
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`base_options`<a id="base_options"></a>
</td>
<td>
Base options for the image classifier task.
</td>
</tr><tr>
<td>
`classification_options`<a id="classification_options"></a>
</td>
<td>
Classification options for the image classifier
task.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>
