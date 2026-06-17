page_type: reference
description: Class that performs dense feature vector extraction on text.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.text.TextEmbedder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="cosine_similarity"/>
<meta itemprop="property" content="create_from_file"/>
<meta itemprop="property" content="create_from_options"/>
<meta itemprop="property" content="embed"/>
<meta itemprop="property" content="get_embedding_dimension"/>
</div>

# tflite_support.task.text.TextEmbedder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/text/text_embedder.py#L42-L128">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Class that performs dense feature vector extraction on text.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.text.TextEmbedder(
    options: <a href="../../../tflite_support/task/text/TextEmbedderOptions"><code>tflite_support.task.text.TextEmbedderOptions</code></a>,
    cpp_embedder: _CppTextEmbedder
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`number_of_output_layers`<a id="number_of_output_layers"></a>
</td>
<td>
Gets the number of output layers of the model.
</td>
</tr><tr>
<td>
`options`<a id="options"></a>
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="cosine_similarity"><code>cosine_similarity</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/text/text_embedder.py#L104-L107">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cosine_similarity(
    u: <a href="../../../tflite_support/task/processor/FeatureVector"><code>tflite_support.task.processor.FeatureVector</code></a>,
    v: <a href="../../../tflite_support/task/processor/FeatureVector"><code>tflite_support.task.processor.FeatureVector</code></a>
) -> float
</code></pre>

Computes cosine similarity [1] between two feature vectors.


<h3 id="create_from_file"><code>create_from_file</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/text/text_embedder.py#L52-L68">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_file(
    file_path: str
) -> 'TextEmbedder'
</code></pre>

Creates the `TextEmbedder` object from a TensorFlow Lite model.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`file_path`
</td>
<td>
Path to the model.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`TextEmbedder` object that's created from the model file.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If failed to create `TextEmbedder` object from the provided
file such as invalid file.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If other types of error occurred.
</td>
</tr>
</table>



<h3 id="create_from_options"><code>create_from_options</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/text/text_embedder.py#L70-L86">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_options(
    options: <a href="../../../tflite_support/task/text/TextEmbedderOptions"><code>tflite_support.task.text.TextEmbedderOptions</code></a>
) -> 'TextEmbedder'
</code></pre>

Creates the `TextEmbedder` object from text embedder options.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`options`
</td>
<td>
Options for the text embedder task.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`TextEmbedder` object that's created from `options`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If failed to create `TextEmbedder` object from
`TextEmbedderOptions` such as missing the model.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If other types of error occurred.
</td>
</tr>
</table>



<h3 id="embed"><code>embed</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/text/text_embedder.py#L88-L102">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>embed(
    text: str
) -> <a href="../../../tflite_support/task/processor/EmbeddingResult"><code>tflite_support.task.processor.EmbeddingResult</code></a>
</code></pre>

Performs actual feature vector extraction on the provided text.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<a href="https://www.tensorflow.org/text/api_docs/python/text"><code>text</code></a>
</td>
<td>
the input text, used to extract the feature vectors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
embedding result.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If any of the input arguments is invalid.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If failed to calculate the embedding vector.
</td>
</tr>
</table>



<h3 id="get_embedding_dimension"><code>get_embedding_dimension</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/text/text_embedder.py#L109-L119">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_embedding_dimension(
    output_index: int
) -> int
</code></pre>

Gets the dimensionality of the embedding output.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`output_index`
</td>
<td>
The output index of output layer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Dimensionality of the embedding output by the output_index'th output
layer. Returns -1 if `output_index` is out of bounds.
</td>
</tr>

</table>
