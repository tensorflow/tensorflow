page_type: reference
description: Class that performs dense feature vector extraction on images.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.vision.ImageEmbedder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="cosine_similarity"/>
<meta itemprop="property" content="create_from_file"/>
<meta itemprop="property" content="create_from_options"/>
<meta itemprop="property" content="embed"/>
<meta itemprop="property" content="get_embedding_by_index"/>
<meta itemprop="property" content="get_embedding_dimension"/>
</div>

# tflite_support.task.vision.ImageEmbedder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/image_embedder.py#L47-L172">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Class that performs dense feature vector extraction on images.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.vision.ImageEmbedder(
    options: <a href="../../../tflite_support/task/vision/ImageEmbedderOptions"><code>tflite_support.task.vision.ImageEmbedderOptions</code></a>,
    cpp_embedder: _CppImageEmbedder
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

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/image_embedder.py#L148-L151">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cosine_similarity(
    u: <a href="../../../tflite_support/task/processor/FeatureVector"><code>tflite_support.task.processor.FeatureVector</code></a>,
    v: <a href="../../../tflite_support/task/processor/FeatureVector"><code>tflite_support.task.processor.FeatureVector</code></a>
) -> float
</code></pre>

Computes cosine similarity [1] between two feature vectors.


<h3 id="create_from_file"><code>create_from_file</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/image_embedder.py#L57-L74">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_file(
    file_path: str
) -> 'ImageEmbedder'
</code></pre>

Creates the `ImageEmbedder` object from a TensorFlow Lite model.


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
`ImageEmbedder` object that's created from the model file.
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
If failed to create `ImageEmbedder` object from the provided
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

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/image_embedder.py#L76-L94">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_options(
    options: <a href="../../../tflite_support/task/vision/ImageEmbedderOptions"><code>tflite_support.task.vision.ImageEmbedderOptions</code></a>
) -> 'ImageEmbedder'
</code></pre>

Creates the `ImageEmbedder` object from image embedder options.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`options`
</td>
<td>
Options for the image embedder task.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`ImageEmbedder` object that's created from `options`.
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
If failed to create `ImageEmbdder` object from
`ImageEmbedderOptions` such as missing the model.
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

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/image_embedder.py#L96-L124">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>embed(
    image: <a href="../../../tflite_support/task/vision/TensorImage"><code>tflite_support.task.vision.TensorImage</code></a>,
    bounding_box: Optional[<a href="../../../tflite_support/task/processor/BoundingBox"><code>tflite_support.task.processor.BoundingBox</code></a>] = None
) -> <a href="../../../tflite_support/task/processor/EmbeddingResult"><code>tflite_support.task.processor.EmbeddingResult</code></a>
</code></pre>

Performs actual feature vector extraction on the provided TensorImage.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`image`
</td>
<td>
Tensor image, used to extract the feature vectors.
</td>
</tr><tr>
<td>
`bounding_box`
</td>
<td>
Bounding box, optional. If set, performed feature vector
extraction only on the provided region of interest. Note that the region
of interest is not clamped, so this method will fail if the region is
out of bounds of the input image.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The embedding result.
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



<h3 id="get_embedding_by_index"><code>get_embedding_by_index</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/image_embedder.py#L126-L146">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_embedding_by_index(
    result: <a href="../../../tflite_support/task/processor/EmbeddingResult"><code>tflite_support.task.processor.EmbeddingResult</code></a>,
    output_index: int
) -> <a href="../../../tflite_support/task/processor/Embedding"><code>tflite_support.task.processor.Embedding</code></a>
</code></pre>

Gets the embedding in the embedding result by `output_index`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`result`
</td>
<td>
embedding result.
</td>
</tr><tr>
<td>
`output_index`
</td>
<td>
output index of the output layer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The Embedding output by the output_index'th layer. In (the most common)
case where a single embedding is produced, you can just call
get_feature_vector_by_index(result, 0).
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
ValueError if the output index is out of bound.
</td>
</tr>

</table>



<h3 id="get_embedding_dimension"><code>get_embedding_dimension</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/image_embedder.py#L153-L163">View source</a>

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
