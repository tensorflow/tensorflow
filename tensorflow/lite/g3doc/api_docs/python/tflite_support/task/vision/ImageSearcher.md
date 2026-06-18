page_type: reference
description: Class to performs image search.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.vision.ImageSearcher" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_from_file"/>
<meta itemprop="property" content="create_from_options"/>
<meta itemprop="property" content="get_user_info"/>
<meta itemprop="property" content="search"/>
</div>

# tflite_support.task.vision.ImageSearcher

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/image_searcher.py#L53-L155">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Class to performs image search.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.vision.ImageSearcher(
    options: <a href="../../../tflite_support/task/vision/ImageSearcherOptions"><code>tflite_support.task.vision.ImageSearcherOptions</code></a>,
    cpp_searcher: _CppImageSearcher
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

It works by performing embedding extraction on images, followed by
nearest-neighbor search in an index of embeddings through ScaNN.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`options`<a id="options"></a>
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="create_from_file"><code>create_from_file</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/image_searcher.py#L67-L91">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_file(
    model_file_path: str, index_file_path: Optional[str] = None
) -> 'ImageSearcher'
</code></pre>

Creates the `ImageSearcher` object from a TensorFlow Lite model.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model_file_path`
</td>
<td>
Path to the model.
</td>
</tr><tr>
<td>
`index_file_path`
</td>
<td>
Path to the index. Only required if the index is not
attached to the output tensor metadata as an AssociatedFile with type
SCANN_INDEX_FILE.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`ImageSearcher` object that's created from `options`.
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
If failed to create `ImageSearcher` object from the provided
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

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/image_searcher.py#L93-L111">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_options(
    options: <a href="../../../tflite_support/task/vision/ImageSearcherOptions"><code>tflite_support.task.vision.ImageSearcherOptions</code></a>
) -> 'ImageSearcher'
</code></pre>

Creates the `ImageSearcher` object from image searcher options.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`options`
</td>
<td>
Options for the image searcher task.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`ImageSearcher` object that's created from `options`.
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
If failed to create `ImageSearcher` object from
`ImageSearcherOptions` such as missing the model.
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



<h3 id="get_user_info"><code>get_user_info</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/image_searcher.py#L144-L151">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_user_info() -> str
</code></pre>

Gets the user info stored in the index file.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Opaque user info stored in the index file (if any), in raw binary form.
Returns an empty string if the index doesn't contain user info.
</td>
</tr>

</table>



<h3 id="search"><code>search</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/image_searcher.py#L113-L142">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>search(
    image: <a href="../../../tflite_support/task/vision/TensorImage"><code>tflite_support.task.vision.TensorImage</code></a>,
    bounding_box: Optional[<a href="../../../tflite_support/task/processor/BoundingBox"><code>tflite_support.task.processor.BoundingBox</code></a>] = None
) -> <a href="../../../tflite_support/task/processor/SearchResult"><code>tflite_support.task.processor.SearchResult</code></a>
</code></pre>

Search for image with similar semantic meaning.

This method performs actual feature extraction on the provided image input,
followed by nearest-neighbor search in the index.

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
Search result.
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
If failed to perform nearest-neighbor search.
</td>
</tr>
</table>
