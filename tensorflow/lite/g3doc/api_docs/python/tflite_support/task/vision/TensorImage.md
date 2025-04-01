page_type: reference
description: Wrapper class for the Image object.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.vision.TensorImage" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_from_array"/>
<meta itemprop="property" content="create_from_buffer"/>
<meta itemprop="property" content="create_from_file"/>
</div>

# tflite_support.task.vision.TensorImage

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/core/tensor_image.py#L22-L139">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Wrapper class for the Image object.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.vision.TensorImage(
    image_data: image_utils.ImageData, is_from_numpy_array: bool = True
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`image_data`<a id="image_data"></a>
</td>
<td>
image_utils.ImageData, contains raw image data, width, height
and channels info.
</td>
</tr><tr>
<td>
`is_from_numpy_array`<a id="is_from_numpy_array"></a>
</td>
<td>
boolean, whether `image_data` is loaded from
numpy array. if False, it means that `image_data` is loaded from
stbi_load** function in C++ and need to free the storage of ImageData in
the destructor.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`buffer`<a id="buffer"></a>
</td>
<td>
Gets the numpy array that represents `self.image_data`.
</td>
</tr><tr>
<td>
`color_space_type`<a id="color_space_type"></a>
</td>
<td>
Gets the color space type of the image.
</td>
</tr><tr>
<td>
`height`<a id="height"></a>
</td>
<td>
Gets the height of the image.
</td>
</tr><tr>
<td>
`width`<a id="width"></a>
</td>
<td>
Gets the width of the image.
</td>
</tr>
</table>



## Methods

<h3 id="create_from_array"><code>create_from_array</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/core/tensor_image.py#L59-L79">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_array(
    array: np.ndarray
) -> 'TensorImage'
</code></pre>

Creates `TensorImage` object from the numpy array.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`array`
</td>
<td>
numpy array with dtype=uint8. Its shape should be either (h, w, 3)
or (1, h, w, 3) for RGB images, either (h, w) or (1, h, w) for GRAYSCALE
images and either (h, w, 4) or (1, h, w, 4) for RGBA images.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`TensorImage` object.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
ValueError if the dytype of the numpy array is not `uint8` or the
dimention is not the valid dimention.
</td>
</tr>

</table>



<h3 id="create_from_buffer"><code>create_from_buffer</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/core/tensor_image.py#L81-L96">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_buffer(
    buffer: str
) -> 'TensorImage'
</code></pre>

Creates `TensorImage` object from the binary buffer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`buffer`
</td>
<td>
Binary memory buffer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`TensorImage` object.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
RuntimeError if the binary buffer can't be decoded into `TensorImage`
object.
</td>
</tr>

</table>



<h3 id="create_from_file"><code>create_from_file</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/vision/core/tensor_image.py#L43-L57">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_file(
    file_name: str
) -> 'TensorImage'
</code></pre>

Creates `TensorImage` object from the image file.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`file_name`
</td>
<td>
Image file name.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`TensorImage` object.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
RuntimeError if the image file can't be decoded.
</td>
</tr>

</table>
