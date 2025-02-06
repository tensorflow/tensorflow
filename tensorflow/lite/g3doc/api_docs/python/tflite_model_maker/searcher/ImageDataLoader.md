page_type: reference
description: DataLoader class for Image Searcher Task.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.searcher.ImageDataLoader" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="append"/>
<meta itemprop="property" content="create"/>
<meta itemprop="property" content="load_from_folder"/>
</div>

# tflite_model_maker.searcher.ImageDataLoader

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/image_searcher_dataloader.py#L33-L155">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



DataLoader class for Image Searcher Task.

Inherits From: [`DataLoader`](../../tflite_model_maker/searcher/DataLoader)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.searcher.ImageDataLoader(
    embedder: image_embedder.ImageEmbedder,
    metadata_type: <a href="../../tflite_model_maker/searcher/MetadataType"><code>tflite_model_maker.searcher.MetadataType</code></a> = <a href="../../tflite_model_maker/searcher/MetadataType#FROM_FILE_NAME"><code>tflite_model_maker.searcher.MetadataType.FROM_FILE_NAME</code></a>
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`embedder`<a id="embedder"></a>
</td>
<td>
Embedder to generate embedding from raw input image.
</td>
</tr><tr>
<td>
`metadata_type`<a id="metadata_type"></a>
</td>
<td>
Type of MetadataLoader to load metadata for each input
data. By default, load the file name as metadata for each input data.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`dataset`<a id="dataset"></a>
</td>
<td>
Gets the dataset.

Due to performance consideration, we don't return a copy, but the returned
`self._dataset` should never be changed.
</td>
</tr><tr>
<td>
`embedder_path`<a id="embedder_path"></a>
</td>
<td>
Gets the path to the TFLite Embedder model file.
</td>
</tr><tr>
<td>
`metadata`<a id="metadata"></a>
</td>
<td>
Gets the metadata.
</td>
</tr>
</table>



## Methods

<h3 id="append"><code>append</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/searcher_dataloader.py#L92-L106">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>append(
    data_loader: 'DataLoader'
) -> None
</code></pre>

Appends the dataset.

Don't check if embedders from the two data loader are the same in this
function. Users are responsible to keep the embedder identical.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`data_loader`
</td>
<td>
The data loader in which the data will be appended.
</td>
</tr>
</table>



<h3 id="create"><code>create</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/image_searcher_dataloader.py#L60-L92">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create(
    image_embedder_path: str,
    metadata_type: <a href="../../tflite_model_maker/searcher/MetadataType"><code>tflite_model_maker.searcher.MetadataType</code></a> = <a href="../../tflite_model_maker/searcher/MetadataType#FROM_FILE_NAME"><code>tflite_model_maker.searcher.MetadataType.FROM_FILE_NAME</code></a>,
    l2_normalize: bool = False
) -> 'DataLoader'
</code></pre>

Creates DataLoader for the Image Searcher task.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`image_embedder_path`
</td>
<td>
Path to the ".tflite" image embedder model.
</td>
</tr><tr>
<td>
`metadata_type`
</td>
<td>
Type of MetadataLoader to load metadata for each input
image based on image path. By default, load the file name as metadata
for each input image.
</td>
</tr><tr>
<td>
`l2_normalize`
</td>
<td>
Whether to normalize the returned feature vector with L2
norm. Use this option only if the model does not already contain a
native L2_NORMALIZATION TF Lite Op. In most cases, this is already the
case and L2 norm is thus achieved through TF Lite inference.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
DataLoader object created for the Image Searcher task.
</td>
</tr>

</table>



<h3 id="load_from_folder"><code>load_from_folder</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/image_searcher_dataloader.py#L94-L155">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_from_folder(
    path: str, mode: str = &#x27;r&#x27;
) -> None
</code></pre>

Loads image data from folder.

Users can load images from different folders one by one. For instance,

```
# Creates data_loader instance.
data_loader = image_searcher_dataloader.DataLoader.create(tflite_path)

# Loads images, first from `image_path1` and secondly from `image_path2`.
data_loader.load_from_folder(image_path1)
data_loader.load_from_folder(image_path2)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`path`
</td>
<td>
image directory to be loaded.
</td>
</tr><tr>
<td>
`mode`
</td>
<td>
mode in which the file is opened, Used when metadata_type is
FROM_DAT_FILE. Only 'r' and 'rb' are supported. 'r' means opening for
reading, 'rb' means opening for reading binary.
</td>
</tr>
</table>



<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/searcher_dataloader.py#L60-L61">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>
