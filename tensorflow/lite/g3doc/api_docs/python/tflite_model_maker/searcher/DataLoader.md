page_type: reference
description: Base DataLoader class for Searcher task.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.searcher.DataLoader" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="append"/>
</div>

# tflite_model_maker.searcher.DataLoader

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/searcher_dataloader.py#L22-L106">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Base DataLoader class for Searcher task.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.searcher.DataLoader(
    embedder_path: Optional[str] = None,
    dataset: Optional[np.ndarray] = None,
    metadata: Optional[List[AnyStr]] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`embedder_path`<a id="embedder_path"></a>
</td>
<td>
Path to the TFLite Embedder model file.
</td>
</tr><tr>
<td>
`dataset`<a id="dataset"></a>
</td>
<td>
Embedding dataset used to build on-device ScaNN index file. The
dataset shape should be (dataset_size, embedding_dim). If None,
`dataset` will be generated from raw input data later.
</td>
</tr><tr>
<td>
`metadata`<a id="metadata"></a>
</td>
<td>
 The metadata for each data in the dataset. The length of
`metadata` should be same as `dataset` and passed in the same order as
`dataset`. If `dataset` is set, `metadata` should be set as well.
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



<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/searcher_dataloader.py#L60-L61">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>
