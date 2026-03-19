page_type: reference
description: DataLoader class for Text Searcher.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.searcher.TextDataLoader" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="append"/>
<meta itemprop="property" content="create"/>
<meta itemprop="property" content="load_from_csv"/>
</div>

# tflite_model_maker.searcher.TextDataLoader

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/text_searcher_dataloader.py#L30-L125">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



DataLoader class for Text Searcher.

Inherits From: [`DataLoader`](../../tflite_model_maker/searcher/DataLoader)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.searcher.TextDataLoader(
    embedder: text_embedder.TextEmbedder
) -> None
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
    <li><a href="https://www.tensorflow.org/lite/models/modify/model_maker/text_searcher">Text Searcher with TensorFlow Lite Model Maker</a></li>
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
`embedder`<a id="embedder"></a>
</td>
<td>
Embedder to generate embedding from raw input image.
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

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/text_searcher_dataloader.py#L43-L69">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create(
    text_embedder_path: str, l2_normalize: bool = False
) -> 'DataLoader'
</code></pre>

Creates DataLoader for the Text Searcher task.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`text_embedder_path`
</td>
<td>
Path to the ".tflite" text embedder model. case and L2
norm is thus achieved through TF Lite inference.
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
DataLoader object created for the Text Searcher task.
</td>
</tr>

</table>



<h3 id="load_from_csv"><code>load_from_csv</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/text_searcher_dataloader.py#L71-L125">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_from_csv(
    path: str,
    text_column: str,
    metadata_column: str,
    delimiter: str = &#x27;,&#x27;,
    quotechar: str = &#x27;\&#x27;&quot;
) -> None
</code></pre>

Loads text data from csv file that includes a "header" line with titles.

Users can load text from different csv files one by one. For instance,

```
# Creates data_loader instance.
data_loader = text_searcher_dataloader.DataLoader.create(tflite_path)

# Loads text, first from `text_path1` and secondly from `text_path2`.
data_loader.load_from_csv(
    text_path1, text_column='text', metadata_column='metadata')
data_loader.load_from_csv(
    text_path2, text_column='text', metadata_column='metadata')
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
Text csv file path to be loaded.
</td>
</tr><tr>
<td>
`text_column`
</td>
<td>
Column name for input text.
</td>
</tr><tr>
<td>
`metadata_column`
</td>
<td>
Column name for user metadata associated with each input
text.
</td>
</tr><tr>
<td>
`delimiter`
</td>
<td>
Character used to separate fields.
</td>
</tr><tr>
<td>
`quotechar`
</td>
<td>
Character used to quote fields containing special characters.
</td>
</tr>
</table>



<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/searcher_dataloader.py#L60-L61">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>
