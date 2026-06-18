page_type: reference
description: Creates the similarity search model with ScaNN.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.searcher.Searcher" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_from_data"/>
<meta itemprop="property" content="create_from_server_scann"/>
<meta itemprop="property" content="export"/>
</div>

# tflite_model_maker.searcher.Searcher

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/searcher.py#L159-L355">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Creates the similarity search model with ScaNN.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.searcher.Searcher(
    serialized_scann_path: str,
    metadata: List[AnyStr],
    embedder_path: Optional[str] = None
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
`serialized_scann_path`<a id="serialized_scann_path"></a>
</td>
<td>
Path to the dir that contains the ScaNN's
artifacts.
</td>
</tr><tr>
<td>
`metadata`<a id="metadata"></a>
</td>
<td>
The metadata for each of the embeddings in the database. Passed
in the same order as the embeddings in ScaNN.
</td>
</tr><tr>
<td>
`embedder_path`<a id="embedder_path"></a>
</td>
<td>
Path to the TFLite Embedder model file.
</td>
</tr>
</table>



## Methods

<h3 id="create_from_data"><code>create_from_data</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/searcher.py#L200-L245">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_data(
    data: <a href="../../tflite_model_maker/searcher/DataLoader"><code>tflite_model_maker.searcher.DataLoader</code></a>,
    scann_options: <a href="../../tflite_model_maker/searcher/ScaNNOptions"><code>tflite_model_maker.searcher.ScaNNOptions</code></a>,
    cache_dir: Optional[str] = None
) -> 'Searcher'
</code></pre>

"Creates the instance from data.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`data`
</td>
<td>
Data used to create scann.
</td>
</tr><tr>
<td>
`scann_options`
</td>
<td>
Options to build the ScaNN index file.
</td>
</tr><tr>
<td>
`cache_dir`
</td>
<td>
The cache directory to save serialized ScaNN and/or the tflite
model. When cache_dir is not set, a temporary folder will be created and
will **not** be removed automatically which makes it can be used later.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Searcher instance.
</td>
</tr>

</table>



<h3 id="create_from_server_scann"><code>create_from_server_scann</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/searcher.py#L180-L198">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create_from_server_scann(
    serialized_scann_path: str,
    metadata: List[AnyStr],
    embedder_path: Optional[str] = None
) -> 'Searcher'
</code></pre>

Creates the instance from the serialized serving scann directory.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`serialized_scann_path`
</td>
<td>
Path to the dir that contains the ScaNN's
artifacts.
</td>
</tr><tr>
<td>
`metadata`
</td>
<td>
The metadata for each of the embeddings in the database. Passed
in the same order as the embeddings in ScaNN.
</td>
</tr><tr>
<td>
`embedder_path`
</td>
<td>
Path to the TFLite Embedder model file.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Searcher instance.
</td>
</tr>

</table>



<h3 id="export"><code>export</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/searcher.py#L247-L355">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export(
    export_format: <a href="../../tflite_model_maker/searcher/ExportFormat"><code>tflite_model_maker.searcher.ExportFormat</code></a>,
    export_filename: str,
    userinfo: AnyStr,
    compression: bool = True
)
</code></pre>

Export the searcher model.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`export_format`
</td>
<td>
Export format that could be tflite or on-device ScaNN index
file, must be <a href="../../tflite_model_maker/audio_classifier/AudioClassifier#DEFAULT_EXPORT_FORMAT"><code>ExportFormat.TFLITE</code></a> or <a href="../../tflite_model_maker/searcher/ExportFormat#SCANN_INDEX_FILE"><code>ExportFormat.SCANN_INDEX_FILE</code></a>.
</td>
</tr><tr>
<td>
`export_filename`
</td>
<td>
File name to save the exported file. The exported file
can be TFLite model or on-device ScaNN index file.
</td>
</tr><tr>
<td>
`userinfo`
</td>
<td>
A special field in the index file that can be an arbitrary
string supplied by the user.
</td>
</tr><tr>
<td>
`compression`
</td>
<td>
Whether to snappy compress the index file.
</td>
</tr>
</table>
