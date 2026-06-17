page_type: reference
description: Options to build ScaNN.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.searcher.ScaNNOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="score_ah"/>
<meta itemprop="property" content="score_brute_force"/>
<meta itemprop="property" content="tree"/>
</div>

# tflite_model_maker.searcher.ScaNNOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/searcher.py#L131-L156">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Options to build ScaNN.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.searcher.ScaNNOptions(
    distance_measure: str,
    tree: Optional[<a href="../../tflite_model_maker/searcher/Tree"><code>tflite_model_maker.searcher.Tree</code></a>] = None,
    score_ah: Optional[<a href="../../tflite_model_maker/searcher/ScoreAH"><code>tflite_model_maker.searcher.ScoreAH</code></a>] = None,
    score_brute_force: Optional[<a href="../../tflite_model_maker/searcher/ScoreBruteForce"><code>tflite_model_maker.searcher.ScoreBruteForce</code></a>] = None
)
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


ScaNN
(<a href="https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html">https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html</a>) is
a highly efficient and scalable vector nearest neighbor retrieval
library from Google Research. We use ScaNN to build the on-device search
index, and do on-device retrieval with a simplified implementation.




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`distance_measure`<a id="distance_measure"></a>
</td>
<td>
How to compute the distance. Allowed values are
'dot_product' and 'squared_l2'. Please note that when distance is
'dot_product', we actually compute the negative dot product between query
and database vectors, to preserve the notion that "smaller is closer".
</td>
</tr><tr>
<td>
`tree`<a id="tree"></a>
</td>
<td>
Configure partitioning. If not set, no partitioning is performed.
</td>
</tr><tr>
<td>
`score_ah`<a id="score_ah"></a>
</td>
<td>
Configure asymmetric hashing. Must defined this or
`score_brute_force`.
</td>
</tr><tr>
<td>
`score_brute_force`<a id="score_brute_force"></a>
</td>
<td>
Configure bruce force. Must defined this or `score_ah`.
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








<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
score_ah<a id="score_ah"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
score_brute_force<a id="score_brute_force"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
tree<a id="tree"></a>
</td>
<td>
`None`
</td>
</tr>
</table>
