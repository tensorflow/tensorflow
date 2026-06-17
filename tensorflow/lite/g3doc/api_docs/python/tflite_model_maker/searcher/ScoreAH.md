page_type: reference
description: Product Quantization (PQ) based in-partition scoring configuration.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.searcher.ScoreAH" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="anisotropic_quantization_threshold"/>
<meta itemprop="property" content="training_iterations"/>
<meta itemprop="property" content="training_sample_size"/>
</div>

# tflite_model_maker.searcher.ScoreAH

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/searcher.py#L85-L118">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Product Quantization (PQ) based in-partition scoring configuration.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.searcher.ScoreAH(
    dimensions_per_block: int,
    anisotropic_quantization_threshold: float = float(&#x27;nan&#x27;),
    training_sample_size: int = 100000,
    training_iterations: int = 10
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


In ScaNN we use PQ to compress the database embeddings, but not the query
embedding. We called it Asymmetric Hashing. See
<a href="https://research.google/pubs/pub41694/">https://research.google/pubs/pub41694/</a>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`dimensions_per_block`<a id="dimensions_per_block"></a>
</td>
<td>
How many dimensions in each PQ block. If the embedding
vector dimensionality is a multiple of this value, there will be
`number_of_dimensions / dimensions_per_block` PQ blocks. Otherwise, the
last block will be the remainder. For example, if a vector has 12
dimensions, and `dimensions_per_block` is 2, then there will be 6
2-dimension blocks. However, if the vector has 13 dimensions and
`dimensions_per_block` is still 2, there will be 6 2-dimension blocks and
one 1-dimension block.
</td>
</tr><tr>
<td>
`anisotropic_quantization_threshold`<a id="anisotropic_quantization_threshold"></a>
</td>
<td>
If this value is set, we will penalize
the quantization error that's parallel to the original vector differently
than the orthogonal error. A generally recommended value for this
parameter would be 0.2. For more details, please look at ScaNN's 2020 ICML
paper https://arxiv.org/abs/1908.10396 and the Google AI Blog post
https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html
</td>
</tr><tr>
<td>
`training_sample_size`<a id="training_sample_size"></a>
</td>
<td>
How many database points to sample for training the
K-Means for PQ centers. A good starting value would be 100k or the whole
dataset if it's smaller than that.
</td>
</tr><tr>
<td>
`training_iterations`<a id="training_iterations"></a>
</td>
<td>
How many iterations to run K-Means for PQ.
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
anisotropic_quantization_threshold<a id="anisotropic_quantization_threshold"></a>
</td>
<td>
`nan`
</td>
</tr><tr>
<td>
training_iterations<a id="training_iterations"></a>
</td>
<td>
`10`
</td>
</tr><tr>
<td>
training_sample_size<a id="training_sample_size"></a>
</td>
<td>
`100000`
</td>
</tr>
</table>
