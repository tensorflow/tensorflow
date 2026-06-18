page_type: reference
description: K-Means partitioning tree configuration.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.searcher.Tree" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="min_partition_size"/>
<meta itemprop="property" content="quantize_centroids"/>
<meta itemprop="property" content="random_init"/>
<meta itemprop="property" content="spherical"/>
<meta itemprop="property" content="training_iterations"/>
<meta itemprop="property" content="training_sample_size"/>
</div>

# tflite_model_maker.searcher.Tree

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/task/searcher.py#L45-L82">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



K-Means partitioning tree configuration.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.searcher.Tree(
    num_leaves: int,
    num_leaves_to_search: int,
    training_sample_size: int = 100000,
    min_partition_size: int = 50,
    training_iterations: int = 12,
    spherical: bool = False,
    quantize_centroids: bool = False,
    random_init: bool = True
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


In ScaNN, we use single layer K-Means tree to partition the database (index)
as a way to reduce search space.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`num_leaves`<a id="num_leaves"></a>
</td>
<td>
How many leaves (partitions) to have on the K-Means tree. In
general, a good starting point would be the square root of the database
size.
</td>
</tr><tr>
<td>
`num_leaves_to_search`<a id="num_leaves_to_search"></a>
</td>
<td>
During inference ScaNN will compare the query vector
against all the partition centroids and select the closest
`num_leaves_to_search` ones to search in. The more leaves to search, the
better the retrieval quality, and higher computational cost.
</td>
</tr><tr>
<td>
`training_sample_size`<a id="training_sample_size"></a>
</td>
<td>
How many database embeddings to sample for the K-Means
training. Generally, you want to use a large enough sample of the database
to train K-Means so that it's representative enough. However, large sample
can also lead to longer training time. A good starting value would be
100k, or the whole dataset if it's smaller than that.
</td>
</tr><tr>
<td>
`min_partition_size`<a id="min_partition_size"></a>
</td>
<td>
Smallest allowable cluster size. Any clusters smaller
than this will be removed, and its data points will be merged with other
clusters. Recommended to be 1/10 of average cluster size (size of database
divided by `num_leaves`)
</td>
</tr><tr>
<td>
`training_iterations`<a id="training_iterations"></a>
</td>
<td>
How many itrations to train K-Means.
</td>
</tr><tr>
<td>
`spherical`<a id="spherical"></a>
</td>
<td>
If true, L2 normalize the K-Means centroids.
</td>
</tr><tr>
<td>
`quantize_centroids`<a id="quantize_centroids"></a>
</td>
<td>
If true, quantize centroids to int8.
</td>
</tr><tr>
<td>
`random_init`<a id="random_init"></a>
</td>
<td>
If true, use random init. Otherwise use K-Means++.
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
min_partition_size<a id="min_partition_size"></a>
</td>
<td>
`50`
</td>
</tr><tr>
<td>
quantize_centroids<a id="quantize_centroids"></a>
</td>
<td>
`False`
</td>
</tr><tr>
<td>
random_init<a id="random_init"></a>
</td>
<td>
`True`
</td>
</tr><tr>
<td>
spherical<a id="spherical"></a>
</td>
<td>
`False`
</td>
</tr><tr>
<td>
training_iterations<a id="training_iterations"></a>
</td>
<td>
`12`
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
