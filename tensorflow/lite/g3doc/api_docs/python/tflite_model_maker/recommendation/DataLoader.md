page_type: reference
description: Recommendation data loader.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_model_maker.recommendation.DataLoader" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="download_and_extract_movielens"/>
<meta itemprop="property" content="from_movielens"/>
<meta itemprop="property" content="gen_dataset"/>
<meta itemprop="property" content="generate_movielens_dataset"/>
<meta itemprop="property" content="get_num_classes"/>
<meta itemprop="property" content="load_vocab"/>
<meta itemprop="property" content="split"/>
</div>

# tflite_model_maker.recommendation.DataLoader

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/recommendation_dataloader.py#L32-L298">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Recommendation data loader.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_model_maker.recommendation.DataLoader(
    dataset, size, vocab
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dataset`<a id="dataset"></a>
</td>
<td>
tf.data.Dataset for recommendation.
</td>
</tr><tr>
<td>
`size`<a id="size"></a>
</td>
<td>
int, dataset size.
</td>
</tr><tr>
<td>
`vocab`<a id="vocab"></a>
</td>
<td>
list of dict, each vocab item is described above.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`size`<a id="size"></a>
</td>
<td>
Returns the size of the dataset.

Note that this function may return None becuase the exact size of the
dataset isn't a necessary parameter to create an instance of this class,
and tf.data.Dataset donesn't support a function to get the length directly
since it's lazy-loaded and may be infinite.
In most cases, however, when an instance of this class is created by helper
functions like 'from_folder', the size of the dataset will be preprocessed,
and this function can return an int representing the size of the dataset.
</td>
</tr>
</table>



## Methods

<h3 id="download_and_extract_movielens"><code>download_and_extract_movielens</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/recommendation_dataloader.py#L141-L144">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>download_and_extract_movielens(
    download_dir
)
</code></pre>

Downloads and extracts movielens dataset, then returns extracted dir.


<h3 id="from_movielens"><code>from_movielens</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/recommendation_dataloader.py#L225-L298">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_movielens(
    data_dir,
    data_tag,
    input_spec: <a href="../../tflite_model_maker/recommendation/spec/InputSpec"><code>tflite_model_maker.recommendation.spec.InputSpec</code></a>,
    generated_examples_dir=None,
    min_timeline_length=3,
    max_context_length=10,
    max_context_movie_genre_length=10,
    min_rating=None,
    train_data_fraction=0.9,
    build_vocabs=True,
    train_filename=&#x27;train_movielens_1m.tfrecord&#x27;,
    test_filename=&#x27;test_movielens_1m.tfrecord&#x27;,
    vocab_filename=&#x27;movie_vocab.json&#x27;,
    meta_filename=&#x27;meta.json&#x27;
)
</code></pre>

Generates data loader from movielens dataset.

The method downloads and prepares dataset, then generates for train/eval.

For `movielens` data format, see:

- function `_generate_fake_data` in `recommendation_testutil.py`
- Or, zip file: <a href="http://files.grouplens.org/datasets/movielens/ml-1m.zip">http://files.grouplens.org/datasets/movielens/ml-1m.zip</a>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`data_dir`
</td>
<td>
str, path to dataset containing (unzipped) text data.
</td>
</tr><tr>
<td>
`data_tag`
</td>
<td>
str, specify dataset in {'train', 'test'}.
</td>
</tr><tr>
<td>
`input_spec`
</td>
<td>
InputSpec, specify data format for input and embedding.
</td>
</tr><tr>
<td>
`generated_examples_dir`
</td>
<td>
str, path to generate preprocessed examples.
(default: same as data_dir)
</td>
</tr><tr>
<td>
`min_timeline_length`
</td>
<td>
int, min timeline length to split train/eval set.
</td>
</tr><tr>
<td>
`max_context_length`
</td>
<td>
int, max context length as one input.
</td>
</tr><tr>
<td>
`max_context_movie_genre_length`
</td>
<td>
int, max context length of movie genre as
one input.
</td>
</tr><tr>
<td>
`min_rating`
</td>
<td>
int or None, include examples with min rating.
</td>
</tr><tr>
<td>
`train_data_fraction`
</td>
<td>
float, percentage of training data [0.0, 1.0].
</td>
</tr><tr>
<td>
`build_vocabs`
</td>
<td>
boolean, whether to build vocabs.
</td>
</tr><tr>
<td>
`train_filename`
</td>
<td>
str, generated file name for training data.
</td>
</tr><tr>
<td>
`test_filename`
</td>
<td>
str, generated file name for test data.
</td>
</tr><tr>
<td>
`vocab_filename`
</td>
<td>
str, generated file name for vocab data.
</td>
</tr><tr>
<td>
`meta_filename`
</td>
<td>
str, generated file name for meta data.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Data Loader.
</td>
</tr>

</table>



<h3 id="gen_dataset"><code>gen_dataset</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/recommendation_dataloader.py#L62-L85">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gen_dataset(
    batch_size=1,
    is_training=False,
    shuffle=False,
    input_pipeline_context=None,
    preprocess=None,
    drop_remainder=True,
    total_steps=None
)
</code></pre>

Generates dataset, and overwrites default drop_remainder = True.


<h3 id="generate_movielens_dataset"><code>generate_movielens_dataset</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/recommendation_dataloader.py#L146-L208">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>generate_movielens_dataset(
    data_dir,
    generated_examples_dir=None,
    train_filename=&#x27;train_movielens_1m.tfrecord&#x27;,
    test_filename=&#x27;test_movielens_1m.tfrecord&#x27;,
    vocab_filename=&#x27;movie_vocab.json&#x27;,
    meta_filename=&#x27;meta.json&#x27;,
    min_timeline_length=3,
    max_context_length=10,
    max_context_movie_genre_length=10,
    min_rating=None,
    train_data_fraction=0.9,
    build_vocabs=True
)
</code></pre>

Generate movielens dataset, and returns a dict contains meta.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`data_dir`
</td>
<td>
str, path to dataset containing (unzipped) text data.
</td>
</tr><tr>
<td>
`generated_examples_dir`
</td>
<td>
str, path to generate preprocessed examples.
(default: same as data_dir)
</td>
</tr><tr>
<td>
`train_filename`
</td>
<td>
str, generated file name for training data.
</td>
</tr><tr>
<td>
`test_filename`
</td>
<td>
str, generated file name for test data.
</td>
</tr><tr>
<td>
`vocab_filename`
</td>
<td>
str, generated file name for vocab data.
</td>
</tr><tr>
<td>
`meta_filename`
</td>
<td>
str, generated file name for meta data.
</td>
</tr><tr>
<td>
`min_timeline_length`
</td>
<td>
int, min timeline length to split train/eval set.
</td>
</tr><tr>
<td>
`max_context_length`
</td>
<td>
int, max context length as one input.
</td>
</tr><tr>
<td>
`max_context_movie_genre_length`
</td>
<td>
int, max context length of movie genre as
one input.
</td>
</tr><tr>
<td>
`min_rating`
</td>
<td>
int or None, include examples with min rating.
</td>
</tr><tr>
<td>
`train_data_fraction`
</td>
<td>
float, percentage of training data [0.0, 1.0].
</td>
</tr><tr>
<td>
`build_vocabs`
</td>
<td>
boolean, whether to build vocabs.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Dict, metadata for the movielens dataset. Containing keys:
`train_file`, `train_size`, `test_file`, `test_size`, vocab_file`,
`vocab_size`, etc.
</td>
</tr>

</table>



<h3 id="get_num_classes"><code>get_num_classes</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/recommendation_dataloader.py#L210-L223">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>get_num_classes(
    meta
) -> int
</code></pre>

Gets number of classes.

0 is reserved. Number of classes is Max Id + 1, e.g., if Max Id = 100,
then classes are [0, 100], that is 101 classes in total.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`meta`
</td>
<td>
dict, containing meta['vocab_max_id'].
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Number of classes.
</td>
</tr>

</table>



<h3 id="load_vocab"><code>load_vocab</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/recommendation_dataloader.py#L90-L122">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>load_vocab(
    vocab_file
) -> collections.OrderedDict
</code></pre>

Loads vocab from file.

The vocab file should be json format of: a list of list[size=4], where the 4
elements are ordered as:
  [id=int, title=str, genres=str joined with '|', count=int]
It is generated when preparing movielens dataset.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`vocab_file`
</td>
<td>
str, path to vocab file.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`vocab`
</td>
<td>
an OrderedDict maps id to item. Each item represents a movie
{
  'id': int,
  'title': str,
  'genres': list[str],
  'count': int,
}
</td>
</tr>
</table>



<h3 id="split"><code>split</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/recommendation_dataloader.py#L87-L88">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>split(
    fraction
)
</code></pre>

Splits dataset into two sub-datasets with the given fraction.

Primarily used for splitting the data set into training and testing sets.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`fraction`
</td>
<td>
float, demonstrates the fraction of the first returned
subdataset in the original data.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The splitted two sub datasets.
</td>
</tr>

</table>



<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/core/data_util/dataloader.py#L126-L130">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>
