description: Splits an already-tokenized tensor of tokens further into WordPiece
tokens.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.keras.layers.WordpieceTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="set_vocabulary"/>
</div>

# text.keras.layers.WordpieceTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/keras/layers/tokenization_layers.py">View
source</a>

Splits an already-tokenized tensor of tokens further into WordPiece tokens.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.keras.layers.WordpieceTokenizer(
    vocabulary=None,
    suffix_indicator=&#x27;##&#x27;,
    max_bytes_per_word=100,
    token_out_type=tf.string,
    unknown_token=&#x27;[UNK]&#x27;,
    pad_value=None,
    merge_wordpiece_dim=True,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

Splits a set of string tokens into subwords as described in
https://arxiv.org/pdf/1609.08144.pdf. This layer does not build the WordPiece
vocabulary; instead, users should set the vocabulary by either passing it to the
init call or by calling set_vocabulary() after the layer is constructed.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`vocabulary`<a id="vocabulary"></a>
</td>
<td>
An optional list of vocabulary terms, or a path to a text file
containing a vocabulary to load into this layer. The file should contain
one token per line. If the list or file contains the same token multiple
times, an error will be thrown.
</td>
</tr><tr>
<td>
`suffix_indicator`<a id="suffix_indicator"></a>
</td>
<td>
(optional) The characters prepended to a wordpiece to
indicate that it is a suffix to another subword. Default is '##'.
</td>
</tr><tr>
<td>
`max_bytes_per_word`<a id="max_bytes_per_word"></a>
</td>
<td>
(optional) Max size of input token. Default is 100.
</td>
</tr><tr>
<td>
`token_out_type`<a id="token_out_type"></a>
</td>
<td>
(optional) The type of the token to return. This can be
`tf.int64` IDs, or `tf.string` subwords. The default is `tf.int64`.
</td>
</tr><tr>
<td>
`unknown_token`<a id="unknown_token"></a>
</td>
<td>
(optional) The string value to substitute for an unknown
token. Default is "[UNK]". If set to `None`, no substitution occurs.
If `token_out_type` is `tf.int64`, the `vocabulary` is used (after
substitution) to convert the unknown token to an integer, resulting in -1
if `unknown_token` is set to `None` or not contained in the `vocabulary`.
</td>
</tr><tr>
<td>
`pad_value`<a id="pad_value"></a>
</td>
<td>
if not None, performs the padding (using pad_value) at the
inner-most dimension (i.e. token dimension) and outputs a padded dense
tensor (default=None).
</td>
</tr><tr>
<td>
`merge_wordpiece_dim`<a id="merge_wordpiece_dim"></a>
</td>
<td>
If False, this layer will output a RaggedTensor
with an additional inner 'wordpiece' dimension, containing the wordpieces
for each token. If set to True, this layer will concatenate and squeeze
along that dimension. Defaults to True.
</td>
</tr>
</table>

## Methods

<h3 id="set_vocabulary"><code>set_vocabulary</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/keras/layers/tokenization_layers.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_vocabulary(
    vocab
)
</code></pre>
