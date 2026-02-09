description: Sentencepiece tokenizer with tf.text interface.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.FastSentencepieceTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="detokenize"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="vocab_size"/>
</div>

# text.FastSentencepieceTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/fast_sentencepiece_tokenizer.py">View
source</a>

Sentencepiece tokenizer with tf.text interface.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.FastSentencepieceTokenizer(
    model, reverse=False, add_bos=False, add_eos=False
)
</code></pre>

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="detokenize"><code>detokenize</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/fast_sentencepiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>detokenize(
    input
)
</code></pre>

Detokenizes tokens into preprocessed text.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input`
</td>
<td>
A `RaggedTensor` or `Tensor` with int32 encoded text with rank >=
1.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A N-1 dimensional string Tensor or RaggedTensor of the detokenized text.
</td>
</tr>

</table>

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/fast_sentencepiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize(
    inputs
)
</code></pre>

The main tokenization function.

<h3 id="vocab_size"><code>vocab_size</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/fast_sentencepiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>vocab_size()
</code></pre>

Returns size of the vocabulary in Sentencepiece model.
