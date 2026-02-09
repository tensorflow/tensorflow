description: A Splitter that uses a state machine to determine sentence breaks.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.StateBasedSentenceBreaker" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="break_sentences"/>
<meta itemprop="property" content="break_sentences_with_offsets"/>
</div>

# text.StateBasedSentenceBreaker

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/state_based_sentence_breaker_op.py">View
source</a>

A `Splitter` that uses a state machine to determine sentence breaks.

<!-- Placeholder for "Used in" -->

`StateBasedSentenceBreaker` splits text into sentences by using a state
machine to determine when a sequence of characters indicates a potential
sentence break.

The state machine consists of an `initial state`, then transitions to a
`collecting terminal punctuation state` once an acronym, an emoticon, or
terminal punctuation (ellipsis, question mark, exclamation point, etc.), is
encountered.

It transitions to the `collecting close punctuation state` when a close
punctuation (close bracket, end quote, etc.) is found.

If non-punctuation is encountered in the collecting terminal punctuation or
collecting close punctuation states, then the state machine exits, returning
false, indicating it has moved past the end of a potential sentence fragment.

## Methods

<h3 id="break_sentences"><code>break_sentences</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/state_based_sentence_breaker_op.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>break_sentences(
    doc
)
</code></pre>

Splits `doc` into sentence fragments and returns the fragments' text.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`doc`
</td>
<td>
A string `Tensor` of shape [batch] with a batch of documents.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`results`
</td>
<td>
A string `RaggedTensor` of shape [batch, (num_sentences)]
with each input broken up into its constituent sentence fragments.
</td>
</tr>
</table>



<h3 id="break_sentences_with_offsets"><code>break_sentences_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/state_based_sentence_breaker_op.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>break_sentences_with_offsets(
    doc
)
</code></pre>

Splits `doc` into sentence fragments, returns text, start & end offsets.


#### Example:

```
                1                  1         2         3
      012345678901234    01234567890123456789012345678901234567
doc: 'Hello...foo bar', 'Welcome to the U.S. don't be surprised'

fragment_text: [
  ['Hello...', 'foo bar'],
  ['Welcome to the U.S.' , 'don't be surprised']
]
start: [[0, 8],[0, 20]]
end: [[8, 15],[19, 38]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`doc`
</td>
<td>
A string `Tensor` of shape `[batch]` or `[batch, 1]`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of `(fragment_text, start, end)` where:
</td>
</tr>
<tr>
<td>
`fragment_text`
</td>
<td>
A string `RaggedTensor` of shape [batch, (num_sentences)]
with each input broken up into its constituent sentence fragments.
</td>
</tr><tr>
<td>
`start`
</td>
<td>
A int64 `RaggedTensor` of shape [batch, (num_sentences)]
where each entry is the inclusive beginning byte offset of a sentence.
</td>
</tr><tr>
<td>
`end`
</td>
<td>
A int64 `RaggedTensor` of shape [batch, (num_sentences)]
where each entry is the exclusive ending byte offset of a sentence.
</td>
</tr>
</table>
