description: Tokenizes a tensor of UTF-8 string into words according to logits.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.SplitMergeFromLogitsTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="split_with_offsets"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.SplitMergeFromLogitsTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/split_merge_from_logits_tokenizer.py">View
source</a>

Tokenizes a tensor of UTF-8 string into words according to logits.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md),
[`Tokenizer`](../text/Tokenizer.md),
[`SplitterWithOffsets`](../text/SplitterWithOffsets.md),
[`Splitter`](../text/Splitter.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.SplitMergeFromLogitsTokenizer(
    force_split_at_break_character=True
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr> <td>
`force_split_at_break_character`<a id="force_split_at_break_character"></a>
</td> <td> a bool that indicates whether to force start a new word after an
ICU-defined whitespace character. Regardless of this parameter, we never include
a whitespace into a token, and we always ignore the split/merge action for the
whitespace character itself. This parameter indicates what happens after a
whitespace. * if force_split_at_break_character is true, create a new word
starting at the first non-space character, regardless of the 0/1 label for that
character, for instance:

~~~
```python
s = [2.0, 1.0]  # sample pair of logits indicating a split action
m = [1.0, 3.0]  # sample pair of logits indicating a merge action

strings=["New York"]
logits=[[s, m, m, s, m, m, m, m]]
output tokens=[["New", "York"]]

strings=["New York"]
logits=[[s, m, m, m, m, m, m, m]]
output tokens=[["New", "York"]]

strings=["New York"],
logits=[[s, m, m, m, s, m, m, m]]
output tokens=[["New", "York"]]
```
~~~

*   otherwise, create a new word / continue the current one depending on the
    action for the first non-whitespace character.

    ```python
    s = [2.0, 1.0]  # sample pair of logits indicating a split action
    m = [1.0, 3.0]  # sample pair of logits indicating a merge action

    strings=["New York"],
    logits=[[s, m, m, s, m, m, m, m]]
    output tokens=[["NewYork"]]

    strings=["New York"],
    logits=[[s, m, m, m, m, m, m, m]]
    output tokens=[["NewYork"]]

    strings=["New York"],
    logits=[[s, m, m, m, s, m, m, m]]
    output tokens=[["New", "York"]]
    ```

    </td>
    </tr>
    </table>

## Methods

<h3 id="split"><code>split</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>split(
    input
)
</code></pre>

Alias for
<a href="../text/Tokenizer.md#tokenize"><code>Tokenizer.tokenize</code></a>.

<h3 id="split_with_offsets"><code>split_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>split_with_offsets(
    input
)
</code></pre>

Alias for
<a href="../text/TokenizerWithOffsets.md#tokenize_with_offsets"><code>TokenizerWithOffsets.tokenize_with_offsets</code></a>.

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/split_merge_from_logits_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize(
    strings, logits
)
</code></pre>

Tokenizes a tensor of UTF-8 strings according to logits.

The logits refer to the split / merge action we should take for each
character.  For more info, see the doc for the logits argument below.

### Example:

```
>>> strings = ['IloveFlume!', 'and tensorflow']
>>> logits = [
... [
...     # 'I'
...     [5.0, -3.2],  # I: split
...     # 'love'
...     [2.2, -1.0],  # l: split
...     [0.2, 12.0],  # o: merge
...     [0.0, 11.0],  # v: merge
...     [-3.0, 3.0],  # e: merge
...     # 'Flume'
...     [10.0, 0.0],  # F: split
...     [0.0, 11.0],  # l: merge
...     [0.0, 11.0],  # u: merge
...     [0.0, 12.0],  # m: merge
...     [0.0, 12.0],  # e: merge
...     # '!'
...     [5.2, -7.0],  # !: split
...     # padding:
...     [1.0, 0.0], [1.0, 1.0], [1.0, 0.0],
... ], [
...     # 'and'
...     [2.0, 0.7],  # a: split
...     [0.2, 1.5],  # n: merge
...     [0.5, 2.3],  # d: merge
...     # ' '
...     [1.7, 7.0],  # <space>: merge
...     # 'tensorflow'
...     [2.2, 0.1],  # t: split
...     [0.2, 3.1],  # e: merge
...     [1.1, 2.5],  # n: merge
...     [0.7, 0.9],  # s: merge
...     [0.6, 1.0],  # o: merge
...     [0.3, 1.0],  # r: merge
...     [0.2, 2.2],  # f: merge
...     [0.7, 3.1],  # l: merge
...     [0.4, 5.0],  # o: merge
...     [0.8, 6.0],  # w: merge
... ]]
>>> tokenizer = SplitMergeFromLogitsTokenizer()
>>> tokenizer.tokenize(strings, logits)
<tf.RaggedTensor [[b'I', b'love', b'Flume', b'!'], [b'and', b'tensorflow']]>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`strings`
</td>
<td>
a 1D `Tensor` of UTF-8 strings.
</td>
</tr><tr>
<td>
`logits`
</td>
<td>
3D Tensor; logits[i,j,0] is the logit for the split action for
j-th character of strings[i].  logits[i,j,1] is the logit for the merge
action for that same character.  For each character, we pick the action
with the greatest logit.  Split starts a new word at this character and
merge adds this character to the previous word.  The shape of this
tensor should be (n, m, 2) where n is the number of strings, and m is
greater or equal with the number of characters from each strings[i].  As
the elements of the strings tensor may have different lengths (in UTF-8
chars), padding may be required to get a dense vector; for each row, the
extra (padding) pairs of logits are ignored.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `RaggedTensor` of strings where `tokens[i, k]` is the string
content of the `k-th` token in `strings[i]`
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`InvalidArgumentError`
</td>
<td>
if one of the input Tensors has the wrong shape.
E.g., if the logits tensor does not have enough elements for one of the
strings.
</td>
</tr>
</table>



<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/split_merge_from_logits_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize_with_offsets(
    strings, logits
)
</code></pre>

Tokenizes a tensor of UTF-8 strings into tokens with [start,end) offsets.

### Example:

```
>>> strings = ['IloveFlume!', 'and tensorflow']
>>> logits = [
... [
...     # 'I'
...     [5.0, -3.2],  # I: split
...     # 'love'
...     [2.2, -1.0],  # l: split
...     [0.2, 12.0],  # o: merge
...     [0.0, 11.0],  # v: merge
...     [-3.0, 3.0],  # e: merge
...     # 'Flume'
...     [10.0, 0.0],  # F: split
...     [0.0, 11.0],  # l: merge
...     [0.0, 11.0],  # u: merge
...     [0.0, 12.0],  # m: merge
...     [0.0, 12.0],  # e: merge
...     # '!'
...     [5.2, -7.0],  # !: split
...     # padding:
...     [1.0, 0.0], [1.0, 1.0], [1.0, 0.0],
... ], [
...     # 'and'
...     [2.0, 0.7],  # a: split
...     [0.2, 1.5],  # n: merge
...     [0.5, 2.3],  # d: merge
...     # ' '
...     [1.7, 7.0],  # <space>: merge
...     # 'tensorflow'
...     [2.2, 0.1],  # t: split
...     [0.2, 3.1],  # e: merge
...     [1.1, 2.5],  # n: merge
...     [0.7, 0.9],  # s: merge
...     [0.6, 1.0],  # o: merge
...     [0.3, 1.0],  # r: merge
...     [0.2, 2.2],  # f: merge
...     [0.7, 3.1],  # l: merge
...     [0.4, 5.0],  # o: merge
...     [0.8, 6.0],  # w: merge
... ]]
>>> tokenizer = SplitMergeFromLogitsTokenizer()
>>> tokens, starts, ends = tokenizer.tokenize_with_offsets(strings, logits)
>>> tokens
<tf.RaggedTensor [[b'I', b'love', b'Flume', b'!'], [b'and', b'tensorflow']]>
>>> starts
<tf.RaggedTensor [[0, 1, 5, 10], [0, 4]]>
>>> ends
<tf.RaggedTensor [[1, 5, 10, 11], [3, 14]]>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`strings`
</td>
<td>
A 1D `Tensor` of UTF-8 strings.
</td>
</tr><tr>
<td>
`logits`
</td>
<td>
3D Tensor; logits[i,j,0] is the logit for the split action for
j-th character of strings[i].  logits[i,j,1] is the logit for the merge
action for that same character.  For each character, we pick the action
with the greatest logit.  Split starts a new word at this character and
merge adds this character to the previous word.  The shape of this
tensor should be (n, m, 2) where n is the number of strings, and m is
greater or equal with the number of characters from each strings[i].  As
the elements of the strings tensor may have different lengths (in UTF-8
chars), padding may be required to get a dense vector; for each row, the
extra (padding) pairs of logits are ignored.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple `(tokens, start_offsets, end_offsets)` where:
* `tokens` is a `RaggedTensor` of strings where `tokens[i, k]` is
  the string content of the `k-th` token in `strings[i]`
* `start_offsets` is a `RaggedTensor` of int64s where
  `start_offsets[i, k]` is the byte offset for the start of the
  `k-th` token in `strings[i]`.
* `end_offsets` is a `RaggedTensor` of int64s where
  `end_offsets[i, k]` is the byte offset immediately after the
  end of the `k-th` token in `strings[i]`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`InvalidArgumentError`
</td>
<td>
if one of the input Tensors has the wrong shape.
E.g., if the tensor logits does not have enough elements for one of the
strings.
</td>
</tr>
</table>





