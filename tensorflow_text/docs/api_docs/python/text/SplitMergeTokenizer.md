description: Tokenizes a tensor of UTF-8 string into words according to labels.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.SplitMergeTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="split_with_offsets"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.SplitMergeTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/split_merge_tokenizer.py">View
source</a>

Tokenizes a tensor of UTF-8 string into words according to labels.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md),
[`Tokenizer`](../text/Tokenizer.md),
[`SplitterWithOffsets`](../text/SplitterWithOffsets.md),
[`Splitter`](../text/Splitter.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.SplitMergeTokenizer()
</code></pre>

<!-- Placeholder for "Used in" -->


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

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/split_merge_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize(
    input, labels, force_split_at_break_character=True
)
</code></pre>

Tokenizes a tensor of UTF-8 strings according to labels.

### Example:

```
>>> strings = ["HelloMonday", "DearFriday"]
>>> labels = [[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
...           [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]]
>>> tokenizer = SplitMergeTokenizer()
>>> tokenizer.tokenize(strings, labels)
<tf.RaggedTensor [[b'Hello', b'Monday'], [b'Dear', b'Friday']]>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr> <td> `input` </td> <td> An N-dimensional `Tensor` or `RaggedTensor` of
UTF-8 strings. </td> </tr><tr> <td> `labels` </td> <td> An (N+1)-dimensional
`Tensor` or `RaggedTensor` of `int32`, with `labels[i1...iN, j]` being the
split(0)/merge(1) label of the j-th character for `input[i1...iN]`. Here split
means create a new word with this character and merge means adding this
character to the previous word. </td> </tr><tr> <td>
`force_split_at_break_character` </td> <td> bool indicates whether to force
start a new word after seeing a ICU defined whitespace character. When seeing
one or more ICU defined whitespace character: * if
`force_split_at_break_character` is set true, then create a new word at the
first non-space character, regardless of the label of that character, for
instance:

```python
  input="New York"
  labels=[0, 1, 1, 0, 1, 1, 1, 1]
  output tokens=["New", "York"]
```

```python
  input="New York"
  labels=[0, 1, 1, 1, 1, 1, 1, 1]
  output tokens=["New", "York"]
```

```python
  input="New York",
  labels=[0, 1, 1, 1, 0, 1, 1, 1]
  output tokens=["New", "York"]
```

*   otherwise, whether to create a new word or not for the first non-space
    character depends on the label of that character, for instance:

    ```python
    input="New York",
    labels=[0, 1, 1, 0, 1, 1, 1, 1]
    output tokens=["NewYork"]
    ```

    ```python
    input="New York",
    labels=[0, 1, 1, 1, 1, 1, 1, 1]
    output tokens=["NewYork"]
    ```

    ```python
    input="New York",
    labels=[0, 1, 1, 1, 0, 1, 1, 1]
    output tokens=["New", "York"]
    ```

    </td>
    </tr>
    </table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `RaggedTensor` of strings where `tokens[i1...iN, j]` is the string
content of the `j-th` token in `input[i1...iN]`
</td>
</tr>

</table>

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/split_merge_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize_with_offsets(
    input, labels, force_split_at_break_character=True
)
</code></pre>

Tokenizes a tensor of UTF-8 strings into tokens with [start,end) offsets.

### Example:

```
>>> strings = ["HelloMonday", "DearFriday"]
>>> labels = [[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
...           [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]]
>>> tokenizer = SplitMergeTokenizer()
>>> tokens, starts, ends = tokenizer.tokenize_with_offsets(strings, labels)
>>> tokens
<tf.RaggedTensor [[b'Hello', b'Monday'], [b'Dear', b'Friday']]>
>>> starts
<tf.RaggedTensor [[0, 5], [0, 4]]>
>>> ends
<tf.RaggedTensor [[5, 11], [4, 10]]>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr> <td> `input` </td> <td> An N-dimensional `Tensor` or `RaggedTensor` of
UTF-8 strings. </td> </tr><tr> <td> `labels` </td> <td> An (N+1)-dimensional
`Tensor` or `RaggedTensor` of int32, with labels[i1...iN, j] being the
split(0)/merge(1) label of the j-th character for input[i1...iN]. Here split
means create a new word with this character and merge means adding this
character to the previous word. </td> </tr><tr> <td>
`force_split_at_break_character` </td> <td> bool indicates whether to force
start a new word after seeing a ICU defined whitespace character. When seeing
one or more ICU defined whitespace character: * if
`force_split_at_break_character` is set true, then create a new word at the
first non-space character, regardless of the label of that character, for
instance:

```python
  input="New York"
  labels=[0, 1, 1, 0, 1, 1, 1, 1]
  output tokens=["New", "York"]
```

```python
  input="New York"
  labels=[0, 1, 1, 1, 1, 1, 1, 1]
  output tokens=["New", "York"]
```

```python
  input="New York",
  labels=[0, 1, 1, 1, 0, 1, 1, 1]
  output tokens=["New", "York"]
```

*   otherwise, whether to create a new word or not for the first non-space
    character depends on the label of that character, for instance:

    ```python
    input="New York",
    labels=[0, 1, 1, 0, 1, 1, 1, 1]
    output tokens=["NewYork"]
    ```

    ```python
    input="New York",
    labels=[0, 1, 1, 1, 1, 1, 1, 1]
    output tokens=["NewYork"]
    ```

    ```python
    input="New York",
    labels=[0, 1, 1, 1, 0, 1, 1, 1]
    output tokens=["New", "York"]
    ```

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
</td>
</tr>
<tr>
<td>
`tokens`
</td>
<td>
is a `RaggedTensor` of strings where `tokens[i1...iN, j]` is
the string content of the `j-th` token in `input[i1...iN]`
</td>
</tr><tr>
<td>
`start_offsets`
</td>
<td>
is a `RaggedTensor` of int64s where
`start_offsets[i1...iN, j]` is the byte offset for the start of the
`j-th` token in `input[i1...iN]`.
</td>
</tr><tr>
<td>
`end_offsets`
</td>
<td>
is a `RaggedTensor` of int64s where
`end_offsets[i1...iN, j]` is the byte offset immediately after the
end of the `j-th` token in `input[i...iN]`.
</td>
</tr>
</table>
