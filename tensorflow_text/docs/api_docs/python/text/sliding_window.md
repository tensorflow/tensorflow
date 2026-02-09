description: Builds a sliding window for data with a specified width.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.sliding_window" />
<meta itemprop="path" content="Stable" />
</div>

# text.sliding_window

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sliding_window_op.py">View
source</a>

Builds a sliding window for `data` with a specified width.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.sliding_window(
    data, width, axis=-1, name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

Returns a tensor constructed from `data`, where each element in
dimension `axis` is a slice of `data` starting at the corresponding
position, with the given width and step size.  I.e.:

* `result.shape.ndims = data.shape.ndims + 1`
* `result[i1..iaxis, a] = data[i1..iaxis, a:a+width]`
  (where `0 <= a < data[i1...iaxis].shape[0] - (width - 1)`).

Note that each result row (along dimension `axis`) has `width - 1` fewer items
than the corresponding `data` row.  If a `data` row has fewer than `width`
items, then the corresponding `result` row will be empty.  If you wish for
the `result` rows to be the same size as the `data` rows, you can use
`pad_along_dimension` to add `width - 1` padding elements before calling
this op.

#### Examples:

Sliding window (width=3) across a sequence of tokens:

```
>>> # input: <string>[sequence_length]
>>> input = tf.constant(["one", "two", "three", "four", "five", "six"])
>>> # output: <string>[sequence_length-2, 3]
>>> sliding_window(data=input, width=3, axis=0)
<tf.Tensor: shape=(4, 3), dtype=string, numpy=
    array([[b'one', b'two', b'three'],
           [b'two', b'three', b'four'],
           [b'three', b'four', b'five'],
           [b'four', b'five', b'six']], dtype=object)>
```

Sliding window (width=2) across the inner dimension of a ragged matrix
containing a batch of token sequences:

```
>>> # input: <string>[num_sentences, (num_words)]
>>> input = tf.ragged.constant(
...     [['Up', 'high', 'in', 'the', 'air'],
...      ['Down', 'under', 'water'],
...      ['Away', 'to', 'outer', 'space']])
>>> # output: <string>[num_sentences, (num_word-1), 2]
>>> sliding_window(input, width=2, axis=-1)
<tf.RaggedTensor [[[b'Up', b'high'], [b'high', b'in'], [b'in', b'the'],
                   [b'the', b'air']], [[b'Down', b'under'],
                   [b'under', b'water']],
                  [[b'Away', b'to'], [b'to', b'outer'],
                   [b'outer', b'space']]]>
```

Sliding window across the second dimension of a 3-D tensor containing batches of
sequences of embedding vectors:

```
>>> # input: <int32>[num_sequences, sequence_length, embedding_size]
>>> input = tf.constant([
...     [[1, 1, 1], [2, 2, 1], [3, 3, 1], [4, 4, 1], [5, 5, 1]],
...     [[1, 1, 2], [2, 2, 2], [3, 3, 2], [4, 4, 2], [5, 5, 2]]])
>>> # output: <int32>[num_sequences, sequence_length-1, 2, embedding_size]
>>> sliding_window(data=input, width=2, axis=1)
<tf.Tensor: shape=(2, 4, 2, 3), dtype=int32, numpy=
    array([[[[1, 1, 1],
             [2, 2, 1]],
            [[2, 2, 1],
             [3, 3, 1]],
            [[3, 3, 1],
             [4, 4, 1]],
            [[4, 4, 1],
             [5, 5, 1]]],
           [[[1, 1, 2],
             [2, 2, 2]],
            [[2, 2, 2],
             [3, 3, 2]],
            [[3, 3, 2],
             [4, 4, 2]],
            [[4, 4, 2],
             [5, 5, 2]]]], dtype=int32)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data`<a id="data"></a>
</td>
<td>
`<dtype> [O1...ON, A, I1...IM]`
A potentially ragged K-dimensional tensor with outer dimensions of size
`O1...ON`; axis dimension of size `A`; and inner dimensions of size
`I1...IM`.  I.e. `K = N + 1 + M`, where `N>=0` and `M>=0`.
</td>
</tr><tr>
<td>
`width`<a id="width"></a>
</td>
<td>
An integer constant specifying the width of the window. Must be
greater than zero.
</td>
</tr><tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
An integer constant specifying the axis along which sliding window
is computed. Negative axis values from `-K` to `-1` are supported.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
The name for this op (optional).
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `K+1` dimensional tensor with the same dtype as `data`, where:

*   `result[i1..iaxis, a]` = `data[i1..iaxis, a:a+width]`
*   `result.shape[:axis]` = `data.shape[:axis]`
*   `result.shape[axis]` = `data.shape[axis] - (width - 1)`
*   `result.shape[axis + 1]` = `width`
*   `result.shape[axis + 2:]` = `data.shape[axis + 1:]` </td> </tr>

</table>
