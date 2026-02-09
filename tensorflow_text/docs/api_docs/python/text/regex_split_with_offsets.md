description: Split input by delimiters that match a regex pattern; returns
offsets.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.regex_split_with_offsets" />
<meta itemprop="path" content="Stable" />
</div>

# text.regex_split_with_offsets

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/regex_split_ops.py">View
source</a>

Split `input` by delimiters that match a regex pattern; returns offsets.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.regex_split_with_offsets(
    input,
    delim_regex_pattern,
    keep_delim_regex_pattern=&#x27;&#x27;,
    name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

`regex_split_with_offsets` will split `input` using delimiters that match a
regex pattern in `delim_regex_pattern`. It will return three tensors: one
containing the split substrings ('result' in the examples below), one containing
the offsets of the starts of each substring ('begin' in the examples below), and
one containing the offsets of the ends of each substring ('end' in the examples
below).

#### Here is an example:

```
>>> text_input=["hello there"]
>>> # split by whitespace
>>> result, begin, end = regex_split_with_offsets(input=text_input,
...                                               delim_regex_pattern="\s")
>>> print("result: %s\nbegin: %s\nend: %s" % (result, begin, end))
result: <tf.RaggedTensor [[b'hello', b'there']]>
begin: <tf.RaggedTensor [[0, 6]]>
end: <tf.RaggedTensor [[5, 11]]>
```

By default, delimiters are not included in the split string results.
Delimiters may be included by specifying a regex pattern
`keep_delim_regex_pattern`. For example:

```
>>> text_input=["hello there"]
>>> # split by whitespace
>>> result, begin, end = regex_split_with_offsets(input=text_input,
...                                             delim_regex_pattern="\s",
...                                             keep_delim_regex_pattern="\s")
>>> print("result: %s\nbegin: %s\nend: %s" % (result, begin, end))
result: <tf.RaggedTensor [[b'hello', b' ', b'there']]>
begin: <tf.RaggedTensor [[0, 5, 6]]>
end: <tf.RaggedTensor [[5, 6, 11]]>
```

If there are multiple delimiters in a row, there are no empty splits emitted.
For example:

```
>>> text_input=["hello  there"]  #  Note the two spaces between the words.
>>> # split by whitespace
>>> result, begin, end = regex_split_with_offsets(input=text_input,
...                                               delim_regex_pattern="\s")
>>> print("result: %s\nbegin: %s\nend: %s" % (result, begin, end))
result: <tf.RaggedTensor [[b'hello', b'there']]>
begin: <tf.RaggedTensor [[0, 7]]>
end: <tf.RaggedTensor [[5, 12]]>
```

See https://github.com/google/re2/wiki/Syntax for the full list of supported
expressions.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`<a id="input"></a>
</td>
<td>
A Tensor or RaggedTensor of string input.
</td>
</tr><tr>
<td>
`delim_regex_pattern`<a id="delim_regex_pattern"></a>
</td>
<td>
A string containing the regex pattern of a delimiter.
</td>
</tr><tr>
<td>
`keep_delim_regex_pattern`<a id="keep_delim_regex_pattern"></a>
</td>
<td>
(optional) Regex pattern of delimiters that should
be kept in the result.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
(optional) Name of the op.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple of RaggedTensors containing:
  (split_results, begin_offsets, end_offsets)
where tokens is of type string, begin_offsets and end_offsets are of type
int64.
</td>
</tr>

</table>
