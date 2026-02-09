description: Split input by delimiters that match a regex pattern.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.regex_split" />
<meta itemprop="path" content="Stable" />
</div>

# text.regex_split

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/regex_split_ops.py">View
source</a>

Split `input` by delimiters that match a regex pattern.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.regex_split(
    input,
    delim_regex_pattern,
    keep_delim_regex_pattern=&#x27;&#x27;,
    name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

`regex_split` will split `input` using delimiters that match a
regex pattern in `delim_regex_pattern`. Here is an example:

```
>>> text_input=["hello there"]
>>> # split by whitespace
>>> regex_split(input=text_input,
...             delim_regex_pattern="\s")
<tf.RaggedTensor [[b'hello', b'there']]>
```

By default, delimiters are not included in the split string results.
Delimiters may be included by specifying a regex pattern
`keep_delim_regex_pattern`. For example:

```
>>> text_input=["hello there"]
>>> # split by whitespace
>>> regex_split(input=text_input,
...             delim_regex_pattern="\s",
...             keep_delim_regex_pattern="\s")
<tf.RaggedTensor [[b'hello', b' ', b'there']]>
```

If there are multiple delimiters in a row, there are no empty splits emitted.
For example:

```
>>> text_input=["hello  there"]  #  Note the two spaces between the words.
>>> # split by whitespace
>>> regex_split(input=text_input,
...             delim_regex_pattern="\s")
<tf.RaggedTensor [[b'hello', b'there']]>
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
A RaggedTensors containing of type string containing the split string
pieces.
</td>
</tr>

</table>
