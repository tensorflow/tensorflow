description: Coerce UTF-8 input strings to structurally valid UTF-8.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.coerce_to_structurally_valid_utf8" />
<meta itemprop="path" content="Stable" />
</div>

# text.coerce_to_structurally_valid_utf8

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/string_ops.py">View
source</a>

Coerce UTF-8 input strings to structurally valid UTF-8.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.coerce_to_structurally_valid_utf8(
    input, replacement_char=_unichr(65533), name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

Any bytes which cause the input string to be invalid UTF-8 are substituted with
the provided replacement character codepoint (default 65533). If you plan on
overriding the default, use a single byte replacement character codepoint to
preserve alignment to the source input string.

In this example, the character \xDEB2 is an invalid UTF-8 bit sequence; the call
to `coerce_to_structurally_valid_utf8` replaces it with \xef\xbf\xbd, which is
the default replacement character encoding. ```

> > > input_data = ["A", b"\xDEB2", "C"]
> > > coerce_to_structurally_valid_utf8(input_data)
> > > <tf.Tensor: shape=(3,), dtype=string, numpy=array([b'A', b'\xef\xbf\xbdB2', b'C'], dtype=object)>
> > > ```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`<a id="input"></a>
</td>
<td>
UTF-8 string tensor to coerce to valid UTF-8.
</td>
</tr><tr>
<td>
`replacement_char`<a id="replacement_char"></a>
</td>
<td>
The replacement character to be used in place of any
invalid byte in the input. Any valid Unicode character may be used. The
default value is the default Unicode replacement character which is
0xFFFD (or U+65533). Note that passing a replacement character
expressible in 1 byte, such as ' ' or '?', will preserve string
alignment to the source since individual invalid bytes will be replaced
with a 1-byte replacement. (optional)
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor of type string with the same shape as the input.
</td>
</tr>

</table>
