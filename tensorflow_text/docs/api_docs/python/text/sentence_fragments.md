description: Find the sentence fragments in a given text. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.sentence_fragments" />
<meta itemprop="path" content="Stable" />
</div>

# text.sentence_fragments

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentence_breaking_ops.py">View
source</a>

Find the sentence fragments in a given text. (deprecated)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.sentence_fragments(
    token_word,
    token_starts,
    token_ends,
    token_properties,
    input_encoding=&#x27;UTF-8&#x27;,
    errors=&#x27;replace&#x27;,
    replacement_char=65533,
    replace_control_characters=False
)
</code></pre>

<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating: Deprecated, use 'StateBasedSentenceBreaker' instead.

A sentence fragment is a potential next sentence determined using
deterministic heuristics based on punctuation, capitalization, and similar
text attributes.

NOTE: This op is deprecated. Use `StateBasedSentenceBreaker` instead.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr> <td> `token_word`<a id="token_word"></a> </td> <td> A Tensor (w/ rank=2) or
a RaggedTensor (w/ ragged_rank=1) containing the token strings. </td> </tr><tr>
<td> `token_starts`<a id="token_starts"></a> </td> <td> A Tensor (w/ rank=2) or
a RaggedTensor (w/ ragged_rank=1) containing offsets where the token starts.
</td> </tr><tr> <td> `token_ends`<a id="token_ends"></a> </td> <td> A Tensor (w/
rank=2) or a RaggedTensor (w/ ragged_rank=1) containing offsets where the token
ends. </td> </tr><tr> <td> `token_properties`<a id="token_properties"></a> </td>
<td> A Tensor (w/ rank=2) or a RaggedTensor (w/ ragged_rank=1) containing a
bitmask.

The values of the bitmask are:

*   0x01 (ILL_FORMED) - Text is ill-formed: typically applies to all tokens of a
    paragraph that is too short or lacks terminal punctuation.
*   0x02 (HEADING)
*   0x04 (BOLD)
*   0x10 (UNDERLINED)
*   0x20 (LIST)
*   0x40 (TITLE)
*   0x80 (EMOTICON)
*   0x100 (ACRONYM) - Token was identified as an acronym. Period-, hyphen-, and
    space-separated acronyms: "U.S.", "U-S", and "U S".
*   0x200 (HYPERLINK) - Indicates that the token (or part of the token) is
    covered by at least one hyperlink. </td> </tr><tr> <td>
    `input_encoding`<a id="input_encoding"></a> </td> <td> String name for the
    unicode encoding that should be used to decode each string. </td> </tr><tr>
    <td> `errors`<a id="errors"></a> </td> <td> Specifies the response when an
    input string can't be converted using the indicated encoding. One of:

*   `'strict'`: Raise an exception for any illegal substrings.

*   `'replace'`: Replace illegal substrings with `replacement_char`.

*   `'ignore'`: Skip illegal substrings.
    </td>
    </tr><tr>
    <td>
    `replacement_char`<a id="replacement_char"></a>
    </td>
    <td>
    The replacement codepoint to be used in place of invalid
    substrings in `input` when `errors='replace'`; and in place of C0 control
    characters in `input` when `replace_control_characters=True`.
    </td>
    </tr><tr>
    <td>
    `replace_control_characters`<a id="replace_control_characters"></a>
    </td>
    <td>
    Whether to replace the C0 control characters
    `(U+0000 - U+001F)` with the `replacement_char`.
    </td>
    </tr>
    </table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A RaggedTensor of `fragment_start`, `fragment_end`, `fragment_properties`
and `terminal_punc_token`.

`fragment_properties` is an int32 bitmask whose values may contain:

*   1 = fragment ends with terminal punctuation
*   2 = fragment ends with multiple terminal punctuations (e.g. "She said
    what?!")
*   3 = Has close parenthesis (e.g. "Mushrooms (they're fungi).")
*   4 = Has sentential close parenthesis (e.g. "(Mushrooms are fungi!)")

    `terminal_punc_token` is a RaggedTensor containing the index of terminal
    punctuation token immediately following the last word in the fragment -- or
    index of the last word itself, if it's an acronym (since acronyms include
    the terminal punctuation). index of the terminal punctuation token. </td>
    </tr>

</table>
