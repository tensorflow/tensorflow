description: Unicode script tokenization layer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.keras.layers.UnicodeScriptTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# text.keras.layers.UnicodeScriptTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/keras/layers/tokenization_layers.py">View
source</a>

Unicode script tokenization layer.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.keras.layers.UnicodeScriptTokenizer(
    keep_whitespace=False, pad_value=None, squeeze_token_dim=True, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

Splits a string when successive tokens change their Unicode script or change
being whitespace or not. By not keeping the whitespace tokens, this allows you
to split on whitespace, and also to split out tokens from different scripts (so,
for instance, a string with both Latin and Japanese characters would be split at
the boundary of the Latin and Japanese characters in addition to any whitespace
boundaries).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`keep_whitespace`<a id="keep_whitespace"></a>
</td>
<td>
A boolean that specifices whether to emit whitespace
tokens (default `False`).
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
`squeeze_token_dim`<a id="squeeze_token_dim"></a>
</td>
<td>
Whether to squeeze the dimension added by tokenization.
When this arg is set to False, the output will have an additional inner
dimension added, containing the tokens in each string; when this arg is
True, the layer will attempt to squeeze that dimension out. If you are
passing one string per batch, you probably want to keep this as True; if
you are passing more than one string per batch or are using this layer in
a context like the Keras `TextVectorization` layer which expects a
tf.strings.split()-stype output, this should be False. Defaults to True.
</td>
</tr>
</table>
