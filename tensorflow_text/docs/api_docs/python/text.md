description: Various tensorflow ops related to text-processing.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__version__"/>
</div>

# Module: text

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/__init__.py">View
source</a>

Various tensorflow ops related to text-processing.

## Modules

[`keras`](./text/keras.md) module: Tensorflow Text Layers for Keras API.

[`metrics`](./text/metrics.md) module: Tensorflow text-processing metrics.

[`tflite_registrar`](./text/tflite_registrar.md) module: tflite_registrar

## Classes

[`class BertTokenizer`](./text/BertTokenizer.md): Tokenizer used for BERT.

[`class ByteSplitter`](./text/ByteSplitter.md): Splits a string tensor into
bytes.

[`class Detokenizer`](./text/Detokenizer.md): Base class for detokenizer
implementations.

[`class FastBertNormalizer`](./text/FastBertNormalizer.md): Normalizes a tensor
of UTF-8 strings.

[`class FastBertTokenizer`](./text/FastBertTokenizer.md): Tokenizer used for
BERT, a faster version with TFLite support.

[`class FastSentencepieceTokenizer`](./text/FastSentencepieceTokenizer.md):
Sentencepiece tokenizer with tf.text interface.

[`class FastWordpieceTokenizer`](./text/FastWordpieceTokenizer.md): Tokenizes a
tensor of UTF-8 string tokens into subword pieces.

[`class FirstNItemSelector`](./text/FirstNItemSelector.md): An `ItemSelector`
that selects the first `n` items in the batch.

[`class HubModuleSplitter`](./text/HubModuleSplitter.md): Splitter that uses a
Hub module.

[`class HubModuleTokenizer`](./text/HubModuleTokenizer.md): Tokenizer that uses
a Hub module.

[`class LastNItemSelector`](./text/LastNItemSelector.md): An `ItemSelector` that
selects the last `n` items in the batch.

[`class MaskValuesChooser`](./text/MaskValuesChooser.md): Assigns values to the
items chosen for masking.

[`class PhraseTokenizer`](./text/PhraseTokenizer.md): Tokenizes a tensor of
UTF-8 string tokens into phrases.

[`class RandomItemSelector`](./text/RandomItemSelector.md): An `ItemSelector`
implementation that randomly selects items in a batch.

[`class Reduction`](./text/Reduction.md): Type of reduction to be done by the
n-gram op.

[`class RegexSplitter`](./text/RegexSplitter.md): `RegexSplitter` splits text on
the given regular expression.

[`class RoundRobinTrimmer`](./text/RoundRobinTrimmer.md): A `Trimmer` that
allocates a length budget to segments via round robin.

[`class SentencepieceTokenizer`](./text/SentencepieceTokenizer.md): Tokenizes a
tensor of UTF-8 strings.

[`class ShrinkLongestTrimmer`](./text/ShrinkLongestTrimmer.md): A `Trimmer` that
truncates the longest segment.

[`class SplitMergeFromLogitsTokenizer`](./text/SplitMergeFromLogitsTokenizer.md):
Tokenizes a tensor of UTF-8 string into words according to logits.

[`class SplitMergeTokenizer`](./text/SplitMergeTokenizer.md): Tokenizes a tensor
of UTF-8 string into words according to labels.

[`class Splitter`](./text/Splitter.md): An abstract base class for splitting
text.

[`class SplitterWithOffsets`](./text/SplitterWithOffsets.md): An abstract base
class for splitters that return offsets.

[`class StateBasedSentenceBreaker`](./text/StateBasedSentenceBreaker.md): A
`Splitter` that uses a state machine to determine sentence breaks.

[`class Tokenizer`](./text/Tokenizer.md): Base class for tokenizer
implementations.

[`class TokenizerWithOffsets`](./text/TokenizerWithOffsets.md): Base class for
tokenizer implementations that return offsets.

[`class Trimmer`](./text/Trimmer.md): Truncates a list of segments using a
pre-determined truncation strategy.

[`class UnicodeCharTokenizer`](./text/UnicodeCharTokenizer.md): Tokenizes a
tensor of UTF-8 strings on Unicode character boundaries.

[`class UnicodeScriptTokenizer`](./text/UnicodeScriptTokenizer.md): Tokenizes
UTF-8 by splitting when there is a change in Unicode script.

[`class WaterfallTrimmer`](./text/WaterfallTrimmer.md): A `Trimmer` that
allocates a length budget to segments in order.

[`class WhitespaceTokenizer`](./text/WhitespaceTokenizer.md): Tokenizes a tensor
of UTF-8 strings on whitespaces.

[`class WordShape`](./text/WordShape_cls.md): Values for the 'pattern' arg of the
wordshape op.

[`class WordpieceTokenizer`](./text/WordpieceTokenizer.md): Tokenizes a tensor
of UTF-8 string tokens into subword pieces.

## Functions

[`boise_tags_to_offsets(...)`](./text/boise_tags_to_offsets.md): Converts the
token offsets and BOISE tags into span offsets and span type.

[`build_fast_bert_normalizer_model(...)`](./text/build_fast_bert_normalizer_model.md):
build_fast_bert_normalizer_model(arg0: bool) -> bytes

[`build_fast_wordpiece_model(...)`](./text/build_fast_wordpiece_model.md):
build_fast_wordpiece_model(arg0: List[str], arg1: int, arg2: str, arg3: str,
arg4: bool, arg5: bool) -> bytes

[`case_fold_utf8(...)`](./text/case_fold_utf8.md): Applies case folding to every
UTF-8 string in the input.

[`coerce_to_structurally_valid_utf8(...)`](./text/coerce_to_structurally_valid_utf8.md): Coerce UTF-8 input strings to structurally valid UTF-8.

[`combine_segments(...)`](./text/combine_segments.md): Combine one or more input
segments for a model's input sequence.

[`concatenate_segments(...)`](./text/concatenate_segments.md): Concatenate input
segments for a model's input sequence.

[`find_source_offsets(...)`](./text/find_source_offsets.md): Maps the input
post-normalized string offsets to pre-normalized offsets.

[`gather_with_default(...)`](./text/gather_with_default.md): Gather slices with `indices=-1` mapped to `default`.

[`greedy_constrained_sequence(...)`](./text/greedy_constrained_sequence.md): Performs greedy constrained sequence on a batch of examples.

[`mask_language_model(...)`](./text/mask_language_model.md): Applies dynamic
language model masking.

[`max_spanning_tree(...)`](./text/max_spanning_tree.md): Finds the maximum
directed spanning tree of a digraph.

[`max_spanning_tree_gradient(...)`](./text/max_spanning_tree_gradient.md):
Returns a subgradient of the MaximumSpanningTree op.

[`ngrams(...)`](./text/ngrams.md): Create a tensor of n-grams based on the input data `data`.

[`normalize_utf8(...)`](./text/normalize_utf8.md): Normalizes each UTF-8 string
in the input tensor using the specified rule.

[`normalize_utf8_with_offsets_map(...)`](./text/normalize_utf8_with_offsets_map.md):
Normalizes each UTF-8 string in the input tensor using the specified rule.

[`offsets_to_boise_tags(...)`](./text/offsets_to_boise_tags.md): Converts the
given tokens and spans in offsets format into BOISE tags.

[`pad_along_dimension(...)`](./text/pad_along_dimension.md): Add padding to the beginning and end of data in a specific dimension.

[`pad_model_inputs(...)`](./text/pad_model_inputs.md): Pad model input and
generate corresponding input masks.

[`regex_split(...)`](./text/regex_split.md): Split `input` by delimiters that
match a regex pattern.

[`regex_split_with_offsets(...)`](./text/regex_split_with_offsets.md): Split
`input` by delimiters that match a regex pattern; returns offsets.

[`sentence_fragments(...)`](./text/sentence_fragments.md): Find the sentence
fragments in a given text. (deprecated)

[`sliding_window(...)`](./text/sliding_window.md): Builds a sliding window for `data` with a specified width.

[`span_alignment(...)`](./text/span_alignment.md): Return an alignment from a set of source spans to a set of target spans.

[`span_overlaps(...)`](./text/span_overlaps.md): Returns a boolean tensor indicating which source and target spans overlap.

[`utf8_binarize(...)`](./text/utf8_binarize.md): Decode UTF8 tokens into code
points and return their bits.

[`viterbi_constrained_sequence(...)`](./text/viterbi_constrained_sequence.md): Performs greedy constrained sequence on a batch of examples.

[`wordshape(...)`](./text/wordshape.md): Determine wordshape features for each input string.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
**version**<a id="__version__"></a>
</td>
<td>
`'2.12.0'`
</td>
</tr>
</table>
