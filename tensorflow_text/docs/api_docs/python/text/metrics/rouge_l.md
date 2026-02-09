description: Computes LCS-based similarity score between the hypotheses and
references.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.metrics.rouge_l" />
<meta itemprop="path" content="Stable" />
</div>

# text.metrics.rouge_l

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/metrics/text_similarity_metric_ops.py">View
source</a>

Computes LCS-based similarity score between the hypotheses and references.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.metrics.rouge_l(
    hypotheses, references, alpha=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

The Rouge-L metric is a score from 0 to 1 indicating how similar two sequences
are, based on the length of the longest common subsequence (LCS). In particular,
Rouge-L is the weighted harmonic mean (or f-measure) combining the LCS precision
(the percentage of the hypothesis sequence covered by the LCS) and the LCS
recall (the percentage of the reference sequence covered by the LCS).

Source: https://www.microsoft.com/en-us/research/publication/
rouge-a-package-for-automatic-evaluation-of-summaries/

This method returns the F-measure, Precision, and Recall for each (hypothesis,
reference) pair.

Alpha is used as a weight for the harmonic mean of precision and recall. A value
of 0 means recall is more important and 1 means precision is more important.
Leaving alpha unset implies alpha=.5, which is the default in the official
ROUGE-1.5.5.pl script. Setting alpha to a negative number triggers a
compatibility mode with the tensor2tensor implementation of ROUGE-L.

```
>>> hypotheses = tf.ragged.constant([["a","b"]])
>>> references = tf.ragged.constant([["b"]])
>>> f, p, r = rouge_l(hypotheses, references, alpha=1)
>>> print("f: %s, p: %s, r: %s" % (f, p, r))
f: tf.Tensor([0.5], shape=(1,), dtype=float32),
p: tf.Tensor([0.5], shape=(1,), dtype=float32),
r: tf.Tensor([1.], shape=(1,), dtype=float32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`hypotheses`<a id="hypotheses"></a>
</td>
<td>
A RaggedTensor with shape [N, (hyp_sentence_len)] and integer or
string values.
</td>
</tr><tr>
<td>
`references`<a id="references"></a>
</td>
<td>
A RaggedTensor with shape [N, (ref_sentence_len)] and integer or
string values.
</td>
</tr><tr>
<td>
`alpha`<a id="alpha"></a>
</td>
<td>
optional float parameter for weighting
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
an (f_measure, p_measure, r_measure) tuple, where each element is a
vector of floats with shape [N]. The i-th float in each vector contains
the similarity measure of hypotheses[i] and references[i].
</td>
</tr>

</table>
