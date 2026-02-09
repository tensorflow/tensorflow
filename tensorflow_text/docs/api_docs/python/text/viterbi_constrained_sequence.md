description: Performs greedy constrained sequence on a batch of examples.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.viterbi_constrained_sequence" />
<meta itemprop="path" content="Stable" />
</div>

# text.viterbi_constrained_sequence

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/viterbi_constrained_sequence_op.py">View
source</a>

Performs greedy constrained sequence on a batch of examples.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.viterbi_constrained_sequence(
    scores,
    sequence_length=None,
    allowed_transitions=None,
    transition_weights=None,
    use_log_space=False,
    use_start_and_end_states=True,
    name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

Constrains a set of predictions based on a set of legal transitions and/or a set
of transition weights, returning the legal sequence that maximizes the product
of the state scores and the transition weights according to the Viterbi
algorithm. If `use_log_space` is True, the Viterbi calculation will be performed
in log space (with sums); if it is False, the Viterbi calculation will be
performed in exp space (with normalized products).

This op also takes a parameter `use_start_and_end_states`, which when true will
add an implicit start and end state to each sequence. These implicit states
allow the user to specify additional weights and permitted transitions to start
and end a sequence (so, for instance, if you wanted to forbid your output from
ending in a certain set of states you could do so).

Inputs to this op can take one of three forms: a single TensorFlow tensor of
scores with no sequence lengths, a TensorFlow tensor of scores along with a
TensorFlow tensor of sequence lengths, or a RaggedTensor. If only the scores
tensor is passed, this op will assume that the sequence lengths are equal to the
size of the tensor (and so use all the data provided). If a scores tensor and
sequence_lengths tensor is provided, the op will only use the data in the scores
tensor as specified by the sequence_lengths tensor. Finally, if a RaggedTensor
is provided, the sequence_lengths will be ignored and the variable length
sequences in the RaggedTensor will be used.

```
>>> scores = np.array([[10.0, 12.0, 6.0, 4.0],
...                    [13.0, 12.0, 11.0, 10.0]], dtype=np.float32)
>>> sequence_length = np.array([2])
>>> transition_weights = np.array([[ .1,  .2,  .3,  .4],
...                                [ .5,  .6,  .7,  .8],
...                                [ .9,  .1, .15,  .2],
...                                [.25, .35, .45, .55]], dtype=np.float32)
>>> allowed_transitions = np.array([[True,  True,  True,  True],
...                                 [True,  True,  True,  True],
...                                 [True, False,  True, False],
...                                 [True,  True,  True,  True]])
>>> viterbi_constrained_sequence(
...      scores=scores,
...      sequence_length=sequence_length,
...      allowed_transitions=allowed_transitions,
...      transition_weights=transition_weights,
...      use_log_space=False,
...      use_start_and_end_states=False)
<tf.RaggedTensor [[1, 3]]>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`scores`<a id="scores"></a>
</td>
<td>
`<float32> [batch_size, num_steps, |num_states|]`
A tensor of scores, where `scores[b, t, s]` is the predicted score for
transitioning to state `s` at step `t` for batch `b`. The |num_states|
dimension must correspond to the num_states attribute for this op. This
input may be ragged; if it is ragged, the ragged tensor should have the
same structure [b, t, s] and only axis 1 should be ragged.
</td>
</tr><tr>
<td>
`sequence_length`<a id="sequence_length"></a>
</td>
<td>
`<{int32, int64}>[batch_size]`
A rank-1 tensor representing the length of the output sequence. If None,
and the 'scores' input is not ragged, sequence lengths will be assumed
to be the length of the score tensor.
</td>
</tr><tr>
<td>
`allowed_transitions`<a id="allowed_transitions"></a>
</td>
<td>
  if use_start_and_end_states is TRUE:
  `<bool>[num_states+1, num_states+1]`
if use_start_and_end_states is FALSE:
  `<bool>[num_states, num_states]`
A rank-2 tensor representing allowed transitions.
- allowed_transitions[i][j] is true if the transition from state i to
    state j is allowed for i and j in 0...(num_states).
- allowed_transitions[num_states][num_states] is ignored.
If use_start_and_end_states is TRUE:
  - allowed_transitions[num_states][j] is true if the sequence is allowed
      to start from state j.
  - allowed_transitions[i][num_states] is true if the sequence is allowed
      to end on state i.
Default - An empty tensor. This allows all sequence states to transition
  to all other sequence states.
</td>
</tr><tr>
<td>
`transition_weights`<a id="transition_weights"></a>
</td>
<td>
  if use_start_and_end_states is TRUE:
  `<float32>[num_states+1, num_states+1]`
if use_start_and_end_states is FALSE:
  `<float32>[num_states, num_states]`
A rank-2 tensor representing transition weights.
- transition_weights[i][j] is the coefficient that a candidate transition
    score will be multiplied by if that transition is from state i to
    state j.
- transition_weights[num_states][num_states] is ignored.
If use_start_and_end_states is TRUE:
  - transition_weights[num_states][j] is the coefficient that will be used
      if the transition starts with state j.
  - transition_weights[i][num_states] is the coefficient that will be used
      if the final state in the sequence is state i.
Default - An empty tensor. This assigns a wieght of 1.0 all transitions
</td>
</tr><tr>
<td>
`use_log_space`<a id="use_log_space"></a>
</td>
<td>
Whether to use log space for the calculation. If false,
calculations will be done in exp-space.
</td>
</tr><tr>
<td>
`use_start_and_end_states`<a id="use_start_and_end_states"></a>
</td>
<td>
If True, sequences will have an implicit start
and end state added.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
The name scope within which this op should be constructed.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An <int32>[batch_size, (num_steps)] ragged tensor containing the appropriate
sequence of transitions. If a sequence is impossible, the value of the
RaggedTensor for that and all following transitions in that sequence shall
be '-1'.
</td>
</tr>

</table>
