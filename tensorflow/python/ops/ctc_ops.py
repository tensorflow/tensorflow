# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CTC (Connectionist Temporal Classification) Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.ops import inplace_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.nn_grad import _BroadcastMul
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


# pylint: disable=protected-access, invalid-name
@tf_export(v1=["nn.ctc_loss"])
def ctc_loss(labels, inputs, sequence_length,
             preprocess_collapse_repeated=False,
             ctc_merge_repeated=True,
             ignore_longer_outputs_than_inputs=False, time_major=True):
  """Computes the CTC (Connectionist Temporal Classification) Loss.

  This op implements the CTC loss as presented in the article:

  [A. Graves, S. Fernandez, F. Gomez, J. Schmidhuber.
  Connectionist Temporal Classification: Labeling Unsegmented Sequence Data
  with Recurrent Neural Networks. ICML 2006, Pittsburgh, USA,
  pp. 369-376.](http://www.cs.toronto.edu/~graves/icml_2006.pdf)

  Input requirements:

  ```
  sequence_length(b) <= time for all b

  max(labels.indices(labels.indices[:, 1] == b, 2))
    <= sequence_length(b) for all b.
  ```

  Notes:

  This class performs the softmax operation for you, so inputs should
  be e.g. linear projections of outputs by an LSTM.

  The `inputs` Tensor's innermost dimension size, `num_classes`, represents
  `num_labels + 1` classes, where num_labels is the number of true labels, and
  the largest value `(num_classes - 1)` is reserved for the blank label.

  For example, for a vocabulary containing 3 labels `[a, b, c]`,
  `num_classes = 4` and the labels indexing is `{a: 0, b: 1, c: 2, blank: 3}`.

  Regarding the arguments `preprocess_collapse_repeated` and
  `ctc_merge_repeated`:

  If `preprocess_collapse_repeated` is True, then a preprocessing step runs
  before loss calculation, wherein repeated labels passed to the loss
  are merged into single labels.  This is useful if the training labels come
  from, e.g., forced alignments and therefore have unnecessary repetitions.

  If `ctc_merge_repeated` is set False, then deep within the CTC calculation,
  repeated non-blank labels will not be merged and are interpreted
  as individual labels.  This is a simplified (non-standard) version of CTC.

  Here is a table of the (roughly) expected first order behavior:

  * `preprocess_collapse_repeated=False`, `ctc_merge_repeated=True`

    Classical CTC behavior: Outputs true repeated classes with blanks in
    between, and can also output repeated classes with no blanks in
    between that need to be collapsed by the decoder.

  * `preprocess_collapse_repeated=True`, `ctc_merge_repeated=False`

    Never learns to output repeated classes, as they are collapsed
    in the input labels before training.

  * `preprocess_collapse_repeated=False`, `ctc_merge_repeated=False`

    Outputs repeated classes with blanks in between, but generally does not
    require the decoder to collapse/merge repeated classes.

  * `preprocess_collapse_repeated=True`, `ctc_merge_repeated=True`

    Untested.  Very likely will not learn to output repeated classes.

  The `ignore_longer_outputs_than_inputs` option allows to specify the behavior
  of the CTCLoss when dealing with sequences that have longer outputs than
  inputs. If true, the CTCLoss will simply return zero gradient for those
  items, otherwise an InvalidArgument error is returned, stopping training.

  Args:
    labels: An `int32` `SparseTensor`.
      `labels.indices[i, :] == [b, t]` means `labels.values[i]` stores
      the id for (batch b, time t).
      `labels.values[i]` must take on values in `[0, num_labels)`.
      See `core/ops/ctc_ops.cc` for more details.
    inputs: 3-D `float` `Tensor`.
      If time_major == False, this will be a `Tensor` shaped:
        `[batch_size, max_time, num_classes]`.
      If time_major == True (default), this will be a `Tensor` shaped:
        `[max_time, batch_size, num_classes]`.
      The logits.
    sequence_length: 1-D `int32` vector, size `[batch_size]`.
      The sequence lengths.
    preprocess_collapse_repeated: Boolean.  Default: False.
      If True, repeated labels are collapsed prior to the CTC calculation.
    ctc_merge_repeated: Boolean.  Default: True.
    ignore_longer_outputs_than_inputs: Boolean. Default: False.
      If True, sequences with longer outputs than inputs will be ignored.
    time_major: The shape format of the `inputs` Tensors.
      If True, these `Tensors` must be shaped `[max_time, batch_size,
      num_classes]`.
      If False, these `Tensors` must be shaped `[batch_size, max_time,
      num_classes]`.
      Using `time_major = True` (default) is a bit more efficient because it
      avoids transposes at the beginning of the ctc_loss calculation.  However,
      most TensorFlow data is batch-major, so by this function also accepts
      inputs in batch-major form.

  Returns:
    A 1-D `float` `Tensor`, size `[batch]`, containing the negative log
      probabilities.

  Raises:
    TypeError: if labels is not a `SparseTensor`.
  """
  # The second, third, etc output tensors contain the gradients.  We use it in
  # _CTCLossGrad() below.
  if not isinstance(labels, sparse_tensor.SparseTensor):
    raise TypeError("Expected labels (first argument) to be a SparseTensor")

  # For internal calculations, we transpose to [time, batch, num_classes]
  if not time_major:
    inputs = array_ops.transpose(inputs, [1, 0, 2])  # (B,T,N) => (T,B,N)

  loss, _ = gen_ctc_ops.ctc_loss(
      inputs,
      labels.indices,
      labels.values,
      sequence_length,
      preprocess_collapse_repeated=preprocess_collapse_repeated,
      ctc_merge_repeated=ctc_merge_repeated,
      ignore_longer_outputs_than_inputs=ignore_longer_outputs_than_inputs)

  return loss


# pylint: disable=unused-argument
@ops.RegisterGradient("CTCLoss")
def _CTCLossGrad(op, grad_loss, _):
  """The derivative provided by CTC Loss.

  Args:
     op: the CTCLoss op.
     grad_loss: The backprop for cost.

  Returns:
     The CTC Loss gradient.
  """
  # Outputs are: loss, grad
  #
  # Currently there is no way to take the second derivative of this op
  # due to the fused implementation's interaction with tf.gradients(),
  # so we make sure we prevent silently incorrect results by raising
  # an error if the second derivative is requested via prevent_gradient.
  grad_without_gradient = array_ops.prevent_gradient(
      op.outputs[1], message="Currently there is no way to take the second "
      " derivative of ctc_loss due to the fused implementation's interaction "
      " with tf.gradients()")
  # Return gradient for inputs and None for
  # labels_indices, labels_values and sequence_length
  return [_BroadcastMul(grad_loss, grad_without_gradient), None, None, None]


@tf_export("nn.ctc_greedy_decoder")
def ctc_greedy_decoder(inputs, sequence_length, merge_repeated=True):
  """Performs greedy decoding on the logits given in input (best path).

  Note: Regardless of the value of merge_repeated, if the maximum index of a
  given time and batch corresponds to the blank index `(num_classes - 1)`, no
  new element is emitted.

  If `merge_repeated` is `True`, merge repeated classes in output.
  This means that if consecutive logits' maximum indices are the same,
  only the first of these is emitted.  The sequence `A B B * B * B` (where '*'
  is the blank label) becomes

    * `A B B B` if `merge_repeated=True`.
    * `A B B B B` if `merge_repeated=False`.

  Args:
    inputs: 3-D `float` `Tensor` sized
      `[max_time, batch_size, num_classes]`.  The logits.
    sequence_length: 1-D `int32` vector containing sequence lengths,
      having size `[batch_size]`.
    merge_repeated: Boolean.  Default: True.

  Returns:
    A tuple `(decoded, neg_sum_logits)` where

    decoded: A single-element list. `decoded[0]`
      is an `SparseTensor` containing the decoded outputs s.t.:

      `decoded.indices`: Indices matrix `(total_decoded_outputs, 2)`.
        The rows store: `[batch, time]`.

      `decoded.values`: Values vector, size `(total_decoded_outputs)`.
        The vector stores the decoded classes.

      `decoded.dense_shape`: Shape vector, size `(2)`.
        The shape values are: `[batch_size, max_decoded_length]`

    neg_sum_logits: A `float` matrix `(batch_size x 1)` containing, for the
        sequence found, the negative of the sum of the greatest logit at each
        timeframe.
  """
  outputs = gen_ctc_ops.ctc_greedy_decoder(
      inputs, sequence_length, merge_repeated=merge_repeated)
  (decoded_ix, decoded_val, decoded_shape, log_probabilities) = outputs
  return ([sparse_tensor.SparseTensor(decoded_ix, decoded_val, decoded_shape)],
          log_probabilities)


@tf_export(v1=["nn.ctc_beam_search_decoder"])
def ctc_beam_search_decoder(inputs, sequence_length, beam_width=100,
                            top_paths=1, merge_repeated=True):
  """Performs beam search decoding on the logits given in input.

  **Note** The `ctc_greedy_decoder` is a special case of the
  `ctc_beam_search_decoder` with `top_paths=1` and `beam_width=1` (but
  that decoder is faster for this special case).

  If `merge_repeated` is `True`, merge repeated classes in the output beams.
  This means that if consecutive entries in a beam are the same,
  only the first of these is emitted.  That is, when the sequence is
  `A B B * B * B` (where '*' is the blank label), the return value is:

    * `A B` if `merge_repeated = True`.
    * `A B B B` if `merge_repeated = False`.

  Args:
    inputs: 3-D `float` `Tensor`, size
      `[max_time x batch_size x num_classes]`.  The logits.
    sequence_length: 1-D `int32` vector containing sequence lengths,
      having size `[batch_size]`.
    beam_width: An int scalar >= 0 (beam search beam width).
    top_paths: An int scalar >= 0, <= beam_width (controls output size).
    merge_repeated: Boolean.  Default: True.

  Returns:
    A tuple `(decoded, log_probabilities)` where

    decoded: A list of length top_paths, where `decoded[j]`
      is a `SparseTensor` containing the decoded outputs:

      `decoded[j].indices`: Indices matrix `(total_decoded_outputs[j] x 2)`
        The rows store: [batch, time].

      `decoded[j].values`: Values vector, size `(total_decoded_outputs[j])`.
        The vector stores the decoded classes for beam j.

      `decoded[j].dense_shape`: Shape vector, size `(2)`.
        The shape values are: `[batch_size, max_decoded_length[j]]`.

    log_probability: A `float` matrix `(batch_size x top_paths)` containing
        sequence log-probabilities.
  """

  decoded_ixs, decoded_vals, decoded_shapes, log_probabilities = (
      gen_ctc_ops.ctc_beam_search_decoder(
          inputs, sequence_length, beam_width=beam_width, top_paths=top_paths,
          merge_repeated=merge_repeated))

  return (
      [sparse_tensor.SparseTensor(ix, val, shape) for (ix, val, shape)
       in zip(decoded_ixs, decoded_vals, decoded_shapes)],
      log_probabilities)


@tf_export("nn.ctc_beam_search_decoder", v1=["nn.ctc_beam_search_decoder_v2"])
def ctc_beam_search_decoder_v2(inputs, sequence_length, beam_width=100,
                               top_paths=1):
  """Performs beam search decoding on the logits given in input.

  **Note** The `ctc_greedy_decoder` is a special case of the
  `ctc_beam_search_decoder` with `top_paths=1` and `beam_width=1` (but
  that decoder is faster for this special case).

  Args:
    inputs: 3-D `float` `Tensor`, size
      `[max_time, batch_size, num_classes]`.  The logits.
    sequence_length: 1-D `int32` vector containing sequence lengths,
      having size `[batch_size]`.
    beam_width: An int scalar >= 0 (beam search beam width).
    top_paths: An int scalar >= 0, <= beam_width (controls output size).

  Returns:
    A tuple `(decoded, log_probabilities)` where

    decoded: A list of length top_paths, where `decoded[j]`
      is a `SparseTensor` containing the decoded outputs:

      `decoded[j].indices`: Indices matrix `[total_decoded_outputs[j], 2]`;
        The rows store: `[batch, time]`.

      `decoded[j].values`: Values vector, size `[total_decoded_outputs[j]]`.
        The vector stores the decoded classes for beam `j`.

      `decoded[j].dense_shape`: Shape vector, size `(2)`.
        The shape values are: `[batch_size, max_decoded_length[j]]`.

    log_probability: A `float` matrix `[batch_size, top_paths]` containing
        sequence log-probabilities.
  """

  # Note, merge_repeated is an invalid optimization that is removed from the
  # public API: it returns low probability paths.
  return ctc_beam_search_decoder(inputs, sequence_length=sequence_length,
                                 beam_width=beam_width, top_paths=top_paths,
                                 merge_repeated=False)


ops.NotDifferentiable("CTCGreedyDecoder")
ops.NotDifferentiable("CTCBeamSearchDecoder")


def _ctc_state_trans(label_seq):
  """Compute CTC alignment model transition matrix.

  Args:
    label_seq: tensor of shape [batch_size, max_seq_length]

  Returns:
    tensor of shape [batch_size, states, states] with a state transition matrix
    computed for each sequence of the batch.
  """

  with ops.name_scope("ctc_state_trans"):
    label_seq = ops.convert_to_tensor(label_seq, name="label_seq")
    batch_size = _get_dim(label_seq, 0)
    num_labels = _get_dim(label_seq, 1)

    num_label_states = num_labels + 1
    num_states = 2 * num_label_states

    label_states = math_ops.range(num_label_states)
    blank_states = label_states + num_label_states

    # Start state to first label.
    start_to_label = [[1, 0]]

    # Blank to label transitions.
    blank_to_label = array_ops.stack([label_states[1:], blank_states[:-1]], 1)

    # Label to blank transitions.
    label_to_blank = array_ops.stack([blank_states, label_states], 1)

    # Scatter transitions that don't depend on sequence.
    indices = array_ops.concat(
        [start_to_label, blank_to_label, label_to_blank], 0)
    values = array_ops.ones([_get_dim(indices, 0)])
    trans = array_ops.scatter_nd(
        indices, values, shape=[num_states, num_states])
    trans += linalg_ops.eye(num_states)  # Self-loops.

    # Label to label transitions. Disallow transitions between repeated labels
    # with no blank state in between.
    batch_idx = array_ops.zeros_like(label_states[2:])
    indices = array_ops.stack(
        [batch_idx, label_states[2:], label_states[1:-1]], 1)
    indices = array_ops.tile(
        array_ops.expand_dims(indices, 0), [batch_size, 1, 1])
    batch_idx = array_ops.expand_dims(math_ops.range(batch_size), 1) * [1, 0, 0]
    indices += array_ops.expand_dims(batch_idx, 1)
    repeats = math_ops.equal(label_seq[:, :-1], label_seq[:, 1:])
    values = 1.0 - math_ops.cast(repeats, dtypes.float32)
    batched_shape = [batch_size, num_states, num_states]
    label_to_label = array_ops.scatter_nd(indices, values, batched_shape)

    return array_ops.expand_dims(trans, 0) + label_to_label


def ctc_state_log_probs(seq_lengths, max_seq_length):
  """Computes CTC alignment initial and final state log probabilities.

  Create the initial/final state values directly as log values to avoid
  having to take a float64 log on tpu (which does not exist).

  Args:
    seq_lengths: int tensor of shape [batch_size], seq lengths in the batch.
    max_seq_length: int, max sequence length possible.

  Returns:
    initial_state_log_probs, final_state_log_probs
  """

  batch_size = _get_dim(seq_lengths, 0)
  num_label_states = max_seq_length + 1
  num_duration_states = 2
  num_states = num_duration_states * num_label_states
  log_0 = math_ops.cast(
      math_ops.log(math_ops.cast(0, dtypes.float64) + 1e-307),
      dtypes.float32)

  initial_state_log_probs = array_ops.one_hot(
      indices=array_ops.zeros([batch_size], dtype=dtypes.int32),
      depth=num_states,
      on_value=0.0,
      off_value=log_0, axis=1)

  label_final_state_mask = array_ops.one_hot(
      seq_lengths, depth=num_label_states, axis=0)
  duration_final_state_mask = array_ops.ones(
      [num_duration_states, 1, batch_size])
  final_state_mask = duration_final_state_mask * label_final_state_mask
  final_state_log_probs = (1.0 - final_state_mask) * log_0
  final_state_log_probs = array_ops.reshape(
      final_state_log_probs, [num_states, batch_size])

  return initial_state_log_probs, array_ops.transpose(final_state_log_probs)


def _ilabel_to_state(labels, num_labels, ilabel_log_probs):
  """Project ilabel log probs to state log probs."""

  num_label_states = _get_dim(labels, 1)
  blank = ilabel_log_probs[:, :, :1]
  blank = array_ops.tile(blank, [1, 1, num_label_states + 1])
  one_hot = array_ops.one_hot(labels, depth=num_labels)
  one_hot = array_ops.expand_dims(one_hot, axis=0)
  ilabel_log_probs = array_ops.expand_dims(ilabel_log_probs, axis=2)
  state_log_probs = math_ops.reduce_sum(ilabel_log_probs * one_hot, axis=3)
  state_log_probs = array_ops.concat([state_log_probs, blank], axis=2)
  return array_ops.pad(
      state_log_probs, [[0, 0], [0, 0], [1, 0]],
      constant_values=math_ops.log(0.0))


def _state_to_olabel(labels, num_labels, states):
  """Sum state log probs to ilabel log probs."""

  num_label_states = _get_dim(labels, 1) + 1
  label_states = states[:, :, 1:num_label_states]
  blank_states = states[:, :, num_label_states:]
  one_hot = array_ops.one_hot(
      labels - 1, depth=(num_labels - 1),
      on_value=0.0, off_value=math_ops.log(0.0))
  one_hot = array_ops.expand_dims(one_hot, axis=0)
  label_states = array_ops.expand_dims(label_states, axis=3)
  label_olabels = math_ops.reduce_logsumexp(label_states + one_hot, axis=2)
  blank_olabels = math_ops.reduce_logsumexp(
      blank_states, axis=2, keepdims=True)
  return array_ops.concat([blank_olabels, label_olabels], axis=-1)


# pylint: disable=redefined-outer-name
def _state_to_olabel_unique(labels, num_labels, states, unique):
  """Sum state log probs to ilabel log probs using unique label indices."""

  num_label_states = _get_dim(labels, 1) + 1
  label_states = states[:, :, 1:num_label_states]
  blank_states = states[:, :, num_label_states:]

  unique_y, unique_idx = unique
  mul_reduce = _sum_states(unique_idx, label_states)

  num_frames = states.shape[0]
  batch_size = states.shape[1]
  num_states = num_label_states - 1
  batch_state_major = array_ops.transpose(mul_reduce, perm=[1, 2, 0])
  batch_state_major = array_ops.reshape(
      batch_state_major, [batch_size * num_states, num_frames])
  batch_offset = math_ops.range(batch_size, dtype=unique_y.dtype) * num_labels
  indices = unique_y + array_ops.expand_dims(batch_offset, axis=-1)
  indices = array_ops.reshape(indices, [-1, 1])
  scatter = array_ops.scatter_nd(
      indices=indices,
      updates=batch_state_major,
      shape=[batch_size * num_labels, num_frames])
  scatter = array_ops.reshape(scatter, [batch_size, num_labels, num_frames])
  scatter = array_ops.where(
      math_ops.equal(scatter, 0.0),
      array_ops.fill(array_ops.shape(scatter), math_ops.log(0.0)),
      scatter)
  label_olabels = array_ops.transpose(scatter, [2, 0, 1])
  label_olabels = label_olabels[:, :, 1:]

  blank_olabels = math_ops.reduce_logsumexp(
      blank_states, axis=2, keepdims=True)

  return array_ops.concat([blank_olabels, label_olabels], axis=-1)


def ctc_loss_and_grad(logits, labels, label_length, logit_length, unique=None):
  """Computes the CTC loss and gradients.

  Most users will want fwd_bwd.ctc_loss

  This function returns the computed gradient, it does not have a gradient
  of its own defined.

  Args:
    logits: tensor of shape [frames, batch_size, num_labels]
    labels: tensor of shape [batch_size, max_label_seq_length]
    label_length: tensor of shape [batch_size]
      Length of reference label sequence in labels.
    logit_length: tensor of shape [batch_size]
      Length of input sequence in logits.
    unique: (optional) unique label indices as computed by unique(labels)
      If supplied, enables an implementation that is faster and more memory
      efficient on TPU.

  Returns:
    loss: tensor of shape [batch_size]
    gradient: tensor of shape [frames, batch_size, num_labels]
  """

  num_labels = _get_dim(logits, 2)
  max_label_seq_length = _get_dim(labels, 1)

  ilabel_log_probs = nn_ops.log_softmax(logits)
  state_log_probs = _ilabel_to_state(labels, num_labels, ilabel_log_probs)
  state_trans_probs = _ctc_state_trans(labels)
  initial_state_log_probs, final_state_log_probs = ctc_state_log_probs(
      label_length, max_label_seq_length)
  fwd_bwd_log_probs, log_likelihood = _forward_backward_log(
      state_trans_log_probs=math_ops.log(state_trans_probs),
      initial_state_log_probs=initial_state_log_probs,
      final_state_log_probs=final_state_log_probs,
      observed_log_probs=state_log_probs,
      sequence_length=logit_length)

  if unique:
    olabel_log_probs = _state_to_olabel_unique(
        labels, num_labels, fwd_bwd_log_probs, unique)
  else:
    olabel_log_probs = _state_to_olabel(labels, num_labels, fwd_bwd_log_probs)

  grad = math_ops.exp(ilabel_log_probs) - math_ops.exp(olabel_log_probs)
  loss = -log_likelihood
  return loss, grad


def _ctc_loss_grad(op, grad_loss, _):
  grad = op.outputs[1]
  grad = [array_ops.reshape(grad_loss, [1, -1, 1]) * grad]
  grad += [None] * (len(op.inputs) - len(grad))
  return grad


def _ctc_loss_shape(op):
  return [op.inputs[2].get_shape(), op.inputs[0].get_shape()]


@tf_export("nn.ctc_loss", v1=["nn.ctc_loss_v2"])
def ctc_loss_v2(labels, logits, label_length, logit_length,
                logits_time_major=True, unique=None,
                blank_index=None, name=None):
  """Computes CTC (Connectionist Temporal Classification) loss.

  This op implements the CTC loss as presented in the article:

  [A. Graves, S. Fernandez, F. Gomez, J. Schmidhuber.
  Connectionist Temporal Classification: Labeling Unsegmented Sequence Data
  with Recurrent Neural Networks. ICML 2006, Pittsburgh, USA,
  pp. 369-376.](http://www.cs.toronto.edu/~graves/icml_2006.pdf)

  Notes:
      - Same as the "Classic CTC" in TensorFlow 1.x's tf.nn.ctc_loss setting of
        preprocess_collapse_repeated=False, ctc_merge_repeated=True
      - Labels may be supplied as either a dense, zero-padded tensor with a
        vector of label sequence lengths OR as a SparseTensor.
      - On TPU and GPU:
          - Only dense padded labels are supported.
      - On CPU:
          - Caller may use SparseTensor or dense padded labels but calling with
            a SparseTensor will be significantly faster.
      - Default blank label is 0 rather num_classes - 1, unless overridden by
        blank_index.

  Args:
    labels: tensor of shape [batch_size, max_label_seq_length] or SparseTensor
    logits: tensor of shape [frames, batch_size, num_labels],
      if logits_time_major == False, shape is [batch_size, frames, num_labels].
    label_length: tensor of shape [batch_size], None if labels is SparseTensor
      Length of reference label sequence in labels.
    logit_length: tensor of shape [batch_size]
      Length of input sequence in logits.
    logits_time_major: (optional) If True (default), logits is shaped
      [time, batch, logits]. If False, shape is [batch, time, logits]
    unique: (optional) Unique label indices as computed by
      ctc_unique_labels(labels).  If supplied, enable a faster, memory
      efficient implementation on TPU.
    blank_index: (optional) Set the class index to use for the blank label.
      Negative values will start from num_classes, ie, -1 will reproduce the
      ctc_loss behavior of using num_classes - 1 for the blank symbol.
      There is some memory/performance overhead to switching from the default
      of 0 as an additional shifted copy of the logits may be created.
    name: A name for this `Op`. Defaults to "ctc_loss_dense".

  Returns:
    loss: tensor of shape [batch_size], negative log probabilities.
  """
  if isinstance(labels, sparse_tensor.SparseTensor):
    if blank_index is None:
      raise ValueError(
          "blank_index must be given when using SparseTensor labels.")

    if blank_index < 0:
      blank_index += _get_dim(logits, 2)

    if blank_index != _get_dim(logits, 2) - 1:
      logits = array_ops.concat([
          logits[:, :, :blank_index],
          logits[:, :, blank_index+1:],
          logits[:, :, blank_index:blank_index+1],
      ], axis=2)
      labels = sparse_tensor.SparseTensor(
          labels.indices,
          array_ops.where(labels.values < blank_index,
                          labels.values,
                          labels.values - 1),
          labels.dense_shape)

    return ctc_loss(labels=labels,
                    inputs=logits,
                    sequence_length=logit_length,
                    time_major=logits_time_major)

  if blank_index is None:
    blank_index = 0

  return ctc_loss_dense(labels=labels,
                        logits=logits,
                        label_length=label_length,
                        logit_length=logit_length,
                        logits_time_major=logits_time_major,
                        unique=unique,
                        blank_index=blank_index,
                        name=name)


def ctc_loss_dense(labels, logits, label_length, logit_length,
                   logits_time_major=True, unique=None,
                   blank_index=0, name=None):
  """Computes CTC (Connectionist Temporal Classification) loss.

  This op implements the CTC loss as presented in the article:

  [A. Graves, S. Fernandez, F. Gomez, J. Schmidhuber.
  Connectionist Temporal Classification: Labeling Unsegmented Sequence Data
  with Recurrent Neural Networks. ICML 2006, Pittsburgh, USA,
  pp. 369-376.](http://www.cs.toronto.edu/~graves/icml_2006.pdf)

  Using the batched forward backward algorithm described in:

  [Sim, K. C., Narayanan, A., Bagby, T., Sainath, T. N., & Bacchiani, M.
  Improving the efficiency of forward-backward algorithm using batched
    computation in TensorFlow.
  Automatic Speech Recognition and Understanding Workshop (ASRU),
    2017 IEEE (pp. 258-264).
  ](https://ieeexplore.ieee.org/iel7/8260578/8268903/08268944.pdf)

  Notes:
    Significant differences from tf.nn.ctc_loss:
      Supports GPU and TPU (tf.nn.ctc_loss supports CPU only):
        For batched operations, GPU and TPU are significantly faster than using
        ctc_loss on CPU.
        This implementation runs on CPU, but significantly slower than ctc_loss.
      Blank label is 0 rather num_classes - 1, unless overridden by blank_index.
      Logits and labels are dense arrays with padding rather than SparseTensor.
      The only mode supported is the same as:
        preprocess_collapse_repeated=False, ctc_merge_repeated=True
        To collapse labels, the caller can preprocess label sequence first.

    The dense implementation supports both CPU, GPU and TPU. A fast path is
    provided that significantly improves memory use for large vocabulary if the
    caller preprocesses label sequences to get unique label indices on the CPU
    (eg. in the data input pipeline) using ctc_ops.unique and simplies this in
    the optional "unique" kwarg. This is especially useful for TPU and GPU but
    also works with if used on CPU.

  Args:
    labels: tensor of shape [batch_size, max_label_seq_length]
    logits: tensor of shape [frames, batch_size, num_labels],
      if logits_time_major == False, shape is [batch_size, frames, num_labels].
    label_length: tensor of shape [batch_size]
      Length of reference label sequence in labels.
    logit_length: tensor of shape [batch_size]
      Length of input sequence in logits.
    logits_time_major: (optional) If True (default), logits is shaped
      [time, batch, logits]. If False, shape is [batch, time, logits]
    unique: (optional) Unique label indices as computed by unique(labels).
      If supplied, enable a faster, memory efficient implementation on TPU.
    blank_index: (optional) Set the class index to use for the blank label.
      Negative values will start from num_classes, ie, -1 will reproduce the
      ctc_loss behavior of using num_classes - 1 for the blank symbol.
      There is some memory/performance overhead to switching from the default
      of 0 as an additional shifted copy of the logits may be created.
    name: A name for this `Op`. Defaults to "ctc_loss_dense".

  Returns:
    loss: tensor of shape [batch_size], negative log probabilities.
  """

  with ops.name_scope(name, "ctc_loss_dense",
                      [logits, labels, label_length, logit_length]):
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    label_length = ops.convert_to_tensor(label_length, name="label_length")
    logit_length = ops.convert_to_tensor(logit_length, name="logit_length")

    if not logits_time_major:
      logits = array_ops.transpose(logits, perm=[1, 0, 2])

    if blank_index != 0:
      if blank_index < 0:
        blank_index += _get_dim(logits, 2)
      logits = array_ops.concat([
          logits[:, :, blank_index:blank_index+1],
          logits[:, :, :blank_index],
          logits[:, :, blank_index+1:],
      ], axis=2)
      labels = array_ops.where(labels < blank_index, labels + 1, labels)

    args = [logits, labels, label_length, logit_length]

    if unique:
      unique_y, unique_idx = unique
      args.extend([unique_y, unique_idx])

    # TODO(tombagby): Update to tfe.defun
    @function.Defun(*[x.dtype for x in args],
                    python_grad_func=_ctc_loss_grad,
                    shape_func=_ctc_loss_shape)
    def compute_ctc_loss(logits_t, labels_t, label_length_t, logit_length_t,
                         *unique_t):
      """Compute CTC loss."""
      logits_t.set_shape(logits.shape)
      labels_t.set_shape(labels.shape)
      label_length_t.set_shape(label_length.shape)
      logit_length_t.set_shape(logit_length.shape)
      kwargs = dict(
          logits=logits_t,
          labels=labels_t,
          label_length=label_length_t,
          logit_length=logit_length_t)
      if unique_t:
        kwargs["unique"] = unique_t
      return ctc_loss_and_grad(**kwargs)

    return compute_ctc_loss(*args)[0]


@tf_export("nn.collapse_repeated")
def collapse_repeated(labels, seq_length, name=None):
  """Merge repeated labels into single labels.

  Args:
    labels: Tensor of shape (batch, max value in seq_length)
    seq_length: Tensor of shape (batch), sequence length of each batch element.
    name: A name for this `Op`. Defaults to "collapse_repeated_labels".

  Returns:
    tuple of Tensor of shape (batch, max_seq_length) with repeated labels
    collapsed and padded to max_seq_length, eg:
        [[A, A, B, B, A],
         [A, B, C, D, E]] => [[A, B, A, 0, 0],
                              [A, B, C, D, E]]
    and int tensor of shape [batch] with new sequence lengths.
  """

  with ops.name_scope(name, "collapse_repeated_labels",
                      [labels, seq_length]):
    labels = ops.convert_to_tensor(labels, name="labels")
    seq_length = ops.convert_to_tensor(seq_length, name="seq_length")

    # Mask labels that don't equal previous label.
    label_mask = array_ops.concat(
        [array_ops.ones_like(labels[:, :1], dtypes.bool),
         math_ops.not_equal(labels[:, 1:], labels[:, :-1])],
        axis=1)

    # Filter labels that aren't in the original sequence.
    maxlen = _get_dim(labels, 1)
    seq_mask = array_ops.sequence_mask(seq_length, maxlen=maxlen)
    label_mask = math_ops.logical_and(label_mask, seq_mask)

    # Count masks for new sequence lengths.
    new_seq_len = math_ops.reduce_sum(
        math_ops.cast(label_mask, dtypes.int32), axis=1)

    # Mask indexes based on sequence length mask.
    new_maxlen = math_ops.reduce_max(new_seq_len)
    idx_mask = array_ops.sequence_mask(new_seq_len, maxlen=new_maxlen)

    # Flatten everything and mask out labels to keep and sparse indices.
    flat_labels = array_ops.reshape(labels, [-1])
    flat_label_mask = array_ops.reshape(label_mask, [-1])
    flat_idx_mask = array_ops.reshape(idx_mask, [-1])
    idx = math_ops.range(_get_dim(flat_idx_mask, 0))

    # Scatter to flat shape.
    flat = array_ops.scatter_nd(
        indices=array_ops.expand_dims(
            array_ops.boolean_mask(idx, flat_idx_mask), axis=1),
        updates=array_ops.boolean_mask(flat_labels, flat_label_mask),
        shape=array_ops.shape(flat_idx_mask))

    # Reshape back to square batch.
    batch_size = _get_dim(labels, 0)
    new_shape = [batch_size, new_maxlen]
    return (array_ops.reshape(flat, new_shape),
            math_ops.cast(new_seq_len, seq_length.dtype))


def dense_labels_to_sparse(dense, length):
  """Convert dense labels with sequence lengths to sparse tensor.

  Args:
    dense: tensor of shape [batch, max_length]
    length: int tensor of shape [batch]
      The length of each sequence in dense.

  Returns:
    tf.SparseTensor with values only for the valid elements of sequences.
  """

  flat_values = array_ops.reshape(dense, [-1])
  flat_indices = math_ops.range(
      array_ops.shape(flat_values, out_type=dtypes.int64)[0])
  mask = array_ops.sequence_mask(length, maxlen=array_ops.shape(dense)[1])
  flat_mask = array_ops.reshape(mask, [-1])
  indices = array_ops.expand_dims(
      array_ops.boolean_mask(flat_indices, flat_mask), 1)
  values = array_ops.boolean_mask(flat_values, flat_mask)
  sparse = sparse_tensor.SparseTensor(
      indices=indices, values=math_ops.cast(values, dtypes.int32),
      dense_shape=array_ops.shape(flat_values, out_type=dtypes.int64))
  reshaped = sparse_ops.sparse_reshape(sparse, array_ops.shape(dense))
  max_length = math_ops.reduce_max(length)
  return sparse_tensor.SparseTensor(
      indices=reshaped.indices,
      values=reshaped.values,
      dense_shape=[
          math_ops.cast(reshaped.dense_shape[0], dtypes.int64),
          math_ops.cast(max_length, dtypes.int64)])


@tf_export("nn.ctc_unique_labels")
def ctc_unique_labels(labels, name=None):
  """Get unique labels and indices for batched labels for tf.nn.ctc_loss.

  For use with tf.nn.ctc_loss_v2 optional argument `unique`: This op can be
  used to preprocess labels in input pipeline to for better speed/memory use
  computing the ctc loss on TPU.

  Example:
    ctc_unique_labels([[3, 4, 4, 3]]) ->
      unique labels padded with 0: [[3, 4, 0, 0]]
      indices of original labels in unique: [0, 1, 1, 0]

  Args:
    labels: tensor of shape [batch_size, max_label_length] padded with 0.
    name: A name for this `Op`. Defaults to "ctc_unique_labels".

  Returns:
    tuple of
      - unique labels, tensor of shape `[batch_size, max_label_length]`
      - indices into unique labels, shape `[batch_size, max_label_length]`
  """

  with ops.name_scope(name, "ctc_unique_labels", [labels]):
    labels = ops.convert_to_tensor(labels, name="labels")
    def _unique(x):
      u = array_ops.unique(x)
      y = array_ops.pad(
          u.y, [[0, _get_dim(u.idx, 0) - _get_dim(u.y, 0)]])
      y = math_ops.cast(y, dtypes.int64)
      return [y, u.idx]
    return functional_ops.map_fn(
        _unique, labels, dtype=[dtypes.int64, dtypes.int32])


def _sum_states(idx, states):
  """Take logsumexp for each unique state out of all label states.

  Args:
    idx: tensor of shape [batch, label_length]
      For each sequence, indices into a set of unique labels as computed by
      calling unique.
    states: tensor of shape [frames, batch, label_length]
      Log probabilities for each label state.

  Returns:
    tensor of shape [frames, batch_size, label_length], log probabilites summed
      for each unique label of the sequence.
  """

  with ops.name_scope("sum_states"):
    idx = ops.convert_to_tensor(idx, name="idx")
    num_states = _get_dim(states, 2)
    states = array_ops.expand_dims(states, axis=2)
    one_hot = array_ops.one_hot(
        idx, depth=num_states, on_value=0.0, off_value=math_ops.log(0.0),
        axis=1)
    return math_ops.reduce_logsumexp(states + one_hot, axis=-1)


def _forward_backward_log(state_trans_log_probs, initial_state_log_probs,
                          final_state_log_probs, observed_log_probs,
                          sequence_length):
  """Forward-backward algorithm computed in log domain.

  Args:
    state_trans_log_probs: tensor of shape [states, states] or
      if different transition matrix per batch [batch_size, states, states]
    initial_state_log_probs: tensor of shape [batch_size, states]
    final_state_log_probs: tensor of shape [batch_size, states]
    observed_log_probs: tensor of shape [frames, batch_size, states]
    sequence_length: tensor of shape [batch_size]

  Returns:
    forward backward log probabilites: tensor of shape [frames, batch, states]
    log_likelihood: tensor of shape [batch_size]

  Raises:
    ValueError: If state_trans_log_probs has unknown or incorrect rank.
  """

  if state_trans_log_probs.shape.ndims == 2:
    perm = [1, 0]
  elif state_trans_log_probs.shape.ndims == 3:
    perm = [0, 2, 1]
  else:
    raise ValueError(
        "state_trans_log_probs rank must be known and == 2 or 3, is: %s" %
        state_trans_log_probs.shape.ndims)

  bwd_state_trans_log_probs = array_ops.transpose(state_trans_log_probs, perm)
  batch_size = _get_dim(observed_log_probs, 1)

  def _forward(state_log_prob, obs_log_prob):
    state_log_prob = array_ops.expand_dims(state_log_prob, axis=1)  # Broadcast.
    state_log_prob += state_trans_log_probs
    state_log_prob = math_ops.reduce_logsumexp(state_log_prob, axis=-1)
    state_log_prob += obs_log_prob
    log_prob_sum = math_ops.reduce_logsumexp(
        state_log_prob, axis=-1, keepdims=True)
    state_log_prob -= log_prob_sum
    return state_log_prob

  fwd = _scan(_forward, observed_log_probs, initial_state_log_probs,
              inclusive=True)

  def _backward(accs, elems):
    """Calculate log probs and cumulative sum masked for sequence length."""
    state_log_prob, cum_log_sum = accs
    obs_log_prob, mask = elems
    state_log_prob += obs_log_prob
    state_log_prob = array_ops.expand_dims(state_log_prob, axis=1)  # Broadcast.
    state_log_prob += bwd_state_trans_log_probs
    state_log_prob = math_ops.reduce_logsumexp(state_log_prob, axis=-1)

    log_prob_sum = math_ops.reduce_logsumexp(
        state_log_prob, axis=-1, keepdims=True)
    state_log_prob -= log_prob_sum

    cum_log_sum += array_ops.squeeze(log_prob_sum) * mask
    batched_mask = array_ops.expand_dims(mask, axis=1)
    out = state_log_prob * batched_mask
    out += final_state_log_probs * (1.0 - batched_mask)
    return out, cum_log_sum

  zero_log_sum = array_ops.zeros([batch_size])
  maxlen = _get_dim(observed_log_probs, 0)
  mask = array_ops.sequence_mask(sequence_length, maxlen, dtypes.float32)
  mask = array_ops.transpose(mask, perm=[1, 0])

  bwd, cum_log_sum = _scan(_backward, (observed_log_probs, mask),
                           (final_state_log_probs, zero_log_sum),
                           reverse=True, inclusive=True)

  fwd_bwd_log_probs = fwd[1:] + bwd[1:]
  fwd_bwd_log_probs_sum = math_ops.reduce_logsumexp(
      fwd_bwd_log_probs, axis=2, keepdims=True)
  fwd_bwd_log_probs -= fwd_bwd_log_probs_sum
  fwd_bwd_log_probs += math_ops.log(array_ops.expand_dims(mask, axis=2))

  log_likelihood = bwd[0, :, 0] + cum_log_sum[0]

  return fwd_bwd_log_probs, log_likelihood


# TODO(tombagby): This is currently faster for the ctc implementation than using
# functional_ops.scan, but could be replaced by that or something similar if
# things change.
def _scan(fn, elems, initial, reverse=False, inclusive=False, final_only=False):
  """Repeatedly applies callable `fn` to a sequence of elements.

  Implemented by functional_ops.While, tpu friendly, no gradient.

  This is similar to functional_ops.scan but significantly faster on tpu/gpu
  for the forward backward use case.

  Examples:
    scan(lambda a, e: a + e, [1.0, 2.0, 3.0], 1.0) => [2.0, 3.0, 4.0]

    Multiple accumulators:
      scan(lambda a, e: (a[0] + e, a[1] * e), [1.0, 2.0, 3.0], (0.0, 1.0))

    Multiple inputs:
      scan(lambda a, e: a + (e[0] * e[1]), (elems1, elems2), 0.0)

  Args:
    fn: callable, fn(accumulators, element) return new accumulator values.
      The (possibly nested) sequence of accumulators is the same as `initial`
      and the return value must have the same structure.
    elems: A (possibly nested) tensor which will be unpacked along the first
      dimension. The resulting slices will be the second argument to fn. The
      first dimension of all nested input tensors must be the same.
    initial: A tensor or (possibly nested) sequence of tensors with initial
      values for the accumulators.
    reverse: (optional) True enables scan and output elems in reverse order.
    inclusive: (optional) True includes the initial accumulator values in the
      output. Length of output will be len(elem sequence) + 1. Not meaningful
      if final_only is True.
    final_only: (optional) When True, return only the final accumulated values,
      not the concatenation of accumulated values for each input.

  Returns:
    A (possibly nested) sequence of tensors with the results of applying fn
    to tensors unpacked from elems and previous accumulator values.
  """

  flat_elems = [ops.convert_to_tensor(x) for x in nest.flatten(elems)]
  num_elems = array_ops.shape(flat_elems[0])[0]
  pack_elems = lambda x: nest.pack_sequence_as(structure=elems, flat_sequence=x)
  flat_initial = [ops.convert_to_tensor(x) for x in nest.flatten(initial)]
  pack = lambda x: nest.pack_sequence_as(structure=initial, flat_sequence=x)
  accum_dtypes = [x.dtype for x in flat_initial]
  num_accums = len(flat_initial)

  # Types for counter, [outputs], [accumulators] loop arguments.
  if final_only:
    loop_dtypes = [dtypes.int32, dtypes.int32] + accum_dtypes
  else:
    loop_dtypes = [dtypes.int32, dtypes.int32] + accum_dtypes + accum_dtypes

  # TODO(tombagby): Update to tfe.defun
  @function.Defun(*loop_dtypes)
  def cond(i, num_elems, *args):
    del args
    return i >= 0 if reverse else i < num_elems

  # The loop *args are [output tensors] + [accumulator tensors] which must
  # be paired. Each output corresponds to one accumulator.
  @function.Defun(*loop_dtypes)
  def body(i, num_elems, *args):
    """Loop body."""
    i.set_shape([])
    if final_only:
      accum = args
    else:
      out, accum = args[:num_accums], args[num_accums:]
    slices = [array_ops.gather(e, i) for e in flat_elems]
    accum = fn(pack(accum), pack_elems(slices))
    flat_accum = nest.flatten(accum)
    if final_only:
      new_out = []
    else:
      update_i = i + 1 if inclusive and not reverse else i
      new_out = [inplace_ops.alias_inplace_update(x, update_i, y)
                 for x, y in zip(out, flat_accum)]
    i = i - 1 if reverse else i + 1
    return [i, num_elems] + new_out + flat_accum

  init_i = (array_ops.shape(flat_elems[0])[0] - 1 if reverse
            else constant_op.constant(0, dtype=dtypes.int32))
  outputs = []
  if not final_only:
    num_outputs = array_ops.shape(flat_elems[0])[0] + (1 if inclusive else 0)
    for initial_accum in flat_initial:
      out_shape = array_ops.concat(
          [[num_outputs], array_ops.shape(initial_accum)], 0)
      out = inplace_ops.empty(out_shape, dtype=initial_accum.dtype, init=True)
      if inclusive:
        out = inplace_ops.alias_inplace_add(
            out, init_i + (1 if reverse else 0), initial_accum)
      outputs.append(out)
  loop_in = [init_i, num_elems] + outputs + flat_initial
  hostmem = [
      i for i, x in enumerate(loop_in)
      if x.dtype.base_dtype in (dtypes.int32, dtypes.int64)
  ]

  # TODO(tombagby): Update to while_v2.
  loop_results = functional_ops.While(loop_in, cond, body, hostmem=hostmem)
  out = loop_results[2:num_accums + 2]
  return pack(out)


def _get_dim(tensor, i):
  """Get value of tensor shape[i] preferring static value if available."""
  return tensor.shape[i].value or array_ops.shape(tensor)[i]
