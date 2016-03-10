/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

// CTC is Connectionist Temporal Classification.  See util/ctc/ for details.

REGISTER_OP("CTCLoss")
    .Input("inputs: float")
    .Input("labels_indices: int64")
    .Input("labels_values: int32")
    .Input("sequence_length: int32")
    .Attr("preprocess_collapse_repeated: bool = false")
    .Attr("ctc_merge_repeated: bool = true")
    .Output("loss: float")
    .Output("gradient: float")
    .Doc(R"doc(
Calculates the CTC Loss (log probability) for each batch entry.  Also calculates
the gradient.  This class performs the softmax operation for you, so inputs
should be e.g. linear projections of outputs by an LSTM.

inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
labels_indices: The indices of a `SparseTensor<int32, 2>`.
  `labels_indices(i, :) == [b, t]` means `labels_values(i)` stores the id for
  `(batch b, time t)`.
labels_values: The values (labels) associated with the given batch and time.
sequence_length: A vector containing sequence lengths (batch).
preprocess_collapse_repeated: Scalar, if true then repeated labels are
  collapsed prior to the CTC calculation.
ctc_merge_repeated: Scalar.  If set to false, *during* CTC calculation
  repeated non-blank labels will not be merged and are interpreted as
  individual labels.  This is a simplified version of CTC.
loss: A vector (batch) containing log-probabilities.
gradient: The gradient of `loss`.  3-D, shape:
  `(max_time x batch_size x num_classes)`.
)doc");

REGISTER_OP("CTCGreedyDecoder")
    .Input("inputs: float")
    .Input("sequence_length: int32")
    .Attr("merge_repeated: bool = false")
    .Output("decoded_indices: int64")
    .Output("decoded_values: int64")
    .Output("decoded_shape: int64")
    .Output("log_probability: float")
    .Doc(R"doc(
Performs greedy decoding on the logits given in inputs.

A note about the attribute merge_repeated: if enabled, when
consecutive logits' maximum indices are the same, only the first of
these is emitted.  Labeling the blank '*', the sequence "A B B * B B"
becomes "A B" if merge_repeated = True and "A B B B B" if
merge_repeated = False.

Regardless of the value of merge_repeated, if the maximum index of a given
time and batch corresponds to the blank, index `(num_classes - 1)`, no new
element is emitted.

inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
sequence_length: A vector containing sequence lengths, size `(batch_size)`.
merge_repeated: If True, merge repeated classes in output.
decoded_indices: Indices matrix, size `(total_decoded_outputs x 2)`,
  of a `SparseTensor<int64, 2>`.  The rows store: [batch, time].
decoded_values: Values vector, size: `(total_decoded_outputs)`,
  of a `SparseTensor<int64, 2>`.  The vector stores the decoded classes.
decoded_shape: Shape vector, size `(2)`, of the decoded SparseTensor.
  Values are: `[batch_size, max_decoded_length]`.
log_probability: Matrix, size `(batch_size x 1)`, containing sequence
  log-probabilities.
)doc");

REGISTER_OP("CTCBeamSearchDecoder")
    .Input("inputs: float")
    .Input("sequence_length: int32")
    .Attr("beam_width: int >= 1")
    .Attr("top_paths: int >= 1")
    .Attr("merge_repeated: bool = true")
    .Output("decoded_indices: top_paths * int64")
    .Output("decoded_values: top_paths * int64")
    .Output("decoded_shape: top_paths * int64")
    .Output("log_probability: float")
    .Doc(R"doc(
Performs beam search decoding on the logits given in input.

A note about the attribute merge_repeated: For the beam search decoder,
this means that if consecutive entries in a beam are the same, only
the first of these is emitted.  That is, when the top path is "A B B B B",
"A B" is returned if merge_repeated = True but "A B B B B" is
returned if merge_repeated = False.

inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
sequence_length: A vector containing sequence lengths, size `(batch)`.
beam_width: A scalar >= 0 (beam search beam width).
top_paths: A scalar >= 0, <= beam_width (controls output size).
merge_repeated: If true, merge repeated classes in output.
decoded_indices: A list (length: top_paths) of indices matrices.  Matrix j,
  size `(total_decoded_outputs[j] x 2)`, has indices of a
  `SparseTensor<int64, 2>`.  The rows store: [batch, time].
decoded_values: A list (length: top_paths) of values vectors.  Vector j,
  size `(length total_decoded_outputs[j])`, has the values of a
  `SparseTensor<int64, 2>`.  The vector stores the decoded classes for beam j.
decoded_shape: A list (length: top_paths) of shape vector.  Vector j,
  size `(2)`, stores the shape of the decoded `SparseTensor[j]`.
  Its values are: `[batch_size, max_decoded_length[j]]`.
log_probability: A matrix, shaped: `(batch_size x top_paths)`.  The
  sequence log-probabilities.
)doc");

}  // namespace tensorflow
