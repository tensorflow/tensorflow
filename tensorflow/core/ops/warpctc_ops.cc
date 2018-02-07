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

REGISTER_OP("WarpCTCLoss")
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

}  // namespace tensorflow
