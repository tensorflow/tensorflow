// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

absl::Status RougeLShapeFn(InferenceContext* c);

REGISTER_OP("RougeL")
    .Input("hyp_values: Tvalues")
    .Input("hyp_splits: Tsplits")
    .Input("ref_values: Tvalues")
    .Input("ref_splits: Tsplits")
    .Input("alpha: float")
    .Output("f_measure: float")
    .Output("p_measure: float")
    .Output("r_measure: float")
    .Attr("Tsplits: {int32, int64} = DT_INT64")
    .Attr("Tvalues: type")
    .SetShapeFn(RougeLShapeFn)
    .Doc(R"doc(
Computes the LCS-based F-measure score between the hypotheses and references.

  Source: https://www.microsoft.com/en-us/research/publication/rouge-a-package-for-automatic-evaluation-of-summaries/

This Op does not impose any tokenization scheme, in order to give callers
more flexibility.

An F-Measure is computed for each (hyp, ref) pair. As such, there must be an
equal number of sentences in the hypotheses and references.

The alpha parameter is used to weight precision and recall. A value of .5
represents matches the default value of the ROUGE-1.5.5.pl script. Negative
values will trigger a compatibility mode with tensor2tensor ROUGE.

A convenient way to compute ROUGE-L over a batch of sentences is to tokenize
them into tf.RaggedTensor format and then call this method with
tokens.values and tokens.row_splits.

The output is a 1D Tensor of shape [S-1], where S is the number of sentence
splits.

hyp_values: a 1D Tensor of shape [H] containing all hypothesis tokens
hyp_splits: a 1D Tensor of shape [S] containing hypothesis sentence splits
ref_values: a 1D Tensor of shape [R] containing all reference tokens
ref_splits: a 1D Tensor of shape [S] containing reference sentence splits
alpha: a 0D scalar Tensor containing the value of the Alpha parameter
f_measure: a 1D Tensor of shape [S-1] containing LCS F-measure scores
p_measure: a 1D Tensor of shape [S-1] containing LCS P-measure scores
r_measure: a 1D Tensor of shape [S-1] containing LCS R-measure scores
)doc");

absl::Status RougeLShapeFn(InferenceContext* c) {
  ShapeHandle unused;

  // Check rank of inner values
  ShapeHandle hyp_values_shape = c->input(0);
  ShapeHandle hyp_splits_shape = c->input(1);
  ShapeHandle ref_values_shape = c->input(2);
  ShapeHandle ref_splits_shape = c->input(3);
  ShapeHandle beta_shape = c->input(4);

  TF_RETURN_IF_ERROR(c->WithRank(hyp_values_shape, 1, &unused));
  TF_RETURN_IF_ERROR(c->WithRank(hyp_splits_shape, 1, &unused));
  TF_RETURN_IF_ERROR(c->WithRank(ref_values_shape, 1, &unused));
  TF_RETURN_IF_ERROR(c->WithRank(ref_splits_shape, 1, &unused));
  TF_RETURN_IF_ERROR(c->WithRank(beta_shape, 0, &unused));

  ShapeHandle output_nrows_plus_one;
  TF_RETURN_IF_ERROR(c->Merge(hyp_splits_shape, ref_splits_shape,
                              &output_nrows_plus_one));

  // Output shape is a 1-D tensor with size equal to number of splits minus 1.
  DimensionHandle dim;
  TF_RETURN_IF_ERROR(c->Subtract(c->Dim(output_nrows_plus_one, 0), 1, &dim));

  // All outputs have the same shape.
  c->set_output(0, c->Vector(dim));
  c->set_output(1, c->Vector(dim));
  c->set_output(2, c->Vector(dim));

  return absl::OkStatus();
}

}  // namespace tensorflow
