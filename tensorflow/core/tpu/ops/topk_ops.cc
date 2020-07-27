/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

using shape_inference::ShapeHandle;

REGISTER_OP("KthOrderStatistic")
    .Input("input: float32")
    .Output("output: float32")
    .Attr("k: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));

      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, -1, &s));
      c->set_output(0, s);
      return Status::OK();
    })
    .Doc(R"doc(
Computes the Kth order statistic of a data set. The current
implementation uses a binary search requiring exactly 32 passes over
the input data. The running time is linear with respect to input
size. The median-of-medians algorithm is probably faster, but is
difficult to implement efficiently in XLA. The implementation imposes
a total ordering on floats. The ordering is consistent with the usual
partial order.  Positive NaNs are greater than positive
infinity. Negative NaNs are less than negative infinity. NaNs with
distinct payloads are treated as distinct. Subnormal numbers are
preserved (not flushed to zero). Positive infinity is greater than all
numbers. Negative infinity is less than all numbers. Positive is
greater than negative zero. There are less than k values greater than
the kth order statistic. There are at least k values greater than or
equal to the Kth order statistic. The semantics are not the same as
top_k_unique.
)doc");

REGISTER_OP("TopKUnique")
    .Input("input: float32")
    .Output("topk: float32")
    .Output("topk_indices: int32")
    .Attr("k: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));

      int32 k;
      TF_RETURN_IF_ERROR(c->GetAttr("k", &k));

      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 1, c->MakeDim(k), &s));
      c->set_output(0, s);
      c->set_output(1, s);
      return Status::OK();
    })
    .Doc(R"doc(
Returns the TopK unique values in the array in sorted order. The
running time is proportional to the product of K and the input
size. Sorting the whole array is more efficient for sufficiently large
values of K. The median-of-medians algorithm is probably faster, but
difficult to implement efficiently in XLA. If there are fewer than K
unique numbers (not NANs), the results are padded with negative
infinity. NaNs are never returned. Subnormal numbers are flushed to
zero. If an element appears at multiple indices, the highest index is
returned. If a TopK element never appears in the input due to padding
values, the indices are padded with negative one. If a padding value
appears in the input and padding is needed, the highest index of the
padding value will be returned. The semantics are not the same as
kth_order_statistic.
)doc");

REGISTER_OP("MakeUnique")
    .Input("input: float32")
    .Output("output: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
      c->set_output(0, input);
      return Status::OK();
    })
    .Doc(R"doc(
Make all elements in the non-Batch dimension unique, but \"close\" to
their initial value. Never returns a sub-normal number. Never returns
zero. The sign of each input element is always identical to the sign
of the corresponding output element. Behavior for infinite elements is
undefined. Behavior for subnormal elements is undefined.
)doc");

REGISTER_OP("TopKWithUnique")
    .Input("input: float32")
    .Output("topk: float32")
    .Output("topk_indices: int32")
    .Attr("k: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));

      int32 k;
      TF_RETURN_IF_ERROR(c->GetAttr("k", &k));

      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 1, c->MakeDim(k), &s));
      c->set_output(0, s);
      c->set_output(1, s);
      return Status::OK();
    })
    .Doc(R"doc(
Returns the TopK values in the array in sorted order. This is a combination
of MakeUnique and TopKUnique. The returned top-K will have its lower bits
replaced by iota, thus it will be close to the original value but not exactly
the same. The running time is proportional to the product of K and the input
size. NaNs are never returned. Subnormal numbers are flushed to zero.)doc");
}  // namespace tensorflow
