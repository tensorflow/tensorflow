/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

// Use a namespace when registering by prepending the
// package's name to the op’s name and separate with a '>'.
// This is the recommendation for out-of-tree ops to avoid name collisions in
// "Best practices for custom operations in TensorFlow"
// https://github.com/tensorflow/community/blob/master/rfcs/20190726-custom-ops.md

// Please use the appropriate namespace for your project
namespace tensorflow {
namespace custom_op_examples {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

Status MultiplexShapeFunction(InferenceContext* c) {
  int64_t num_cond_a;
  TF_RETURN_IF_ERROR(c->GetAttr("N", &num_cond_a));
  tensorflow::shape_inference::ShapeHandle unused;
  // There are N `cond` inputs, N `a` inputs and 1 `b` input in this order.
  // Check that all `cond` inputs and all `a` inputs have the same shape
  // as the `b` input.
  int64_t last = 2 * num_cond_a;
  for (int64_t i = 0; i < last; i++) {
    TF_RETURN_IF_ERROR(c->Merge(c->input(i), c->input(last), &unused));
  }
  c->set_output(0, c->input(last));
  return OkStatus();
}

REGISTER_OP("Examples>MultiplexDense")
    .Input("cond: N * bool")
    .Input("a_values: N * T")
    .Input("b_values: T")
    .Output("output_values: T")
    .Attr("T: type")
    .Attr("N: int = 1")
    .SetShapeFn(MultiplexShapeFunction)
    .Doc(R"doc(
Return elements chosen from `a_values` or `b_values` depending on `cond`.

When `a_values` and `cond` are tenors (i.e. N=1), this is similar to `np.where`
and `tf.where`. When `a_values` and `cond` are lists of tensors (i.e. N>1),
this is similar to `np.select`. In either case these are simplified to only
handle dense tensors, no optional parameters, no broadcasting, etc..

cond: tf.Tensor or list of tf.Tensor of type bool. If it is a list, `a_values`
      must be a list of the same length. Where True, yield the corresponding
      element from `a_values` (with priority to the first one encountered in
      lists), otherwise yield `b_values`.
a_values: tf.Tensor or list of tf.Tensor. Each tensor has the same type and
          shape as `b_values`. If it is a list, `cond` must be a list of the
          same length.
b_values: tf.Tensor with the same type and shape as the `a_values` if it is a
          tensor or as every element of `a_values` if `a_values` is a list.
output_values: A tf.Tensor with elements from `a_values` where `cond` is True,
               and elements from `b` elsewhere.
)doc");

}  // namespace custom_op_examples
}  // namespace tensorflow
