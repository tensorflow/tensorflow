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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/errors.h"

// Use a namespace when registering by prepending the
// package's name to the op’s name and separate with a '>'.
// This is the recommendation for out-of-tree ops to avoid name collisions in
// "Best practices for custom operations in TensorFlow"
// https://github.com/tensorflow/community/blob/master/rfcs/20190726-custom-ops.md

REGISTER_OP("Examples1>MultiplexDense")
    .Input("cond: bool")
    .Input("a_values: T")
    .Input("b_values: T")
    .Output("output_values: T")
    .Attr("T: type")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      // Determine the output shape and also assert that inputs 0 and 1 have
      // the same shape.
      tensorflow::shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &out));
      // Assert that inputs 0 and 2 have the same shape, i.e. that all inputs
      // have the same shape. This is optional, but it is desirable
      // to raise errors about inconsistent input shapes early when using
      // graph mode.
      tensorflow::shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(2), &unused));

      c->set_output(0, out);
      return ::tensorflow::OkStatus();
    })
    .Doc(R"doc(
Return elements chosen from `a` or `b` depending on `cond`.

This is similar to `np.where` and `tf.where`, but simplified to only handle
the case of dense tensors, no optional parameters, no broadcasting, etc..

cond: tf.Tensor of type bool.
a_values: tf.Tensor with the same type and shape as `b_values`.
b_values: tf.Tensor with the same type and shape as `a_values`.

      Where True, yield `a_values`, otherwise yield `b_values`.
output_values: A tf.Tensor with elements from `a` where `cond` is True, and
               elements from `b` elsewhere.
)doc");
