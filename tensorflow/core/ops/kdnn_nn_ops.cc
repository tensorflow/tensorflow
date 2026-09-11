/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Op definitions for the KDNN-backed variants of TF ops. Each one is a
// drop-in replacement for the corresponding public op, intended to be
// substituted in by the Grappler remapper when:
//
//   * IsKDNNEnabled() returns true (build + env-var allow it), AND
//   * the input tensor passes the op's predicate.
//
// NOTE Do not invoke these operators directly in Python. The graph
// rewrite pass is expected to emit them.

#ifdef KERNEL_KDNN

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace {

using shape_inference::InferenceContext;
using shape_inference::UnchangedShape;
using shape_inference::UnchangedShapeWithRankAtLeast;

}  // namespace

// Sigmoid: y = 1 / (1 + exp(-x))  — KDNN SIMD implementation.
REGISTER_OP("_KdnnSigmoid")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {half, bfloat16, float} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      // In-place is allowed: x == y is a valid call.
      if (c->num_inputs() != 1) return absl::FailedPreconditionError("1 input");
      if (c->num_outputs() != 1) {
        return absl::FailedPreconditionError("1 output");
      }
      c->set_output(0, c->input(0));
      return absl::OkStatus();
    })
    .Doc(R"doc(
KDNN version of Sigmoid operator. Uses KDNN's element-wise SIMD op.

NOTE Do not invoke this operator directly in Python. The Grappler remapper
is expected to substitute this for the public `Sigmoid` op when KDNN is
enabled and the input tensor's dtype is supported (float / bfloat16 /
half on aarch64).
)doc");

}  // namespace tensorflow

#endif  // KERNEL_KDNN
