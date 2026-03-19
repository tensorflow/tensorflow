/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/rng_alg.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"
#include "tsl/platform/errors.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("StochasticCastToInt")
    .Input("input: Tin")
    .Input("key: uint64")
    .Input("counter: uint64")
    .Input("alg: int32")
    .Output("output: Tout")
    .Attr("Tin: {half, bfloat16, float32, float64}")
    .Attr("Tout: {int8, int16, int32}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle key;
      ShapeHandle shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &key));    // key shape
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &shape));  // counter shape
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &shape));  // alg shape
      DimensionHandle dim;
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(key, 0), RNG_KEY_SIZE, &dim));  // alg dim
      c->set_output(0, c->input(0));                          // out shape
      return absl::OkStatus();
    });

// TODO(b/232442915): Add support for rounding floats to lower precision floats.

}  // namespace tensorflow
