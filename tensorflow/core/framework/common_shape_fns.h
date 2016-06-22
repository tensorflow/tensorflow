/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#ifndef THIRD_PARTY_TENSORFLOW_CORE_OPS_COMMON_SHAPE_FNS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_OPS_COMMON_SHAPE_FNS_H_

#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace shape_inference {

// Transfers shape of input(0) to output(0).
Status UnchangedShape(shape_inference::InferenceContext* c);

// Transfers shape of input(0) to output(0), after asserting its rank is <rank>.
Status UnchangedShapeWithRank(shape_inference::InferenceContext* c, int32 rank);

// Transfers shape of input(0) to output(0), after asserting its rank >= <rank>.
Status UnchangedShapeWithRankAtLeast(shape_inference::InferenceContext* c,
                                     int32 rank);

inline Status UnchangedShapeWithRank(shape_inference::InferenceContext* c,
                                     int32 rank) {
  const Shape* out;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), rank, &out));
  c->set_output(0, out);
  return Status::OK();
}

inline Status UnchangedShapeWithRankAtLeast(
    shape_inference::InferenceContext* c, int32 rank) {
  const Shape* out;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), rank, &out));
  c->set_output(0, out);
  return Status::OK();
}

}  // namespace shape_inference
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_OPS_COMMON_SHAPE_FNS_H_
