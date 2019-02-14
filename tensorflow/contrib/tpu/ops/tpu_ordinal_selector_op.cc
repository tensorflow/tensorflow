/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

namespace tensorflow {

REGISTER_OP("TPUOrdinalSelector")
    .Output("device_ordinals: int32")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0,
                    c->Vector(shape_inference::InferenceContext::kUnknownDim));
      return Status::OK();
    })
    .Doc(R"doc(
A TPU core selector Op.

This Op produces a set of TPU cores (for warm-up) or a single TPU core
(for regular inference) to execute the TPU program on. The output is
consumed by TPUPartitionedCall.

device_ordinals: A vector 1 or more TPU cores.
)doc");

}  // namespace tensorflow
