/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
namespace tfrt_stub {

// Note: Change with care. `PwStreamResults` is persisted in user SavedModel, so
// strict forward and backward compatibility is required. Follow best practices
// in go/use_tensorflow/custom_ops/custom_ops_overview.md#op_compatibility.
REGISTER_OP("PwStreamResults")
    .Input("args: T")
    .Attr("T: list(type) >= 0")
    .Attr("names: list(string) >= 0")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(R"(
Sends tensors back to the serving controller.

This op can be used in Pathways Serving to send results back to the serving
controller in the middle of computation. This allows returning intermediate
results to serving clients in a streamed fashion without waiting for the entire
signature computation to finish.

This op blocks until the controller returns an ACK. So if two streamed results
need to be ordered, inserting a control edge between the two ops makes it
possible to guarantee such ordering.

names: Names of tensors. The cardinality must match that of `args`.
)");

}  // namespace tfrt_stub
}  // namespace tensorflow
