/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

REGISTER_OP("XRTExecute")
    .Attr("Ninputs: int >= 0")
    .Input("computation_handle: int64")
    .Input("execution_config: string")
    .Input("input_handles: Ninputs * int64")
    .Output("output_handle: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<shape_inference::ShapeHandle> input_handle_shapes;
      TF_RETURN_IF_ERROR(c->input("input_handles", &input_handle_shapes));
      for (size_t i = 0; i < input_handle_shapes.size(); ++i) {
        shape_inference::ShapeHandle unused;
        TF_RETURN_IF_ERROR(
            c->WithRankAtMost(input_handle_shapes[i], 1, &unused));
      }
      return tensorflow::shape_inference::ScalarShape(c);
    })
    .Doc(
        R"(
Runs a previously-compiled computation on a core. If
execution_config.release_input_handles is true, the input handles are invalid
after this op runs.

'computation_handle' is an id returned by XRTCompile.
'execution_config' is a serialized xrt::TPUExecutionConfig proto.
'input_handles' is a list of ids of allocations, one per input to the compiled
computation.
'output_handle' is an identifier for the result of the compiled computation.
'Ninputs' is the number of input handles.
)");

}  // namespace tensorflow
