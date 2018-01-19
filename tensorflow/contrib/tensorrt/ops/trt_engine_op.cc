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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

namespace shape_inference {
extern Status TRTEngineOpShapeInference(InferenceContext* c);
}

REGISTER_OP("TRTEngineOp")
    .Attr("serialized_engine: string")
    .Attr("input_nodes: list(string)")
    .Attr("output_nodes: list(string)")
    .Attr("InT: list({int8, float16, float32})")
    .Attr("OutT: list({int8, float16, float32})")
    .Input("in_tensor: InT")
    .Output("out_tensor: OutT")
    .SetShapeFn(shape_inference::TRTEngineOpShapeInference);

}  // namespace tensorflow
