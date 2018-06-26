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

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

namespace shape_inference {
extern Status TRTEngineOpShapeInference(InferenceContext* c);
}

REGISTER_OP("TRTEngineOp")
    .Attr("serialized_segment: string")
    .Attr("input_shapes: list(shape)")
    .Attr("output_shapes: list(shape)")
    .Attr("segment_funcdef_name: string")
    .Attr("InT: list({int8,float16,float32})")
    .Attr("OutT: list({int8,float16,float32})")
    .Attr("static_engine: bool = true")
    .Attr("fixed_input_size: bool = true")
    .Attr("cached_engine_batches: list(int) = []")
    .Attr("max_cached_engines_count: int = 1")
    .Attr("workspace_size_bytes: int")
    .Attr("precision_mode: {'FP32', 'FP16', 'INT8', 'INT8CALIB'}")
    .Attr("calibration_data: string = ''")
    .Input("in_tensor: InT")
    .Output("out_tensor: OutT");
    // TODO(Sami): shape inference not working for concrete input shape 
    //.SetShapeFn(shape_inference::TRTEngineOpShapeInference);

}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
