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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
namespace tensorflow {

REGISTER_OP("TRTCalibOp")
    .Attr("serialized_segment: string")
    .Attr("segment_funcdef_name: string")
    .Attr("input_shapes: list(shape)")
    .Attr("output_shapes: list(shape)")
    .Attr("InT: list({int8, float16, float32})")
    .Attr("OutT: list({int8, float16, float32})")
    .Attr("workspace_size_bytes: int")
    .Input("in_tensor: InT")
    .Output("out_tensor: OutT")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c)->tensorflow::Status {
      std::vector<tensorflow::TensorShapeProto> shapes;
      auto status=c->GetAttr("output_shapes", &shapes);
      if(!status.ok()){
        LOG(ERROR)<<"getting output_shapes failed with "<<status;
        return status;
      }
      for (int i = 0; i < shapes.size(); i++) {
        tensorflow::shape_inference::ShapeHandle shape;
        status=c->MakeShapeFromShapeProto(shapes.at(i),&shape);
        if(!status.ok()){
          LOG(ERROR)<<"stting output shape "<<i<<" failed with "<<status;
          return status;
        }
        
        c->set_output(i, shape);
      }
      return Status::OK();
    });

}  // namespace tensorflow
