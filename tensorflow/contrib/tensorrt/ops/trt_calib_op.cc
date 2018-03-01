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
    .Attr("segment_nodes: list(string)")         // names of the ops in segment
    .Attr("segment_output_names: list(string)")  // names of the output ops in
                                                 // segment
    .Attr("input_names: list(string)")           // names of the inputs for
                                                 // passing into tensorrt
    .Attr("resource_name: string")
    .Attr("InT: list({int8, float16, float32})")
    .Input("in_tensor: InT")
    .Output("out_tensor: InT")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      for (int i = 0; i < c->num_inputs(); i++) {
        c->set_output(i, c->input(i));
      }
      return Status::OK();
    });

}  // namespace tensorflow
