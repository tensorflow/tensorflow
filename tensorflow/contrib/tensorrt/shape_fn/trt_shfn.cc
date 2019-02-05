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

#include "tensorflow/contrib/tensorrt/shape_fn/trt_shfn.h"

#include <string>
#include <vector>

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorflow/compiler/tf2tensorrt/plugin/trt_plugin_factory.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace shape_inference {

tensorflow::Status TRTEngineOpShapeInference(InferenceContext* c) {
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->UnknownShape());
  }

  // Check the sanity of the input shapes.
  std::vector<tensorflow::TensorShape> input_shapes;
  TF_RETURN_IF_ERROR(c->GetAttr("input_shapes", &input_shapes));
  if (input_shapes.size() != c->num_inputs()) {
    return tensorflow::errors::InvalidArgument(
        "The actual number of inputs doesn't match the number of input "
        "shapes set in the attr: ",
        c->num_inputs(), " vs ", input_shapes.size());
  }
  bool input_match = true;
  for (int i = 0; i < c->num_inputs(); ++i) {
    ShapeHandle handle;
    TF_RETURN_IF_ERROR(
        c->MakeShapeFromTensorShape(input_shapes.at(i), &handle));
    ShapeHandle merged;
    if (!c->Merge(c->input(i), handle, &merged).ok()) {
      // Input shape doesn't match what was set in attr, fine.
      input_match = false;
    }
  }

  // Check the sanity of the output shapes.
  std::vector<tensorflow::TensorShape> output_shapes;
  TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));
  if (output_shapes.size() != c->num_outputs()) {
    return tensorflow::errors::InvalidArgument(
        "The actual number of outputs doesn't match the number of output "
        "shapes set in the attr: ",
        c->num_outputs(), " vs ", output_shapes.size());
  }
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    ShapeHandle handle;
    TF_RETURN_IF_ERROR(
        c->MakeShapeFromTensorShape(output_shapes.at(i), &handle));
    if (input_match) c->set_output(i, handle);
  }
  return Status::OK();
}

}  // namespace shape_inference
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
