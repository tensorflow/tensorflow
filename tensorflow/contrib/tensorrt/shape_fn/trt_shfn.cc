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
#include "tensorflow/contrib/tensorrt/log/trt_logger.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace shape_inference {

tensorflow::Status TRTEngineOpShapeInference(InferenceContext* context) {
  tensorflow::tensorrt::Logger logger;
  string serialized_engine;
  TF_RETURN_IF_ERROR(context->GetAttr("serialized_engine", &serialized_engine));
  nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(logger);
  nvinfer1::ICudaEngine* trt_engine = infer->deserializeCudaEngine(
      serialized_engine.c_str(), serialized_engine.size(), nullptr);

  int num_batch = -1;
  std::vector<::tensorflow::DataType> input_type;
  TF_RETURN_IF_ERROR(context->GetAttr("InT", &input_type));
  for (size_t i = 0; i < context->num_inputs(); i++) {
    // Check if input shape is legit
    auto input_shape = context->input(i);
    for (int j = 0; j < context->Rank(input_shape); j++) {
      auto dim_handler = context->Dim(input_shape, j);
      if (j == 0) {
        if (i == 0) {
          num_batch = context->Value(dim_handler);
        } else if (num_batch != context->Value(dim_handler)) {
          // TODO(jie): TensorRT engine requires consistent batch between inputs
          //            tensors. Segmenter should be aware of this.
          LOG(FATAL) << "TensorRT engine requires consistent batch size";
        }
      }
    }
  }

  // Arrange input here
  std::vector<string> input_nodes;
  TF_RETURN_IF_ERROR(context->GetAttr("input_nodes", &input_nodes));

  // Arrange output here
  std::vector<string> output_nodes;
  TF_RETURN_IF_ERROR(context->GetAttr("output_nodes", &output_nodes));
  for (size_t i = 0; i < output_nodes.size(); i++) {
    int binding_index = trt_engine->getBindingIndex(output_nodes[i].c_str());
    ShapeHandle output_shape;
    std::vector<DimensionHandle> dim_vec;
    dim_vec.emplace_back(context->MakeDim(num_batch));
    if (binding_index != -1) {
      auto dims = trt_engine->getBindingDimensions(binding_index);
      for (int j = 0; j < dims.nbDims; j++) {
        dim_vec.emplace_back(context->MakeDim(dims.d[j]));
      }
    } else {
      LOG(FATAL) << "TensorRT engine cannot find binding: " << output_nodes[i];
    }
    output_shape = context->MakeShape(dim_vec);
    context->set_output(i, output_shape);
  }

  return Status::OK();
}

}  // namespace shape_inference
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
