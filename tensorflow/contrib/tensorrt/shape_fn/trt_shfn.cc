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

#include <string>
#include <vector>

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorflow/contrib/tensorrt/log/trt_logger.h"
#include "tensorflow/contrib/tensorrt/shape_fn/trt_shfn.h"
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace shape_inference {
tensorflow::Status TRTEngineOpShapeInference(InferenceContext* c) {
  tensorflow::tensorrt::Logger gLogger;
  string serialized_engine;
  c->GetAttr("serialized_engine", &serialized_engine);
  nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(gLogger);
  nvinfer1::ICudaEngine* trt_engine = infer->deserializeCudaEngine(
      serialized_engine.c_str(), serialized_engine.size(), nullptr);


  int nbBatch = -1;
  // debug print out input arrays
  std::vector<::tensorflow::DataType> input_type;
  c->GetAttr("InT", &input_type);
  for (size_t i = 0; i < c->num_inputs(); i++) {
    // check if input shape is legit
    auto input_shape = c->input(i);
    for (int j = 0; j < c->Rank(input_shape); j++) {
      auto dimHandler = c->Dim(input_shape, j);
      if (j == 0) {
        if (i == 0)
          nbBatch = c->Value(dimHandler);
        else if (nbBatch != c->Value(dimHandler))
          // TODO(jie): TensorRT engine requires consistent batch between inputs
          //            tensors. Segmenter should be aware of this.
          LOG(FATAL) << "TensorRT engine requires consistent batch size";
      }
    }
  }

  // arrange input here
  std::vector<string> input_nodes;
  c->GetAttr("input_nodes", &input_nodes);

  // arrange output here
  std::vector<string> output_nodes;
  c->GetAttr("output_nodes", &output_nodes);
  for (size_t i = 0; i < output_nodes.size(); i++) {
    int binding_index =
        trt_engine->getBindingIndex(output_nodes[i].c_str());
    ShapeHandle output_shape;
    std::vector<DimensionHandle> vecDim;
    vecDim.emplace_back(c->MakeDim(nbBatch));
    if (binding_index != -1) {
      auto dims = trt_engine->getBindingDimensions(binding_index);
      for (int j = 0; j < dims.nbDims; j++)
        vecDim.emplace_back(c->MakeDim(dims.d[j]));
    } else {
      LOG(FATAL) << "TensorRT engine cannot find binding: "
                 << output_nodes[i];
    }
    output_shape = c->MakeShape(vecDim);
    c->set_output(i, output_shape);
  }

  return Status::OK();
}
}  // namespace shape_inference
}  // namespace tensorflow

#endif // GOOGLE_TENSORRT
#endif // GOOGLE_CUDA
