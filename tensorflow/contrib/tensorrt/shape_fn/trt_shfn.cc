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
#include "NvInfer.h"
#include "tensorflow/contrib/tensorrt/log/trt_logger.h"

namespace tensorflow {
namespace shape_inference {
tensorflow::Status TRTEngineOpShapeInference(InferenceContext* c) {
  tensorflow::tensorrt::Logger gLogger;
  string serialized_engine;
  c->GetAttr("serialized_engine", &serialized_engine);
  nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(gLogger);
  nvinfer1::ICudaEngine* trt_engine = infer->deserializeCudaEngine(
      serialized_engine.c_str(), serialized_engine.size(), nullptr);

  // debug print out engine binding;
  std::stringstream oss;
  for (int i = 0; i < trt_engine->getNbBindings(); i++) {
    LOG(INFO) << "index: " << i
              << ", binding name: " << trt_engine->getBindingName(i);

    bool input_flag = trt_engine->bindingIsInput(i);
    oss << "input?: " << (input_flag ? "Y" : "N");

    oss << "Dimension: ";
    auto dims = trt_engine->getBindingDimensions(i);
    oss << " nbDims: " << dims.nbDims << " -> ";
    for (int j = 0; j < dims.nbDims; j++) oss << dims.d[j] << ", ";
    LOG(INFO) << oss.str();
    oss.str("");
    switch (trt_engine->getBindingDataType(i)) {
      case nvinfer1::DataType::kFLOAT:
        LOG(INFO) << "data type: float" << std::endl;
        break;
      case nvinfer1::DataType::kHALF:
        LOG(INFO) << "data type: half" << std::endl;
        break;
      case nvinfer1::DataType::kINT8:
        LOG(INFO) << "data type: int8" << std::endl;
        break;
    }
  }

  int nbBatch = -1;
  // debug print out input arrays
  std::vector<::tensorflow::DataType> input_type;
  c->GetAttr("InT", &input_type);
  oss.str("");
  for (size_t i = 0; i < c->num_inputs(); i++) {
    // check if input shape is legit
    auto input_shape = c->input(i);
    int index = i;
    oss << "input:" << i << " type: " << input_type[index] << " shape: ";
    for (int j = 0; j < c->Rank(input_shape); j++) {
      auto dimHandler = c->Dim(input_shape, j);
      if (c->ValueKnown(dimHandler))
        oss << c->Value(dimHandler) << ", ";
      else
        oss << "?" << c->Value(dimHandler) << ", ";
      if (j == 0) {
        if (i == 0)
          nbBatch = c->Value(dimHandler);
        else if (nbBatch != c->Value(dimHandler))
          LOG(WARNING) << "!!!!!!nbBatch does not match!!!!!!";
        // assert(nbBatch == c->Value(dimHandler);
      }
    }
    LOG(INFO) << oss.str();
  }

  // arrange input here
  std::vector<string> input_nodes;
  c->GetAttr("input_nodes", &input_nodes);
  for (size_t i = 0; i < input_nodes.size(); i++) {
    int index = i;
    LOG(INFO) << "input:" << i << " name: " << input_nodes[index];
  }

  // arrange output here
  std::vector<string> output_nodes;
  c->GetAttr("output_nodes", &output_nodes);
  oss.str("");
  for (size_t i = 0; i < output_nodes.size(); i++) {
    int index = i;
    int binding_index =
        trt_engine->getBindingIndex(output_nodes[index].c_str());
    oss << "string name " << output_nodes[index];
    ShapeHandle output_shape;
    std::vector<DimensionHandle> vecDim;
    vecDim.emplace_back(c->MakeDim(nbBatch));
    if (binding_index != -1) {
      oss << "got binding " << binding_index;
      auto dims = trt_engine->getBindingDimensions(binding_index);
      for (int j = 0; j < dims.nbDims; j++)
        vecDim.emplace_back(c->MakeDim(dims.d[j]));
    } else {
      oss << "no binding ";
    }
    output_shape = c->MakeShape(vecDim);
    c->set_output(i, output_shape);
    LOG(INFO) << oss.str();
  }

  return Status::OK();
}
}  // namespace shape_inference
}  // namespace tensorflow
