/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/selectors/subgraph.h"

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"

namespace tflite {
namespace gpu {

int GPUOperationsSubgraph::AddTensor(const TensorDescriptor& desc) {
  new_tensors.push_back(desc);
  return -new_tensors.size();
}

int GPUOperationsSubgraph::AddTensor(const BHWC& shape,
                                     const TensorDescriptor& desc) {
  TensorDescriptor desc_with_shape = desc;
  desc_with_shape.SetBHWCShape(shape);
  return AddTensor(desc_with_shape);
}

int GPUOperationsSubgraph::AddTensor(const OHWI& shape,
                                     const TensorDescriptor& desc) {
  const BHWC shape_as_bhwc(shape.o, shape.h, shape.w, shape.i);
  return AddTensor(shape_as_bhwc, desc);
}

std::unique_ptr<GPUOperation>* InitSingleOpSubgraph(
    const std::vector<Value*>& inputs, const std::vector<Value*>& outputs,
    GPUOperationsSubgraph* gpu_subgraph) {
  gpu_subgraph->operations.clear();
  gpu_subgraph->new_tensors.clear();
  gpu_subgraph->operations.push_back({});
  for (int i = 0; i < inputs.size(); ++i) {
    gpu_subgraph->operations[0].input_ids.push_back(inputs[i]->id);
  }
  for (int i = 0; i < outputs.size(); ++i) {
    gpu_subgraph->operations[0].output_ids.push_back(outputs[i]->id);
  }

  return &gpu_subgraph->operations[0].operation;
}

}  // namespace gpu
}  // namespace tflite
