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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SELECTORS_SUBGRAPH_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SELECTORS_SUBGRAPH_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"

namespace tflite {
namespace gpu {

struct GPUOperationWithRefs {
  std::unique_ptr<GPUOperation> operation;

  // input and output ids can be positive or negative.
  // if we have positive id, we will use preallocated tensor from GraphFloat32
  // otherwise, we will use ids for newly allocated tensors
  std::vector<int> input_ids;
  std::vector<int> output_ids;
  std::string name;
};

struct GPUOperationsSubgraph {
  std::vector<GPUOperationWithRefs> operations;
  std::vector<TensorDescriptor> new_tensors;

  int AddTensor(const TensorDescriptor& desc);
  int AddTensor(const BHWC& shape, const TensorDescriptor& desc);
  int AddTensor(const OHWI& shape, const TensorDescriptor& desc);
};

std::unique_ptr<GPUOperation>* InitSingleOpSubgraph(
    const std::vector<Value*>& inputs, const std::vector<Value*>& outputs,
    GPUOperationsSubgraph* gpu_subgraph);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SELECTORS_SUBGRAPH_H_
