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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_OBJECT_READER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_OBJECT_READER_H_

#include <cstdint>
#include <unordered_map>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder_helper.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace gpu {

// If quantized tensors exist in the graph & quant_conversion_map is non-null,
// the mapping between the original tensors (fixed-point) & GPU values (fp) is
// stored in quant_conversion_map.
class ObjectReader {
 public:
  static absl::Status ReadNonConstantTensor(
      TfLiteContext* context, std::unordered_map<int, Value*>* tensor_to_value,
      std::unordered_map<int, int>* quant_conversion_map, GraphFloat32* graph,
      uint32_t tensor_idx, Value** value = nullptr);

  ObjectReader(GraphFloat32* graph, TfLiteContext* context,
               const TfLiteNode* node,
               std::unordered_map<int, Value*>* tensor_to_value,
               std::unordered_map<int, int>* quant_conversion_map = nullptr)
      : graph_(graph),
        context_(context),
        node_(node),
        tensor_to_value_(tensor_to_value),
        quant_conversion_map_(quant_conversion_map) {}

  absl::Status ReadValue(uint32_t idx, Value** value);

  absl::Status ReadValueByTensorIdx(uint32_t tensor_idx, Value** value);

  int GetNumberOfRuntimeInputs() const;

  absl::Status GetTensorDims(uint32_t idx, TfLiteIntArray* dimensions) const;

  template <typename TensorT>
  absl::Status ReadTensor(uint32_t idx, TensorT* t) const {
    const int32_t tensor_idx = node_->inputs->data[idx];
    const TfLiteTensor* tflite_tensor = context_->tensors + tensor_idx;
    t->data.resize(NumElements(tflite_tensor));
    RETURN_IF_ERROR(CreateVectorCopyData(*tflite_tensor, &t->data[0]));

    // Axis and data layout depend on operation this tensor is used in. So,
    // postpone resolutions until operations are parsed.
    t->id = tensor_idx;
    return SetAllDimensions(tflite_tensor->dims, &t->shape);
  }

  absl::Status AddOutput(const Node* node, int id);

  absl::Status AddOutputs(const Node* node);

  absl::Status AddInput(const Node* node, uint32_t idx);

  TfLiteTensor* GetInputTensor(int index) const;

  TfLiteTensor* GetOutputTensor(int index) const;

  absl::Status VerifyInputsConstsOutputs(const TfLiteNode* node,
                                         int runtime_inputs, int const_inputs,
                                         int outputs);

 private:
  GraphFloat32* graph_;
  TfLiteContext* context_;
  const TfLiteNode* node_;
  std::unordered_map<int, Value*>* tensor_to_value_;
  std::unordered_map<int, int>* quant_conversion_map_;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_OBJECT_READER_H_
