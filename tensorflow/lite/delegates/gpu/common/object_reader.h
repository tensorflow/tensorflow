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

#include "absl/container/flat_hash_map.h"
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
      TfLiteContext* context, absl::flat_hash_map<int, Value*>* tensor_to_value,
      absl::flat_hash_map<int, int>* quant_conversion_map, GraphFloat32* graph,
      uint32_t tensor_idx, Value** value = nullptr);

  ObjectReader(GraphFloat32* graph, TfLiteContext* context,
               const TfLiteNode* node,
               absl::flat_hash_map<int, Value*>* tensor_to_value,
               absl::flat_hash_map<int, int>* quant_conversion_map = nullptr)
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
    if (idx < 0 || idx >= node_->inputs->size) {
      // If larger, this can be an older model with fewer input tensors than the
      // current implementation.
      return absl::OutOfRangeError("Invalid data index found.");
    }
    const int32_t tensor_idx = node_->inputs->data[idx];
    if (tensor_idx < 0) {
      return absl::InvalidArgumentError(
          "Invalid data index found. Possibly an unset optional tensor is "
          "being read.");
    }

    const TfLiteTensor* tflite_tensor = context_->tensors + tensor_idx;
    if (tflite_tensor->sparsity != nullptr) {
      return absl::InvalidArgumentError("Sparsity is not supported on GPU.");
    }
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

  absl::Status AddUpdate(const Node* node, uint32_t idx);

  TfLiteTensor* GetInputTensor(int index) const;

  TfLiteTensor* GetOutputTensor(int index) const;

  absl::Status VerifyInputsConstsOutputs(const TfLiteNode* node,
                                         int runtime_inputs, int const_inputs,
                                         int outputs);

 private:
  GraphFloat32* graph_;
  TfLiteContext* context_;
  const TfLiteNode* node_;
  absl::flat_hash_map<int, Value*>* tensor_to_value_;
  absl::flat_hash_map<int, int>* quant_conversion_map_;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_OBJECT_READER_H_
