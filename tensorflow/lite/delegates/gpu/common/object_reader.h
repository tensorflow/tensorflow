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
#include <cstring>
#include <vector>

#include "fp16.h"  // from @FP16
#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder_helper.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h"
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

  absl::Status GetTensorId(uint32_t input_id, int* tensor_id) const;

  absl::Status GetTensorDims(uint32_t idx, TfLiteIntArray* dimensions) const;

  template <typename TensorT>
  absl::Status ReadTensor(uint32_t index, TensorT* tensor) const {
    if (index < 0 || index >= node_->inputs->size) {
      // If larger, this can be an older model with fewer input tensors than the
      // current implementation.
      return absl::OutOfRangeError("Invalid data index found.");
    }
    const int32_t tensor_id = node_->inputs->data[index];
    if (tensor_id < 0) {
      return absl::InvalidArgumentError(
          "Invalid data index found. Possibly an unset optional tensor is "
          "being read.");
    }
    const TfLiteTensor* tflite_tensor = context_->tensors + tensor_id;
    tensor->data.resize(NumElements(tflite_tensor));
    if (tflite_tensor->sparsity) {
      std::vector<int> dims;
      dims.reserve(tflite_tensor->dims->size);
      for (int i = 0; i < tflite_tensor->dims->size; ++i) {
        dims.push_back(tflite_tensor->dims->data[i]);
      }
      switch (tflite_tensor->type) {
        case kTfLiteFloat32: {
          internal::sparsity::FormatConverter<float> converter(
              dims, *tflite_tensor->sparsity);
          converter.SparseToDense(
              static_cast<const float*>(tflite_tensor->data.data));
          const std::vector<float> out = converter.GetData();
          std::memcpy(&tensor->data[0], out.data(), out.size() * sizeof(float));
          break;
        }
        case kTfLiteFloat16: {
          internal::sparsity::FormatConverter<Eigen::half> converter(
              dims, *tflite_tensor->sparsity);
          converter.SparseToDense(
              static_cast<const Eigen::half*>(tflite_tensor->data.data));
          const std::vector<Eigen::half> out = converter.GetData();
          std::transform(out.begin(), out.end(), tensor->data.begin(),
                         [](const Eigen::half& x) {
                           return fp16_ieee_to_fp32_value(
                               Eigen::numext::bit_cast<uint16_t>(x));
                         });
          break;
        }
        default: {
          return absl::InvalidArgumentError(
              "Unexpected data type in sparse tensor");
        }
      }
    } else {
      RETURN_IF_ERROR(CreateVectorCopyData(*tflite_tensor, &tensor->data[0]));
    }

    // Axis and data layout depend on operation this tensor is used in. So,
    // postpone resolutions until operations are parsed.
    tensor->id = tensor_id;
    return SetAllDimensions(tflite_tensor->dims, &tensor->shape);
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
