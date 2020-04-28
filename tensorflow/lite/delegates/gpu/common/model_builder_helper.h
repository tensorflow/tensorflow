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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_HELPER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_HELPER_H_

#include <set>
#include <string>
#include <unordered_map>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/kernels/internal/reference/dequantize.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace gpu {

class GraphWithDequantPartitionHelper : public delegates::GraphPartitionHelper {
 public:
  GraphWithDequantPartitionHelper(
      TfLiteContext* context, delegates::IsNodeSupportedFn is_node_supported_fn)
      : GraphPartitionHelper(context, std::move(is_node_supported_fn)) {}

  TfLiteStatus Partition(
      std::set<std::string>* unsupported_nodes_info) override;

  // Returns a list of node indices of all nodes from the first n largest
  // partitions. If there are fewer paritions than n, all nodes will be
  // returned. The partition is ranked according to the number of nodes.
  std::vector<int> GetNodesOfFirstNLargestPartitions(int n);

 protected:
  bool IsNodeSupported(TfLiteContext* context, TfLiteNode* node,
                       TfLiteRegistration* registration, int node_id,
                       std::string* unsupported_details) override;

 private:
  // Record 'node' if it is a dequant op (i.e. a fp16 one here) and return true.
  // When it's not a dequant op, remap its inputs to the inputs of the preceding
  // dequant if there's a one and returns false. 'orig_inputs' records original
  // input tensor ids of this node if any input is remapped.
  bool RecordAndRemapInputTensors(int32_t op_code, int node_id,
                                  TfLiteNode* node,
                                  std::vector<int>* orig_inputs);

  // Restore inputs of 'node' to 'orig_inputs' only if two sizes match.
  void RestoreToOrigInputTensors(TfLiteNode* node,
                                 const std::vector<int>& orig_inputs);

  // Remap input tensors of every node in 'nodes' (i.e. node indices) if some of
  // them are from dequant ops.
  void RemapInputTensors(const std::vector<int>& nodes) const;

  void RemoveSingleDequantNodePartitions();

  void RemoveReservedDequantsFromNodes(std::vector<int>* nodes);

  // Remap input tensors of a single 'node' if some of come from a dequant op.
  // If 'orig_inputs' isn't nullptr, it records original input tensor ids of
  // this node if any input is remapped.
  void RemapInputTensors(TfLiteNode* node, std::vector<int>* orig_inputs) const;

  // A map recording dequantize nodes's input/output tensors of this selected
  // graph. The key is the output tensor id, and the value is the input tensor
  // id.
  std::unordered_map<int, int> dequant_nodes_;

  // A set of dequant nodes as in node indices that have to be preserved in the
  // graph.
  std::set<int> dequant_nodes_to_save_;
};

absl::Status GetNodeAndRegistration(TfLiteContext* context, int node_id,
                                    TfLiteNode** tflite_node,
                                    TfLiteRegistration** registration);

DataType ToDataType(TfLiteType type);

absl::Status ExtractTensorShape(const TfLiteTensor& tflite_tensor, BHWC* bhwc);

absl::Status ConvertTfLiteTensorToTensorRef(const TfLiteTensor& tflite_tensor,
                                            TensorRef<BHWC>* tensor_ref);

// Populates quantization parameters for non-constant UInt8/Int8 tensors.
// This helps the delegate emulate quantized inference with
// QuantizeAndDequantize.
absl::Status PopulateQuantParams(const TfLiteTensor& tensor,
                                 QuantizationParams* quant_params);

int GetNumberOfRuntimeInputsForNode(const TfLiteContext* context,
                                    const TfLiteNode* tflite_node);

int GetNumberOfConstInputsForNode(const TfLiteContext* context,
                                  const TfLiteNode* tflite_node);

int GetNumberOfRuntimeOutputsForNode(const TfLiteContext* context,
                                     const TfLiteNode* tflite_node);

absl::Status CheckInputsOutputs(const TfLiteContext* context,
                                const TfLiteNode* tflite_node,
                                int runtime_inputs, int outputs);

absl::Status CheckInputsConstsOutputs(const TfLiteContext* context,
                                      const TfLiteNode* tflite_node,
                                      int runtime_inputs, int const_inputs,
                                      int outputs);

void ConvertFloat16ToFloat32(size_t num_elements, const uint16_t* src,
                             float* dst);

template <typename T>
void DequantizeConstantTensor(const TfLiteTensor& tensor, const T* source_data,
                              float* dequantized_data) {
  TfLiteAffineQuantization* quant_params =
      reinterpret_cast<TfLiteAffineQuantization*>(tensor.quantization.params);
  if (quant_params->scale->size > 1) {
    // Tensor is per-channel quantized.
    PerChannelDequantizationParams op_params;
    op_params.zero_point = quant_params->zero_point->data;
    op_params.scale = quant_params->scale->data;
    op_params.quantized_dimension = quant_params->quantized_dimension;
    reference_ops::PerChannelDequantize(op_params, GetTensorShape(&tensor),
                                        source_data, GetTensorShape(&tensor),
                                        dequantized_data);
  } else {
    DequantizationParams op_params;
    op_params.zero_point = tensor.params.zero_point;
    op_params.scale = tensor.params.scale;
    reference_ops::Dequantize(op_params, GetTensorShape(&tensor), source_data,
                              GetTensorShape(&tensor), dequantized_data);
  }
}

template <typename T>
absl::Status CreateVectorCopyData(const TfLiteTensor& tensor, T* tensor_data) {
  if (tensor.bytes % sizeof(T) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input data size ", tensor.bytes,
                     " is not aligned to expected type: ", sizeof(T)));
  }
  std::memcpy(tensor_data, tensor.data.uint8, tensor.bytes);
  return absl::OkStatus();
}

template <>
absl::Status CreateVectorCopyData<float>(const TfLiteTensor& tensor,
                                         float* tensor_data);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, Scalar* shape);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, Linear* shape);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, HWC* shape);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, HW* shape);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, OHWI* shape);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, BHWC* shape);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_HELPER_H_
