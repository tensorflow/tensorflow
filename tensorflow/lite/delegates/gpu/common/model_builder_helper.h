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

#include <stddef.h>

#include <cstdint>
#include <cstring>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/kernels/internal/reference/dequantize.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace gpu {

absl::Status GetNodeAndRegistration(TfLiteContext* context, int node_id,
                                    TfLiteNode** tflite_node,
                                    TfLiteRegistration** registration);

DataType ToDataType(TfLiteType type);

absl::Status ExtractTensorShape(const TfLiteTensor& tflite_tensor, BHWC* bhwc);

absl::Status ExtractAxisFromIndex(const TfLiteTensor& tflite_tensor, int index,
                                  Axis* axis);

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
inline void DequantizeConstantTensor(const TfLiteTensor& tensor,
                                     const T* source_data,
                                     float* dequantized_data) {
  TfLiteAffineQuantization* quant_params =
      static_cast<TfLiteAffineQuantization*>(tensor.quantization.params);
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
absl::Status CreateVectorCopyData(const TfLiteTensor& src, T* dst) {
  if (src.bytes % sizeof(T) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input data size ", src.bytes,
                     " is not aligned to expected type: ", sizeof(T)));
  }
  if (const int n = tflite::NumElements(&src); n * sizeof(T) == src.bytes) {
    std::memcpy(dst, src.data.raw_const, src.bytes);
    return absl::OkStatus();
  } else {
    switch (src.type) {
      case kTfLiteNoType:
        return absl::InvalidArgumentError("src has no type.");
      case kTfLiteFloat32:
        for (int i = 0; i < n; ++i) {
          dst[i] = tflite::GetTensorData<float>(&src)[i];
        }
        return absl::OkStatus();
      case kTfLiteInt32:
        for (int i = 0; i < n; ++i) {
          dst[i] = tflite::GetTensorData<int32_t>(&src)[i];
        }
        return absl::OkStatus();
      case kTfLiteUInt8:
        for (int i = 0; i < n; ++i) {
          dst[i] = tflite::GetTensorData<uint8_t>(&src)[i];
        }
        return absl::OkStatus();
      case kTfLiteInt64:
        for (int i = 0; i < n; ++i) {
          dst[i] = tflite::GetTensorData<int64_t>(&src)[i];
        }
        return absl::OkStatus();
      case kTfLiteString:
        return absl::UnimplementedError("src can't be string.");
      case kTfLiteBool:
        for (int i = 0; i < n; ++i) {
          dst[i] = tflite::GetTensorData<bool>(&src)[i];
        }
        return absl::OkStatus();
      case kTfLiteInt16:
        for (int i = 0; i < n; ++i) {
          dst[i] = tflite::GetTensorData<int16_t>(&src)[i];
        }
        return absl::OkStatus();
      case kTfLiteComplex64:
        return absl::UnimplementedError("src can't be complex64.");
      case kTfLiteInt8:
        for (int i = 0; i < n; ++i) {
          dst[i] = tflite::GetTensorData<int8_t>(&src)[i];
        }
        return absl::OkStatus();
      case kTfLiteFloat16:
        return absl::UnimplementedError("src can't be float16.");
      case kTfLiteBFloat16:
        return absl::UnimplementedError("src can't be bfloat16.");
      case kTfLiteFloat64:
        for (int i = 0; i < n; ++i) {
          dst[i] = tflite::GetTensorData<double>(&src)[i];
        }
        return absl::OkStatus();
      case kTfLiteComplex128:
        return absl::UnimplementedError("src can't be complex128.");
      case kTfLiteUInt64:
        for (int i = 0; i < n; ++i) {
          dst[i] = tflite::GetTensorData<uint64_t>(&src)[i];
        }
        return absl::OkStatus();
      case kTfLiteResource:
        return absl::UnimplementedError("src can't be resource.");
      case kTfLiteVariant:
        return absl::UnimplementedError("src can't be variant.");
      case kTfLiteUInt32:
        for (int i = 0; i < n; ++i) {
          dst[i] = tflite::GetTensorData<uint32_t>(&src)[i];
        }
        return absl::OkStatus();
      case kTfLiteUInt16:
        for (int i = 0; i < n; ++i) {
          dst[i] = tflite::GetTensorData<uint16_t>(&src)[i];
        }
        return absl::OkStatus();
      case kTfLiteInt4:
        return absl::UnimplementedError("src can't be int4.");
    }
  }
}

template <>
absl::Status CreateVectorCopyData<float>(const TfLiteTensor& src, float* dst);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, Scalar* shape);

absl::Status CheckIfLinearConvertible(const TfLiteIntArray* dimensions);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, Linear* shape);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, HWC* shape);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, HW* shape);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, OHWI* shape);

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, BHWC* shape);

// If there is fused activation present, then there will be another node created
// that will have identical output as the given node. New operation node will
// depend on the given node output.
absl::Status MaybeFuseActivation(TfLiteFusedActivation fused_activation,
                                 GraphFloat32* graph, Node* node);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_HELPER_H_
