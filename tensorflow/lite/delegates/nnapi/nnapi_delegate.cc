/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// This section needs to be before the import of nnapi_delegate_kernel
// because the code changes according to  the definition of
// TFLITE_NNAPI_ALLOW_MMAP_SHARING
#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif
#if defined __ANDROID__ || defined __unix__
#define TFLITE_NNAPI_ALLOW_MMAP_SHARING
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_kernel.h"
#include "tensorflow/lite/delegates/nnapi/quant_lstm_sup.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/nnapi/nnapi_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace {

// TODO(b/80621585): Consider printing error string, but don't for now to
// minimize binary size.
#define RETURN_TFLITE_ERROR_IF_NN_ERROR(context, code, p_errno)               \
  do {                                                                        \
    const auto _code = (code);                                                \
    if (_code != ANEURALNETWORKS_NO_ERROR) {                                  \
      context->ReportError(context, "NN API returned error (%d, line %d).\n", \
                           _code, __LINE__);                                  \
      *p_errno = _code;                                                       \
      return kTfLiteError;                                                    \
    }                                                                         \
  } while (0)

bool IsFloat(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return true;
    default:
      return false;
  }
}

bool IsFloatOrUInt8(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
    case kTfLiteUInt8:
      return true;
    default:
      return false;
  }
}

bool IsQuantized(TfLiteType type) {
  switch (type) {
    case kTfLiteUInt8:
    case kTfLiteInt8:
      return true;
    default:
      // kTfLiteInt16 isn't supported as quantized type yet.
      return false;
  }
}

bool IsScalarInputSupported(int builtin_code) {
  switch (builtin_code) {
    case kTfLiteBuiltinAdd:
    case kTfLiteBuiltinMul:
    case kTfLiteBuiltinSub:
    case kTfLiteBuiltinDiv:
    case kTfLiteBuiltinEqual:
    case kTfLiteBuiltinNotEqual:
    case kTfLiteBuiltinGreater:
    case kTfLiteBuiltinGreaterEqual:
    case kTfLiteBuiltinLess:
    case kTfLiteBuiltinLessEqual:
      return true;
    default:
      return false;
  }
}

// Check if the operation requires explict conversion from int8 to uint8 values.
bool NeedInt8Conversion(const TfLiteContext* context, int builtin_code,
                        const TfLiteNode* node) {
  const int input_id = node->inputs->data[0];
  const TfLiteType input_type = context->tensors[input_id].type;
  switch (builtin_code) {
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinDepthwiseConv2d:
    case kTfLiteBuiltinFullyConnected: {
      if (input_type == kTfLiteInt8) {
        const int weights_id = node->inputs->data[1];
        const auto& weights_tensor = context->tensors[weights_id];
        if ((weights_tensor.type == kTfLiteInt8 ||
             weights_tensor.type == kTfLiteUInt8) &&
            weights_tensor.quantization.type == kTfLiteAffineQuantization) {
          return true;
        }
      }
      return false;
    }
    case kTfLiteBuiltinSelect: {
      const auto value_type = context->tensors[node->inputs->data[1]].type;
      return value_type == kTfLiteInt8;
    }
    case kTfLiteBuiltinAdd:
    case kTfLiteBuiltinArgMax:
    case kTfLiteBuiltinArgMin:
    case kTfLiteBuiltinAveragePool2d:
    case kTfLiteBuiltinBatchToSpaceNd:
    case kTfLiteBuiltinConcatenation:
    case kTfLiteBuiltinEqual:
    case kTfLiteBuiltinExpandDims:
    case kTfLiteBuiltinGreater:
    case kTfLiteBuiltinGreaterEqual:
    case kTfLiteBuiltinHardSwish:
    case kTfLiteBuiltinL2Normalization:
    case kTfLiteBuiltinLess:
    case kTfLiteBuiltinLessEqual:
    case kTfLiteBuiltinLogistic:
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinMaxPool2d:
    case kTfLiteBuiltinMean:
    case kTfLiteBuiltinMinimum:
    case kTfLiteBuiltinMul:
    case kTfLiteBuiltinNotEqual:
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinPadv2:
    case kTfLiteBuiltinReduceMax:
    case kTfLiteBuiltinReduceMin:
    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinReluN1To1:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinResizeBilinear:
    case kTfLiteBuiltinResizeNearestNeighbor:
    case kTfLiteBuiltinReshape:
    case kTfLiteBuiltinSlice:
    case kTfLiteBuiltinSoftmax:
    case kTfLiteBuiltinSpaceToBatchNd:
    case kTfLiteBuiltinSpaceToDepth:
    case kTfLiteBuiltinDepthToSpace:
    case kTfLiteBuiltinStridedSlice:
    case kTfLiteBuiltinSub:
    case kTfLiteBuiltinTanh:
    case kTfLiteBuiltinTile:
    case kTfLiteBuiltinTopkV2:
    case kTfLiteBuiltinTranspose: {
      return input_type == kTfLiteInt8;
    }
    default:
      return false;
  }
}

constexpr int kLstmFullKernelInputSize = 24;
// The 20 input version is deprecated and kept only to
// support old model. The latest version of the LSTM Full Kernel
// is the one with 24 inputs
constexpr int kLstmFullKernelNoOptionalParamsInputSize = 20;
constexpr int kLstmBasicKernelInputSize = 5;

inline bool isLstmBasicKernel(const TfLiteNode* node) {
  return node->inputs->size == kLstmBasicKernelInputSize;
}

inline bool isLstmFullKernel(const TfLiteNode* node) {
  return node->inputs->size == kLstmFullKernelInputSize ||
         node->inputs->size == kLstmFullKernelNoOptionalParamsInputSize;
}

bool IsHybridOperator(const TfLiteContext* context, int builtin_code,
                      const TfLiteNode* node) {
  switch (builtin_code) {
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinFullyConnected: {
      const int input_id = node->inputs->data[0];
      const int filter_id = node->inputs->data[1];
      const TfLiteType input_type = context->tensors[input_id].type;
      const TfLiteType filter_type = context->tensors[filter_id].type;
      return IsFloat(input_type) && IsQuantized(filter_type);
    }
    case kTfLiteBuiltinLstm: {
      const int input_id = node->inputs->data[0];
      // Input #1 is optional so use #2 to determine if hybrid.
      const int weights_id = node->inputs->data[2];
      const TfLiteType input_type = context->tensors[input_id].type;
      const TfLiteType weights_type = context->tensors[weights_id].type;
      return isLstmFullKernel(node) && IsFloat(input_type) &&
             IsQuantized(weights_type);
    }
    case kTfLiteBuiltinUnidirectionalSequenceLstm: {
      const int input_id = node->inputs->data[0];
      // Input #1 is optional so use #2 to determine if hybrid.
      const int weights_id = node->inputs->data[2];
      const TfLiteType input_type = context->tensors[input_id].type;
      const TfLiteType weights_type = context->tensors[weights_id].type;
      return IsFloat(input_type) && IsQuantized(weights_type);
    }
    case kTfLiteBuiltinBidirectionalSequenceLstm: {
      const int input_id = node->inputs->data[0];
      // Input #1 is optional so use #2 to determine if hybrid.
      const int weights_id = node->inputs->data[2];
      const TfLiteType input_type = context->tensors[input_id].type;
      const TfLiteType weights_type = context->tensors[weights_id].type;
      return IsFloat(input_type) && IsQuantized(weights_type);
    }
    case kTfLiteBuiltinUnidirectionalSequenceRnn: {
      const int input_id = node->inputs->data[0];
      const int weights_id = node->inputs->data[1];
      const TfLiteType input_type = context->tensors[input_id].type;
      const TfLiteType weights_type = context->tensors[weights_id].type;
      return IsFloat(input_type) && IsQuantized(weights_type);
    }
    default:
      return false;
  }
}

constexpr size_t kDefaultByteAlignmentForNNAPI = 16;

static size_t getNumPaddingBytes(size_t byte_size) {
  size_t num_padding_bytes = 0;
  if (byte_size % kDefaultByteAlignmentForNNAPI) {
    num_padding_bytes = kDefaultByteAlignmentForNNAPI -
                        (byte_size % kDefaultByteAlignmentForNNAPI);
  }
  return num_padding_bytes;
}

// Return NNAPI device handle with the provided null-terminated device name. If
// no matching device could be found, nullptr will be returned.
ANeuralNetworksDevice* GetDeviceHandle(TfLiteContext* context,
                                       const char* device_name_ptr) {
  if (!device_name_ptr) return nullptr;
  ANeuralNetworksDevice* device_handle = nullptr;
  std::string device_name(device_name_ptr);
  uint32_t num_devices = 0;
  NnApiImplementation()->ANeuralNetworks_getDeviceCount(&num_devices);

  for (uint32_t i = 0; i < num_devices; i++) {
    ANeuralNetworksDevice* device = nullptr;
    const char* buffer = nullptr;
    NnApiImplementation()->ANeuralNetworks_getDevice(i, &device);
    NnApiImplementation()->ANeuralNetworksDevice_getName(device, &buffer);
    if (device_name == buffer) {
      device_handle = device;
      break;
    }
  }
  if (!device_handle) {
    context->ReportError(context,
                         "Could not find the specified NNAPI accelerator: %s. "
                         "Must be one of: {%s}.",
                         device_name_ptr,
                         nnapi::GetStringDeviceNamesList().c_str());
  }
  return device_handle;
}

// Compute the hash of a TfLiteIntArray.
uint64_t GetHash(const TfLiteIntArray* int_array) {
  constexpr auto kHashConst = 0x9e3779b97f4a7800ULL;
  uint64_t result = 0;
  for (auto i : TfLiteIntArrayView(int_array)) {
    result = result ^ (i + kHashConst + (result << 10) + (result >> 4));
  }
  return result;
}

bool HasZeroes(TfLiteIntArrayView array) {
  for (auto value : array) {
    if (value == 0) {
      return true;
    }
  }
  return false;
}

// Bit mask for tensor flags.
enum {
  NN_TENSOR_FLAG_SCALAR_AS_TENSOR = 1U << 0,
  NN_TENSOR_FLAG_INT8_CONVERSION = 1U << 1,
};

}  // namespace

namespace delegate {
namespace nnapi {

// RAII NN API Execution Destructor for use with std::unique_ptr
struct NNFreeExecution {
  void operator()(ANeuralNetworksExecution* execution) {
    NnApiImplementation()->ANeuralNetworksExecution_free(execution);
  }
};

class DequantizeMapping {
 public:
  int DequantizedAnnIndex(int ann_index, TfLiteType type) const {
    for (const auto& element : mapping_) {
      if (ann_index == std::get<0>(element) && type == std::get<1>(element)) {
        return std::get<2>(element);
      }
    }
    return -1;
  }

  void Add(int ann_index, TfLiteType type, int dequantized_ann_index) {
    // This assumes it is not already mapped.
    mapping_.emplace_back(ann_index, type, dequantized_ann_index);
  }

 private:
  // Each tuple specifies the ANN (quantized) tensor index, the desired
  // floating-point type and the matching ANN (dequantized) tensor index. This
  // could use a map but instead std::vector is used to keep code size lower.
  std::vector<std::tuple<int, TfLiteType, int>> mapping_;
};

// Abstract builder for building an op in the NN API graph. This handles
// the disparity between TFLite and NN API operand types. NN API has singular
// operands for both tensors and parameters, and TFLite separates the two.
class NNAPIOpBuilder {
 public:
  NNAPIOpBuilder(const NnApi* nnapi, TfLiteContext* context,
                 OperandMapping* tensor_mapping,
                 DequantizeMapping* dequantize_mapping,
                 std::map<const MMAPAllocation*, ANeuralNetworksMemory*>*
                     allocation_mapping,
                 ANeuralNetworksModel* nn_model, int* nnapi_errno)
      : nnapi_(nnapi),
        context_(context),
        operand_mapping_(tensor_mapping),
        dequantize_mapping_(dequantize_mapping),
        allocation_memory_mapping_(allocation_mapping),
        nn_model_(nn_model),
        nnapi_errno_(nnapi_errno) {}

  TfLiteStatus AddScalarBoolOperand(bool value) {
    return AddScalarOperand<bool>(value, ANEURALNETWORKS_BOOL);
  }

  TfLiteStatus AddScalarInt32Operand(int32_t value) {
    return AddScalarOperand<int32_t>(value, ANEURALNETWORKS_INT32);
  }

  TfLiteStatus AddScalarFloat32Operand(float value) {
    return AddScalarOperand<float>(value, ANEURALNETWORKS_FLOAT32);
  }

  TfLiteStatus AddVectorInt32Operand(const int32_t* values,
                                     uint32_t num_values) {
    return AddVectorOperand<int32_t>(values, num_values,
                                     ANEURALNETWORKS_TENSOR_INT32,
                                     /*scale=*/0.f, /*zero_point=*/0);
  }

  TfLiteStatus AddVectorInt32Operand(const int32_t* values, uint32_t num_values,
                                     float scale, int32_t zero_point) {
    return AddVectorOperand<int32_t>(
        values, num_values, ANEURALNETWORKS_TENSOR_INT32, scale, zero_point);
  }

  TfLiteStatus AddVectorFloat32Operand(const float* values,
                                       uint32_t num_values) {
    return AddVectorOperand<float>(values, num_values,
                                   ANEURALNETWORKS_TENSOR_FLOAT32);
  }

  TfLiteStatus AddPoolingParams(void* data) {
    auto builtin = reinterpret_cast<TfLitePoolParams*>(data);
    AddScalarInt32Operand(builtin->padding);
    AddScalarInt32Operand(builtin->stride_width);
    AddScalarInt32Operand(builtin->stride_height);
    AddScalarInt32Operand(builtin->filter_width);
    AddScalarInt32Operand(builtin->filter_height);
    AddScalarInt32Operand(builtin->activation);
    return kTfLiteOk;
  }

  TfLiteStatus AddTensorInput(int tensor_index, bool hybrid_op,
                              int tensor_flags = 0) {
    return AddTensor(tensor_index, hybrid_op, &augmented_inputs_, tensor_flags);
  }

  TfLiteStatus AddTensorOutput(int tensor_index, int tensor_flags = 0) {
    return AddTensor(tensor_index, /*hybrid_op=*/false, &augmented_outputs_,
                     tensor_flags);
  }

  TfLiteStatus AddAdditionalFloat32OutputTensor(uint32_t dimension_count) {
    std::vector<uint32_t> dims(dimension_count, 0);
    return AddFloat32OutputTensor(dimension_count, dims.data(), nullptr);
  }

  TfLiteStatus AddStateFloat32Tensor(int tensor_index,
                                     int* ann_tensor_index_out) {
    TfLiteTensor* tensor = &context_->tensors[tensor_index];
    return AddFloat32OutputTensor(
        tensor->dims->size, reinterpret_cast<uint32_t*>(tensor->dims->data),
        ann_tensor_index_out);
  }

  // Add a constant tensor with a single element, intended for broadcast capable
  // ops.
  TfLiteStatus AddSingleValueConstantTensor(float value, bool is_quantized) {
    if (!is_quantized) {
      return AddVectorFloat32Operand(&value, 1);
    } else {
      // in the case that we need to add a quantized tensor, set the value to
      // 64, zero_point to be 0 and adjust scale accordingly.
      const uint8_t quant8_value = 64;
      return AddVectorOperand<uint8_t>(&quant8_value, 1,
                                       ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                                       value / quant8_value, 0);
    }
  }

  // Calculate the scale and zero_point for 8-bit unsigned tensor, given float
  // min and max. zero_point is clamped to [0, 255].
  TfLiteStatus CalculateQuantizationParams(float min, float max, float* scale,
                                           int* zero_point) {
    if (max < min) return kTfLiteError;
    *scale = (max - min) / 255.f;
    if (min > 0.f) {
      *zero_point = 0;
    } else if (max < 0.f) {
      *zero_point = 255;
    } else {
      *zero_point = (0.f - min) / (*scale);
    }
    return kTfLiteOk;
  }

  // Lower hardswish according to the following equation:
  // hard_swish[x] = x (ReLU6(x + 3)) / 6 == x * (Relu_N1_to_1(x/3) * 3 + 3) / 6
  // = 0.5x * Relu_N1_to_1(x/3) + 0.5x
  TfLiteStatus AddHardSwish(int lite_input_index, int lite_output_index,
                            bool need_int8_conversion) {
    const TfLiteTensor& tensor = context_->tensors[lite_input_index];
    float input_scale = tensor.params.scale;
    int input_zero_point = tensor.params.zero_point;
    float input_min = 0.f;
    float input_max = 0.f;
    int tensor_flags = 0;
    if (need_int8_conversion) {
      tensor_flags = tensor_flags | NN_TENSOR_FLAG_INT8_CONVERSION;
      input_zero_point += 128;
    }
    bool is_quantized = false;
    int nn_type = ANEURALNETWORKS_TENSOR_FLOAT32;
    if (tensor.type == kTfLiteInt8 || tensor.type == kTfLiteUInt8) {
      is_quantized = true;
      nn_type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
      input_min = (0 - input_zero_point) * input_scale;
      input_max = (255 - input_zero_point) * input_scale;
    }

    // Stage1 : s1 = Relu1(x * 1/3)
    float s1_output_min = 0.f;
    float s1_output_max = 0.f;
    int s1_out_ann_index = 0;
    {
      float s1_output_scale = 0.f;
      int s1_output_zero_point = 0;
      if (is_quantized) {
        // clamp the output range to [-1, 1] if needed.
        s1_output_min = input_min / 3.f < -1.f ? -1.f : input_min / 3.f;
        s1_output_max = input_max / 3.f > 1.f ? 1.f : input_max / 3.f;
        CalculateQuantizationParams(s1_output_min, s1_output_max,
                                    &s1_output_scale, &s1_output_zero_point);
      }
      TF_LITE_ENSURE_OK(context_,
                        AddTensorInput(lite_input_index, false, tensor_flags));
      const float value3f = 1.f / 3.f;
      TF_LITE_ENSURE_OK(context_,
                        AddSingleValueConstantTensor(value3f, is_quantized));
      TF_LITE_ENSURE_OK(context_,
                        AddScalarInt32Operand(ANEURALNETWORKS_FUSED_RELU1));
      TF_LITE_ENSURE_OK(
          context_,
          AddAdditionalOutputTensor(
              tensor.dims->size, reinterpret_cast<uint32_t*>(tensor.dims->data),
              nn_type, s1_output_scale, s1_output_zero_point,
              &s1_out_ann_index));
      TF_LITE_ENSURE_OK(context_, FinalizeAddOperation(ANEURALNETWORKS_MUL));
    }

    // Stage2 : s2 = x / 2
    float s2_output_min = input_min / 2.f;
    float s2_output_max = input_max / 2.f;
    int s2_out_ann_index = 0;
    {
      float s2_output_scale = input_scale / 2.0f;
      int s2_output_zero_point = input_zero_point;
      TF_LITE_ENSURE_OK(context_,
                        AddTensorInput(lite_input_index, false, tensor_flags));
      const float value2f = 0.5f;
      TF_LITE_ENSURE_OK(context_,
                        AddSingleValueConstantTensor(value2f, is_quantized));
      TF_LITE_ENSURE_OK(context_,
                        AddScalarInt32Operand(ANEURALNETWORKS_FUSED_NONE));
      TF_LITE_ENSURE_OK(
          context_,
          AddAdditionalOutputTensor(
              tensor.dims->size, reinterpret_cast<uint32_t*>(tensor.dims->data),
              nn_type, s2_output_scale, s2_output_zero_point,
              &s2_out_ann_index));
      TF_LITE_ENSURE_OK(context_, FinalizeAddOperation(ANEURALNETWORKS_MUL));
    }

    // Stage 3 : s3 = s1 * s2
    int s3_out_ann_index = 0;
    {
      augmented_inputs_.push_back(s1_out_ann_index);
      augmented_inputs_.push_back(s2_out_ann_index);
      TF_LITE_ENSURE_OK(context_,
                        AddScalarInt32Operand(ANEURALNETWORKS_FUSED_NONE));
      float s3_output_scale = 0.f;
      int s3_output_zero_point = 0;
      if (is_quantized) {
        // the min for stage 3 is always 0.0f.
        float s3_output_min = 0.f;
        // the max for stage 3 is max(s1_min * s2_min, s1_max * s3_max).
        float s3_output_max =
            s1_output_max * s2_output_max > s1_output_min * s2_output_min
                ? s1_output_max * s2_output_max
                : s1_output_min * s2_output_min;
        CalculateQuantizationParams(s3_output_min, s3_output_max,
                                    &s3_output_scale, &s3_output_zero_point);
      }
      TF_LITE_ENSURE_OK(
          context_,
          AddAdditionalOutputTensor(
              tensor.dims->size, reinterpret_cast<uint32_t*>(tensor.dims->data),
              nn_type, s3_output_scale, s3_output_zero_point,
              &s3_out_ann_index));
      TF_LITE_ENSURE_OK(context_, FinalizeAddOperation(ANEURALNETWORKS_MUL));
    }

    // Stage 4: y = s3 + s2
    {
      augmented_inputs_.push_back(s2_out_ann_index);
      augmented_inputs_.push_back(s3_out_ann_index);
      TF_LITE_ENSURE_OK(context_,
                        AddScalarInt32Operand(ANEURALNETWORKS_FUSED_NONE));
      TF_LITE_ENSURE_OK(context_,
                        AddTensorOutput(lite_output_index, tensor_flags));
      TF_LITE_ENSURE_OK(context_, FinalizeAddOperation(ANEURALNETWORKS_ADD));
    }

    return kTfLiteOk;
  }

  // Adds a Dequantize operator and replaces the input tensor index with the
  // dequantized version. If the dequantized version of the operator already
  // exists then it is not added again.
  TfLiteStatus AddDequantize(int nn_input_index, int lite_index,
                             TfLiteType dequantized_type) {
    const int ann_index = operand_mapping_->lite_index_to_ann(lite_index);
    int dequantized_ann_index =
        dequantize_mapping_->DequantizedAnnIndex(ann_index, dequantized_type);

    if (dequantized_ann_index == -1) {
      // The dequantized version does not exist yet, it has to be added: a new
      // Dequantize operation is added, yielding a new tensor.
      const TfLiteTensor& tensor = context_->tensors[lite_index];
      ANeuralNetworksOperandType operand_type{
          ANEURALNETWORKS_TENSOR_FLOAT32,
          static_cast<uint32_t>(tensor.dims->size),
          reinterpret_cast<uint32_t*>(tensor.dims->data), 0.f, 0};
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context_,
          nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type),
          nnapi_errno_);
      dequantized_ann_index = operand_mapping_->add_new_non_tensor_operand();

      // Add Dequantize operation.
      const uint32_t dequantize_input[1] = {static_cast<uint32_t>(ann_index)};
      const uint32_t dequantize_output[1] = {
          static_cast<uint32_t>(dequantized_ann_index)};
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context_,
          nnapi_->ANeuralNetworksModel_addOperation(
              nn_model_, ANEURALNETWORKS_DEQUANTIZE, 1, dequantize_input, 1,
              dequantize_output),
          nnapi_errno_);
      dequantize_mapping_->Add(ann_index, dequantized_type,
                               dequantized_ann_index);
    }

    // The input for the original operation is modified so that the operation
    // now uses the dequantized tensor as input.
    augmented_inputs_[nn_input_index] = dequantized_ann_index;

    return kTfLiteOk;
  }

  // Finish emitting the op (of type `type`) into the NN API.
  TfLiteStatus FinalizeAddOperation(ANeuralNetworksOperationType type) {
    // Actually add a NN API operation
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_addOperation(
            nn_model_, type, static_cast<uint32_t>(augmented_inputs_.size()),
            augmented_inputs_.data(),
            static_cast<uint32_t>(augmented_outputs_.size()),
            augmented_outputs_.data()),
        nnapi_errno_);
    augmented_inputs_.clear();
    augmented_outputs_.clear();
    return kTfLiteOk;
  }

  TfLiteStatus AddSingleValueTensorAsScalarOperand(int tensor_index,
                                                   int nn_type) {
    const TfLiteTensor* tensor = &context_->tensors[tensor_index];
    TF_LITE_ENSURE_EQ(context_, NumElements(tensor), 1);

    ANeuralNetworksOperandType operand_type{.type = nn_type};
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type),
        nnapi_errno_);
    int ann_tensor_index = operand_mapping_->lite_index_to_ann(tensor_index);
    if (ann_tensor_index != -1) {
      augmented_inputs_.push_back(ann_tensor_index);
      return kTfLiteOk;
    }
    // Allocate a new tensor index
    ann_tensor_index = operand_mapping_->add_new_ann_tensor_index(tensor_index);
    augmented_inputs_.push_back(ann_tensor_index);

    const TfLiteType tensor_type = tensor->type;
    TfLiteType nn_type_equivalent;
    TF_LITE_ENSURE_OK(context_, GetEquivalentToANNType(context_, nn_type,
                                                       &nn_type_equivalent));
    if (tensor_type != nn_type_equivalent) {
      operand_mapping_->add_type_conversion(tensor_index, nn_type_equivalent);
    }
    return kTfLiteOk;
  }

  template <typename T>
  TfLiteStatus AddNewInputConstantTensor(
      int32_t nn_type, TfLiteType type, const TfLiteIntArray* dims,
      const std::vector<T>& tensor_value,
      const TfLiteQuantizationParams& quant_params, int* tensor_index) {
    TF_LITE_ENSURE_OK(context_,
                      context_->AddTensors(context_, 1, tensor_index));

    TfLiteTensor* new_tensor = &context_->tensors[*tensor_index];
    new_tensor->type = type;
    new_tensor->allocation_type = kTfLiteDynamic;
    new_tensor->params = quant_params;

    // Not removing the new tensor in case of resizing errors since it will
    // be cleared by the context
    TF_LITE_ENSURE_OK(
        context_,
        context_->ResizeTensor(
            context_, new_tensor,
            // Resize Tensor takes ownership of the dims array passed as param
            TfLiteIntArrayCopy(dims)));

    memcpy(new_tensor->data.raw,
           reinterpret_cast<const char*>(tensor_value.data()),
           tensor_value.size() * sizeof(T));

    const uint32_t tensor_rank = static_cast<uint32_t>(dims->size);
    const uint32_t* tensor_dims = reinterpret_cast<const uint32_t*>(dims->data);
    ANeuralNetworksOperandType operand_type{nn_type, tensor_rank, tensor_dims,
                                            quant_params.scale,
                                            quant_params.zero_point};

    const int ann_tensor_index =
        operand_mapping_->add_delegate_generated_input_ann_tensors_operand();

    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type),
        nnapi_errno_);

    augmented_inputs_.push_back(ann_tensor_index);

    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_setOperandValue(
            nn_model_, ann_tensor_index, new_tensor->data.raw,
            new_tensor->bytes),
        nnapi_errno_);

    return kTfLiteOk;
  }

  template <typename T>
  TfLiteStatus AddNewInputConstantTensor(
      int32_t nn_type, TfLiteType type, std::initializer_list<int> dims,
      const std::vector<T>& tensor_value,
      const TfLiteQuantizationParams& quant_params, int* tensor_index) {
    TfLiteIntArray* dim_array = TfLiteIntArrayCreate(dims.size());
    dim_array->size = dims.size();
    std::copy(dims.begin(), dims.end(), dim_array->data);

    const auto result = AddNewInputConstantTensor(
        nn_type, type, dim_array, tensor_value, quant_params, tensor_index);
    TfLiteIntArrayFree(dim_array);
    return result;
  }

 private:
  // Returns a TF Lite type which has the same memory representation as a
  // provided NN API type.
  TfLiteStatus GetEquivalentToANNType(TfLiteContext* context, int nn_type,
                                      TfLiteType* type) {
    switch (nn_type) {
      case ANEURALNETWORKS_INT32:
        *type = kTfLiteInt32;
        return kTfLiteOk;
      case ANEURALNETWORKS_FLOAT32:
        *type = kTfLiteFloat32;
        return kTfLiteOk;
      default:
        context->ReportError(context,
                             "NN API Delegate: Can't get an equivalent TF Lite "
                             "type for provided NN API type: %d.\n",
                             nn_type);
        return kTfLiteError;
    }
  }

  template <typename T>
  TfLiteStatus AddScalarOperand(T value, int32_t nn_type) {
    ANeuralNetworksOperandType operand_type{.type = nn_type};
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type),
        nnapi_errno_);
    const int ann_index = operand_mapping_->add_new_non_tensor_operand();
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_setOperandValue(nn_model_, ann_index,
                                                     &value, sizeof(T)),
        nnapi_errno_);
    augmented_inputs_.push_back(ann_index);
    return kTfLiteOk;
  }

  template <typename T>
  TfLiteStatus AddVectorOperand(const T* values, uint32_t num_values,
                                int32_t nn_type, float scale,
                                int32_t zero_point) {
    ANeuralNetworksOperandType operand_type{.type = nn_type,
                                            .dimensionCount = 1,
                                            .dimensions = &num_values,
                                            .scale = scale,
                                            .zeroPoint = zero_point};

    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type),
        nnapi_errno_);

    const int ann_index = operand_mapping_->add_new_non_tensor_operand();
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_setOperandValue(
            nn_model_, ann_index, values, sizeof(T) * num_values),
        nnapi_errno_);
    augmented_inputs_.push_back(ann_index);
    return kTfLiteOk;
  }

  template <typename T>
  TfLiteStatus AddVectorOperand(const T* values, uint32_t num_values,
                                int32_t nn_type) {
    return AddVectorOperand(values, num_values, nn_type, /*scale=*/0.f,
                            /*zero_point=*/0);
  }

  TfLiteStatus AddFloat32OutputTensor(uint32_t dimension_count,
                                      const uint32_t* dimension_data,
                                      int* ann_index_out) {
    return AddAdditionalOutputTensor(
        dimension_count, dimension_data, ANEURALNETWORKS_TENSOR_FLOAT32,
        /*scale=*/0.f, /*zero_point=*/0, ann_index_out);
  }

  TfLiteStatus AddAdditionalOutputTensor(uint32_t dimension_count,
                                         const uint32_t* dimension_data,
                                         int32_t nn_type, float scale,
                                         int32_t zero_point,
                                         int* ann_index_out) {
    ANeuralNetworksOperandType operand_type{
        .type = nn_type,
        .dimensionCount = dimension_count,
        .dimensions = dimension_data,
        .scale = scale,
        .zeroPoint = zero_point,
    };
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type),
        nnapi_errno_);
    const int ann_index = operand_mapping_->add_new_non_tensor_operand();
    augmented_outputs_.push_back(ann_index);
    if (ann_index_out) *ann_index_out = ann_index;
    return kTfLiteOk;
  }

  // Adds a new NN API tensor that shadows the TF Lite tensor `tensor_index`.
  // This returns the NN API tensor index corresponding to the created tensor.
  // If another caller previously created a NN API tensor for `tensor_index`
  // then the existing one is returned.
  TfLiteStatus AddTensor(int tensor_index, bool hybrid_op,
                         std::vector<uint32_t>* indices, int tensor_flags = 0) {
    const bool scalar_as_tensor =
        tensor_flags & NN_TENSOR_FLAG_SCALAR_AS_TENSOR;
    const bool need_int8_conversion =
        tensor_flags & NN_TENSOR_FLAG_INT8_CONVERSION;
    int ann_tensor_index = operand_mapping_->lite_index_to_ann(tensor_index);
    if (ann_tensor_index != -1) {
      indices->push_back(ann_tensor_index);
      return kTfLiteOk;
    }
    // Allocate a new tensor index
    ann_tensor_index = operand_mapping_->add_new_ann_tensor_index(tensor_index);

    // Parameters needed for new type.
    int32_t nn_type = 0;
    float scale = 0.0f;
    int32_t zeroPoint = 0;
    TfLiteTensor* tensor = &context_->tensors[tensor_index];
    TfLiteType tensor_type = tensor->type;
    if (hybrid_op && (tensor_type == kTfLiteUInt8)) {
      // For legacy reason, UINT8 weights in hybrid operators are actually INT8
      // values and should be interpreted as such.
      tensor_type = kTfLiteInt8;
    }
    switch (tensor_type) {
      case kTfLiteNoType:
        // Tensors added during initialization of Ops don't have a type yet and
        // should not be registered with the NNAPI.
        indices->push_back(-1);
        return kTfLiteOk;
      case kTfLiteFloat32:
        nn_type = ANEURALNETWORKS_TENSOR_FLOAT32;
        break;
      case kTfLiteUInt8:
      case kTfLiteInt8:
        // If explicit int8 conversion is needed, we still need
        // ANEURALNETWORKS_TENSOR_QUANT8_ASYMM type.
        nn_type = (tensor_type == kTfLiteUInt8 || need_int8_conversion)
                      ? ANEURALNETWORKS_TENSOR_QUANT8_ASYMM
                      : ANEURALNETWORKS_TENSOR_QUANT8_SYMM;
        scale = tensor->params.scale;
        zeroPoint = tensor->params.zero_point;
        if (need_int8_conversion) {
          zeroPoint += 128;
          operand_mapping_->add_type_conversion(tensor_index, kTfLiteUInt8);
        }
        if (scale == 0) {
          // TENSOR_QUANT8_ASYMM and ANEURALNETWORKS_TENSOR_QUANT8_ASYMM
          // with zero scale are not valid in NNAPI.
          scale = 1;
        }
        break;
      case kTfLiteInt32:
        nn_type = ANEURALNETWORKS_TENSOR_INT32;
        scale = tensor->params.scale;
        zeroPoint = tensor->params.zero_point;
        break;
      case kTfLiteBool:
        nn_type = ANEURALNETWORKS_TENSOR_BOOL8;
        break;
      case kTfLiteInt16:
        nn_type = ANEURALNETWORKS_TENSOR_QUANT16_SYMM;
        scale = tensor->params.scale;
        zeroPoint = tensor->params.zero_point;
        break;
      default:
        context_->ReportError(
            context_, "Failed to add NN API tensor: type %s is not supported.",
            TfLiteTypeGetName(tensor_type));
        return kTfLiteError;
    }
    uint32_t tensor_rank = static_cast<uint32_t>(tensor->dims->size);
    uint32_t* tensor_dims = reinterpret_cast<uint32_t*>(tensor->dims->data);
    if (scalar_as_tensor && tensor_rank == 0) {
      // Use rank 1, shape {1} operand for TFLite scalar tensors.
      tensor_rank = 1;
      tensor_dims = &tensor_rank;
    }
    if (tensor_rank == 0) {
      // if the tensor_rank is 0, the dimension ptr must be nullptr.
      tensor_dims = nullptr;
    }
    ANeuralNetworksSymmPerChannelQuantParams ann_perchannel_params;
    if (tensor_type == kTfLiteInt8 || tensor_type == kTfLiteUInt8) {
      if (tensor->quantization.type == kTfLiteAffineQuantization) {
        TfLiteAffineQuantization* quantization_params =
            static_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
        if (quantization_params->scale->size > 1) {
          // Set up per-channel quantization.
          ann_perchannel_params = {
              .channelDim = static_cast<uint32_t>(
                  quantization_params->quantized_dimension),
              .scaleCount =
                  static_cast<uint32_t>(quantization_params->scale->size),
              .scales = quantization_params->scale->data,
          };
          nn_type = ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL;
          scale = 0.0f;
          zeroPoint = 0;
        }
      }
    }

    ANeuralNetworksOperandType operand_type{nn_type, tensor_rank, tensor_dims,
                                            scale, zeroPoint};
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type),
        nnapi_errno_);

    if (nn_type == ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL) {
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context_,
          nnapi_->ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(
              nn_model_, ann_tensor_index, &ann_perchannel_params),
          nnapi_errno_);
    }
    if (tensor->allocation_type == kTfLiteMmapRo) {
      if (IsQuantized(tensor_type) && need_int8_conversion) {
        // We need to to add a tensor and convert the weights into uint8.
        // Currently this is only needed for fully_connected. The new_tensor is
        // needed for lifetime management for the converted weights.
        int new_tensor_index = -1;
        TF_LITE_ENSURE_OK(context_,
                          context_->AddTensors(context_, 1, &new_tensor_index));
        TfLiteTensor* new_tensor = &context_->tensors[new_tensor_index];
        new_tensor->type = kTfLiteUInt8;
        new_tensor->allocation_type = kTfLiteDynamic;
        new_tensor->params.scale = scale;
        new_tensor->params.zero_point = zeroPoint;
        // Not removing the new tensor in case of resizing errors since it will
        // be cleared by the context
        TF_LITE_ENSURE_OK(
            context_, context_->ResizeTensor(context_, new_tensor,
                                             // Resize Tensor takes ownership of
                                             // the dims array passed as param
                                             TfLiteIntArrayCopy(tensor->dims)));
        // Convert the int8 value into corresponding uint8 value;
        const auto num_elements = NumElements(tensor);
        for (int i = 0; i < num_elements; ++i) {
          new_tensor->data.uint8[i] = static_cast<const uint8_t>(
              static_cast<int32_t>(tensor->data.int8[i]) + 128);
        }
        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            context_,
            nnapi_->ANeuralNetworksModel_setOperandValue(
                nn_model_, ann_tensor_index, new_tensor->data.raw,
                new_tensor->bytes),
            nnapi_errno_);
#ifdef TFLITE_NNAPI_ALLOW_MMAP_SHARING
      } else if (tensor->allocation &&
                 static_cast<const Allocation*>(tensor->allocation)->type() ==
                     Allocation::Type::kMMap) {
        const MMAPAllocation* mmap_alloc =
            static_cast<const MMAPAllocation*>(tensor->allocation);
        if (allocation_memory_mapping_->count(mmap_alloc) == 0) {
          ANeuralNetworksMemory* ann_memory_handle = nullptr;
          nnapi_->ANeuralNetworksMemory_createFromFd(
              mmap_alloc->bytes(), PROT_READ, mmap_alloc->fd(), 0,
              &ann_memory_handle);
          allocation_memory_mapping_->insert(
              std::make_pair(mmap_alloc, ann_memory_handle));
        }
        ANeuralNetworksMemory* ann_memory_handle =
            allocation_memory_mapping_->at(mmap_alloc);
        // Compute the offset to the base pointer of the MMAPAllocation.
        auto offset = reinterpret_cast<const uint8_t*>(tensor->data.raw) -
                      reinterpret_cast<const uint8_t*>(mmap_alloc->base());
        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            context_,
            nnapi_->ANeuralNetworksModel_setOperandValueFromMemory(
                nn_model_, ann_tensor_index, ann_memory_handle, offset,
                tensor->bytes),
            nnapi_errno_);
#endif
      } else {
        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            context_,
            nnapi_->ANeuralNetworksModel_setOperandValue(
                nn_model_, ann_tensor_index, tensor->data.raw, tensor->bytes),
            nnapi_errno_);
      }
    }

    indices->push_back(ann_tensor_index);
    return kTfLiteOk;
  }

  // Access to NNAPI.
  const NnApi* const nnapi_;

  // TfLiteContext for error handling.
  TfLiteContext* const context_;

  // Tracks relationship between indices.
  OperandMapping* const operand_mapping_;

  // Keeps mapping of ANN quantized tensor and float data type to equivalent
  // dequantized ANN tensor. For example, tensor #4 (UINT8) + FLOAT32 could map
  // to tensor #10 (FLOAT32) because a DEQUANTIZE operator was added to convert
  // tensor #4 to a FLOAT32 tensor.
  DequantizeMapping* const dequantize_mapping_;

  std::map<const MMAPAllocation*, ANeuralNetworksMemory*>* const
      allocation_memory_mapping_;

  // The NNAPI model.
  ANeuralNetworksModel* const nn_model_;

  // Inputs and outputs for the current op. These are augmented in the sense
  // that NN API uses operands for all arguments, not just tensors, unlike
  // TensorFlow Lite.
  std::vector<uint32_t> augmented_inputs_;
  std::vector<uint32_t> augmented_outputs_;

  // Return status code of the latest NNAPI call.
  int* nnapi_errno_;
};  // namespace nnapi

namespace {
struct OpValidationContext {
  bool is_valid;
  std::vector<NNAPIValidationFailure>* validation_failures;
};

#define EXPECT_INPUT_TYPE_IN(actual_type, ...)                    \
  ExpectTypeIn(actual_type, {__VA_ARGS__},                        \
               NNAPIValidationFailureType::kUnsupportedInputType, \
               "Input type not in expected list " #__VA_ARGS__, &val_ctx)

inline void AddValidationFailure(NNAPIValidationFailureType failure_type,
                                 const char* message,
                                 OpValidationContext* val_ctx) {
  val_ctx->is_valid = false;

#ifdef NNAPI_VERBOSE_VALIDATION
  if (val_ctx->validation_failures) {
    val_ctx->validation_failures->push_back({failure_type, message});
  }
#endif
}

template <typename... Args>
inline void AddValidationFailureFmt(OpValidationContext* val_ctx,
                                    NNAPIValidationFailureType failure_type,
                                    const char* message_fmt, Args... args) {
  val_ctx->is_valid = false;
#ifdef NNAPI_VERBOSE_VALIDATION
  if (val_ctx->validation_failures) {
    size_t req_buf_size = snprintf(nullptr, 0, message_fmt, args...) + 1;
    std::unique_ptr<char[]> tmp_buf(new char[req_buf_size]);
    snprintf(tmp_buf.get(), req_buf_size, message_fmt, args...);

    val_ctx->validation_failures->push_back({failure_type, tmp_buf.get()});
  }
#endif
}

inline bool Expect(bool condition, NNAPIValidationFailureType failure_type,
                   const char* message, OpValidationContext* val_ctx) {
  if (!condition) {
    AddValidationFailure(failure_type, message, val_ctx);
    return false;
  }
  return true;
}

template <typename... Args>
inline bool ExpectFmt(bool condition, OpValidationContext* val_ctx,
                      NNAPIValidationFailureType failure_type,
                      const char* message_fmt, Args... args) {
  if (!condition) {
    AddValidationFailureFmt(val_ctx, failure_type, message_fmt, args...);
    return false;
  }
  return true;
}

inline bool ExpectTypeIn(TfLiteType actual_type,
                         std::initializer_list<TfLiteType> allowed_types,
                         NNAPIValidationFailureType failure_type,
                         const char* msg, OpValidationContext* val_ctx) {
  return Expect(std::find(allowed_types.begin(), allowed_types.end(),
                          actual_type) != allowed_types.end(),
                failure_type, msg, val_ctx);
}

inline bool ExpectMinAndroidSdkVersion(int curr_version, int min_version,
                                       OpValidationContext* val_ctx) {
  return ExpectFmt(curr_version >= min_version, val_ctx,
                   NNAPIValidationFailureType::kUnsupportedAndroidVersion,
                   "Android sdk version less than %d", min_version);
}

inline bool ExpectMaxOpVersion(int curr_version, int max_version,
                               OpValidationContext* val_ctx) {
  return ExpectFmt(curr_version <= max_version, val_ctx,
                   NNAPIValidationFailureType::kUnsupportedOperatorVersion,
                   "OP Version higher than %d", max_version);
}

inline bool ExpectOpVersion(int curr_version, int max_version,
                            OpValidationContext* val_ctx) {
  return ExpectFmt(curr_version <= max_version, val_ctx,
                   NNAPIValidationFailureType::kUnsupportedOperatorVersion,
                   "OP Version different from %d", max_version);
}

inline bool ExpectIsFloatOperator(const TfLiteContext* context,
                                  const TfLiteNode* node,
                                  OpValidationContext* val_ctx) {
  const auto input_type = context->tensors[node->inputs->data[0]].type;
  return Expect(IsFloat(input_type),
                NNAPIValidationFailureType::kUnsupportedInputType,
                "Input should be Float", val_ctx);
}

bool ExpectIsFloatOrUint8Operator(const TfLiteContext* context,
                                  const TfLiteNode* node,
                                  OpValidationContext* val_ctx) {
  const auto input_type = context->tensors[node->inputs->data[0]].type;
  return Expect(IsFloatOrUInt8(input_type),
                NNAPIValidationFailureType::kUnsupportedInputType,
                "Input should be Float or UINT8", val_ctx);
}

bool ExpectIsFloatOrQuant8Operator(const TfLiteContext* context,
                                   const TfLiteNode* node,
                                   OpValidationContext* val_ctx) {
  const auto input_type = context->tensors[node->inputs->data[0]].type;
  return Expect(IsFloat(input_type) || IsQuantized(input_type),
                NNAPIValidationFailureType::kUnsupportedInputType,
                "Input should be Float or Quant8", val_ctx);
}

// When using NN API version 1.0 or 1.1, the condition below must be true for
// quantized versions of the following ops:
// * CONV_2D
// * DEPTHWISE_CONV_2D
// * FULLY_CONNECTED (where filter actually stands for weights)
// The condition is relaxed and no longer required since version 1.2.
bool ExpectIsRestrictedScalesCompliant(const TfLiteContext* context,
                                       const TfLiteNode* node,
                                       OpValidationContext* val_ctx) {
  const int input_id = node->inputs->data[0];
  const int filter_id = node->inputs->data[1];
  const int output_id = node->outputs->data[0];
  const float input_scale = context->tensors[input_id].params.scale;
  const float filter_scale = context->tensors[filter_id].params.scale;
  const float output_scale = context->tensors[output_id].params.scale;
  return Expect(input_scale * filter_scale < output_scale,
                NNAPIValidationFailureType::kNotRestrictedScaleCompliant,
                "When using NN API version 1.0 or 1.1, input_scale * "
                "filter_scale < output_scale:",
                val_ctx);
}

}  // namespace

// Return a function that knows how to translate a node into its operands
// when called. You can use this function to see if a node is supported
// (i.e. if the returned MappingFn is null, then the node is not supported).
bool NNAPIDelegateKernel::Validate(
    const TfLiteContext* context, int builtin_code, int version,
    int android_sdk_version, const TfLiteNode* node,
    bool is_accelerator_specified,
    std::vector<NNAPIValidationFailure>* map_failures) {
  OpValidationContext val_ctx{true, map_failures};

  switch (builtin_code) {
    case kTfLiteBuiltinAdd: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinArgMax:
    case kTfLiteBuiltinArgMin: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      // Those operators were introduced in NNAPI 1.2.
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const TfLiteType input_type =
          context->tensors[node->inputs->data[(0)]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat16, kTfLiteFloat32,
                           kTfLiteInt32, kTfLiteUInt8, kTfLiteInt8);

      const auto& axis_tensor = context->tensors[node->inputs->data[1]];
      if (axis_tensor.type == kTfLiteInt64) {
        Expect(
            axis_tensor.allocation_type == kTfLiteMmapRo &&
                *axis_tensor.data.i64 <= std::numeric_limits<int32_t>::max() &&
                *axis_tensor.data.i64 >= std::numeric_limits<int32_t>::min(),
            NNAPIValidationFailureType::kUnsupportedInputType,
            "NNAPI only supports axis as int32. If the axis type is int64 and "
            "constant we can convert it to int32 if the value isn't too "
            "large.",
            &val_ctx);
      } else {
        Expect(axis_tensor.type == kTfLiteInt32,
               NNAPIValidationFailureType::kUnsupportedInputType,
               "Axis should be Int32", &val_ctx);
      }
      if (builtin_code == kTfLiteBuiltinArgMax) {
        auto builtin =
            reinterpret_cast<TfLiteArgMaxParams*>(node->builtin_data);
        Expect(builtin->output_type == kTfLiteInt32,
               NNAPIValidationFailureType::kUnsupportedOutputType,
               "NNAPI only supports int32 output.", &val_ctx);
      } else {
        auto builtin =
            reinterpret_cast<TfLiteArgMinParams*>(node->builtin_data);
        Expect(builtin->output_type == kTfLiteInt32,
               NNAPIValidationFailureType::kUnsupportedOutputType,
               "NNAPI only supports int32 output.", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinMul: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinAveragePool2d: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      auto builtin = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
      // TODO(b/138756912): Large filter window would overflow on the
      // reference CPU path.
      Expect(is_accelerator_specified ||
                 (builtin->filter_width * builtin->filter_height <= 256),
             NNAPIValidationFailureType::kUnsupportedOperandSize,
             "Large filter window would overflow on the reference CPU path",
             &val_ctx);
    } break;
    case kTfLiteBuiltinMaxPool2d: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinL2Pool2d: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectIsFloatOperator(context, node, &val_ctx);

      if (android_sdk_version < kMinSdkVersionForNNAPI12) {
        auto builtin = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
        Expect(builtin->activation == kTfLiteActNone,
               NNAPIValidationFailureType::kUnsupportedOperandValue,
               "Before NNAPI 1.2 fused activation for l2_pool may not be "
               "supported.",
               &val_ctx);
      }
    } break;
    case kTfLiteBuiltinConv2d: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
      if (android_sdk_version < kMinSdkVersionForNNAPI12) {
        Expect(!IsHybridOperator(context, builtin_code, node),
               NNAPIValidationFailureType::kUnsupportedHybridOperator,
               "Hybrid operators not supported before NNAPI 1.2", &val_ctx);
        ExpectIsFloatOrUint8Operator(context, node, &val_ctx);

        const auto& filter_tensor = context->tensors[node->inputs->data[1]];
        if (filter_tensor.quantization.type == kTfLiteAffineQuantization) {
          TfLiteAffineQuantization* quantization_params =
              static_cast<TfLiteAffineQuantization*>(
                  filter_tensor.quantization.params);
          Expect(quantization_params->scale->size <= 1,
                 NNAPIValidationFailureType::kUnsupportedQuantizationType,
                 "Per-channel quantized convolution not supported before NNAPI "
                 "1.2.",
                 &val_ctx);
        }
      }
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
          input_type == kTfLiteUInt8) {
        ExpectIsRestrictedScalesCompliant(context, node, &val_ctx);
      }
      auto builtin = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
      // TODO(b/132950584): Add support for Conv2D with omitted bias.
      Expect(node->inputs->size == 3,
             NNAPIValidationFailureType::kMissingRequiredOperand,
             "Conv2D with omitted bias not supported", &val_ctx);
      if (builtin->dilation_width_factor != 1 ||
          builtin->dilation_height_factor != 1) {
        Expect(android_sdk_version >= kMinSdkVersionForNNAPI12,
               NNAPIValidationFailureType::kUnsupportedOperandValue,
               "NNAPI supports dilated Conv2D since NNAPI 1.2.", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinDepthwiseConv2d: {
      ExpectMaxOpVersion(version, 3, &val_ctx);

      if (android_sdk_version < kMinSdkVersionForNNAPI12) {
        ExpectIsFloatOrUint8Operator(context, node, &val_ctx);

        const auto input_type = context->tensors[node->inputs->data[0]].type;
        if (input_type == kTfLiteUInt8) {
          ExpectIsRestrictedScalesCompliant(context, node, &val_ctx);
        }

        auto builtin =
            reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
        Expect(builtin->dilation_width_factor == 1 &&
                   builtin->dilation_height_factor == 1,
               NNAPIValidationFailureType::kUnsupportedOperandValue,
               "dilation_width_factor and dilation_height_factor expected to "
               "be equal to 1",
               &val_ctx);
      }
    } break;
    case kTfLiteBuiltinFullyConnected: {
      ExpectMaxOpVersion(version, 4, &val_ctx);
      // TODO(b/132950584): Add support for FullyConnected with no bias.
      Expect(node->inputs->size == 3 &&
                 node->inputs->data[2] != kTfLiteOptionalTensor,
             NNAPIValidationFailureType::kMissingRequiredOperand,
             "FullyConnected with no bias not supported", &val_ctx);
      const auto output_type = context->tensors[node->outputs->data[0]].type;
      Expect(output_type != kTfLiteInt16,
             NNAPIValidationFailureType::kUnsupportedOutputType,
             "Unsupported output of type kTfLiteInt16", &val_ctx);
      if (android_sdk_version < kMinSdkVersionForNNAPI12) {
        Expect(!IsHybridOperator(context, builtin_code, node),
               NNAPIValidationFailureType::kUnsupportedHybridOperator,
               "Hybrid operators not supported before NNAPI 1.2", &val_ctx);
        ExpectIsFloatOrUint8Operator(context, node, &val_ctx);
      }
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
          input_type == kTfLiteUInt8) {
        ExpectIsRestrictedScalesCompliant(context, node, &val_ctx);
      }
      auto builtin =
          reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
      Expect(!builtin->keep_num_dims,
             NNAPIValidationFailureType::kUnsupportedOperandValue,
             "keep_num_dims == true not supported", &val_ctx);
    } break;
    case kTfLiteBuiltinHardSwish: {
      // Add support for hardswish. For Pre-Q devices, deconstructing it into
      // basic ops. Though for some nnapi accelerators using optimized tflite
      // kernels might even be faster.
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinSoftmax: {
      ExpectOpVersion(version, 2, &val_ctx);
      const auto& input = context->tensors[node->outputs->data[0]];
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      const int input_rank = input.dims->size;
      Expect(input_rank <= 4,
             NNAPIValidationFailureType::kUnsupportedOperandRank,
             "Input rank should be <= 4", &val_ctx);
      if (android_sdk_version < kMinSdkVersionForNNAPI12) {
        Expect(
            input_rank == 2 || input_rank == 4,
            NNAPIValidationFailureType::kUnsupportedOperandRank,
            "Before API level 29 only 2D and 4D input tensors were supported.",
            &val_ctx);
      }
    } break;
    case kTfLiteBuiltinReshape: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      Expect(node->inputs->size >= 2,
             NNAPIValidationFailureType::kMissingRequiredOperand,
             "Expected at least 2 inputs", &val_ctx);
      if (node->inputs->size >= 2) {
        Expect(context->tensors[node->inputs->data[1]].allocation_type ==
                   kTfLiteMmapRo,
               NNAPIValidationFailureType::kInputTensorShouldHaveConstantShape,
               "The shape input tensor must be constant.", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinResizeBilinear: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const auto& input = context->tensors[node->inputs->data[0]];
      const auto output_dims = context->tensors[node->outputs->data[0]].dims;
      Expect(input.dims->size == 4,
             NNAPIValidationFailureType::kUnsupportedOperandRank,
             "Input should have rank 4", &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      Expect(node->inputs->size >= 2,
             NNAPIValidationFailureType::kUnsupportedOperatorVariant,
             "Expected at least 2 inputs", &val_ctx);
      if (node->inputs->size >= 2) {
        Expect(context->tensors[node->inputs->data[1]].allocation_type ==
                   kTfLiteMmapRo,
               NNAPIValidationFailureType::kInputTensorShouldHaveConstantShape,
               "The size input tensor must be constant.", &val_ctx);
      }
      if (android_sdk_version < kMinSdkVersionForNNAPI12) {
        Expect(output_dims->data[1] == output_dims->data[2],
               NNAPIValidationFailureType::kUnsupportedOperandValue,
               "Require width == height due to driver differences in NNAPI "
               "< 1.2",
               &val_ctx);
      }
      auto builtin =
          reinterpret_cast<TfLiteResizeBilinearParams*>(node->builtin_data);
      Expect(!builtin->align_corners,
             NNAPIValidationFailureType::kUnsupportedOperandValue,
             "NNAPI does not support align_corners == true.", &val_ctx);
      if (android_sdk_version < kMinSdkVersionForNNAPI12) {
        Expect(input.type == kTfLiteFloat32,
               NNAPIValidationFailureType::kUnsupportedInputType,
               "NNAPI 1.0 & 1.1 only supports float input.", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinResizeNearestNeighbor: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      auto builtin = reinterpret_cast<TfLiteResizeNearestNeighborParams*>(
          node->builtin_data);
      Expect(!builtin->align_corners,
             NNAPIValidationFailureType::kUnsupportedOperandValue,
             "NNAPI does not support align_corners == true.", &val_ctx);
    } break;
    case kTfLiteBuiltinSqueeze: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI11,
                                 &val_ctx);
      auto builtin = reinterpret_cast<TfLiteSqueezeParams*>(node->builtin_data);
      if (android_sdk_version == kMinSdkVersionForNNAPI11) {
        Expect(builtin->num_squeeze_dims != 0,
               NNAPIValidationFailureType::kUnsupportedOperandValue,
               "NNAPI 1.1 does not support null squeeze_dims properly.",
               &val_ctx);
      }
    } break;
    case kTfLiteBuiltinUnidirectionalSequenceLstm: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);

      Expect(!IsHybridOperator(context, builtin_code, node),
             NNAPIValidationFailureType::kUnsupportedHybridOperator,
             "Hybrid version of this op is not supported by NN API.", &val_ctx);

      Expect(node->inputs->size == 20 || node->inputs->size == 24,
             NNAPIValidationFailureType::kUnsupportedOperatorVariant,
             "Supporting only operation with 20 or 24 inputs", &val_ctx);
    } break;
    case kTfLiteBuiltinL2Normalization: {
      ExpectMaxOpVersion(version, 2, &val_ctx);

      if (android_sdk_version < kMinSdkVersionForNNAPI12) {
        ExpectIsFloatOperator(context, node, &val_ctx);

        const auto& input = context->tensors[node->inputs->data[0]];
        Expect(input.dims->size == 4,
               NNAPIValidationFailureType::kUnsupportedOperatorVariant,
               "Expected 4 inputs", &val_ctx);
      }
      auto builtin = reinterpret_cast<TfLiteL2NormParams*>(node->builtin_data);
      Expect(builtin->activation == kTfLiteActNone,
             NNAPIValidationFailureType::kNoActivationExpected,
             "Expected no activation", &val_ctx);
    } break;
    case kTfLiteBuiltinLocalResponseNormalization: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinLshProjection: {
      ExpectOpVersion(version, 1, &val_ctx);

      if (reinterpret_cast<TfLiteLSHProjectionParams*>(node->builtin_data)
              ->type == kTfLiteLshProjectionSparse) {
        // NNAPI does not support sparse projection correctly pre-Q
        // (b/111751836).
        Expect(android_sdk_version >= kMinSdkVersionForNNAPI12,
               NNAPIValidationFailureType::kUnsupportedInputType,
               "NNAPI does not support sparse projection correctly pre-Q",
               &val_ctx);
        Expect(node->inputs->size == 2,
               NNAPIValidationFailureType::kUnsupportedOperatorVariant,
               " NNAPI does not support weights for sparse projects.",
               &val_ctx);
      }
    } break;
    case kTfLiteBuiltinConcatenation: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      Expect(reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data)
                     ->activation == kTfLiteActNone,
             NNAPIValidationFailureType::kNoActivationExpected,
             "No activation function supported", &val_ctx);
      Expect(context->tensors[node->inputs->data[0]].dims->size <= 4,
             NNAPIValidationFailureType::kUnsupportedOperandRank,
             "Input rank should be less than 4", &val_ctx);

      if (context->tensors[node->inputs->data[0]].type == kTfLiteUInt8 &&
          android_sdk_version < kMinSdkVersionForNNAPI12) {
        auto first_param = context->tensors[node->inputs->data[0]].params;
        for (int i = 1; i < node->inputs->size; i++) {
          auto curr_param = context->tensors[node->inputs->data[i]].params;
          if (!Expect(curr_param.scale == first_param.scale &&
                          curr_param.zero_point == first_param.zero_point,
                      NNAPIValidationFailureType::kUnsupportedOperandValue,
                      "NNAPI 1.0-1 only supported concatenating quantized "
                      "tensor of the same scale and offset.",
                      &val_ctx)) {
            break;
          }
        }
      }
    } break;
    case kTfLiteBuiltinDequantize: {
      Expect(version == 1 || version == 2,
             NNAPIValidationFailureType::kUnsupportedOperatorVersion,
             "Supported op versions are 1 and 2 only", &val_ctx);

      const auto& input = context->tensors[node->inputs->data[0]];
      Expect(input.type != kTfLiteFloat16,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "kTfLiteFloat16 not supported as input", &val_ctx);

      const auto zero_point = input.params.zero_point;
      Expect(input.type != kTfLiteInt8 ||
                 (zero_point == 0 &&
                  android_sdk_version >= kMinSdkVersionForNNAPI12),
             NNAPIValidationFailureType::kUnsupportedInputType,
             "NN API supports int8 type since version 1.2 but only for "
             "symmetric quantization.",
             &val_ctx);
    } break;
    case kTfLiteBuiltinFloor: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinReluN1To1:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinLogistic: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinTanh: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const TfLiteType input_type =
          context->tensors[node->inputs->data[0]].type;
      Expect(IsFloat(input_type) ||
                 (IsQuantized(input_type) &&
                  android_sdk_version >= kMinSdkVersionForNNAPI12),
             NNAPIValidationFailureType::kUnsupportedInputType,
             " NNAPI only support float tanh.", &val_ctx);
    } break;
    case kTfLiteBuiltinSub: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const TfLiteType input_type =
          context->tensors[node->inputs->data[0]].type;
      Expect((android_sdk_version >= kMinSdkVersionForNNAPI11 &&
              IsFloat(input_type)) ||
                 (android_sdk_version >= kMinSdkVersionForNNAPI12 &&
                  IsQuantized(input_type)),
             NNAPIValidationFailureType::kUnsupportedInputType,
             "NNAPI only support float sub.", &val_ctx);
    } break;
    case kTfLiteBuiltinDiv: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI11,
                                 &val_ctx);
      Expect(context->tensors[node->inputs->data[0]].type == kTfLiteFloat32,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "NNAPI only support float div.", &val_ctx);
    } break;
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinPadv2: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI11,
                                 &val_ctx);

      const TfLiteIntArrayView input_shape(
          context->tensors[node->inputs->data[0]].dims);
      Expect(!HasZeroes(input_shape),
             NNAPIValidationFailureType::kUnsupportedOperandValue,
             "NN API pad ops do not support input tensors with no elements",
             &val_ctx);

      Expect(node->inputs->size >= 2,
             NNAPIValidationFailureType::kUnsupportedOperatorVariant,
             "Expecting at least 2 inputs", &val_ctx);

      if (node->inputs->size == 3) {
        // This is going to be mapped with a PadV2
        Expect(
            android_sdk_version >= kMinSdkVersionForNNAPI12,
            NNAPIValidationFailureType::kUnsupportedOperatorVariant,
            "Specification of the padding value is supported from NNAPI 1.2.",
            &val_ctx);
      } else {  // this is going to be mapped as Pad
        if (android_sdk_version < kMinSdkVersionForNNAPI12) {
          Expect(context->tensors[node->inputs->data[0]].type == kTfLiteFloat32,
                 NNAPIValidationFailureType::kUnsupportedInputType,
                 "Only Float32 inputs are supported before NNAPI 1.2",
                 &val_ctx);
        }
      }
    } break;
    case kTfLiteBuiltinUnidirectionalSequenceRnn: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      Expect(!IsHybridOperator(context, builtin_code, node),
             NNAPIValidationFailureType::kUnsupportedHybridOperator,
             "Hybrid version of this op is not supported by NN API.", &val_ctx);
    } break;
    case kTfLiteBuiltinSpaceToBatchNd: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI11,
                                 &val_ctx);
    } break;
    case kTfLiteBuiltinBatchToSpaceNd: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI11,
                                 &val_ctx);
      auto crops = context->tensors[node->inputs->data[2]];
      auto crops_data = crops.data.i32;
      Expect(crops_data && crops.bytes == 16 && crops_data[0] == 0 &&
                 crops_data[1] == 0 && crops_data[2] == 0 && crops_data[3] == 0,
             NNAPIValidationFailureType::kUnsupportedOperandValue,
             "All crops should be 0.", &val_ctx);
    } break;
    case kTfLiteBuiltinStridedSlice: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI11,
                                 &val_ctx);
    } break;
    case kTfLiteBuiltinTranspose: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI11,
                                 &val_ctx);
      // Note that the permutation input tensor value dictates the output
      // dimensions.
      // TODO(b/110888333): Support dynamically-sized tensors in delegates.
      Expect((node->inputs->size > 1) &&
                 (context->tensors[node->inputs->data[1]].allocation_type ==
                  kTfLiteMmapRo),
             NNAPIValidationFailureType::kInputTensorShouldHaveConstantShape,
             "Dynamically-sized tensors not supported.", &val_ctx);
    } break;
    case kTfLiteBuiltinAbs:
    case kTfLiteBuiltinExp:
    case kTfLiteBuiltinLog:
    case kTfLiteBuiltinRsqrt:
    case kTfLiteBuiltinPow: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      ExpectIsFloatOperator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinSlice: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      const auto begin_type = context->tensors[node->inputs->data[1]].type;
      const auto size_type = context->tensors[node->inputs->data[2]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteInt32,
                           kTfLiteUInt8, kTfLiteInt8);
      Expect(begin_type == kTfLiteInt32,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "Begin type should be Int32", &val_ctx);
      Expect(size_type == kTfLiteInt32,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "Size type should be Int32", &val_ctx);
    } break;
    case kTfLiteBuiltinSin: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      ExpectIsFloatOperator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinTransposeConv: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
    } break;
    case kTfLiteBuiltinSqrt: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      ExpectIsFloatOperator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinRnn: {
      ExpectOpVersion(version, 1, &val_ctx);
      Expect(node->inputs->size == 5,
             NNAPIValidationFailureType::kUnsupportedOperatorVariant,
             "Expected 5 input", &val_ctx);
      if (node->inputs->size >= 2) {
        Expect(
            context->tensors[node->inputs->data[/*kWeightsTensor*/ 1]].type ==
                kTfLiteFloat32,
            NNAPIValidationFailureType::kUnsupportedInputType,
            "NNAPI only support float32 weights.", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinSpaceToDepth: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const TfLiteType input_type =
          context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8);
    } break;
    case kTfLiteBuiltinSvdf: {
      ExpectOpVersion(version, 1, &val_ctx);
      Expect(node->inputs->size == 5,
             NNAPIValidationFailureType::kUnsupportedOperandRank,
             "Expected input of rank 5", &val_ctx);
      if (node->inputs->size >= 2) {
        Expect(
            context->tensors[node->inputs->data[/*kWeightsTensor*/ 1]].type ==
                kTfLiteFloat32,
            NNAPIValidationFailureType::kUnsupportedInputType,
            "NNAPI only support float32 weights.", &val_ctx);
      }
      Expect(android_sdk_version >= kMinSdkVersionForNNAPI11,
             NNAPIValidationFailureType::kUnsupportedOperandRank,
             "SVDF does not support rank > 1 on NNAPI 1.0.", &val_ctx);
      Expect(context->tensors[node->inputs->data[/*kWeightsFeatureTensor*/ 1]]
                     .type == kTfLiteFloat32,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "Weights should be Float32", &val_ctx);
    } break;
    case kTfLiteBuiltinLstm: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
      Expect(
          android_sdk_version >= kMinSdkVersionForNNAPI11,
          NNAPIValidationFailureType::kUnsupportedAndroidVersion,
          "NNAPI 1.0 has a bug for optional tensors which would affect LSTM.",
          &val_ctx);
      Expect(android_sdk_version >= kMinSdkVersionForNNAPI12 ||
                 !IsHybridOperator(context, builtin_code, node),
             NNAPIValidationFailureType::kUnsupportedHybridOperator,
             "Hybrid operators not supported before NNAPI 1.2.", &val_ctx);

      const auto weight_input_index =
          isLstmBasicKernel(node) ? 2 /*  basic::kInputWeights */
                                  : 4 /* full::kInputToOutputWeightsTensor */;

      const TfLiteType weight_type =
          context->tensors[node->inputs->data[weight_input_index]].type;

      if (isLstmBasicKernel(node)) {
        Expect(weight_type == kTfLiteUInt8,
               NNAPIValidationFailureType::kUnsupportedInputType,
               "Basic LSTM Kernels support only UINT8 weights", &val_ctx);

        const auto input_quantization_params =
            context->tensors[node->inputs->data[0]].params;
        Expect(input_quantization_params.scale == 1. / 128. &&
                   input_quantization_params.zero_point == 128,
               NNAPIValidationFailureType::kUnsupportedQuantizationParameters,
               "Invalid input quantization", &val_ctx);

        const auto output_quantization_params =
            context->tensors[node->outputs->data[0]].params;
        Expect(output_quantization_params.scale == 1. / 128. &&
                   output_quantization_params.zero_point == 128,
               NNAPIValidationFailureType::kUnsupportedQuantizationParameters,
               "Invalid output quantization", &val_ctx);

        const auto cell_state_quantization_params =
            context->tensors[node->outputs->data[1]].params;
        Expect(cell_state_quantization_params.scale == 16. / 32768. ||
                   cell_state_quantization_params.zero_point == 0,
               NNAPIValidationFailureType::kUnsupportedQuantizationParameters,
               "Invalid cell state quantization", &val_ctx);

        auto is_const_tensor = [&node, &context](int tensor_idx) {
          return context->tensors[node->inputs->data[tensor_idx]]
                     .allocation_type == kTfLiteMmapRo;
        };

        Expect(is_const_tensor(2 /* kInputWeights */),
               NNAPIValidationFailureType::kInputTensorShouldHaveConstantShape,
               "Weights tensor should be constant", &val_ctx);
        Expect(is_const_tensor(3 /* kInputBiases */),
               NNAPIValidationFailureType::kInputTensorShouldHaveConstantShape,
               "Biases tensor should be constant", &val_ctx);

        return val_ctx.is_valid;
      } else {
        if (node->inputs->size == 24) {
          ExpectMinAndroidSdkVersion(android_sdk_version,
                                     kMinSdkVersionForNNAPI12, &val_ctx);
        }

        Expect(weight_type == kTfLiteFloat32 || weight_type == kTfLiteUInt8,
               NNAPIValidationFailureType::kUnsupportedInputType,
               "Weight has to be Float32 or UINT8", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinMean: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI11,
                                 &val_ctx);
      if (android_sdk_version >= kMinSdkVersionForNNAPI12) {
        Expect(context->tensors[node->inputs->data[0]].type == kTfLiteFloat32 ||
                   IsQuantized(context->tensors[node->inputs->data[0]].type),
               NNAPIValidationFailureType::kUnsupportedInputType,
               "Expected Float32 or Quantized input", &val_ctx);
      } else {
        Expect(context->tensors[node->inputs->data[0]].type == kTfLiteFloat32,
               NNAPIValidationFailureType::kUnsupportedInputType,
               "Expected Float32 input", &val_ctx);
      }
      Expect(context->tensors[node->outputs->data[0]].dims->size > 0,
             NNAPIValidationFailureType::kUnsupportedOutputType,
             "NNAPI does not support generating a scalar as output for MEAN.",
             &val_ctx);

      auto input_param = context->tensors[node->inputs->data[0]].params;
      auto output_param = context->tensors[node->outputs->data[0]].params;
      Expect(input_param.scale == output_param.scale &&
                 input_param.zero_point == output_param.zero_point,
             NNAPIValidationFailureType::kUnsupportedOutputType,
             "NNAPI requires that the input and output have the same "
             "quantization parameters.",
             &val_ctx);
    } break;
    case kTfLiteBuiltinEmbeddingLookup: {
      ExpectOpVersion(version, 1, &val_ctx);
      Expect(context->tensors[node->inputs->data[1]].type == kTfLiteFloat32,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "NNAPI only support float32 values.", &val_ctx);
    } break;
    case kTfLiteBuiltinHashtableLookup: {
      ExpectOpVersion(version, 1, &val_ctx);
      Expect(context->tensors[node->outputs->data[0]].type == kTfLiteFloat32,
             NNAPIValidationFailureType::kUnsupportedOutputType,
             "NNAPI only support float32 output.", &val_ctx);
    } break;
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinMinimum: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8, kTfLiteInt32);
    } break;
    case kTfLiteBuiltinCast: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const TfLiteType input_type =
          context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteInt32,
                           kTfLiteUInt8);
      const TfLiteType output_type =
          context->tensors[node->outputs->data[0]].type;
      ExpectTypeIn(output_type, {kTfLiteFloat32, kTfLiteInt32, kTfLiteUInt8},
                   NNAPIValidationFailureType::kUnsupportedOutputType,
                   "Output type should be one of kTfLiteFloat32, kTfLiteInt32, "
                   "kTfLiteUInt8.",
                   &val_ctx);
    } break;
    case kTfLiteBuiltinPrelu: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      ExpectIsFloatOrUint8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinTile: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteInt8,
                           kTfLiteUInt8, kTfLiteInt32);
      const auto multipliers_type =
          context->tensors[node->inputs->data[1]].type;
      Expect(multipliers_type == kTfLiteInt32,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "Multipliers should be Int32", &val_ctx);
    } break;
    case kTfLiteBuiltinLogicalOr:
    case kTfLiteBuiltinLogicalAnd:
    case kTfLiteBuiltinLogicalNot: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      Expect(input_type == kTfLiteBool,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "Input should be bool", &val_ctx);
    } break;
    case kTfLiteBuiltinLess:
    case kTfLiteBuiltinLessEqual:
    case kTfLiteBuiltinGreater:
    case kTfLiteBuiltinGreaterEqual:
    case kTfLiteBuiltinEqual:
    case kTfLiteBuiltinNotEqual: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI11,
                                 &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8, kTfLiteBool, kTfLiteInt32);
    } break;
    case kTfLiteBuiltinNeg: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI11,
                                 &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteInt32);
    } break;
    case kTfLiteBuiltinTopkV2: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const auto& input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteInt32,
                           kTfLiteUInt8, kTfLiteInt8);
      const auto& k_param = context->tensors[node->inputs->data[1]];
      Expect(k_param.type == kTfLiteInt32 &&
                 k_param.allocation_type == kTfLiteMmapRo,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "K param should be a constant of type Int32", &val_ctx);
    } break;
    case kTfLiteBuiltinSelect: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI11,
                                 &val_ctx);
      const auto value_type = context->tensors[node->inputs->data[1]].type;
      EXPECT_INPUT_TYPE_IN(value_type, kTfLiteFloat32, kTfLiteInt32,
                           kTfLiteUInt8, kTfLiteInt8);
      TfLiteIntArray* condition_shape =
          context->tensors[node->inputs->data[0]].dims;
      TfLiteIntArray* input_shape =
          context->tensors[node->inputs->data[1]].dims;
      Expect(TfLiteIntArrayEqual(condition_shape, input_shape),
             NNAPIValidationFailureType::kUnsupportedOperandValue,
             "Condition and inputs tensors shuld have the same shape",
             &val_ctx);
    } break;
    case kTfLiteBuiltinGather: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      const auto& positions = context->tensors[node->inputs->data[1]];
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteFloat16,
                           kTfLiteInt32, kTfLiteUInt8);
      ExpectTypeIn(positions.type,
                   {kTfLiteFloat32, kTfLiteFloat16, kTfLiteInt32, kTfLiteUInt8},
                   NNAPIValidationFailureType::kUnsupportedInputType,
                   "Positions type should be one of kTfLiteFloat32, "
                   "kTfLiteFloat16, kTfLiteInt32, kTfLiteUInt8",
                   &val_ctx);
      Expect(positions.dims->size != 0,
             NNAPIValidationFailureType::kUnsupportedOperandRank,
             "0-dimension args are not supported by NNAPI.", &val_ctx);
    } break;
    case kTfLiteBuiltinBidirectionalSequenceLstm: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      Expect(!IsHybridOperator(context, builtin_code, node),
             NNAPIValidationFailureType::kUnsupportedHybridOperator,
             "Hybrid version of this op is not supported by NN API.", &val_ctx);
    } break;
    case kTfLiteBuiltinExpandDims: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteFloat16,
                           kTfLiteInt32, kTfLiteUInt8, kTfLiteInt8);
      const auto axis = context->tensors[node->inputs->data[1]];
      Expect(axis.type == kTfLiteInt32 && axis.allocation_type == kTfLiteMmapRo,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "NNAPI only supports constant int32 axis tensor.", &val_ctx);
    } break;
    case kTfLiteBuiltinSplit: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      // Tensor indices: split_dim: 0, value: 1
      const TfLiteTensor& input = context->tensors[node->inputs->data[1]];
      EXPECT_INPUT_TYPE_IN(input.type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt32);
      const TfLiteTensor& axis = context->tensors[node->inputs->data[0]];
      Expect(axis.type == kTfLiteInt32 && axis.allocation_type == kTfLiteMmapRo,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "NNAPI only supports constant int32 axis tensor.", &val_ctx);
    } break;
    case kTfLiteBuiltinLogSoftmax: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      Expect(input_type == kTfLiteFloat32,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "Input should be Float32.", &val_ctx);
    } break;
    case kTfLiteBuiltinQuantize: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const auto value_type = context->tensors[node->inputs->data[0]].type;
      Expect(value_type == kTfLiteFloat32,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "Value should be Float32.", &val_ctx);
      const auto output_type = context->tensors[node->outputs->data[0]].type;
      Expect(output_type == kTfLiteUInt8,
             NNAPIValidationFailureType::kUnsupportedOutputType,
             "Output should be kTfLiteUInt8.", &val_ctx);
      const auto quantization_params =
          context->tensors[node->outputs->data[0]].params;
      Expect(quantization_params.scale > 0.f,
             NNAPIValidationFailureType::kUnsupportedQuantizationParameters,
             "Quantization scale should be > 0.", &val_ctx);
    } break;
    case kTfLiteBuiltinReduceAny:
    case kTfLiteBuiltinReduceMin:
    case kTfLiteBuiltinReduceMax: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      Expect(context->tensors[node->outputs->data[0]].dims->size != 0,
             NNAPIValidationFailureType::kUnsupportedOutputType,
             "NNAPI does not support generating a scalar as output.", &val_ctx);
      if (builtin_code == kTfLiteBuiltinReduceProd) {
        const auto input_type = context->tensors[node->inputs->data[0]].type;
        Expect(input_type == kTfLiteFloat32,
               NNAPIValidationFailureType::kUnsupportedInputType,
               "NNAPI only supports floating point REDUCE_PROD.", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinDepthToSpace: {
      const TfLiteType input_type =
          context->tensors[node->inputs->data[0]].type;
      if (version <= 1 &&
          (input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
           input_type == kTfLiteInt8)) {
        return [](const NNAPIOpMappingArgs& mapping_args)
                   -> ANeuralNetworksOperationType {
          auto builtin = reinterpret_cast<TfLiteDepthToSpaceParams*>(
              mapping_args.node->builtin_data);
          mapping_args.builder->AddScalarInt32Operand(builtin->block_size);
          return ANEURALNETWORKS_DEPTH_TO_SPACE;
        };
      }
    } break;
    case kTfLiteBuiltinReduceProd:
    case kTfLiteBuiltinSum: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      Expect(context->tensors[node->outputs->data[0]].dims->size != 0,
             NNAPIValidationFailureType::kUnsupportedOutputType,
             "NNAPI does not support generating a scalar as output", &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      Expect(input_type == kTfLiteFloat32,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "NNAPI only supports floating point input.", &val_ctx);
    } break;
    default:
      // All other operators are not mapped.
      AddValidationFailure(NNAPIValidationFailureType::kUnsupportedOperator,
                           "Unsupported operation type.", &val_ctx);
  }
  return val_ctx.is_valid;
}

TfLiteStatus NNAPIDelegateKernel::Map(
    TfLiteContext* context, int builtin_code, int version,
    int android_sdk_version, const NNAPIOpMappingArgs& mapping_args,
    ANeuralNetworksOperationType* nn_op_type) {
  switch (builtin_code) {
    case kTfLiteBuiltinAdd: {
      auto builtin =
          reinterpret_cast<TfLiteAddParams*>(mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = ANEURALNETWORKS_ADD;
    } break;
    case kTfLiteBuiltinArgMax: {
      *nn_op_type = ANEURALNETWORKS_ARGMAX;
    } break;
    case kTfLiteBuiltinArgMin: {
      *nn_op_type = ANEURALNETWORKS_ARGMIN;
    } break;
    case kTfLiteBuiltinMul: {
      auto builtin =
          reinterpret_cast<TfLiteMulParams*>(mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = ANEURALNETWORKS_MUL;
    } break;
    case kTfLiteBuiltinAveragePool2d: {
      mapping_args.builder->AddPoolingParams(mapping_args.node->builtin_data);
      *nn_op_type = ANEURALNETWORKS_AVERAGE_POOL_2D;
    } break;
    case kTfLiteBuiltinMaxPool2d: {
      mapping_args.builder->AddPoolingParams(mapping_args.node->builtin_data);
      *nn_op_type = ANEURALNETWORKS_MAX_POOL_2D;
    } break;
    case kTfLiteBuiltinL2Pool2d: {
      mapping_args.builder->AddPoolingParams(mapping_args.node->builtin_data);
      *nn_op_type = ANEURALNETWORKS_L2_POOL_2D;
    } break;
    case kTfLiteBuiltinConv2d: {
      auto builtin =
          reinterpret_cast<TfLiteConvParams*>(mapping_args.node->builtin_data);
      // NNAPI supports dilated Conv2D since NNAPI 1.2.
      if (builtin->dilation_width_factor != 1 ||
          builtin->dilation_height_factor != 1) {
        auto builtin = reinterpret_cast<TfLiteConvParams*>(
            mapping_args.node->builtin_data);
        mapping_args.builder->AddScalarInt32Operand(builtin->padding);
        mapping_args.builder->AddScalarInt32Operand(builtin->stride_width);
        mapping_args.builder->AddScalarInt32Operand(builtin->stride_height);
        mapping_args.builder->AddScalarInt32Operand(builtin->activation);
        mapping_args.builder->AddScalarBoolOperand(false);  // Use NHWC format
        mapping_args.builder->AddScalarInt32Operand(
            builtin->dilation_width_factor);
        mapping_args.builder->AddScalarInt32Operand(
            builtin->dilation_height_factor);
      } else {
        auto builtin = reinterpret_cast<TfLiteConvParams*>(
            mapping_args.node->builtin_data);
        mapping_args.builder->AddScalarInt32Operand(builtin->padding);
        mapping_args.builder->AddScalarInt32Operand(builtin->stride_width);
        mapping_args.builder->AddScalarInt32Operand(builtin->stride_height);
        mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      }
      *nn_op_type = ANEURALNETWORKS_CONV_2D;
    } break;
    case kTfLiteBuiltinDepthwiseConv2d: {
      auto builtin = reinterpret_cast<TfLiteDepthwiseConvParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->padding);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_width);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_height);
      mapping_args.builder->AddScalarInt32Operand(builtin->depth_multiplier);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      if (builtin->dilation_width_factor != 1 ||
          builtin->dilation_height_factor != 1) {
        mapping_args.builder->AddScalarBoolOperand(false);  // Use NHWC format.
        mapping_args.builder->AddScalarInt32Operand(
            builtin->dilation_width_factor);
        mapping_args.builder->AddScalarInt32Operand(
            builtin->dilation_height_factor);
      }
      *nn_op_type = ANEURALNETWORKS_DEPTHWISE_CONV_2D;
    } break;
    case kTfLiteBuiltinFullyConnected: {
      auto builtin = reinterpret_cast<TfLiteFullyConnectedParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = ANEURALNETWORKS_FULLY_CONNECTED;
    } break;
    case kTfLiteBuiltinSoftmax: {
      auto builtin = reinterpret_cast<TfLiteSoftmaxParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarFloat32Operand(builtin->beta);
      // Optional scalar specifying the dimension the activation would be
      // performed on is not added. Default to -1.
      *nn_op_type = ANEURALNETWORKS_SOFTMAX;
    } break;
    case kTfLiteBuiltinReshape: {
      *nn_op_type = ANEURALNETWORKS_RESHAPE;
    } break;
    case kTfLiteBuiltinResizeBilinear: {
      const int output_id = mapping_args.node->outputs->data[0];
      auto& output = mapping_args.context->tensors[output_id];
      const int output_height = output.dims->data[1];
      const int output_width = output.dims->data[2];
      mapping_args.builder->AddScalarInt32Operand(output_width);
      mapping_args.builder->AddScalarInt32Operand(output_height);
      *nn_op_type = ANEURALNETWORKS_RESIZE_BILINEAR;
    } break;
    case kTfLiteBuiltinResizeNearestNeighbor: {
      const TfLiteTensor& new_shape =
          mapping_args.context->tensors[mapping_args.node->inputs->data[1]];
      // NNAPI uses scalar inputs for height and width.
      mapping_args.builder->AddScalarInt32Operand(new_shape.data.i32[1]);
      mapping_args.builder->AddScalarInt32Operand(new_shape.data.i32[0]);
      mapping_args.builder->AddScalarBoolOperand(false);  // Use NHWC format

      *nn_op_type = ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR;
    } break;
    case kTfLiteBuiltinSqueeze: {
      auto builtin = reinterpret_cast<TfLiteSqueezeParams*>(
          mapping_args.node->builtin_data);
      // Note that we add the squeeze dimensions even if the dimensions
      // were unspecified (empty), as NNAPI requires the operand.
      mapping_args.builder->AddVectorInt32Operand(
          builtin->num_squeeze_dims ? builtin->squeeze_dims : nullptr,
          static_cast<uint32_t>(builtin->num_squeeze_dims));
      *nn_op_type = ANEURALNETWORKS_SQUEEZE;
    } break;
    case kTfLiteBuiltinUnidirectionalSequenceLstm: {
      auto builtin = reinterpret_cast<TfLiteUnidirectionalSequenceLSTMParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      mapping_args.builder->AddScalarFloat32Operand(builtin->cell_clip);
      mapping_args.builder->AddScalarFloat32Operand(builtin->proj_clip);
      mapping_args.builder->AddScalarBoolOperand(builtin->time_major);
      const bool hybrid_op = IsHybridOperator(
          mapping_args.context, kTfLiteBuiltinUnidirectionalSequenceLstm,
          mapping_args.node);
      if (mapping_args.node->inputs->size == 24) {
        // Add layer normalization tensors if they are provided.
        for (int i = 20; i < 24; ++i) {
          const int input_index = mapping_args.node->inputs->data[i];
          if (input_index != kTfLiteOptionalTensor) {
            mapping_args.builder->AddTensorInput(input_index, hybrid_op);
          } else {
            mapping_args.builder->AddVectorFloat32Operand(nullptr, 0);
          }
        }
      } else {
        for (int i = 0; i < 4; ++i) {
          mapping_args.builder->AddVectorFloat32Operand(nullptr, 0);
        }
      }

      *nn_op_type = ANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_LSTM;
    } break;
    case kTfLiteBuiltinL2Normalization: {
      *nn_op_type = ANEURALNETWORKS_L2_NORMALIZATION;
    } break;
    case kTfLiteBuiltinLocalResponseNormalization: {
      auto builtin = reinterpret_cast<TfLiteLocalResponseNormParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->radius);
      mapping_args.builder->AddScalarFloat32Operand(builtin->bias);
      mapping_args.builder->AddScalarFloat32Operand(builtin->alpha);
      mapping_args.builder->AddScalarFloat32Operand(builtin->beta);
      *nn_op_type = ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION;
    } break;
    case kTfLiteBuiltinLshProjection: {
      auto builtin = reinterpret_cast<TfLiteLSHProjectionParams*>(
          mapping_args.node->builtin_data);
      int type = builtin->type;
      // In Android Q+, NNAPI uses 3 to denote
      // kTfLiteLshProjectionSparse.
      const int kNNAPILshProjectionSparse = 3;
      if (builtin->type == kTfLiteLshProjectionSparse) {
        type = kNNAPILshProjectionSparse;
        // Add NNAPI null weight operand.
        mapping_args.builder->AddVectorFloat32Operand(nullptr, 0);
      }
      mapping_args.builder->AddScalarInt32Operand(type);
      *nn_op_type = ANEURALNETWORKS_LSH_PROJECTION;
    } break;
    case kTfLiteBuiltinConcatenation: {
      auto builtin = reinterpret_cast<TfLiteConcatenationParams*>(
          mapping_args.node->builtin_data);
      int axis = builtin->axis < 0
                     ? mapping_args.context
                               ->tensors[mapping_args.node->inputs->data[0]]
                               .dims->size +
                           builtin->axis
                     : builtin->axis;
      mapping_args.builder->AddScalarInt32Operand(axis);
      *nn_op_type = ANEURALNETWORKS_CONCATENATION;
    } break;
    case kTfLiteBuiltinDequantize: {
      *nn_op_type = ANEURALNETWORKS_DEQUANTIZE;
    } break;
    case kTfLiteBuiltinFloor: {
      *nn_op_type = ANEURALNETWORKS_FLOOR;
    } break;
    case kTfLiteBuiltinRelu: {
      *nn_op_type = ANEURALNETWORKS_RELU;
    } break;
    case kTfLiteBuiltinReluN1To1: {
      *nn_op_type = ANEURALNETWORKS_RELU1;
    } break;
    case kTfLiteBuiltinRelu6: {
      *nn_op_type = ANEURALNETWORKS_RELU6;
    } break;
    case kTfLiteBuiltinLogistic: {
      *nn_op_type = ANEURALNETWORKS_LOGISTIC;
    } break;
    case kTfLiteBuiltinTanh: {
      *nn_op_type = ANEURALNETWORKS_TANH;
    } break;
    case kTfLiteBuiltinSub: {
      auto builtin =
          reinterpret_cast<TfLiteSubParams*>(mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = ANEURALNETWORKS_SUB;
    } break;
    case kTfLiteBuiltinDiv: {
      auto builtin =
          reinterpret_cast<TfLiteDivParams*>(mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = ANEURALNETWORKS_DIV;
    } break;
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinPadv2: {
      // We want to map to PAD as much as possible since it is more widely
      // supported. We map to PadV2 only when there is the need to specify
      // the padding value
      if (mapping_args.node->inputs->size == 2) {
        *nn_op_type = ANEURALNETWORKS_PAD;
      } else {
        const int constant_value_id = mapping_args.node->inputs->data[2];
        if (constant_value_id == kTfLiteOptionalTensor) {
          *nn_op_type = ANEURALNETWORKS_PAD;
        } else {
          *nn_op_type = ANEURALNETWORKS_PAD_V2;
        }
      }
    } break;
    case kTfLiteBuiltinUnidirectionalSequenceRnn: {
      auto builtin = reinterpret_cast<TfLiteSequenceRNNParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      mapping_args.builder->AddScalarInt32Operand(builtin->time_major);
      *nn_op_type = ANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_RNN;
    } break;
    case kTfLiteBuiltinSpaceToBatchNd: {
      *nn_op_type = ANEURALNETWORKS_SPACE_TO_BATCH_ND;
    } break;
    case kTfLiteBuiltinBatchToSpaceNd: {
      *nn_op_type = ANEURALNETWORKS_BATCH_TO_SPACE_ND;
    } break;
    case kTfLiteBuiltinStridedSlice: {
      auto builtin = reinterpret_cast<TfLiteStridedSliceParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->begin_mask);
      mapping_args.builder->AddScalarInt32Operand(builtin->end_mask);
      mapping_args.builder->AddScalarInt32Operand(builtin->shrink_axis_mask);
      *nn_op_type = ANEURALNETWORKS_STRIDED_SLICE;
    } break;
    case kTfLiteBuiltinTranspose: {
      *nn_op_type = ANEURALNETWORKS_TRANSPOSE;
    } break;
    case kTfLiteBuiltinAbs: {
      *nn_op_type = ANEURALNETWORKS_ABS;
    } break;
    case kTfLiteBuiltinExp: {
      *nn_op_type = ANEURALNETWORKS_EXP;
    } break;
    case kTfLiteBuiltinLog: {
      *nn_op_type = ANEURALNETWORKS_LOG;
    } break;
    case kTfLiteBuiltinRsqrt: {
      *nn_op_type = ANEURALNETWORKS_RSQRT;
    } break;
    case kTfLiteBuiltinPow: {
      *nn_op_type = ANEURALNETWORKS_POW;
    } break;
    case kTfLiteBuiltinSlice: {
      *nn_op_type = ANEURALNETWORKS_SLICE;
    } break;
    case kTfLiteBuiltinSin: {
      *nn_op_type = ANEURALNETWORKS_SIN;
    } break;
    case kTfLiteBuiltinTransposeConv: {
      const bool hybrid_op = IsHybridOperator(
          mapping_args.context, kTfLiteBuiltinTransposeConv, mapping_args.node);
      mapping_args.builder->AddTensorInput(
          mapping_args.node->inputs->data[/*kDataInputTensor*/ 2], hybrid_op);
      mapping_args.builder->AddTensorInput(
          mapping_args.node->inputs->data[/*kWeightsTensor*/ 1], hybrid_op);

      // NNAPI requires a bias tensor, so we allocate a new tensor to fill
      // it with zeroes. It is deleted with other tensors in the context
      // during subgraph destructor call.
      int bias_index = -1;
      mapping_args.context->AddTensors(mapping_args.context, 1, &bias_index);
      TfLiteTensor* bias_tensor = &mapping_args.context->tensors[bias_index];
      const auto input_type =
          mapping_args.context
              ->tensors[mapping_args.node->inputs->data[/*kDataInputTensor*/ 2]]
              .type;
      if (input_type == kTfLiteFloat32) {
        bias_tensor->type = kTfLiteFloat32;
      } else {
        bias_tensor->type = kTfLiteInt32;
      }

      // Create an array with a required bias shape and resize the bias
      // tensor.
      TfLiteIntArray* bias_shape = TfLiteIntArrayCreate(1);
      const TfLiteTensor& output_shape =
          mapping_args.context->tensors[mapping_args.node->inputs
                                            ->data[/*kOutputShapeTensor*/ 0]];
      const int output_depth = output_shape.data.i32[3];
      bias_shape->data[0] = output_depth;
      bias_tensor->allocation_type = kTfLiteDynamic;
      mapping_args.context->ResizeTensor(mapping_args.context, bias_tensor,
                                         bias_shape);

      // Set tensor's values to zeroes and add it using AddVector*, so
      // that the values are copied to NNAPI. We don't use the AddTensor
      // function because it doesn't copy values and the tensor we just
      // created is not in the node->inputs.
      if (input_type == kTfLiteFloat32) {
        memset(bias_tensor->data.f, 0, output_depth * sizeof(float));
        mapping_args.builder->AddVectorFloat32Operand(bias_tensor->data.f,
                                                      output_depth);
      } else {
        memset(bias_tensor->data.i32, 0, output_depth * sizeof(int));
        const TfLiteTensor& input_tensor =
            mapping_args.context->tensors[mapping_args.node->inputs
                                              ->data[/*kDataInputTensor*/ 2]];
        const TfLiteTensor& filter_tensor =
            mapping_args.context->tensors[mapping_args.node->inputs
                                              ->data[/*kWeightsTensor*/ 1]];
        // NNAPI requires bias scale to be a product of an input scale and
        // a filter scale.
        bias_tensor->params.scale =
            input_tensor.params.scale * filter_tensor.params.scale;
        mapping_args.builder->AddVectorInt32Operand(
            bias_tensor->data.i32, output_depth,
            input_tensor.params.scale * filter_tensor.params.scale,
            /*zero_point=*/0);
      }

      mapping_args.builder->AddTensorInput(
          mapping_args.node->inputs->data[/*kOutputShapeTensor*/ 0], hybrid_op);

      auto builtin = reinterpret_cast<TfLiteTransposeConvParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->padding);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_width);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_height);
      mapping_args.builder->AddScalarInt32Operand(
          /*ANEURALNETWORKS_FUSED_NONE*/ 0);
      // Use NHWC layout for input and output.
      mapping_args.builder->AddScalarBoolOperand(false);
      *nn_op_type = ANEURALNETWORKS_TRANSPOSE_CONV;
    } break;
    case kTfLiteBuiltinSqrt: {
      *nn_op_type = ANEURALNETWORKS_SQRT;
    } break;
    case kTfLiteBuiltinRnn: {
      // NNAPI need both state_in and state_out.
      int ann_index;
      mapping_args.builder->AddStateFloat32Tensor(
          mapping_args.node->inputs->data[/*kHiddenStateTensor*/ 4],
          &ann_index);
      mapping_args.model_state_outputs->push_back(ann_index);
      mapping_args.model_state_tfl_inputs->push_back(
          mapping_args.node->inputs->data[/*kHiddenStateTensor*/ 4]);
      auto builtin =
          reinterpret_cast<TfLiteRNNParams*>(mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = ANEURALNETWORKS_RNN;
    } break;
    case kTfLiteBuiltinSpaceToDepth: {
      auto builtin = reinterpret_cast<TfLiteSpaceToDepthParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->block_size);
      *nn_op_type = ANEURALNETWORKS_SPACE_TO_DEPTH;
    } break;
    case kTfLiteBuiltinSvdf: {
      // NNAPI need both state_in and state_out.
      int ann_index;
      mapping_args.builder->AddStateFloat32Tensor(
          mapping_args.node->inputs->data[/*kInputActivationStateTensor*/ 4],
          &ann_index);
      mapping_args.model_state_outputs->push_back(ann_index);
      mapping_args.model_state_tfl_inputs->push_back(
          mapping_args.node->inputs->data[/*kInputActivationStateTensor*/ 4]);

      auto builtin =
          reinterpret_cast<TfLiteSVDFParams*>(mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->rank);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = ANEURALNETWORKS_SVDF;
    } break;
    case kTfLiteBuiltinLstm: {
      if (isLstmBasicKernel(mapping_args.node)) {
        const auto output_dims =
            mapping_args.context->tensors[mapping_args.node->outputs->data[1]]
                .dims;

        // Inputs kInputData
        mapping_args.builder->AddTensorInput(
            mapping_args.node->inputs->data[0 /* kInputData */],
            /* hybrid_op */ false,
            /* scalar_as_tensor */ false);

        // The 8 weights tensors are set decomposing the
        // kInputWeights param
        const auto weight_tensor =
            mapping_args.context->tensors[mapping_args.node->inputs
                                              ->data[2 /* kInputWeights */]];

        std::vector<uint8_t> recurrent_to_input;
        std::vector<uint8_t> input_to_input;
        std::vector<uint8_t> recurrent_to_cell;
        std::vector<uint8_t> input_to_cell;
        std::vector<uint8_t> recurrent_to_forget;
        std::vector<uint8_t> input_to_forget;
        std::vector<uint8_t> recurrent_to_output;
        std::vector<uint8_t> input_to_output;
        tflite::delegate::nnapi::DecomposeQuantLstmWeightsTensor(
            weight_tensor.data.uint8, weight_tensor.dims, &recurrent_to_input,
            &input_to_input, &recurrent_to_cell, &input_to_cell,
            &recurrent_to_forget, &input_to_forget, &recurrent_to_output,
            &input_to_output);

        TfLiteIntArray* recurrent_weight_dims = TfLiteIntArrayCreate(2);
        TfLiteIntArray* input_weight_dims = TfLiteIntArrayCreate(2);
        tflite::delegate::nnapi::SetWeightSubmatrixDims(
            weight_tensor.dims, recurrent_weight_dims, input_weight_dims);

        int new_tensor_index = -1;

        mapping_args.builder->AddNewInputConstantTensor<uint8_t>(
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, kTfLiteUInt8,
            input_weight_dims, input_to_input, weight_tensor.params,
            &new_tensor_index);

        mapping_args.builder->AddNewInputConstantTensor<uint8_t>(
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, kTfLiteUInt8,
            input_weight_dims, input_to_forget, weight_tensor.params,
            &new_tensor_index);

        mapping_args.builder->AddNewInputConstantTensor<uint8_t>(
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, kTfLiteUInt8,
            input_weight_dims, input_to_cell, weight_tensor.params,
            &new_tensor_index);

        mapping_args.builder->AddNewInputConstantTensor<uint8_t>(
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, kTfLiteUInt8,
            input_weight_dims, input_to_output, weight_tensor.params,
            &new_tensor_index);

        mapping_args.builder->AddNewInputConstantTensor<uint8_t>(
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, kTfLiteUInt8,
            recurrent_weight_dims, recurrent_to_input, weight_tensor.params,
            &new_tensor_index);

        mapping_args.builder->AddNewInputConstantTensor<uint8_t>(
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, kTfLiteUInt8,
            recurrent_weight_dims, recurrent_to_forget, weight_tensor.params,
            &new_tensor_index);

        mapping_args.builder->AddNewInputConstantTensor<uint8_t>(
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, kTfLiteUInt8,
            recurrent_weight_dims, recurrent_to_cell, weight_tensor.params,
            &new_tensor_index);

        mapping_args.builder->AddNewInputConstantTensor<uint8_t>(
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, kTfLiteUInt8,
            recurrent_weight_dims, recurrent_to_output, weight_tensor.params,
            &new_tensor_index);

        TfLiteIntArrayFree(input_weight_dims);
        TfLiteIntArrayFree(recurrent_weight_dims);

        // Biases have to be split in four.
        const auto bias_size = output_dims->data[1];
        const TfLiteTensor& biases_tensor =
            mapping_args.context->tensors[mapping_args.node->inputs
                                              ->data[3 /* kInputBiases */]];

        std::vector<int32_t> input_bias;
        std::vector<int32_t> cell_bias;
        std::vector<int32_t> forget_bias;
        std::vector<int32_t> output_bias;
        delegate::nnapi::DecomposeBiasTensor(biases_tensor.data.i32, bias_size,
                                             &input_bias, &cell_bias,
                                             &forget_bias, &output_bias);

        int input_bias_tensor = -1;
        mapping_args.builder->AddNewInputConstantTensor<int32_t>(
            ANEURALNETWORKS_TENSOR_INT32, kTfLiteInt32, {bias_size}, input_bias,
            biases_tensor.params, &input_bias_tensor);
        int forget_bias_tensor = -1;
        mapping_args.builder->AddNewInputConstantTensor(
            ANEURALNETWORKS_TENSOR_INT32, kTfLiteInt32, {bias_size},
            forget_bias, biases_tensor.params, &forget_bias_tensor);
        int cell_gate_bias_tensor = -1;
        mapping_args.builder->AddNewInputConstantTensor(
            ANEURALNETWORKS_TENSOR_INT32, kTfLiteInt32, {bias_size}, cell_bias,
            biases_tensor.params, &cell_gate_bias_tensor);
        int output_gate_bias_tensor = -1;
        mapping_args.builder->AddNewInputConstantTensor(
            ANEURALNETWORKS_TENSOR_INT32, kTfLiteInt32, {bias_size},
            output_bias, biases_tensor.params, &output_gate_bias_tensor);

        mapping_args.builder->AddTensorInput(
            mapping_args.node->inputs->data[4 /* kInputPrevState */],
            /* hybrid_op */ false,
            /* scalar_as_tensor */ false);

        // kInputPrevActivation
        mapping_args.builder->AddTensorInput(
            mapping_args.node->inputs->data[1 /* kInputPrevActivation */],
            /* hybrid_op */ false,
            /* scalar_as_tensor */ false);

        // Configuring the copy from the activation, state outputs
        // to their associated inputs
        mapping_args.feedback_loops->push_back(std::make_tuple(
            0 /*kOutputActivation*/, 1 /*kInputPrevActivation*/));

        mapping_args.feedback_loops->push_back(
            std::make_tuple(1 /*kOutputState*/, 4 /*kInputPrevState*/));

        // OUTPUTS
        // Setting only the first two since the remaining ones are
        // ignored by NNAPI
        mapping_args.builder->AddTensorOutput(
            mapping_args.node->outputs->data[1 /* kOutputState */], 0);

        mapping_args.builder->AddTensorOutput(
            mapping_args.node->outputs
                ->data[0 /* kOutputkOutputActivationState */],
            0);

        *nn_op_type = ANEURALNETWORKS_QUANTIZED_16BIT_LSTM;
      } else {
        auto builtin = reinterpret_cast<TfLiteLSTMParams*>(
            mapping_args.node->builtin_data);
        mapping_args.builder->AddScalarInt32Operand(builtin->activation);
        mapping_args.builder->AddScalarFloat32Operand(builtin->cell_clip);
        mapping_args.builder->AddScalarFloat32Operand(builtin->proj_clip);

        // Current NNAPI implementation requires the scratch_buffer as
        // output.
        mapping_args.builder->AddAdditionalFloat32OutputTensor(2);

        // NNAPI need both state_in and state_out for cell_state and
        // output_state.
        int ann_index;
        mapping_args.builder->AddStateFloat32Tensor(
            mapping_args.node->inputs->data[/*kInputActivationStateTensor*/ 18],
            &ann_index);
        mapping_args.model_state_outputs->push_back(ann_index);
        mapping_args.model_state_tfl_inputs->push_back(
            mapping_args.node->inputs
                ->data[/*kInputActivationStateTensor*/ 18]);
        mapping_args.builder->AddStateFloat32Tensor(
            mapping_args.node->inputs->data[/*kInputCellStateTensor*/ 19],
            &ann_index);
        mapping_args.model_state_outputs->push_back(ann_index);
        mapping_args.model_state_tfl_inputs->push_back(
            mapping_args.node->inputs->data[/*kInputCellStateTensor*/ 19]);

        const bool hybrid_op = IsHybridOperator(
            mapping_args.context, kTfLiteBuiltinLstm, mapping_args.node);

        if (mapping_args.node->inputs->size == 24) {
          for (int i = 20; i < 24; ++i) {
            const auto input_index = mapping_args.node->inputs->data[i];
            if (input_index != kTfLiteOptionalTensor) {
              mapping_args.builder->AddTensorInput(input_index, hybrid_op);
            } else {
              mapping_args.builder->AddVectorFloat32Operand(nullptr, 0);
            }
          }
        }

        *nn_op_type = ANEURALNETWORKS_LSTM;
      }
    } break;
    case kTfLiteBuiltinMean: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          mapping_args.node->builtin_data);
      int32_t keep_dims = 0;
      if (builtin->keep_dims) keep_dims = 1;
      mapping_args.builder->AddScalarInt32Operand(keep_dims);
      *nn_op_type = ANEURALNETWORKS_MEAN;
    } break;
    case kTfLiteBuiltinEmbeddingLookup: {
      *nn_op_type = ANEURALNETWORKS_EMBEDDING_LOOKUP;
    } break;
    case kTfLiteBuiltinHashtableLookup: {
      *nn_op_type = ANEURALNETWORKS_HASHTABLE_LOOKUP;
    } break;
    case kTfLiteBuiltinMaximum: {
      *nn_op_type = ANEURALNETWORKS_MAXIMUM;
    } break;
    case kTfLiteBuiltinMinimum: {
      *nn_op_type = ANEURALNETWORKS_MINIMUM;
    } break;
    case kTfLiteBuiltinCast: {
      *nn_op_type = ANEURALNETWORKS_CAST;
    } break;
    case kTfLiteBuiltinPrelu: {
      *nn_op_type = ANEURALNETWORKS_PRELU;
    } break;
    case kTfLiteBuiltinTile: {
      *nn_op_type = ANEURALNETWORKS_TILE;
    } break;
    case kTfLiteBuiltinLogicalOr: {
      *nn_op_type = ANEURALNETWORKS_LOGICAL_OR;
    } break;
    case kTfLiteBuiltinLogicalAnd: {
      *nn_op_type = ANEURALNETWORKS_LOGICAL_AND;
    } break;
    case kTfLiteBuiltinLogicalNot: {
      *nn_op_type = ANEURALNETWORKS_LOGICAL_NOT;
    } break;
    case kTfLiteBuiltinLess: {
      *nn_op_type = ANEURALNETWORKS_LESS;
    } break;
    case kTfLiteBuiltinLessEqual: {
      *nn_op_type = ANEURALNETWORKS_LESS_EQUAL;
    } break;
    case kTfLiteBuiltinGreater: {
      *nn_op_type = ANEURALNETWORKS_GREATER;
    } break;
    case kTfLiteBuiltinGreaterEqual: {
      *nn_op_type = ANEURALNETWORKS_GREATER_EQUAL;
    } break;
    case kTfLiteBuiltinEqual: {
      *nn_op_type = ANEURALNETWORKS_EQUAL;
    } break;
    case kTfLiteBuiltinNotEqual: {
      *nn_op_type = ANEURALNETWORKS_NOT_EQUAL;
    } break;
    case kTfLiteBuiltinNeg: {
      *nn_op_type = ANEURALNETWORKS_NEG;
    } break;
    case kTfLiteBuiltinTopkV2: {
      const TfLiteTensor& k_param =
          mapping_args.context->tensors[mapping_args.node->inputs->data[1]];
      mapping_args.builder->AddScalarInt32Operand(*k_param.data.i32);
      *nn_op_type = ANEURALNETWORKS_TOPK_V2;
    } break;
    case kTfLiteBuiltinSelect: {
      *nn_op_type = ANEURALNETWORKS_SELECT;
    } break;
    case kTfLiteBuiltinGather: {
      auto builtin = reinterpret_cast<TfLiteGatherParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddTensorInput(mapping_args.node->inputs->data[0],
                                           /* hybrid_op */ false,
                                           /* scalar_as_tensor */ false);

      mapping_args.builder->AddScalarInt32Operand(builtin->axis);

      mapping_args.builder->AddTensorInput(mapping_args.node->inputs->data[1],
                                           /* hybrid_op */ false,
                                           /* scalar_as_tensor */ false);

      *nn_op_type = ANEURALNETWORKS_GATHER;
    } break;
    case kTfLiteBuiltinBidirectionalSequenceLstm: {
      auto builtin = reinterpret_cast<TfLiteBidirectionalSequenceLSTMParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      mapping_args.builder->AddScalarFloat32Operand(builtin->cell_clip);
      mapping_args.builder->AddScalarFloat32Operand(builtin->proj_clip);
      mapping_args.builder->AddScalarBoolOperand(builtin->merge_outputs);
      mapping_args.builder->AddScalarBoolOperand(builtin->time_major);
      // TF Lite doesn't support layer normalization in bidirectional
      // sequence LSTM, so we insert optional tensors for NNAPI.
      for (int i = 0; i < 8; ++i) {
        mapping_args.builder->AddVectorFloat32Operand(nullptr, 0);
      }
      *nn_op_type = ANEURALNETWORKS_BIDIRECTIONAL_SEQUENCE_LSTM;
    } break;
    case kTfLiteBuiltinExpandDims: {
      const TfLiteTensor& axis_param =
          mapping_args.context->tensors[mapping_args.node->inputs->data[1]];
      mapping_args.builder->AddScalarInt32Operand(*axis_param.data.i32);
      *nn_op_type = ANEURALNETWORKS_EXPAND_DIMS;
    } break;
    case kTfLiteBuiltinSplit: {
      const TfLiteTensor& axis =
          mapping_args.context->tensors[mapping_args.node->inputs->data[0]];
      auto builtin =
          reinterpret_cast<TfLiteSplitParams*>(mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(*axis.data.i32);
      mapping_args.builder->AddScalarInt32Operand(builtin->num_splits);
      *nn_op_type = ANEURALNETWORKS_SPLIT;
    } break;
    case kTfLiteBuiltinLogSoftmax: {
      // Scaling and axis are hardcoded to respectively 1 and -1
      // in TFLite.
      mapping_args.builder->AddScalarFloat32Operand(1);
      mapping_args.builder->AddScalarInt32Operand(-1);
      *nn_op_type = ANEURALNETWORKS_LOG_SOFTMAX;
    } break;
    case kTfLiteBuiltinQuantize: {
      *nn_op_type = ANEURALNETWORKS_QUANTIZE;
    } break;
    case kTfLiteBuiltinReduceAny: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarBoolOperand(builtin->keep_dims);
      *nn_op_type = ANEURALNETWORKS_REDUCE_ANY;
    } break;
    case kTfLiteBuiltinReduceMin: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarBoolOperand(builtin->keep_dims);
      *nn_op_type = ANEURALNETWORKS_REDUCE_MIN;
    } break;
    case kTfLiteBuiltinReduceMax: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarBoolOperand(builtin->keep_dims);
      *nn_op_type = ANEURALNETWORKS_REDUCE_MAX;
    } break;
    case kTfLiteBuiltinReduceProd: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarBoolOperand(builtin->keep_dims);
      *nn_op_type = ANEURALNETWORKS_REDUCE_PROD;
    } break;
    case kTfLiteBuiltinSum: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarBoolOperand(builtin->keep_dims);
      *nn_op_type = ANEURALNETWORKS_REDUCE_SUM;
    } break;
    default:
      // All other operators are not mapped.
      return kTfLiteError;
  }
  return kTfLiteOk;
}

// Initialize the kernel (a NN model).
TfLiteStatus NNAPIDelegateKernel::Init(TfLiteContext* context,
                                       const TfLiteDelegateParams* params,
                                       int* nnapi_errno) {
  for (auto node_index : TfLiteIntArrayView(params->nodes_to_replace)) {
    nodes_.push_back(node_index);
  }

  const auto delegate_options =
      StatefulNnApiDelegate::GetOptions(params->delegate);
  const char* device_name_ptr = delegate_options.accelerator_name;
  if (nnapi_->android_sdk_version >= kMinSdkVersionForNNAPI12) {
    if (device_name_ptr != nullptr) {
      // User specified an accelerator to use.
      ANeuralNetworksDevice* nnapi_device =
          GetDeviceHandle(context, device_name_ptr);
      if (nnapi_device == nullptr) {
        return kTfLiteError;
      }
      nnapi_devices_.push_back(nnapi_device);
    } else if (delegate_options.disallow_nnapi_cpu) {
      std::string nnapi_cpu("nnapi-reference");
      uint32_t num_devices = 0;
      NnApiImplementation()->ANeuralNetworks_getDeviceCount(&num_devices);

      for (uint32_t i = 0; i < num_devices; i++) {
        ANeuralNetworksDevice* device = nullptr;
        const char* buffer = nullptr;
        NnApiImplementation()->ANeuralNetworks_getDevice(i, &device);
        NnApiImplementation()->ANeuralNetworksDevice_getName(device, &buffer);
        if (nnapi_cpu != buffer) {
          nnapi_devices_.push_back(device);
        }
      }
      if (nnapi_devices_.empty()) {
        context->ReportError(
            context, "NNAPI delegate requested but no accelerators available.");
        return kTfLiteError;
      }
    }
  }

  // Mark the handle backed tensors.
  tensor_memory_map_ =
      &StatefulNnApiDelegate::GetTensorMemoryMap(params->delegate);

  if (!nn_model_) {
    ANeuralNetworksModel* model = nullptr;
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context, nnapi_->ANeuralNetworksModel_create(&model), nnapi_errno);
    nn_model_.reset(model);

    TF_LITE_ENSURE_STATUS(BuildGraph(context, params->input_tensors,
                                     params->output_tensors, nnapi_errno));
  }

  if (!nn_compilation_) {
    ANeuralNetworksCompilation* compilation = nullptr;
    if (!nnapi_devices_.empty()) {
      // Compile for the selected accelerator.
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context,
          nnapi_->ANeuralNetworksCompilation_createForDevices(
              nn_model_.get(), nnapi_devices_.data(), nnapi_devices_.size(),
              &compilation),
          nnapi_errno);
    } else {
      RETURN_TFLITE_ERROR_IF_NN_ERROR(context,
                                      nnapi_->ANeuralNetworksCompilation_create(
                                          nn_model_.get(), &compilation),
                                      nnapi_errno);
    }

    auto preference = delegate_options.execution_preference;
    if (preference !=
        StatefulNnApiDelegate::Options::ExecutionPreference::kUndefined) {
      const int preference_result =
          nnapi_->ANeuralNetworksCompilation_setPreference(compilation,
                                                           preference);
      if (preference_result != ANEURALNETWORKS_NO_ERROR) {
        nnapi_->ANeuralNetworksCompilation_free(compilation);
        compilation = nullptr;
      }
      RETURN_TFLITE_ERROR_IF_NN_ERROR(context, preference_result, nnapi_errno);
    }

    const char* cache_dir = delegate_options.cache_dir;
    const char* model_token = delegate_options.model_token;
    if (nnapi_->android_sdk_version >= kMinSdkVersionForNNAPI12 && cache_dir &&
        model_token) {
      // Compilation caching could be enabled, try construct the uint8
      // token.
      // TODO(133342794): use a generic token generator class.
      uint64_t token_parts[4];
      // bits from model_token.
      token_parts[0] = std::hash<std::string>{}(model_token);
      // bits from params->nodes_to_replace.
      token_parts[1] = GetHash(params->nodes_to_replace);
      // bits from params->input_tensors.
      token_parts[2] = GetHash(params->input_tensors);
      // bits from params->output_tensors.
      token_parts[3] = GetHash(params->output_tensors);
      // NNAPI requires the token to be 256bit long.
      std::vector<uint8_t> nnapi_cache_token(32, 0);
      // Copy the token bits.
      uint8_t* p = reinterpret_cast<uint8_t*>(token_parts);
      for (int i = 0; i < 4 * sizeof(uint64_t); i++) {
        nnapi_cache_token[i] = p[i];
      }
      const int set_caching_result =
          nnapi_->ANeuralNetworksCompilation_setCaching(
              compilation, cache_dir, nnapi_cache_token.data());
      if (set_caching_result != ANEURALNETWORKS_NO_ERROR) {
        nnapi_->ANeuralNetworksCompilation_free(compilation);
        compilation = nullptr;
      }
      RETURN_TFLITE_ERROR_IF_NN_ERROR(context, set_caching_result, nnapi_errno);
    }
    const int finish_result =
        nnapi_->ANeuralNetworksCompilation_finish(compilation);
    if (finish_result != ANEURALNETWORKS_NO_ERROR) {
      nnapi_->ANeuralNetworksCompilation_free(compilation);
      compilation = nullptr;
    }
    RETURN_TFLITE_ERROR_IF_NN_ERROR(context, finish_result, nnapi_errno);
    nn_compilation_.reset(compilation);
  }
  return kTfLiteOk;
}

TfLiteStatus NNAPIDelegateKernel::Prepare(TfLiteContext* context,
                                          TfLiteNode* node, int* nnapi_errno) {
  if (!nn_compilation_) {
    // Compilation failed earlier, return error.
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus NNAPIDelegateKernel::Invoke(TfLiteContext* context,
                                         TfLiteNode* node, int* nnapi_errno) {
  ANeuralNetworksExecution* execution = nullptr;
  RETURN_TFLITE_ERROR_IF_NN_ERROR(context,
                                  nnapi_->ANeuralNetworksExecution_create(
                                      nn_compilation_.get(), &execution),
                                  nnapi_errno);
  std::unique_ptr<ANeuralNetworksExecution, NNFreeExecution>
      execution_unique_ptr(execution);

  // Set the input tensor buffers. Note: we access tflite tensors using
  // absolute indices but NN api indices inputs by relative indices.
  int relative_input_index = 0;

  size_t input_offset = 0;
  for (auto absolute_input_index : TfLiteIntArrayView(node->inputs)) {
    if (absolute_input_index == kTfLiteOptionalTensor) {
      continue;
    }
    TfLiteTensor* tensor = &context->tensors[absolute_input_index];
    if (tensor->allocation_type != kTfLiteMmapRo) {
      if (tensor->buffer_handle != kTfLiteNullBufferHandle &&
          tensor->buffer_handle < tensor_memory_map_->size()) {
        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            context,
            nnapi_->ANeuralNetworksExecution_setInputFromMemory(
                execution, relative_input_index, nullptr,
                tensor_memory_map_->at(tensor->buffer_handle).memory, 0,
                tensor->bytes),
            nnapi_errno);
        relative_input_index++;
        continue;
      }
      TfLiteType ann_type_equivalent =
          operand_mapping_.lite_index_to_ann_type_conversion(
              absolute_input_index);
      int tensor_size = 0;
      if (ann_type_equivalent != kTfLiteNoType) {
        const auto num_elements = NumElements(tensor);
        uint8_t* input_ptr = nn_input_memory_->get_data_ptr() + input_offset;
        if (tensor->type == kTfLiteUInt8 &&
            ann_type_equivalent == kTfLiteInt32) {
          for (int i = 0; i < num_elements; ++i) {
            reinterpret_cast<int32_t*>(input_ptr)[i] =
                static_cast<const int32_t>(tensor->data.uint8[i]);
          }
        } else if (tensor->type == kTfLiteInt8 &&
                   ann_type_equivalent == kTfLiteUInt8) {
          // Explicitly convert int8 values to uint8 values.
          for (int i = 0; i < num_elements; ++i) {
            input_ptr[i] = static_cast<const uint8_t>(
                static_cast<int32_t>(tensor->data.int8[i]) + 128);
          }
        } else if (tensor->type == kTfLiteInt8 &&
                   ann_type_equivalent == kTfLiteInt32) {
          for (int i = 0; i < num_elements; ++i) {
            reinterpret_cast<int32_t*>(input_ptr)[i] =
                static_cast<const int32_t>(tensor->data.int8[i]) + 128;
          }
        } else {
          context->ReportError(
              context,
              "NN API Delegate: unsupported tensor types conversion: "
              "from type code %d to type code %d.\n",
              tensor->type, ann_type_equivalent);
          return kTfLiteError;
        }
        size_t type_size;
        TF_LITE_ENSURE_OK(
            context, GetSizeOfType(context, ann_type_equivalent, &type_size));
        tensor_size = NumElements(tensor) * type_size;
        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            context,
            nnapi_->ANeuralNetworksExecution_setInputFromMemory(
                execution, relative_input_index, nullptr,
                nn_input_memory_->get_handle(), input_offset, tensor_size),
            nnapi_errno);
      } else {
        // copy data to pre-allocated shared memory.
        memcpy(nn_input_memory_->get_data_ptr() + input_offset,
               tensor->data.raw, tensor->bytes);
        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            context,
            nnapi_->ANeuralNetworksExecution_setInputFromMemory(
                execution, relative_input_index, nullptr,
                nn_input_memory_->get_handle(), input_offset, tensor->bytes),
            nnapi_errno);
        tensor_size = tensor->bytes;
      }
      input_offset += tensor_size;
      input_offset += getNumPaddingBytes(tensor_size);
      relative_input_index++;
    }
  }

  // Set the output tensor buffers.
  int relative_output_index = 0;
  size_t output_offset = 0;
  for (auto output_index : TfLiteIntArrayView(node->outputs)) {
    // If the NNAPI implementation doesn't have some of the outputs
    // they are left unmapped and we should not try to read their value here
    if (operand_mapping_.lite_index_to_ann(output_index) == -1) {
      continue;
    }
    TfLiteTensor* tensor = &context->tensors[output_index];
    if (tensor->buffer_handle != kTfLiteNullBufferHandle &&
        tensor->buffer_handle < tensor_memory_map_->size()) {
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context,
          nnapi_->ANeuralNetworksExecution_setOutputFromMemory(
              execution, relative_output_index, nullptr,
              tensor_memory_map_->at(tensor->buffer_handle).memory, 0,
              tensor->bytes),
          nnapi_errno);

    } else {
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context,
          nnapi_->ANeuralNetworksExecution_setOutputFromMemory(
              execution, relative_output_index, nullptr,
              nn_output_memory_->get_handle(), output_offset, tensor->bytes),
          nnapi_errno);
      output_offset += tensor->bytes;
      output_offset += getNumPaddingBytes(tensor->bytes);
    }
    relative_output_index++;
  }

  // The state_out of previous invocation need to be mapped to state_in of
  // current invocation.
  for (size_t i = 0; i < model_state_tfl_inputs_.size(); i++) {
    int state_tensor_idx = model_state_tfl_inputs_[i];
    TfLiteTensor* tensor = &context->tensors[state_tensor_idx];
    // Here we are using a deep copy for state_in tensors so that we are not
    // reading and writing into the same buffer during a invocation.
    // TODO(110369471): using double shared buffer to minimize the copies.
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context,
        nnapi_->ANeuralNetworksExecution_setOutput(
            execution, relative_output_index, nullptr, tensor->data.raw,
            tensor->bytes),
        nnapi_errno);
    relative_output_index++;
  }
  // Invoke ANN in blocking fashion.
  if (nnapi_->android_sdk_version < kMinSdkVersionForNNAPI12) {
    ANeuralNetworksEvent* event = nullptr;
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context,
        nnapi_->ANeuralNetworksExecution_startCompute(execution, &event),
        nnapi_errno);
    const int wait_result = nnapi_->ANeuralNetworksEvent_wait(event);
    nnapi_->ANeuralNetworksEvent_free(event);
    RETURN_TFLITE_ERROR_IF_NN_ERROR(context, wait_result, nnapi_errno);
  } else {
    // Use synchronous execution for NNAPI 1.2+.
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context, nnapi_->ANeuralNetworksExecution_compute(execution),
        nnapi_errno);
  }

  // copy results from shared memory to the destination.
  output_offset = 0;
  for (auto output_index : TfLiteIntArrayView(node->outputs)) {
    TfLiteTensor* tensor = &context->tensors[output_index];
    if (tensor->buffer_handle != kTfLiteNullBufferHandle) {
      continue;
    }
    TfLiteType ann_type_equivalent =
        operand_mapping_.lite_index_to_ann_type_conversion(output_index);
    if (tensor->type == kTfLiteInt8 && ann_type_equivalent == kTfLiteUInt8) {
      // Explicitly convert uint8 values to int8 values.
      uint8_t* output_ptr = reinterpret_cast<uint8_t*>(
          nn_output_memory_->get_data_ptr() + output_offset);
      const auto num_elements = NumElements(tensor);
      for (int i = 0; i < num_elements; ++i) {
        output_ptr[i] =
            static_cast<uint8_t>(static_cast<int32_t>(output_ptr[i]) - 128);
      }
    }
    memcpy(tensor->data.raw, nn_output_memory_->get_data_ptr() + output_offset,
           tensor->bytes);
    output_offset += tensor->bytes;
    output_offset += getNumPaddingBytes(tensor->bytes);
  }

  // copy output of all output tensors in feedback_loops_ into the
  // associated input
  for (auto feedback_loop : feedback_loops_) {
    int output_tensor_idx;
    int input_tensor_idx;
    std::tie(output_tensor_idx, input_tensor_idx) = feedback_loop;
    TfLiteTensor* src =
        &context->tensors[node->outputs->data[output_tensor_idx]];
    TfLiteTensor* dest =
        &context->tensors[node->inputs->data[input_tensor_idx]];

    memcpy(dest->data.raw, src->data.raw, src->bytes);
  }

  return kTfLiteOk;
}

void NNAPIDelegateKernel::AddDequantizeOperatorsWhereNeeded(
    const TfLiteContext* context, int builtin_code, const TfLiteNode* node,
    NNAPIOpBuilder* builder, int* nnapi_errno) {
  // Depending on the operator and the input data format, Dequantize
  // operators may need to be added. For example when the input is
  // floating-point but weights are quantized then the weights will first be
  // dequantized to the same format as the input before being passed to the
  // operator.

  // The tensor determining whether the inputs should be floating-point.
  int input_tensor_index = -1;
  std::vector<int> inputs_to_potentially_dequantize;

  switch (builtin_code) {
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinFullyConnected: {
      input_tensor_index = 0;
      // Weights and bias are inputs #1 and #2 respectively and may require
      // dequantization.
      inputs_to_potentially_dequantize = {1, 2};
      break;
    }
    case kTfLiteBuiltinLstm: {
      input_tensor_index = 0;
      inputs_to_potentially_dequantize = {1,  2,  3,  4,  5,  6,  7,
                                          8,  9,  10, 11, 12, 13, 14,
                                          15, 16, 17, 20, 21, 22, 23};
      break;
    }
    default:
      return;
  }

  int tensor_id = node->inputs->data[input_tensor_index];
  if (tensor_id < 0) return;

  // Nothing to do if the input is not floating-point.
  if (!IsFloat(context->tensors[tensor_id].type)) return;

  for (int i : inputs_to_potentially_dequantize) {
    if (i < 0 || i >= node->inputs->size) continue;  // Ignore invalid index.
    tensor_id = node->inputs->data[i];
    if (tensor_id < 0) continue;  // Ignore optional input.

    const TfLiteType type = context->tensors[tensor_id].type;
    // Nothing to do for this tensor if it's not quantized.
    if (!IsQuantized(type)) continue;

    // Insert Dequantize operator if it hasn't been done already and change
    // the node's input accordingly.
    builder->AddDequantize(i, node->inputs->data[i], type);
  }
}

TfLiteStatus NNAPIDelegateKernel::AddOpsAndTensors(TfLiteContext* context,
                                                   int* nnapi_errno) {
  DequantizeMapping dequantize_mapping;
  // The operand builder allows creating a single op. It is created outside
  // the for loop to avoid reallocating the vectors.
  NNAPIOpBuilder builder(nnapi_, context, &operand_mapping_,
                         &dequantize_mapping, &allocation_memory_mapping_,
                         nn_model_.get(), nnapi_errno);
  // Add Tensors.
  for (auto node_index : nodes_) {
    // Obtain the op and registration.
    TfLiteNode* node;
    TfLiteRegistration* reg;
    TF_LITE_ENSURE_STATUS(
        context->GetNodeAndRegistration(context, node_index, &node, &reg));

    const bool hybrid_op = IsHybridOperator(context, reg->builtin_code, node);
    const bool scalar_as_tensor = IsScalarInputSupported(reg->builtin_code);
    const bool need_int8_conversion =
        NeedInt8Conversion(context, reg->builtin_code, node);
    int input_tensor_flags = 0;
    if (scalar_as_tensor) {
      input_tensor_flags |= NN_TENSOR_FLAG_SCALAR_AS_TENSOR;
    }

    // h_swish will be lowered into supported NNAPI operations.
    if (reg->builtin_code == kTfLiteBuiltinHardSwish) {
      builder.AddHardSwish(node->inputs->data[0], node->outputs->data[0],
                           need_int8_conversion);
      continue;
    }
    // Map inputs to NN API tensor indices.
    for (int input_pos = 0; input_pos < node->inputs->size; ++input_pos) {
      const auto input_index = node->inputs->data[input_pos];
      if (need_int8_conversion &&
          (input_pos == 0 ||
           reg->builtin_code == kTfLiteBuiltinFullyConnected ||
           reg->builtin_code == kTfLiteBuiltinAdd ||
           reg->builtin_code == kTfLiteBuiltinMul ||
           reg->builtin_code == kTfLiteBuiltinSub ||
           reg->builtin_code == kTfLiteBuiltinConcatenation ||
           reg->builtin_code == kTfLiteBuiltinMaximum ||
           reg->builtin_code == kTfLiteBuiltinMinimum ||
           reg->builtin_code == kTfLiteBuiltinLess ||
           reg->builtin_code == kTfLiteBuiltinLessEqual ||
           reg->builtin_code == kTfLiteBuiltinGreater ||
           reg->builtin_code == kTfLiteBuiltinGreaterEqual ||
           reg->builtin_code == kTfLiteBuiltinEqual ||
           reg->builtin_code == kTfLiteBuiltinNotEqual ||
           reg->builtin_code == kTfLiteBuiltinSelect)) {
        // Only selected inputs require int8 conversion.
        TF_LITE_ENSURE_STATUS(builder.AddTensorInput(
            input_index, hybrid_op,
            input_tensor_flags | NN_TENSOR_FLAG_INT8_CONVERSION));
        continue;
      }
      if (reg->builtin_code == kTfLiteBuiltinLstm && isLstmFullKernel(node) &&
          input_pos >= 20) {
        // Skip layer normalization weights. They are added in the Map
        // function (after all the other inputs added there) since layer
        // normalization weights are the last four inputs of the LSTM op in
        // NNAPI.
        continue;
      }
      if (reg->builtin_code == kTfLiteBuiltinLstm && isLstmBasicKernel(node)) {
        // Configuring all inputs in the Map function
        continue;
      }
      if (reg->builtin_code == kTfLiteBuiltinUnidirectionalSequenceLstm) {
        if (input_pos >= 20) {
          // Skip layer normalization weights. They are added in the Map
          // function (after all the other inputs added there) since layer
          // normalization weights are the last four inputs of the
          // unidirectional sequence LSTM op in NNAPI.
          continue;
        }
        if (input_index == kTfLiteOptionalTensor) {
          TF_LITE_ENSURE_STATUS(builder.AddVectorFloat32Operand(nullptr, 0));
          continue;
        }
      }
      if ((reg->builtin_code == kTfLiteBuiltinSplit) &&
          (input_index == node->inputs->data[0])) {
        // Skip the axis input tensor; it will be added as a scalar operand
        // by the Map() mapping.
        continue;
      }
      if (reg->builtin_code == kTfLiteBuiltinTransposeConv) {
        // Everything is added during Map since input tensors
        // have different order.
        continue;
      }

      // Pad and Padv2 have an optional parameter for a pad value which has
      // to be converted to a scalar type in NN API.
      if ((reg->builtin_code == kTfLiteBuiltinPadv2 ||
           reg->builtin_code == kTfLiteBuiltinPad) &&
          node->inputs->size == 3 && input_pos == 2) {
        const int constant_value_id = node->inputs->data[2];
        if (constant_value_id == kTfLiteOptionalTensor) {
          continue;
        }
        const TfLiteTensor constant_value = context->tensors[constant_value_id];

        switch (constant_value.type) {
          case kTfLiteFloat32:
            if (constant_value.allocation_type == kTfLiteMmapRo) {
              builder.AddScalarFloat32Operand(*constant_value.data.f);
            } else {
              builder.AddSingleValueTensorAsScalarOperand(
                  constant_value_id, ANEURALNETWORKS_FLOAT32);
            }
            break;
          case kTfLiteUInt8:
            if (constant_value.allocation_type == kTfLiteMmapRo) {
              builder.AddScalarInt32Operand(
                  static_cast<int32_t>(*constant_value.data.uint8));
            } else {
              builder.AddSingleValueTensorAsScalarOperand(
                  constant_value_id, ANEURALNETWORKS_INT32);
            }
            break;
          case kTfLiteInt8:
            if (constant_value.allocation_type == kTfLiteMmapRo) {
              builder.AddScalarInt32Operand(
                  static_cast<int32_t>(*constant_value.data.int8) + 128);
            } else {
              builder.AddSingleValueTensorAsScalarOperand(
                  constant_value_id, ANEURALNETWORKS_INT32);
            }
            break;
          default:
            context->ReportError(context,
                                 "Unsupported type of pad value for pad_v2\n");
            return kTfLiteError;
        }
        continue;
      }

      if (input_index == kTfLiteOptionalTensor &&
          (reg->builtin_code == kTfLiteBuiltinLstm ||
           reg->builtin_code == kTfLiteBuiltinSvdf ||
           reg->builtin_code == kTfLiteBuiltinBidirectionalSequenceLstm)) {
        // properly handle the optional tensor for LSTM and SVDF.
        // currently only support float32.
        TF_LITE_ENSURE_STATUS(builder.AddVectorFloat32Operand(nullptr, 0));
      } else if (reg->builtin_code == kTfLiteBuiltinResizeBilinear ||
                 reg->builtin_code == kTfLiteBuiltinResizeNearestNeighbor) {
        if (input_pos == 0) {
          // Only the first input tensor is added. The second one,
          // specifying the output height and width, is not added and
          // instead the height and width will be added individually as
          // scalars by the mapping function returned by Map().
          TF_LITE_ENSURE_STATUS(builder.AddTensorInput(input_index, hybrid_op));
        }
      } else if (reg->builtin_code == kTfLiteBuiltinTopkV2 && input_pos > 0) {
        // The K parameter tensor is not handled here but by the functor
        // returned by Map, the input tensor is instead added in
        // the else clause below
        continue;
      } else if (reg->builtin_code == kTfLiteBuiltinGather) {
        // Everything is added during Map since input tensors
        // have different order.
        continue;
      } else if (reg->builtin_code == kTfLiteBuiltinExpandDims &&
                 input_pos == 1) {
        // The axis param is added during Map
        continue;
      } else if (reg->builtin_code == kTfLiteBuiltinBatchToSpaceNd &&
                 input_pos == 2) {
        // NNAPI does not support crops.
        // The Map function will check if all crops are zero.
        continue;
      } else if (reg->builtin_code == kTfLiteBuiltinArgMin ||
                 reg->builtin_code == kTfLiteBuiltinArgMax) {
        // The first input tensor is added as is. The second one, specifying
        // the axis, needs to be converted to a scalar since TFLite uses a
        // tensor but NNAPI uses a scalar as the axis.
        if (input_pos == 0) {
          TF_LITE_ENSURE_STATUS(builder.AddTensorInput(input_index, hybrid_op));
        } else {
          const int axis_id = node->inputs->data[1];
          const TfLiteTensor& axis_tensor = context->tensors[axis_id];
          switch (axis_tensor.type) {
            case kTfLiteInt32:
              if (axis_tensor.allocation_type == kTfLiteMmapRo) {
                TF_LITE_ENSURE_STATUS(builder.AddScalarInt32Operand(
                    static_cast<int32_t>(*axis_tensor.data.i32)));
              } else {
                TF_LITE_ENSURE_STATUS(
                    builder.AddSingleValueTensorAsScalarOperand(
                        axis_id, ANEURALNETWORKS_INT32));
              }
              break;
            case kTfLiteInt64:
              // Map() function already makes sure int64 input is constant.
              TF_LITE_ENSURE_STATUS(builder.AddScalarInt32Operand(
                  static_cast<int32_t>(*axis_tensor.data.i64)));
              break;
            default:
              return kTfLiteError;
          }
        }
      } else {
        TF_LITE_ENSURE_STATUS(
            builder.AddTensorInput(input_index, hybrid_op, input_tensor_flags));
      }
    }
    // Get op type and operands
    // Fails if the Map function failed
    int nn_op_type;
    TF_LITE_ENSURE_STATUS(Map(context, reg->builtin_code, reg->version,
                              nnapi_->android_sdk_version,
                              {context, &builder, node, &model_state_outputs_,
                               &model_state_tfl_inputs_, &feedback_loops_},
                              &nn_op_type));

    // Map outputs to NN API tensor indices.
    int output_tensor_flags = 0;
    if (need_int8_conversion) {
      output_tensor_flags |= NN_TENSOR_FLAG_INT8_CONVERSION;
    }
    for (int output_pos = 0; output_pos < node->outputs->size; ++output_pos) {
      const auto output_index = node->outputs->data[output_pos];

      // Outputs for  basic LSTM cell are set in the Map function since
      if (reg->builtin_code == kTfLiteBuiltinLstm && isLstmBasicKernel(node)) {
        continue;
      }

      TF_LITE_ENSURE_STATUS(
          builder.AddTensorOutput(output_index, output_tensor_flags));
    }

    // Dequantize operators may have to be added in case inputs are to be
    // floating-point.
    AddDequantizeOperatorsWhereNeeded(context, reg->builtin_code, node,
                                      &builder, nnapi_errno);

    builder.FinalizeAddOperation(nn_op_type);
  }
  return kTfLiteOk;
}

TfLiteStatus NNAPIDelegateKernel::BuildGraph(
    TfLiteContext* context, const TfLiteIntArray* input_tensors,
    const TfLiteIntArray* output_tensors, int* nnapi_errno) {
  // Build the ops and tensors.
  TF_LITE_ENSURE_STATUS(AddOpsAndTensors(context, nnapi_errno));
  // Map input and output tensor indices to ANN
  std::vector<uint32_t> inputs;
  inputs.reserve(input_tensors->size);
  std::vector<uint32_t> outputs;
  outputs.reserve(output_tensors->size);

  size_t total_input_byte_size = 0;
  // Make the TensorFlow Lite inputs and outputs to ann_indices.
  for (int i : TfLiteIntArrayView(input_tensors)) {
    // Constant tensors are not NNAPI inputs.
    if (i != kTfLiteOptionalTensor &&
        context->tensors[i].allocation_type != kTfLiteMmapRo &&
        // The delegate might not have mapped this input (this can
        // happen if one tensor is split in several ones)
        operand_mapping_.lite_index_to_ann(i) != -1) {
      inputs.push_back(operand_mapping_.lite_index_to_ann(i));
      if (context->tensors[i].buffer_handle != kTfLiteNullBufferHandle) {
        continue;
      }
      const TfLiteType nn_type_conversion =
          operand_mapping_.lite_index_to_ann_type_conversion(i);
      int tensor_size = 0;
      if (nn_type_conversion == kTfLiteNoType) {
        tensor_size = context->tensors[i].bytes;
      } else {
        size_t type_size;
        TF_LITE_ENSURE_OK(
            context, GetSizeOfType(context, nn_type_conversion, &type_size));
        tensor_size = NumElements(&context->tensors[i]) * type_size;
      }
      total_input_byte_size += tensor_size;
      total_input_byte_size += getNumPaddingBytes(tensor_size);
    }
  }

  size_t total_output_byte_size = 0;
  for (int i : TfLiteIntArrayView(output_tensors)) {
    const int output_tensor_ann_index = operand_mapping_.lite_index_to_ann(i);
    // Unmapped outputs are not added
    if (output_tensor_ann_index != -1) {
      outputs.push_back(output_tensor_ann_index);
    }
    if (context->tensors[i].buffer_handle != kTfLiteNullBufferHandle) {
      continue;
    }
    total_output_byte_size += context->tensors[i].bytes;
    total_output_byte_size += getNumPaddingBytes(context->tensors[i].bytes);
  }

  // Add state output tensors as model outputs.
  for (int i : model_state_outputs_) {
    outputs.push_back(i);
  }

  // Tell ANN to declare inputs/outputs
  RETURN_TFLITE_ERROR_IF_NN_ERROR(
      context,
      nnapi_->ANeuralNetworksModel_identifyInputsAndOutputs(
          nn_model_.get(), inputs.size(), inputs.data(), outputs.size(),
          outputs.data()),
      nnapi_errno);

  // Set relaxed computation mode for fp32 if possible.
  if (nnapi_->android_sdk_version >= kMinSdkVersionForNNAPI11) {
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context,
        nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16(
            nn_model_.get(), context->allow_fp32_relax_to_fp16),
        nnapi_errno);
  }

  // Finalize the model
  RETURN_TFLITE_ERROR_IF_NN_ERROR(
      context, nnapi_->ANeuralNetworksModel_finish(nn_model_.get()),
      nnapi_errno);

  // Create shared memory pool for inputs and outputs.
  nn_input_memory_.reset(
      new NNMemory(nnapi_, "input_pool", total_input_byte_size));
  nn_output_memory_.reset(
      new NNMemory(nnapi_, "output_pool", total_output_byte_size));

  return kTfLiteOk;
}

}  // namespace nnapi
}  // namespace delegate

using ::tflite::delegate::nnapi::NNAPIDelegateKernel;

StatefulNnApiDelegate::StatefulNnApiDelegate(Options options)
    : TfLiteDelegate(TfLiteDelegateCreate()),
      delegate_data_(
          Data{.execution_preference = options.execution_preference}) {
  if (options.accelerator_name) {
    delegate_data_.accelerator_name = options.accelerator_name;
  }
  if (options.cache_dir) {
    delegate_data_.cache_dir = options.cache_dir;
  }
  if (options.model_token) {
    delegate_data_.model_token = options.model_token;
  }
  delegate_data_.disallow_nnapi_cpu = options.disallow_nnapi_cpu;
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Created TensorFlow Lite delegate for NNAPI.");
  Prepare = DoPrepare;
  CopyFromBufferHandle = DoCopyFromBufferHandle;
  CopyToBufferHandle = DoCopyToBufferHandle;
  FreeBufferHandle = DoFreeBufferHandle;
  data_ = &delegate_data_;
}

StatefulNnApiDelegate::StatefulNnApiDelegate()
    : StatefulNnApiDelegate(Options()) {}

const StatefulNnApiDelegate::Options StatefulNnApiDelegate::GetOptions(
    TfLiteDelegate* delegate) {
  auto delegate_data = reinterpret_cast<Data*>(delegate->data_);
  StatefulNnApiDelegate::Options options;
  options.execution_preference = delegate_data->execution_preference;
  options.accelerator_name = delegate_data->accelerator_name.empty()
                                 ? nullptr
                                 : delegate_data->accelerator_name.c_str();
  options.cache_dir = delegate_data->cache_dir.empty()
                          ? nullptr
                          : delegate_data->cache_dir.c_str();
  options.model_token = delegate_data->model_token.empty()
                            ? nullptr
                            : delegate_data->model_token.c_str();
  options.disallow_nnapi_cpu = delegate_data->disallow_nnapi_cpu;
  return options;
}

const std::vector<StatefulNnApiDelegate::MemoryRegistration>&
StatefulNnApiDelegate::GetTensorMemoryMap(TfLiteDelegate* delegate) {
  auto delegate_data = reinterpret_cast<Data*>(delegate->data_);
  return delegate_data->tensor_memory_map;
}

TfLiteBufferHandle StatefulNnApiDelegate::RegisterNnapiMemory(
    ANeuralNetworksMemory* memory, CopyToHostTensorFnPtr callback,
    void* callback_context) {
  int map_size = delegate_data_.tensor_memory_map.size();
  for (int i = 0; i < map_size; i++) {
    if (delegate_data_.tensor_memory_map[i].memory == nullptr) {
      delegate_data_.tensor_memory_map[i] = {memory, callback,
                                             callback_context};
      return i;
    }
  }
  delegate_data_.tensor_memory_map.push_back(
      {memory, callback, callback_context});
  return map_size;
}

TfLiteStatus StatefulNnApiDelegate::DoCopyFromBufferHandle(
    TfLiteContext* context, TfLiteDelegate* delegate,
    TfLiteBufferHandle buffer_handle, TfLiteTensor* tensor) {
  auto delegate_data = reinterpret_cast<Data*>(delegate->data_);
  if (buffer_handle < 0 ||
      buffer_handle >= delegate_data->tensor_memory_map.size()) {
    return kTfLiteError;
  }
  auto memory = delegate_data->tensor_memory_map[buffer_handle].memory;
  auto callback = delegate_data->tensor_memory_map[buffer_handle].callback;
  auto callback_context =
      delegate_data->tensor_memory_map[buffer_handle].callback_context;
  if (!memory || !callback) {
    return kTfLiteError;
  }
  return callback(tensor, memory, 0, tensor->bytes, callback_context);
}

TfLiteStatus StatefulNnApiDelegate::DoCopyToBufferHandle(
    TfLiteContext* context, TfLiteDelegate* delegate,
    TfLiteBufferHandle buffer_handle, TfLiteTensor* tensor) {
  return kTfLiteError;
}

void StatefulNnApiDelegate::DoFreeBufferHandle(TfLiteContext* context,
                                               TfLiteDelegate* delegate,
                                               TfLiteBufferHandle* handle) {
  auto delegate_data = reinterpret_cast<Data*>(delegate->data_);
  if (*handle >= 0 && *handle < delegate_data->tensor_memory_map.size()) {
    delegate_data->tensor_memory_map[*handle] = {nullptr, nullptr, nullptr};
    *handle = kTfLiteNullBufferHandle;
  }
}

int StatefulNnApiDelegate::GetNnApiErrno() const {
  return delegate_data_.nnapi_errno;
}

using ::tflite::delegate::nnapi::kMinSdkVersionForNNAPI;
using ::tflite::delegate::nnapi::kMinSdkVersionForNNAPI12;

TfLiteStatus StatefulNnApiDelegate::DoPrepare(TfLiteContext* context,
                                              TfLiteDelegate* delegate) {
  int* nnapi_errno = &(static_cast<Data*>(delegate->data_)->nnapi_errno);

  // Resetting the error code when the delegate is initialized
  // by TFLite. This causes the error to be reset if reusing the same
  // StatefulNnApiDelegate after a failure
  *nnapi_errno = 0;

  // Do not check nodes_ if NN API is unavailable.
  const NnApi* nnapi = NnApiImplementation();
  if (nnapi->android_sdk_version < kMinSdkVersionForNNAPI ||
      !nnapi->nnapi_exists) {
    return kTfLiteOk;
  }
  bool is_accelerator_specified = false;
  // For NNAPI 1.2+, check if there is any accelerator available.
  // If not, don't delegate to NNAPI's CPU reference implementation.
  if (nnapi->android_sdk_version >= kMinSdkVersionForNNAPI12) {
    // Check if user specified an acclelerator to use.
    const char* device_name_ptr = GetOptions(delegate).accelerator_name;
    if (device_name_ptr) {
      if (!GetDeviceHandle(context, device_name_ptr)) {
        return kTfLiteError;
      } else {
        // also check if the selected device is not CPU reference impl.
        const string kNnapiReferenceImplName = "nnapi-reference";
        is_accelerator_specified = kNnapiReferenceImplName != device_name_ptr;
      }
    } else {
      // If no accelerator is specified, only use NNAPI if an accelerator is
      // available. Any available accelerator will make the device_count larger
      // than 1. More sophisticated check and whitelisting can be added later.
      uint32_t device_count = 0;
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context, nnapi->ANeuralNetworks_getDeviceCount(&device_count),
          nnapi_errno);
      if (device_count <= 1) {
        return kTfLiteOk;
      }
    }
  }
  // Allocate one element in vector already since TensorFlow Lite uses
  // the first value as the number of nodes. The actual value will be set
  // later, after the vector has been filled.
  std::vector<int> supported_nodes(1);
  // We don't care about all nodes_, we only care about ones in the
  // current plan.
  TfLiteIntArray* plan;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));

  int android_sdk_version = NnApiImplementation()->android_sdk_version;
  // Check for every node if it is supported
  for (int node_index : TfLiteIntArrayView(plan)) {
    TfLiteNode* node;
    TfLiteRegistration* registration;
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));
    if (NNAPIDelegateKernel::Validate(
            context, registration->builtin_code, registration->version,
            android_sdk_version, node, is_accelerator_specified)) {
      supported_nodes.push_back(node_index);
    }
  }
  // First element in vector must be the number of actual nodes.
  supported_nodes[0] = supported_nodes.size() - 1;

  // If there are no delegated nodes, short-circuit node replacement.
  if (!supported_nodes[0]) {
    return kTfLiteOk;
  }

  // NN API Delegate Registration (the pseudo kernel that will invoke NN
  // API node sub sets)
  static const TfLiteRegistration nnapi_delegate_kernel = {
      .init = [](TfLiteContext* context, const char* buffer,
                 size_t length) -> void* {
        const TfLiteDelegateParams* params =
            reinterpret_cast<const TfLiteDelegateParams*>(buffer);
        int* nnapi_errno =
            &(static_cast<Data*>(params->delegate->data_)->nnapi_errno);
        NNAPIDelegateKernel* kernel_state = new NNAPIDelegateKernel;
        kernel_state->Init(context, params, nnapi_errno);
        return kernel_state;
      },

      .free = [](TfLiteContext* context, void* buffer) -> void {
        delete reinterpret_cast<NNAPIDelegateKernel*>(buffer);
      },

      .prepare = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        NNAPIDelegateKernel* state =
            reinterpret_cast<NNAPIDelegateKernel*>(node->user_data);
        int* nnapi_errno =
            &(static_cast<Data*>(node->delegate->data_)->nnapi_errno);
        return state->Prepare(context, node, nnapi_errno);
      },

      .invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        NNAPIDelegateKernel* state =
            reinterpret_cast<NNAPIDelegateKernel*>(node->user_data);
        int* nnapi_errno =
            &(static_cast<Data*>(node->delegate->data_)->nnapi_errno);
        return state->Invoke(context, node, nnapi_errno);
      },

      .profiling_string = nullptr,
      .builtin_code = kTfLiteBuiltinDelegate,
      .custom_name = "TfLiteNnapiDelegate",
      .version = 1,
  };

  // Request TFLite to partition the graph and make kernels
  // for each independent node sub set a new nnapi_delegate_kernel.
  return context->ReplaceNodeSubsetsWithDelegateKernels(
      context, nnapi_delegate_kernel,
      reinterpret_cast<TfLiteIntArray*>(supported_nodes.data()), delegate);
}

// Returns a singleton NNAPI Delegate that can check for support of ops.
TfLiteDelegate* NnApiDelegate() {
  static StatefulNnApiDelegate* delegate = new StatefulNnApiDelegate();
  return delegate;
}

}  // namespace tflite
