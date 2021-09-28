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
#include <cstdio>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/serialization.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/nnapi/sl/public/NeuralNetworksSupportLibraryImpl.h"

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

#if defined __ANDROID__ || defined __unix__
#define TFLITE_NNAPI_ALLOW_MMAP_SHARING
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "fp16.h"  // from @FP16
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_kernel.h"
#include "tensorflow/lite/delegates/nnapi/quant_lstm_sup.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/nnapi/nnapi_util.h"
#include "tensorflow/lite/util.h"
#ifdef NNAPI_VERBOSE_VALIDATION
#include "tensorflow/lite/schema/schema_generated.h"
#endif
#include <farmhash.h>

namespace tflite {
namespace {

static const char kNnapiId[] = "nnapi_";

// Returns a string ID unique to what accelerator is run by NNAPI, based on
// user params. Assumes that the default accelerator is same across runs.
// Used for caching nodes to be delegated for a model.
std::string NnApiBackendId(
    const StatefulNnApiDelegate::Options& delegate_options) {
  std::string delegate_id = kNnapiId;
  if (delegate_options.accelerator_name) {
    delegate_id += delegate_options.accelerator_name;
  }
  return delegate_id;
}

// Returns the enum name corresponding to the given error code if the given
// value corresponds to an of the error codes in the enumeration above or
// an message with the unknown code.
// LINT.IfChange(NnApiErrorDescription)
std::string NnApiErrorDescription(int error_code) {
  switch (error_code) {
    case ANEURALNETWORKS_NO_ERROR:
      return "ANEURALNETWORKS_NO_ERROR";
    case ANEURALNETWORKS_OUT_OF_MEMORY:
      return "ANEURALNETWORKS_OUT_OF_MEMORY";
    case ANEURALNETWORKS_INCOMPLETE:
      return "ANEURALNETWORKS_INCOMPLETE";
    case ANEURALNETWORKS_UNEXPECTED_NULL:
      return "ANEURALNETWORKS_UNEXPECTED_NULL";
    case ANEURALNETWORKS_BAD_DATA:
      return "ANEURALNETWORKS_BAD_DATA";
    case ANEURALNETWORKS_OP_FAILED:
      return "ANEURALNETWORKS_OP_FAILED";
    case ANEURALNETWORKS_BAD_STATE:
      return "ANEURALNETWORKS_BAD_STATE";
    case ANEURALNETWORKS_UNMAPPABLE:
      return "ANEURALNETWORKS_UNMAPPABLE";
    case ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE:
      return "ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE";
    case ANEURALNETWORKS_UNAVAILABLE_DEVICE:
      return "ANEURALNETWORKS_UNAVAILABLE_DEVICE";
    case ANEURALNETWORKS_MISSED_DEADLINE_TRANSIENT:
      return "ANEURALNETWORKS_MISSED_DEADLINE_TRANSIENT";
    case ANEURALNETWORKS_MISSED_DEADLINE_PERSISTENT:
      return "ANEURALNETWORKS_MISSED_DEADLINE_PERSISTENT";
    case ANEURALNETWORKS_RESOURCE_EXHAUSTED_TRANSIENT:
      return "ANEURALNETWORKS_RESOURCE_EXHAUSTED_TRANSIENT";
    case ANEURALNETWORKS_RESOURCE_EXHAUSTED_PERSISTENT:
      return "ANEURALNETWORKS_RESOURCE_EXHAUSTED_PERSISTENT";
    case ANEURALNETWORKS_DEAD_OBJECT:
      return "ANEURALNETWORKS_DEAD_OBJECT";
    default:
      return "Unknown NNAPI error code: " + std::to_string(error_code);
  }
}
// LINT.ThenChange()

#define RETURN_TFLITE_ERROR_IF_NN_ERROR(context, code, call_desc, p_errno)  \
  do {                                                                      \
    const auto _code = (code);                                              \
    const auto _call_desc = (call_desc);                                    \
    if (_code != ANEURALNETWORKS_NO_ERROR) {                                \
      const auto error_desc = NnApiErrorDescription(_code);                 \
      TF_LITE_KERNEL_LOG(context,                                           \
                         "NN API returned error %s at line %d while %s.\n", \
                         error_desc.c_str(), __LINE__, _call_desc);         \
      *p_errno = _code;                                                     \
      return kTfLiteError;                                                  \
    }                                                                       \
  } while (0)

#define RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(context, code, call_desc, \
                                                   p_tensor, p_errno)        \
  do {                                                                       \
    const auto _code = (code);                                               \
    const auto _call_desc = (call_desc);                                     \
    if (_code != ANEURALNETWORKS_NO_ERROR) {                                 \
      const auto error_desc = NnApiErrorDescription(_code);                  \
      TF_LITE_KERNEL_LOG(context,                                            \
                         "NN API returned error %s at line %d while %s "     \
                         "for tensor '%s'.\n",                               \
                         error_desc.c_str(), __LINE__, _call_desc,           \
                         (p_tensor)->name ? (p_tensor)->name : "no-name");   \
      *p_errno = _code;                                                      \
      return kTfLiteError;                                                   \
    }                                                                        \
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

bool IsInt32(TfLiteType type) {
  switch (type) {
    case kTfLiteInt32:
      return true;
    default:
      return false;
  }
}

bool IsFloatOrQuantized(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
    case kTfLiteUInt8:
    case kTfLiteInt8:
      return true;
    default:
      return false;
  }
}

bool IsFloatOrInt32(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
    case kTfLiteInt32:
      return true;
    default:
      return false;
  }
}

bool IsFloatQuantizedOrInt32(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
    case kTfLiteUInt8:
    case kTfLiteInt8:
    case kTfLiteInt32:
      return true;
    default:
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
    case kTfLiteBuiltinPow:
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinMinimum:
    case kTfLiteBuiltinPrelu:
    case kTfLiteBuiltinLeakyRelu:
      return true;
    default:
      return false;
  }
}

// Check if the operation requires explicit conversion from int8 to uint8
// values.
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
    case kTfLiteBuiltinTransposeConv: {
      // Transpose convolution has a different order of inputs:
      // 0: output_shape, 1: filter, 2: input, 3: bias.
      const int input_id = 2;
      const TfLiteType input_type = context->tensors[input_id].type;
      if (input_type == kTfLiteInt8) {
        return true;
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
    case kTfLiteBuiltinGather:
    case kTfLiteBuiltinGreater:
    case kTfLiteBuiltinGreaterEqual:
    case kTfLiteBuiltinHardSwish:
    case kTfLiteBuiltinL2Normalization:
    case kTfLiteBuiltinLeakyRelu:
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
    case kTfLiteBuiltinPrelu:
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

bool IsMeanWithDifferentInputOutputQuantization(const TfLiteContext* context,
                                                const TfLiteNode* node) {
  const auto& input = context->tensors[node->inputs->data[0]];
  const auto& output = context->tensors[node->outputs->data[0]];
  return input.params.scale != output.params.scale ||
         input.params.zero_point != output.params.zero_point;
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

bool IsDequantizeConstFloat16(TfLiteContext* context, const TfLiteNode* node,
                              const TfLiteRegistration* registration) {
  return registration->builtin_code == kTfLiteBuiltinDequantize &&
         context->tensors[node->inputs->data[0]].type ==
             TfLiteType::kTfLiteFloat16 &&
         IsConstantTensor(&context->tensors[node->inputs->data[0]]);
}

bool IsDequantizeNonConstFloat16(TfLiteContext* context, const TfLiteNode* node,
                                 const TfLiteRegistration* registration) {
  return registration->builtin_code == kTfLiteBuiltinDequantize &&
         context->tensors[node->inputs->data[0]].type ==
             TfLiteType::kTfLiteFloat16 &&
         !IsConstantTensor(&context->tensors[node->inputs->data[0]]);
}

bool IsDensifyConstTensor(TfLiteContext* context, const TfLiteNode* node,
                          const TfLiteRegistration* registration) {
  return registration->builtin_code == kTfLiteBuiltinDensify &&
         IsConstantTensor(&context->tensors[node->inputs->data[0]]);
}

bool HasUnspecifiedDimension(const TfLiteTensor* tensor) {
  if (tensor->dims_signature) {
    for (int i : TfLiteIntArrayView(tensor->dims_signature)) {
      if (i == -1) return true;
    }
  }
  return false;
}

ANeuralNetworksOperandType ConvertTensorTypeToNNType(
    const TfLiteTensor* tensor, TfLiteType ann_type_equivalent) {
  int32_t nn_type = 0;
  float scale = 0.0f;
  int32_t zero_point = 0;
  switch (tensor->type) {
    case kTfLiteFloat32:
      nn_type = ANEURALNETWORKS_TENSOR_FLOAT32;
      break;
    case kTfLiteUInt8:
      nn_type = ann_type_equivalent == kTfLiteInt32
                    ? ANEURALNETWORKS_TENSOR_INT32
                    : ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
      scale = tensor->params.scale;
      zero_point = tensor->params.zero_point;
      if (scale == 0) {
        // TENSOR_QUANT8_ASYMM and ANEURALNETWORKS_TENSOR_QUANT8_ASYMM
        // with zero scale are not valid in NNAPI.
        scale = 1;
      }
      break;
    case kTfLiteInt8:
      nn_type = ANEURALNETWORKS_TENSOR_QUANT8_SYMM;
      scale = tensor->params.scale;
      zero_point = tensor->params.zero_point;
      if (ann_type_equivalent == kTfLiteUInt8) {
        nn_type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
        zero_point += 128;
      } else if (ann_type_equivalent == kTfLiteInt32) {
        nn_type = ANEURALNETWORKS_TENSOR_INT32;
        zero_point += 128;
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
      zero_point = tensor->params.zero_point;
      break;
    case kTfLiteBool:
      nn_type = ANEURALNETWORKS_TENSOR_BOOL8;
      break;
    case kTfLiteInt16:
      nn_type = ANEURALNETWORKS_TENSOR_QUANT16_SYMM;
      scale = tensor->params.scale;
      zero_point = tensor->params.zero_point;
      break;
    default:
      break;
  }
  uint32_t tensor_rank = static_cast<uint32_t>(tensor->dims->size);
  uint32_t* tensor_dims = reinterpret_cast<uint32_t*>(tensor->dims->data);
  static uint32_t scalar_rank = 1;
  // treat scalar input as single cell tensor in NNAPI.
  if (tensor_rank == 0) {
    tensor_rank = scalar_rank;
    tensor_dims = &scalar_rank;
  }
  ANeuralNetworksOperandType nn_operand_type{
      .type = nn_type,
      .dimensionCount = tensor_rank,
      .dimensions = tensor_dims,
      .scale = scale,
      .zeroPoint = zero_point,
  };
  return nn_operand_type;
}

// NNAPI in API 31 hard-code the preferred alignment/padding with 64 bytes.
constexpr size_t kDefaultByteAlignmentForNNAPI = 64;

static size_t GetNumPaddingBytes(size_t byte_size) {
  size_t num_padding_bytes = 0;
  if (byte_size % kDefaultByteAlignmentForNNAPI) {
    num_padding_bytes = kDefaultByteAlignmentForNNAPI -
                        (byte_size % kDefaultByteAlignmentForNNAPI);
  }
  return num_padding_bytes;
}

static size_t GetNNTensorSize(size_t tensor_size, bool allow_padding) {
  size_t padding_bytes = GetNumPaddingBytes(tensor_size);
  size_t nn_tensor_size = tensor_size;
  if (allow_padding) {
    nn_tensor_size += padding_bytes;
  }
  return nn_tensor_size;
}

// Return NNAPI device handle with the provided null-terminated device name.
// Returns kTfLiteError in case of any NNAPI error and if no device with the
// given name can be found.
TfLiteStatus GetDeviceHandle(const NnApi* nnapi, TfLiteContext* context,
                             const char* device_name_ptr,
                             ANeuralNetworksDevice** result, int* nnapi_errno) {
  if (!device_name_ptr) return kTfLiteError;
  *result = nullptr;
  std::string device_name(device_name_ptr);
  uint32_t num_devices = 0;
  nnapi->ANeuralNetworks_getDeviceCount(&num_devices);

  for (uint32_t i = 0; i < num_devices; i++) {
    ANeuralNetworksDevice* device = nullptr;
    const char* buffer = nullptr;
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context, nnapi->ANeuralNetworks_getDevice(i, &device),
        "Searching for target device", nnapi_errno);

    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context, nnapi->ANeuralNetworksDevice_getName(device, &buffer),
        "Searching for target device", nnapi_errno);

    if (device_name == buffer) {
      *result = device;
      return kTfLiteOk;
    }
  }

  context->ReportError(context,
                       "Could not find the specified NNAPI accelerator: %s. "
                       "Must be one of: {%s}.",
                       device_name_ptr,
                       nnapi::GetStringDeviceNamesList(nnapi).c_str());
  return kTfLiteError;
}

// Compute the hash of a TfLiteIntArray.
uint64_t GetHash(const TfLiteIntArray* int_array, uint64_t combine_with = 0) {
  constexpr auto kHashConst = 0x9e3779b97f4a7800ULL;
  uint64_t result = combine_with;
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

// In SPLIT_V, it is legal to specify -1 in size_splits representing an unknown
// split size taking as many values as possible. This function computes and
// returns the actual value of this unknown size, or returns -1 if all split
// sizes are known. The caller is responsible for making sure the size_splits
// and axis tensor are constants.
int ComputeSplitVUnknownSplitSize(const TfLiteContext* context,
                                  const TfLiteNode* node) {
  const auto& input = context->tensors[node->inputs->data[0]];
  const auto& size_splits_tensor = context->tensors[node->inputs->data[1]];
  const auto& axis_tensor = context->tensors[node->inputs->data[2]];

  const auto* size_splits = size_splits_tensor.data.i32;
  int num_splits = size_splits_tensor.dims->data[0];
  bool has_unknown_split_size = false;
  int sum_of_known_split_sizes = 0;
  for (int i = 0; i < num_splits; i++) {
    if (size_splits[i] == -1) {
      has_unknown_split_size = true;
    } else {
      sum_of_known_split_sizes += size_splits[i];
    }
  }

  int axis = axis_tensor.data.i32[0];
  axis = axis < 0 ? axis + input.dims->size : axis;
  int total_size = input.dims->data[axis];
  return has_unknown_split_size ? total_size - sum_of_known_split_sizes : -1;
}

// Bit mask for tensor flags.
enum {
  NN_TENSOR_FLAG_SCALAR_AS_TENSOR = 1U << 0,
  NN_TENSOR_FLAG_INT8_CONVERSION = 1U << 1,
  NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED = 1U << 2,
  NN_TENSOR_FLAG_FORCE_PER_CHANNEL = 1U << 3,
  NN_TENSOR_FLAG_HALF_TO_FLOAT_CONVERSION = 1U << 4,
};

// Returns the feature level to target when delegating to the given devices.
// The feature level is the max of the ones supported by the devices or
// the current NNAPI runtime feature level if no device is present.
TfLiteStatus GetTargetFeatureLevel(
    TfLiteContext* context, const NnApi* nnapi,
    const std::vector<ANeuralNetworksDevice*>& device_handles,
    int* target_feature_level, int* nnapi_errno) {
  *target_feature_level = nnapi->nnapi_runtime_feature_level;
  int64_t devices_feature_level = -1;
  for (const auto* device_handle : device_handles) {
    int64_t curr_device_feature_level;
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context,
        nnapi->ANeuralNetworksDevice_getFeatureLevel(
            device_handle, &curr_device_feature_level),
        "Searching for target device", nnapi_errno);

    devices_feature_level =
        std::max(curr_device_feature_level, devices_feature_level);
  }

  if ((devices_feature_level > 0) &&
      // This second check is necessary since if the nnapi-reference device is
      // in the list of target devices the devices_feature_level value will be
      // 1000.
      (devices_feature_level < nnapi->nnapi_runtime_feature_level)) {
    TFLITE_LOG(TFLITE_LOG_INFO,
               "Changing NNAPI Feature Level %lld to "
               "supported by target devices: %lld",
               nnapi->android_sdk_version, devices_feature_level);

    *target_feature_level = devices_feature_level;
  }

  return kTfLiteOk;
}

// Returns true if this delegate is configured to use a specific set of devices.
// This will happen either if:
// - accelerator_name option has been specified
// - NNAPI CPU implementation has been explicitly disabled.
// If exclude_nnapi_reference is true this method will return false if the
// accelerator_name in the delegate options is equal to "nnapi-reference"
bool ShouldUseTargetDevices(StatefulNnApiDelegate::Options delegate_options,
                            const NnApi* nnapi,
                            bool exclude_nnapi_reference = false) {
  const char* device_name_ptr = delegate_options.accelerator_name;
  std::string nnapi_cpu("nnapi-reference");
  bool has_selected_accelerator = device_name_ptr != nullptr;
  if (exclude_nnapi_reference && has_selected_accelerator) {
    if (nnapi_cpu == device_name_ptr) return false;
  }
  return (delegate_options.disallow_nnapi_cpu &&
          nnapi->android_sdk_version >=
              delegate::nnapi::kMinSdkVersionForNNAPI12) ||
         has_selected_accelerator;
}

// Fills the given result vector with the list of devices the given delegate
// is referring to.
// There are three possible results:
// - an empty array (not the full list of available accelerators,
//   for efficiency reasons) if no accelerator is chosen and the
//   disallow_nnapi_cpu delegate option is false.
// - A single element array with the target processor, if an accelerator name
//   is specified in the delegate options.
// - The full list of devices available on device less the nnapi reference
//   implementation if the delegate option disallow_nnapi_cpu has been
//   specified.
TfLiteStatus GetTargetDevices(TfLiteContext* context, TfLiteDelegate* delegate,
                              const NnApi* nnapi, int* nnapi_errno,
                              std::vector<ANeuralNetworksDevice*>* result) {
  if (nnapi->android_sdk_version < delegate::nnapi::kMinSdkVersionForNNAPI12) {
    return kTfLiteError;
  }

  const auto delegate_options = StatefulNnApiDelegate::GetOptions(delegate);
  const char* device_name_ptr = delegate_options.accelerator_name;

  if (device_name_ptr != nullptr) {
    // User specified an accelerator to use.
    ANeuralNetworksDevice* nnapi_device = nullptr;
    TF_LITE_ENSURE_STATUS(GetDeviceHandle(nnapi, context, device_name_ptr,
                                          &nnapi_device, nnapi_errno));
    result->push_back(nnapi_device);
  } else if (delegate_options.disallow_nnapi_cpu) {
    std::string nnapi_cpu("nnapi-reference");
    uint32_t num_devices = 0;
    nnapi->ANeuralNetworks_getDeviceCount(&num_devices);

    for (uint32_t i = 0; i < num_devices; i++) {
      ANeuralNetworksDevice* device = nullptr;
      const char* buffer = nullptr;
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context, nnapi->ANeuralNetworks_getDevice(i, &device),
          "Getting list of available devices", nnapi_errno);
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context, nnapi->ANeuralNetworksDevice_getName(device, &buffer),
          "Getting list of available devices", nnapi_errno);
      if (nnapi_cpu != buffer) {
        result->push_back(device);
      }
    }
  }

  return kTfLiteOk;
}

}  // namespace

namespace delegate {
namespace nnapi {

#ifdef TFLITE_NNAPI_ALLOW_MMAP_SHARING
NNMemory::NNMemory(const NnApi* nnapi, const char* name, size_t size) {
  if (name && size > 0) {
    nnapi_ = nnapi;
    byte_size_ = size;
#ifdef __ANDROID__
    fd_ = nnapi_->ASharedMemory_create(name, size);
#else
    // For non-Android platforms ASharedMemory_create needs unique name to
    // create a shared memory object (see nnapi_implementation.cc).
    char shm_name_buffer[L_tmpnam];
    if (tmpnam(shm_name_buffer) == nullptr) {
      shm_name_buffer[0] = '\0';
    }
    // tmpnam will produce a string containing with slashes, but shm_open
    // won't like that.
    shm_region_name_ = std::string(name) + std::string(shm_name_buffer);
    std::replace(shm_region_name_.begin(), shm_region_name_.end(), '/', '-');
    fd_ = nnapi_->ASharedMemory_create(shm_region_name_.c_str(), size);
#endif

    data_ptr_ = reinterpret_cast<uint8_t*>(
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    nnapi_->ANeuralNetworksMemory_createFromFd(size, PROT_READ | PROT_WRITE,
                                               fd_, 0, &nn_memory_handle_);
  }
}
#else
NNMemory::NNMemory(const NnApi* /*nnapi*/, const char* /*name*/,
                   size_t /*size*/)
    : nnapi_(nullptr) {}
#endif

NNMemory::~NNMemory() {
#ifdef TFLITE_NNAPI_ALLOW_MMAP_SHARING
  if (data_ptr_) {
    munmap(data_ptr_, byte_size_);
  }
  if (nn_memory_handle_) {
    nnapi_->ANeuralNetworksMemory_free(nn_memory_handle_);
  }
#ifdef __ANDROID__
  if (fd_ >= 0) close(fd_);
#else
  if (!shm_region_name_.empty()) shm_unlink(shm_region_name_.c_str());
#endif
#endif
}

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
                 std::vector<int>* nnapi_to_tflite_op_mapping,
                 ANeuralNetworksModel* nn_model, int* nnapi_errno,
                 bool allow_dynamic_dimensions)
      : nnapi_(nnapi),
        context_(context),
        operand_mapping_(tensor_mapping),
        dequantize_mapping_(dequantize_mapping),
        allocation_memory_mapping_(allocation_mapping),
        nnapi_to_tflite_op_mapping_(nnapi_to_tflite_op_mapping),
        nn_model_(nn_model),
        nnapi_errno_(nnapi_errno),
        allow_dynamic_dimensions_(allow_dynamic_dimensions) {}

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

  TfLiteStatus AddVectorInt16Operand(const int16_t* values,
                                     uint32_t num_values) {
    return AddVectorOperand<int16_t>(values, num_values,
                                     ANEURALNETWORKS_TENSOR_QUANT16_SYMM,
                                     /*scale=*/1.f, /*zero_point=*/0);
  }

  TfLiteStatus AddVectorInt8Operand(const int8_t* values, uint32_t num_values) {
    return AddVectorOperand<int8_t>(values, num_values,
                                    ANEURALNETWORKS_TENSOR_QUANT8_SYMM,
                                    /*scale=*/1.f, /*zero_point=*/0);
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

  TfLiteStatus AddStateInt16Tensor(int tensor_index,
                                   int* ann_tensor_index_out) {
    TfLiteTensor* tensor = &context_->tensors[tensor_index];
    return AddAdditionalOutputTensor(
        tensor->dims->size, reinterpret_cast<uint32_t*>(tensor->dims->data),
        ANEURALNETWORKS_TENSOR_QUANT16_SYMM, tensor->params.scale,
        tensor->params.zero_point, ann_tensor_index_out);
  }

  TfLiteStatus AddStateInt8AsymTensor(int tensor_index,
                                      int* ann_tensor_index_out) {
    TfLiteTensor* tensor = &context_->tensors[tensor_index];
    return AddAdditionalOutputTensor(
        tensor->dims->size, reinterpret_cast<uint32_t*>(tensor->dims->data),
        ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, tensor->params.scale,
        tensor->params.zero_point, ann_tensor_index_out);
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
  TfLiteStatus TransformHardSwishIntoSupportedOps(int lite_input_index,
                                                  int lite_output_index,
                                                  bool need_int8_conversion,
                                                  int lite_node_index) {
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
      TF_LITE_ENSURE_OK(
          context_, FinalizeAddOperation(ANEURALNETWORKS_MUL, lite_node_index));
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
      TF_LITE_ENSURE_OK(
          context_, FinalizeAddOperation(ANEURALNETWORKS_MUL, lite_node_index));
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
      TF_LITE_ENSURE_OK(
          context_, FinalizeAddOperation(ANEURALNETWORKS_MUL, lite_node_index));
    }

    // Stage 4: y = s3 + s2
    {
      augmented_inputs_.push_back(s2_out_ann_index);
      augmented_inputs_.push_back(s3_out_ann_index);
      TF_LITE_ENSURE_OK(context_,
                        AddScalarInt32Operand(ANEURALNETWORKS_FUSED_NONE));
      TF_LITE_ENSURE_OK(context_,
                        AddTensorOutput(lite_output_index, tensor_flags));
      TF_LITE_ENSURE_OK(
          context_, FinalizeAddOperation(ANEURALNETWORKS_ADD, lite_node_index));
    }

    return kTfLiteOk;
  }

  // Adds the operation to the model and maps the operation to the originating
  // TFLite one.
  TfLiteStatus AddOperationToModel(ANeuralNetworksOperationType type,
                                   uint32_t input_count, const uint32_t* inputs,
                                   uint32_t output_count,
                                   const uint32_t* outputs,
                                   int lite_node_index) {
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_addOperation(
            nn_model_, type, input_count, inputs, output_count, outputs),
        "adding operation", nnapi_errno_);
    nnapi_to_tflite_op_mapping_->push_back(lite_node_index);
    return kTfLiteOk;
  }

  // Adds a Dequantize operator and replaces the input tensor index with the
  // dequantized version. If the dequantized version of the operator already
  // exists then it is not added again.
  TfLiteStatus AddDequantize(int nn_input_index, int lite_tensor_index,
                             TfLiteType dequantized_type, int lite_node_index) {
    const int ann_index =
        operand_mapping_->lite_index_to_ann(lite_tensor_index);
    int dequantized_ann_index =
        dequantize_mapping_->DequantizedAnnIndex(ann_index, dequantized_type);

    if (dequantized_ann_index == -1) {
      // The dequantized version does not exist yet, it has to be added: a new
      // Dequantize operation is added, yielding a new tensor.
      const TfLiteTensor& tensor = context_->tensors[lite_tensor_index];
      ANeuralNetworksOperandType operand_type{
          ANEURALNETWORKS_TENSOR_FLOAT32,
          static_cast<uint32_t>(tensor.dims->size),
          reinterpret_cast<uint32_t*>(tensor.dims->data), 0.f, 0};
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context_,
          nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type),
          "adding operand", nnapi_errno_);
      dequantized_ann_index = operand_mapping_->add_new_non_tensor_operand();

      // Add Dequantize operation.
      const uint32_t dequantize_input[1] = {static_cast<uint32_t>(ann_index)};
      const uint32_t dequantize_output[1] = {
          static_cast<uint32_t>(dequantized_ann_index)};
      TF_LITE_ENSURE_OK(
          context_, AddOperationToModel(ANEURALNETWORKS_DEQUANTIZE,
                                        /*input_count=*/1, dequantize_input,
                                        /*output_count=*/1, dequantize_output,
                                        lite_node_index));
      dequantize_mapping_->Add(ann_index, dequantized_type,
                               dequantized_ann_index);
    }

    // The input for the original operation is modified so that the operation
    // now uses the dequantized tensor as input.
    augmented_inputs_[nn_input_index] = dequantized_ann_index;

    return kTfLiteOk;
  }

  // Add a RESHAPE op which reshapes an NNAPI intermediate output to the
  // dimensions of the TFLite output tensor.
  TfLiteStatus AppendReshape(int nn_input_index, int lite_out_tensor_index,
                             int lite_node_index) {
    augmented_inputs_.push_back(nn_input_index);
    auto& output_tensor = context_->tensors[lite_out_tensor_index];
    TF_LITE_ENSURE_STATUS(
        AddVectorInt32Operand(output_tensor.dims->data,
                              static_cast<uint32_t>(output_tensor.dims->size)));
    TF_LITE_ENSURE_OK(context_,
                      AddTensorOutput(lite_out_tensor_index,
                                      NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED));
    TF_LITE_ENSURE_STATUS(
        FinalizeAddOperation(ANEURALNETWORKS_RESHAPE, lite_node_index));
    return kTfLiteOk;
  }

  // Add a ADD op to requantize an NNAPI intermediate output to the scale and
  // zero point of the TFLite output tensor.
  TfLiteStatus AppendRequantize(int nn_input_index, int lite_out_tensor_index,
                                int lite_node_index, int tensor_flags = 0) {
    augmented_inputs_.push_back(nn_input_index);
    auto& output_tensor = context_->tensors[lite_out_tensor_index];

    // Create a zero vector with the same type as the output type. There is only
    // one single element in the vector, and it is broadcastable with any
    // tensor.
    TF_LITE_ENSURE(context_, IsQuantized(output_tensor.type));
    bool need_int8_conversion = tensor_flags & NN_TENSOR_FLAG_INT8_CONVERSION;
    int nn_type = (output_tensor.type == kTfLiteUInt8 || need_int8_conversion)
                      ? ANEURALNETWORKS_TENSOR_QUANT8_ASYMM
                      : ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED;
    int8_t zero = 0;
    TF_LITE_ENSURE_STATUS(AddVectorOperand(&zero, /*num_values=*/1, nn_type,
                                           /*scale=*/1.0f, /*zero_point=*/0));

    TF_LITE_ENSURE_STATUS(AddScalarInt32Operand(ANEURALNETWORKS_FUSED_NONE));
    TF_LITE_ENSURE_STATUS(AddTensorOutput(lite_out_tensor_index, tensor_flags));
    TF_LITE_ENSURE_STATUS(
        FinalizeAddOperation(ANEURALNETWORKS_ADD, lite_node_index));
    return kTfLiteOk;
  }

  // Lower PACK into CONCAT + RESHAPE when possible
  TfLiteStatus TransformPackIntoSupportedOps(int lite_node_index,
                                             TfLiteNode* node,
                                             TfLiteRegistration* reg) {
    // Add input tensors for CONCAT, and calculate the dimensions for the
    // output.
    int concat_output_ann_index = -1;
    TfLitePackParams* builtin =
        reinterpret_cast<TfLitePackParams*>(node->builtin_data);
    auto& input_tensor = context_->tensors[node->inputs->data[0]];
    int axis = builtin->axis < 0 ? input_tensor.dims->size + builtin->axis + 1
                                 : builtin->axis;
    TF_LITE_ENSURE(context_, axis < input_tensor.dims->size);
    uint32_t concat_dim_size = 0;
    for (int input_pos = 0; input_pos < node->inputs->size; ++input_pos) {
      const auto input_index = node->inputs->data[input_pos];
      concat_dim_size +=
          context_->tensors[node->inputs->data[input_pos]].dims->data[axis];
      TF_LITE_ENSURE_STATUS(
          AddTensorInput(input_index, /*hybrid_op=*/false,
                         NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED));
    }
    TF_LITE_ENSURE_STATUS(AddScalarInt32Operand(axis));
    std::vector<uint32_t> concat_output_shape(input_tensor.dims->size, 0);
    for (int i = 0; i < concat_output_shape.size(); i++) {
      if (i == axis) {
        concat_output_shape[i] = concat_dim_size;
      } else {
        concat_output_shape[i] = input_tensor.dims->data[i];
      }
    }
    TF_LITE_ENSURE_STATUS(AddIntermediateOutputTensor(
        input_tensor.type, concat_output_shape.size(),
        concat_output_shape.data(), input_tensor.params.scale,
        input_tensor.params.zero_point, &concat_output_ann_index));
    TF_LITE_ENSURE_STATUS(
        FinalizeAddOperation(ANEURALNETWORKS_CONCATENATION, lite_node_index));

    // Reshape the output tensor
    TF_LITE_ENSURE_STATUS(AppendReshape(
        concat_output_ann_index, node->outputs->data[0], lite_node_index));
    return kTfLiteOk;
  }

  // Lower SPLIT_V into SLICEs.
  TfLiteStatus TransformSplitVIntoSupportedOps(int lite_node_index,
                                               TfLiteNode* node,
                                               TfLiteRegistration* reg) {
    auto& input = context_->tensors[node->inputs->data[0]];
    int input_rank = input.dims->size;

    const auto& size_splits_tensor = context_->tensors[node->inputs->data[1]];
    const auto* size_splits = size_splits_tensor.data.i32;
    int num_splits = size_splits_tensor.dims->data[0];
    int axis = context_->tensors[node->inputs->data[2]].data.i32[0];
    axis = axis < 0 ? axis + input_rank : axis;
    TF_LITE_ENSURE(context_, axis >= 0);
    TF_LITE_ENSURE(context_, axis < input_rank);
    int unknown_split_size = ComputeSplitVUnknownSplitSize(context_, node);

    // Keep track of the start index of a slice.
    int slice_begin_index = 0;
    for (int split_index = 0; split_index < num_splits; split_index++) {
      int split_size = size_splits[split_index] == -1
                           ? unknown_split_size
                           : size_splits[split_index];
      TF_LITE_ENSURE(context_, split_size > 0);

      // Parameters of SLICE.
      std::vector<int> begin_indices(input_rank);
      std::vector<int> slice_sizes(input_rank);
      for (int i = 0; i < input_rank; i++) {
        if (i == axis) {
          // Take only the splitted size.
          begin_indices[i] = slice_begin_index;
          slice_sizes[i] = split_size;
        } else {
          // Take the full size.
          begin_indices[i] = 0;
          slice_sizes[i] = input.dims->data[i];
        }
      }
      slice_begin_index += split_size;

      // Build NNAPI SLICE inputs and output.
      TF_LITE_ENSURE_STATUS(AddTensorInput(
          node->inputs->data[0],
          /*hybrid_op=*/false, NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED));
      TF_LITE_ENSURE_STATUS(
          AddVectorInt32Operand(begin_indices.data(), begin_indices.size()));
      TF_LITE_ENSURE_STATUS(
          AddVectorInt32Operand(slice_sizes.data(), slice_sizes.size()));
      int lite_output_index = node->outputs->data[split_index];
      TF_LITE_ENSURE_STATUS(AddTensorOutput(
          lite_output_index, NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED));

      TF_LITE_ENSURE_STATUS(
          FinalizeAddOperation(ANEURALNETWORKS_SLICE, lite_node_index));
    }
    return kTfLiteOk;
  }

  // Lower SQUARED_DIFFERENCE into SUB and MUL.
  TfLiteStatus TransformSquaredDifferenceIntoSupportedOps(
      int lite_node_index, TfLiteNode* node, TfLiteRegistration* reg) {
    const TfLiteTensor& lhs = context_->tensors[node->inputs->data[0]];
    const TfLiteTensor& output = context_->tensors[node->outputs->data[0]];

    // Stage1 : diff = lhs - rhs
    int diff_out_ann_index = 0;
    {
      // For quantized data type, choose a proper scale and zero point based on
      // the output range.
      float max_output = 0.f;
      int diff_output_zero_point = 0;
      int diff_output_nn_type = ANEURALNETWORKS_TENSOR_FLOAT32;
      switch (lhs.type) {
        case kTfLiteFloat32:
          diff_output_nn_type = ANEURALNETWORKS_TENSOR_FLOAT32;
          break;
        case kTfLiteInt32:
          diff_output_nn_type = ANEURALNETWORKS_TENSOR_INT32;
          break;
        case kTfLiteUInt8:
          max_output = (255 - output.params.zero_point) * output.params.scale;
          diff_output_zero_point = 128;
          diff_output_nn_type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
          break;
        case kTfLiteInt8:
          max_output = (127 - output.params.zero_point) * output.params.scale;
          diff_output_zero_point = 0;
          diff_output_nn_type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED;
          break;
        default:
          return kTfLiteError;
      }
      // Final output range: [0, max_output], and output = diff^2,
      // -> diff range: [-sqrt(max_output), sqrt(max_output)]
      // This range corresponds to [1, 255] for uint8 with zero_point = 128,
      // or [-127, 127] for int8 with zero_point = 0.
      float diff_output_scale = 2.0f * std::sqrt(max_output) / 254.0f;

      TF_LITE_ENSURE_OK(
          context_, AddTensorInput(node->inputs->data[0], /*hybrid_op=*/false,
                                   NN_TENSOR_FLAG_SCALAR_AS_TENSOR |
                                       NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED));
      TF_LITE_ENSURE_OK(
          context_, AddTensorInput(node->inputs->data[1], /*hybrid_op=*/false,
                                   NN_TENSOR_FLAG_SCALAR_AS_TENSOR |
                                       NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED));
      TF_LITE_ENSURE_OK(context_,
                        AddScalarInt32Operand(ANEURALNETWORKS_FUSED_NONE));
      TF_LITE_ENSURE_OK(
          context_,
          AddAdditionalOutputTensor(
              output.dims->size, reinterpret_cast<uint32_t*>(output.dims->data),
              diff_output_nn_type, diff_output_scale, diff_output_zero_point,
              &diff_out_ann_index));
      TF_LITE_ENSURE_OK(
          context_, FinalizeAddOperation(ANEURALNETWORKS_SUB, lite_node_index));
    }

    // Stage2 : out = diff * diff
    {
      augmented_inputs_.push_back(diff_out_ann_index);
      augmented_inputs_.push_back(diff_out_ann_index);
      TF_LITE_ENSURE_OK(context_,
                        AddScalarInt32Operand(ANEURALNETWORKS_FUSED_NONE));
      TF_LITE_ENSURE_OK(context_,
                        AddTensorOutput(node->outputs->data[0],
                                        NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED));
      TF_LITE_ENSURE_OK(
          context_, FinalizeAddOperation(ANEURALNETWORKS_MUL, lite_node_index));
    }

    return kTfLiteOk;
  }

  // Finish emitting the op (of type `type`) into the NN API.
  TfLiteStatus FinalizeAddOperation(ANeuralNetworksOperationType type,
                                    int lite_node_index) {
    // Actually add a NN API operation
    TF_LITE_ENSURE_OK(context_,
                      AddOperationToModel(
                          type, static_cast<uint32_t>(augmented_inputs_.size()),
                          augmented_inputs_.data(),
                          static_cast<uint32_t>(augmented_outputs_.size()),
                          augmented_outputs_.data(), lite_node_index));
    augmented_inputs_.clear();
    augmented_outputs_.clear();
    return kTfLiteOk;
  }

  TfLiteStatus AddSingleValueTensorAsScalarOperand(int tensor_index,
                                                   int nn_type) {
    const TfLiteTensor* tensor = &context_->tensors[tensor_index];
    TF_LITE_ENSURE_EQ(context_, NumElements(tensor), 1);

    ANeuralNetworksOperandType operand_type{.type = nn_type};
    RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(
        context_,
        nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type),
        "adding operand", tensor, nnapi_errno_);
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
        "adding operand", nnapi_errno_);

    augmented_inputs_.push_back(ann_tensor_index);

    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_setOperandValue(
            nn_model_, ann_tensor_index, new_tensor->data.raw,
            new_tensor->bytes),
        "setting new operand value", nnapi_errno_);

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

  TfLiteStatus AddIntermediateOutputTensor(TfLiteType tfl_type,
                                           uint32_t dimension_count,
                                           const uint32_t* dimension_data,
                                           float scale, int32_t zero_point,
                                           int* ann_index_out,
                                           bool need_int8_conversion = false) {
    int32_t nn_type;
    switch (tfl_type) {
      case kTfLiteFloat32:
        nn_type = ANEURALNETWORKS_TENSOR_FLOAT32;
        break;
      case kTfLiteInt8:
        nn_type = need_int8_conversion
                      ? ANEURALNETWORKS_TENSOR_QUANT8_ASYMM
                      : ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED;
        break;
      case kTfLiteUInt8:
        nn_type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
        break;
      default:
        return kTfLiteError;
    }
    if (need_int8_conversion) {
      zero_point += 128;
    }
    TF_LITE_ENSURE_STATUS(
        AddAdditionalOutputTensor(dimension_count, dimension_data, nn_type,
                                  scale, zero_point, ann_index_out));
    return kTfLiteOk;
  }

  void ClearInputOuputLists() {
    augmented_inputs_.clear();
    augmented_outputs_.clear();
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
        "adding operand", nnapi_errno_);
    const int ann_index = operand_mapping_->add_new_non_tensor_operand();
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_setOperandValue(nn_model_, ann_index,
                                                     &value, sizeof(T)),
        "setting new operand value", nnapi_errno_);
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
        "adding operand", nnapi_errno_);

    const int ann_index = operand_mapping_->add_new_non_tensor_operand();
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_setOperandValue(
            nn_model_, ann_index, values, sizeof(T) * num_values),
        "settings new operand value", nnapi_errno_);
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
        "adding operand", nnapi_errno_);
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
    const bool use_int8_asymm_signed =
        tensor_flags & NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED;
    const bool force_per_channel =
        tensor_flags & NN_TENSOR_FLAG_FORCE_PER_CHANNEL;
    const bool need_half2float_conversion =
        tensor_flags & NN_TENSOR_FLAG_HALF_TO_FLOAT_CONVERSION;

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
    ANeuralNetworksSymmPerChannelQuantParams ann_perchannel_params;
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
      case kTfLiteFloat16:
        nn_type = ANEURALNETWORKS_TENSOR_FLOAT16;
        if (need_half2float_conversion) {
          nn_type = ANEURALNETWORKS_TENSOR_FLOAT32;
          operand_mapping_->add_type_conversion(tensor_index, kTfLiteFloat32);
        }
        break;
      case kTfLiteUInt8:
        nn_type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
        scale = tensor->params.scale;
        zeroPoint = tensor->params.zero_point;
        if (scale == 0) {
          // ANEURALNETWORKS_TENSOR_QUANT8_ASYMM with zero scale is not valid in
          // NNAPI.
          scale = 1;
        }
        break;
      case kTfLiteInt8:
        // If explicit int8 conversion is needed, we still need
        // ANEURALNETWORKS_TENSOR_QUANT8_ASYMM type.
        if (use_int8_asymm_signed) {
          nn_type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED;
        } else if (need_int8_conversion) {
          nn_type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
        } else {
          nn_type = ANEURALNETWORKS_TENSOR_QUANT8_SYMM;
        }
        scale = tensor->params.scale;
        zeroPoint = tensor->params.zero_point;
        if (tensor->quantization.type == kTfLiteAffineQuantization) {
          TfLiteAffineQuantization* quantization_params =
              static_cast<TfLiteAffineQuantization*>(
                  tensor->quantization.params);
          if (quantization_params->scale->size > 1 || force_per_channel) {
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
          } else if (quantization_params->scale->size == 1) {
            scale = quantization_params->scale->data[0];
            zeroPoint = quantization_params->zero_point->data[0];
          }
        }
        if (nn_type != ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL) {
          if (need_int8_conversion) {
            zeroPoint += 128;
            operand_mapping_->add_type_conversion(tensor_index, kTfLiteUInt8);
          }
          if (scale == 0) {
            // QUANT8 tensors with zero scale are not valid in NNAPI.
            scale = 1;
          }
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
    bool has_unspecified_dimensions = HasUnspecifiedDimension(tensor);
    uint32_t tensor_rank = static_cast<uint32_t>(tensor->dims->size);
    std::vector<uint32_t> dims_unspecified(tensor_rank, 0);
    if (has_unspecified_dimensions) {
      for (int i = 0; i < tensor->dims_signature->size; i++) {
        dims_unspecified[i] = tensor->dims_signature->data[i] == -1
                                  ? 0
                                  : tensor->dims_signature->data[i];
      }
    }
    uint32_t* tensor_dims =
        has_unspecified_dimensions && allow_dynamic_dimensions_
            ? dims_unspecified.data()
            : reinterpret_cast<uint32_t*>(tensor->dims->data);
    if (scalar_as_tensor && tensor_rank == 0) {
      // Use rank 1, shape {1} operand for TFLite scalar tensors.
      tensor_rank = 1;
      tensor_dims = &tensor_rank;
    }
    if (tensor_rank == 0) {
      // if the tensor_rank is 0, the dimension ptr must be nullptr.
      tensor_dims = nullptr;
    }

    ANeuralNetworksOperandType operand_type{nn_type, tensor_rank, tensor_dims,
                                            scale, zeroPoint};
    RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(
        context_,
        nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type),
        "adding operand", tensor, nnapi_errno_);

    if (nn_type == ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL) {
      RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(
          context_,
          nnapi_->ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(
              nn_model_, ann_tensor_index, &ann_perchannel_params),
          "setting new operand per channel quantization params", tensor,
          nnapi_errno_);
    }
    if (tensor->allocation_type == kTfLiteMmapRo) {
      if (IsQuantized(tensor_type) && need_int8_conversion &&
          nn_type != ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL) {
        // We need to add a tensor and convert the weights into uint8.
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
        RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(
            context_,
            nnapi_->ANeuralNetworksModel_setOperandValue(
                nn_model_, ann_tensor_index, new_tensor->data.raw,
                new_tensor->bytes),
            "setting new operand value", tensor, nnapi_errno_);
      } else if (tensor_type == kTfLiteFloat16 && need_half2float_conversion) {
        // We need to convert the constant fp16 weights to fp32. The new_tensor
        // is needed for lifetime management for the converted weights.
        int new_tensor_index = -1;
        TF_LITE_ENSURE_OK(context_,
                          context_->AddTensors(context_, 1, &new_tensor_index));
        TfLiteTensor* new_tensor = &context_->tensors[new_tensor_index];
        new_tensor->type = kTfLiteFloat32;
        new_tensor->allocation_type = kTfLiteDynamic;
        // Not removing the new tensor in case of resizing errors since it will
        // be cleared by the context
        TF_LITE_ENSURE_OK(
            context_, context_->ResizeTensor(context_, new_tensor,
                                             // Resize Tensor takes ownership of
                                             // the dims array passed as param
                                             TfLiteIntArrayCopy(tensor->dims)));
        // Convert the fp16 value into corresponding fp32 value;
        const auto num_elements = NumElements(tensor);
        for (int i = 0; i < num_elements; ++i) {
          new_tensor->data.f[i] = fp16_ieee_to_fp32_value(
              reinterpret_cast<uint16_t*>(tensor->data.data)[i]);
        }
        RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(
            context_,
            nnapi_->ANeuralNetworksModel_setOperandValue(
                nn_model_, ann_tensor_index, new_tensor->data.data,
                new_tensor->bytes),
            "setting new operand value", tensor, nnapi_errno_);
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
        RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(
            context_,
            nnapi_->ANeuralNetworksModel_setOperandValueFromMemory(
                nn_model_, ann_tensor_index, ann_memory_handle, offset,
                tensor->bytes),
            "setting new operand value from memory", tensor, nnapi_errno_);
#endif
      } else {
        RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(
            context_,
            nnapi_->ANeuralNetworksModel_setOperandValue(
                nn_model_, ann_tensor_index, tensor->data.data, tensor->bytes),
            "setting new operand value", tensor, nnapi_errno_);
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

  // Tracks for every operation in the NNAPI model the source TfLite model
  // node index.
  std::vector<int>* const nnapi_to_tflite_op_mapping_;

  // The NNAPI model.
  ANeuralNetworksModel* const nn_model_;

  // Inputs and outputs for the current op. These are augmented in the sense
  // that NN API uses operands for all arguments, not just tensors, unlike
  // TensorFlow Lite.
  std::vector<uint32_t> augmented_inputs_;
  std::vector<uint32_t> augmented_outputs_;

  // Return status code of the latest NNAPI call.
  int* nnapi_errno_;

  // Whether to allow dynamic batch size without re-compilation.
  bool allow_dynamic_dimensions_;
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
  return Expect(IsFloatOrQuantized(input_type),
                NNAPIValidationFailureType::kUnsupportedInputType,
                "Input should be Float or Quant8", val_ctx);
}

bool ExpectIsFloatOrInt32Operator(const TfLiteContext* context,
                                  const TfLiteNode* node,
                                  OpValidationContext* val_ctx) {
  const auto input_type = context->tensors[node->inputs->data[0]].type;
  return Expect(IsFloatOrInt32(input_type),
                NNAPIValidationFailureType::kUnsupportedInputType,
                "Input should be Float or Int32", val_ctx);
}

bool ExpectIsFloatQuant8OrInt32Operator(const TfLiteContext* context,
                                        const TfLiteNode* node,
                                        OpValidationContext* val_ctx) {
  const auto input_type = context->tensors[node->inputs->data[0]].type;
  return Expect(IsFloatQuantizedOrInt32(input_type),
                NNAPIValidationFailureType::kUnsupportedInputType,
                "Input should be Float, Quant8, or Int32", val_ctx);
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
                "filter_scale < output_scale.",
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
      if (android_sdk_version >= kMinSdkVersionForNNAPI13) {
        ExpectIsFloatQuant8OrInt32Operator(context, node, &val_ctx);
        if (IsInt32(context->tensors[node->inputs->data[0]].type)) {
          Expect(reinterpret_cast<TfLiteAddParams*>(node->builtin_data)
                         ->activation == kTfLiteActNone,
                 NNAPIValidationFailureType::kNoActivationExpected,
                 "No activation function supported", &val_ctx);
        }
      } else {
        ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      }
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
      if (is_accelerator_specified) {
        ExpectMaxOpVersion(version, 3, &val_ctx);
      } else {
        ExpectMaxOpVersion(version, 2, &val_ctx);
      }
      if (android_sdk_version >= kMinSdkVersionForNNAPI13) {
        ExpectIsFloatQuant8OrInt32Operator(context, node, &val_ctx);
        if (IsInt32(context->tensors[node->inputs->data[0]].type)) {
          Expect(reinterpret_cast<TfLiteMulParams*>(node->builtin_data)
                         ->activation == kTfLiteActNone,
                 NNAPIValidationFailureType::kNoActivationExpected,
                 "No activation function supported", &val_ctx);
        }
      } else {
        ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      }
    } break;
    case kTfLiteBuiltinAveragePool2d: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      auto builtin = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
      // TODO(b/138756912): Large filter window would overflow on the
      // quantized reference CPU path.
      if (IsQuantized(context->tensors[node->inputs->data[0]].type)) {
        Expect(is_accelerator_specified ||
                   (builtin->filter_width * builtin->filter_height <= 256),
               NNAPIValidationFailureType::kUnsupportedOperandSize,
               "Large filter window would overflow on the reference CPU path",
               &val_ctx);
      }
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
      ExpectMaxOpVersion(version, 5, &val_ctx);
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
      ExpectMaxOpVersion(version, 5, &val_ctx);
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
      if (builtin->keep_num_dims) {
        ExpectMinAndroidSdkVersion(android_sdk_version,
                                   kMinSdkVersionForNNAPI13, &val_ctx);
      }
    } break;
    case kTfLiteBuiltinHardSwish: {
      // Add support for hardswish. For Pre-Q devices, deconstructing it into
      // basic ops. Though for some nnapi accelerators using optimized tflite
      // kernels might even be faster.
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinSoftmax: {
      ExpectOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      const auto& output = context->tensors[node->outputs->data[0]];
      ExpectTypeIn(output.type, {kTfLiteFloat32, kTfLiteUInt8, kTfLiteInt8},
                   NNAPIValidationFailureType::kUnsupportedOutputType,
                   "Output type should be one of kTfLiteFloat32, kTfLiteUInt8, "
                   "kTfLiteInt8.",
                   &val_ctx);
      const auto& input = context->tensors[node->inputs->data[0]];
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
      if (node->inputs->size >= 2) {
        Expect(context->tensors[node->inputs->data[1]].allocation_type ==
                   kTfLiteMmapRo,
               NNAPIValidationFailureType::kInputTensorShouldHaveConstantShape,
               "The shape input tensor must be constant.", &val_ctx);
      }
      if (node->inputs->size == 1) {
        // reject scalar reshaping
        auto* params =
            reinterpret_cast<TfLiteReshapeParams*>(node->builtin_data);
        int num_dimensions = params->num_dimensions;
        if (num_dimensions == 1 && params->shape[0] == 0) {
          // Legacy tflite models use a shape parameter of [0] to indicate
          // scalars.
          num_dimensions = 0;
        }
        Expect(num_dimensions > 0,
               NNAPIValidationFailureType::kUnsupportedOperandRank,
               "New shape rank should be > 0", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinResizeBilinear: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
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
      if (android_sdk_version <= kMinSdkVersionForNNAPI12) {
        Expect(!builtin->align_corners,
               NNAPIValidationFailureType::kUnsupportedOperandValue,
               "NNAPI does not support align_corners == true.", &val_ctx);
        Expect(!builtin->half_pixel_centers,
               NNAPIValidationFailureType::kUnsupportedOperandValue,
               "NNAPI does not support half_pixel_centers == true.", &val_ctx);
      }
      if (android_sdk_version < kMinSdkVersionForNNAPI12) {
        Expect(input.type == kTfLiteFloat32,
               NNAPIValidationFailureType::kUnsupportedInputType,
               "NNAPI 1.0 & 1.1 only supports float input.", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinResizeNearestNeighbor: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
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
      auto builtin = reinterpret_cast<TfLiteResizeNearestNeighborParams*>(
          node->builtin_data);
      if (android_sdk_version <= kMinSdkVersionForNNAPI12) {
        Expect(!builtin->align_corners,
               NNAPIValidationFailureType::kUnsupportedOperandValue,
               "NNAPI does not support align_corners == true.", &val_ctx);
        Expect(!builtin->half_pixel_centers,
               NNAPIValidationFailureType::kUnsupportedOperandValue,
               "NNAPI does not support half_pixel_centers == true.", &val_ctx);
      }
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

      const auto& input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat16, kTfLiteFloat32,
                           kTfLiteUInt8, kTfLiteInt8);

      if (input_type == kTfLiteUInt8 &&
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
      // Allow dequantizing fp16->fp32.
      if (android_sdk_version >= kMinSdkVersionForNNAPI13 &&
          context->tensors[node->inputs->data[0]].type == kTfLiteFloat16 &&
          context->tensors[node->inputs->data[0]].allocation_type !=
              kTfLiteMmapRo) {
        return true;
      }
      Expect(version == 1 || version == 2,
             NNAPIValidationFailureType::kUnsupportedOperatorVersion,
             "Supported op versions are 1 and 2 only", &val_ctx);

      const auto& input = context->tensors[node->inputs->data[0]];
      if (android_sdk_version < kMinSdkVersionForNNAPI12) {
        EXPECT_INPUT_TYPE_IN(input.type, kTfLiteUInt8);
      } else {
        EXPECT_INPUT_TYPE_IN(input.type, kTfLiteUInt8, kTfLiteInt8);

        if (android_sdk_version == kMinSdkVersionForNNAPI12 &&
            input.type == kTfLiteInt8) {
          const auto zero_point = input.params.zero_point;
          Expect(zero_point == 0,
                 NNAPIValidationFailureType::kUnsupportedInputType,
                 "NN API supports int8 type since version 1.2 but only for "
                 "symmetric quantization.",
                 &val_ctx);
        }
      }
    } break;
    case kTfLiteBuiltinDensify: {
      // Allow densifying sparse weights.
      if (android_sdk_version >= kMinSdkVersionForNNAPI13 &&
          context->tensors[node->inputs->data[0]].allocation_type ==
              kTfLiteMmapRo) {
        return true;
      }
      return false;
    } break;
    case kTfLiteBuiltinFloor: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinReluN1To1:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinLogistic: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
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
      ExpectMaxOpVersion(version, 3, &val_ctx);
      const TfLiteType input_type =
          context->tensors[node->inputs->data[0]].type;
      Expect((android_sdk_version >= kMinSdkVersionForNNAPI11 &&
              IsFloat(input_type)) ||
                 (android_sdk_version >= kMinSdkVersionForNNAPI12 &&
                  IsQuantized(input_type)) ||
                 (android_sdk_version >= kMinSdkVersionForNNAPI13 &&
                  IsInt32(input_type)),
             NNAPIValidationFailureType::kUnsupportedInputType,
             "NNAPI only support float sub.", &val_ctx);
      if (IsInt32(input_type)) {
        Expect(reinterpret_cast<TfLiteSubParams*>(node->builtin_data)
                       ->activation == kTfLiteActNone,
               NNAPIValidationFailureType::kNoActivationExpected,
               "No activation function supported", &val_ctx);
      }
      const int input0_rank =
          context->tensors[node->inputs->data[0]].dims->size;
      const int input1_rank =
          context->tensors[node->inputs->data[1]].dims->size;
      Expect(input0_rank <= 4 && input1_rank <= 4,
             NNAPIValidationFailureType::kUnsupportedOperandRank,
             "Input rank must be <= 4", &val_ctx);
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
      ExpectMaxOpVersion(version, 3, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      Expect((node->inputs->size > 1) &&
                 (context->tensors[node->inputs->data[0]].allocation_type ==
                  kTfLiteMmapRo) &&
                 (context->tensors[node->inputs->data[1]].allocation_type ==
                  kTfLiteMmapRo),
             NNAPIValidationFailureType::kInputTensorShouldHaveConstantShape,
             "Dynamically-sized tensors not supported.", &val_ctx);
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

        if (android_sdk_version >= kMinSdkVersionForNNAPI13) {
          Expect(weight_type == kTfLiteFloat32 || weight_type == kTfLiteUInt8 ||
                     weight_type == kTfLiteInt8,
                 NNAPIValidationFailureType::kUnsupportedInputType,
                 "Weight has to be Float32 or UINT8 or INT8", &val_ctx);
        } else {
          Expect(weight_type == kTfLiteFloat32 || weight_type == kTfLiteUInt8,
                 NNAPIValidationFailureType::kUnsupportedInputType,
                 "Weight has to be Float32 or UINT8", &val_ctx);
        }
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
      ExpectMaxOpVersion(version, 3, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8, kTfLiteInt32);
      const TfLiteTensor& operand0 = context->tensors[node->inputs->data[0]];
      if (operand0.dims->size == 0) {
        Expect(operand0.allocation_type == kTfLiteMmapRo,
               NNAPIValidationFailureType::kUnsupportedInputType,
               "Scalar operand should be constant", &val_ctx);
      }
      const TfLiteTensor& operand1 = context->tensors[node->inputs->data[1]];
      if (operand1.dims->size == 0) {
        Expect(operand1.allocation_type == kTfLiteMmapRo,
               NNAPIValidationFailureType::kUnsupportedInputType,
               "Scalar operand should be constant", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinCast: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const TfLiteType input_type =
          context->tensors[node->inputs->data[0]].type;
      const TfLiteType output_type =
          context->tensors[node->outputs->data[0]].type;
      if (android_sdk_version >= kMinSdkVersionForNNAPI13) {
        EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteInt32,
                             kTfLiteUInt8, kTfLiteInt8);

        ExpectTypeIn(
            output_type,
            {kTfLiteFloat32, kTfLiteInt32, kTfLiteUInt8, kTfLiteInt8},
            NNAPIValidationFailureType::kUnsupportedOutputType,
            "Output type should be one of kTfLiteFloat32, kTfLiteInt32, "
            "kTfLiteUInt8, kTfLiteInt8.",
            &val_ctx);
      } else {
        EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteInt32,
                             kTfLiteUInt8);

        ExpectTypeIn(
            output_type, {kTfLiteFloat32, kTfLiteInt32, kTfLiteUInt8},
            NNAPIValidationFailureType::kUnsupportedOutputType,
            "Output type should be one of kTfLiteFloat32, kTfLiteInt32, "
            "kTfLiteUInt8.",
            &val_ctx);
      }
    } break;
    case kTfLiteBuiltinLeakyRelu:
    case kTfLiteBuiltinPrelu: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8);
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
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8, kTfLiteBool, kTfLiteInt32);
    } break;
    case kTfLiteBuiltinNeg: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
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
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
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
             "Condition and inputs tensors should have the same shape",
             &val_ctx);
    } break;
    case kTfLiteBuiltinGather: {
      ExpectOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      const auto& positions = context->tensors[node->inputs->data[1]];

      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteFloat16,
                           kTfLiteInt32, kTfLiteUInt8, kTfLiteInt8);

      Expect(positions.type == kTfLiteInt32,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "Positions type should be one of kTfLiteInt32", &val_ctx);
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
      ExpectOpVersion(version, 3, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      // Tensor indices: split_dim: 0, value: 1
      const TfLiteTensor& input = context->tensors[node->inputs->data[1]];
      if (android_sdk_version >= kMinSdkVersionForNNAPI13) {
        EXPECT_INPUT_TYPE_IN(input.type, kTfLiteFloat32, kTfLiteUInt8,
                             kTfLiteInt8, kTfLiteInt32);
      } else {
        EXPECT_INPUT_TYPE_IN(input.type, kTfLiteFloat32, kTfLiteUInt8,
                             kTfLiteInt32);
      }
      const TfLiteTensor& axis = context->tensors[node->inputs->data[0]];
      Expect(axis.type == kTfLiteInt32 && axis.allocation_type == kTfLiteMmapRo,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "NNAPI only supports constant int32 axis tensor.", &val_ctx);
    } break;
    case kTfLiteBuiltinSplitV: {
      ExpectOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI13,
                                 &val_ctx);
      // Tensor indices: value: 0, size_splits: 1, axis: 2
      const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
      const TfLiteTensor& size_splits = context->tensors[node->inputs->data[1]];
      const TfLiteTensor& axis = context->tensors[node->inputs->data[2]];
      EXPECT_INPUT_TYPE_IN(input.type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8, kTfLiteInt32);
      bool size_splits_is_int32_const_vector =
          size_splits.type == kTfLiteInt32 && size_splits.dims->size == 1 &&
          size_splits.allocation_type == kTfLiteMmapRo;
      bool axis_is_int32_const =
          axis.type == kTfLiteInt32 && axis.allocation_type == kTfLiteMmapRo;
      Expect(size_splits_is_int32_const_vector,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "NNAPI only supports constant int32 size_splits vector.",
             &val_ctx);
      Expect(axis_is_int32_const,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "NNAPI only supports constant int32 axis tensor.", &val_ctx);
      if (size_splits_is_int32_const_vector && axis_is_int32_const) {
        Expect(std::all_of(size_splits.data.i32,
                           size_splits.data.i32 + size_splits.dims->data[0],
                           [](auto size) { return size != 0; }),
               NNAPIValidationFailureType::kUnsupportedInputType,
               "NNAPI only supports non-zero split sizes.", &val_ctx);
        Expect(ComputeSplitVUnknownSplitSize(context, node) != 0,
               NNAPIValidationFailureType::kUnsupportedInputType,
               "NNAPI only supports non-zero split sizes.", &val_ctx);
      }
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
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const auto value_type = context->tensors[node->inputs->data[0]].type;
      Expect(value_type == kTfLiteFloat32 || IsQuantized(value_type),
             NNAPIValidationFailureType::kUnsupportedInputType,
             "Value should be quantized or Float32.", &val_ctx);
      if (IsQuantized(value_type)) {
        const auto quantization_params =
            context->tensors[node->inputs->data[0]].params;
        Expect(quantization_params.scale > 0.f,
               NNAPIValidationFailureType::kUnsupportedQuantizationParameters,
               "Quantization scale should be > 0.", &val_ctx);
      }
      const auto output_type = context->tensors[node->outputs->data[0]].type;
      if (android_sdk_version < kMinSdkVersionForNNAPI13) {
        Expect(output_type == kTfLiteUInt8,
               NNAPIValidationFailureType::kUnsupportedOutputType,
               "Output should be kTfLiteUInt8.", &val_ctx);
      } else {
        ExpectTypeIn(output_type, {kTfLiteUInt8, kTfLiteInt8},
                     NNAPIValidationFailureType::kUnsupportedOutputType,
                     "Output should be kTfLiteUInt8.", &val_ctx);
      }
      const auto quantization_params =
          context->tensors[node->outputs->data[0]].params;
      Expect(quantization_params.scale > 0.f,
             NNAPIValidationFailureType::kUnsupportedQuantizationParameters,
             "Quantization scale should be > 0.", &val_ctx);
    } break;
    case kTfLiteBuiltinReduceAny: {
      ExpectOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      Expect(context->tensors[node->outputs->data[0]].dims->size != 0,
             NNAPIValidationFailureType::kUnsupportedOutputType,
             "NNAPI does not support generating a scalar as output.", &val_ctx);
    } break;
    case kTfLiteBuiltinReduceMin:
    case kTfLiteBuiltinReduceMax: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI12,
                                 &val_ctx);
      const auto input_tensor = context->tensors[node->inputs->data[0]];
      const auto input_type = input_tensor.type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8);
      Expect(input_tensor.dims->size != 0,
             NNAPIValidationFailureType::kUnsupportedOutputType,
             "NNAPI does not support generating a scalar as output.", &val_ctx);
    } break;
    case kTfLiteBuiltinDepthToSpace: {
      const TfLiteType input_type =
          context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8);
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
    case kTfLiteBuiltinElu: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI13,
                                 &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      Expect(input_type == kTfLiteFloat32,
             NNAPIValidationFailureType::kUnsupportedInputType,
             "NNAPI only supports floating point input.", &val_ctx);
    } break;
    case kTfLiteBuiltinFill: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI13,
                                 &val_ctx);
      const auto& dims_tensor = context->tensors[node->inputs->data[0]];
      Expect(IsConstantTensor(&dims_tensor),
             NNAPIValidationFailureType::kUnsupportedInputType,
             "NNAPI doesn't support dynamic dimensions tensor.", &val_ctx);
      EXPECT_INPUT_TYPE_IN(dims_tensor.type, kTfLiteInt32, kTfLiteInt64);
      if (IsConstantTensor(&dims_tensor)) {
        Expect(dims_tensor.dims->data[0] != 0,
               NNAPIValidationFailureType::kUnsupportedOperandValue,
               "NNAPI doesn't support generating scalars from FILL", &val_ctx);
        if (dims_tensor.type == kTfLiteInt64) {
          bool fit_in_int32 =
              std::all_of(dims_tensor.data.i64,
                          dims_tensor.data.i64 + dims_tensor.dims->data[0],
                          [](int64_t dim) {
                            return std::numeric_limits<int32_t>::min() <= dim &&
                                   dim <= std::numeric_limits<int32_t>::max();
                          });
          Expect(fit_in_int32,
                 NNAPIValidationFailureType::kUnsupportedOperandValue,
                 "NNAPI only supports int32 dimensions tensor. If the "
                 "dimensions type is int64 and they are constant we can "
                 "convert them to int32 if the value isn't too large.",
                 &val_ctx);
        }
      }
      const auto& value_tensor = context->tensors[node->inputs->data[1]];
      EXPECT_INPUT_TYPE_IN(value_tensor.type, kTfLiteFloat32, kTfLiteInt32,
                           kTfLiteInt64);
      if (value_tensor.type == kTfLiteInt64 &&
          IsConstantTensor(&value_tensor)) {
        Expect(
            *value_tensor.data.i64 <= std::numeric_limits<int32_t>::max() &&
                *value_tensor.data.i64 >= std::numeric_limits<int32_t>::min(),
            NNAPIValidationFailureType::kUnsupportedInputType,
            "NNAPI only supports int32 input. If the input type is int64 and "
            "constant we can convert it to int32 if the value isn't too "
            "large.",
            &val_ctx);
      }
    } break;
    case kTfLiteBuiltinPack: {
      ExpectOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI13,
                                 &val_ctx);
      const auto input_type = context->tensors[node->inputs->data[0]].type;
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteInt32, kTfLiteFloat32,
                           kTfLiteInt8);
      auto builtin = reinterpret_cast<TfLitePackParams*>(node->builtin_data);
      Expect(builtin->axis != -1 &&
                 builtin->axis !=
                     context->tensors[node->inputs->data[0]].dims->size,
             NNAPIValidationFailureType::kUnsupportedOperandValue,
             "NNAPI does not support axis being the last dimension", &val_ctx);
    } break;
    case kTfLiteBuiltinSquaredDifference: {
      ExpectOpVersion(version, 2, &val_ctx);
      ExpectMinAndroidSdkVersion(android_sdk_version, kMinSdkVersionForNNAPI11,
                                 &val_ctx);
      const auto input0_type = context->tensors[node->inputs->data[0]].type;
      if (android_sdk_version >= kMinSdkVersionForNNAPI13) {
        EXPECT_INPUT_TYPE_IN(input0_type, kTfLiteFloat32, kTfLiteUInt8,
                             kTfLiteInt8, kTfLiteInt32);
      } else if (android_sdk_version >= kMinSdkVersionForNNAPI12) {
        EXPECT_INPUT_TYPE_IN(input0_type, kTfLiteFloat32, kTfLiteUInt8);
      } else {
        EXPECT_INPUT_TYPE_IN(input0_type, kTfLiteFloat32);
      }
      const int input0_rank =
          context->tensors[node->inputs->data[0]].dims->size;
      const int input1_rank =
          context->tensors[node->inputs->data[1]].dims->size;
      Expect(input0_rank <= 4 && input1_rank <= 4,
             NNAPIValidationFailureType::kUnsupportedOperandRank,
             "NNAPI does not support input rank greater than 4", &val_ctx);
    } break;
    default:
      // All other operators are not mapped.
      AddValidationFailure(NNAPIValidationFailureType::kUnsupportedOperator,
                           "Unsupported operation type.", &val_ctx);
  }
  return val_ctx.is_valid;
}  // NOLINT(readability/fn_size)

TfLiteStatus NNAPIDelegateKernel::Map(
    TfLiteContext* context, int builtin_code, int version,
    int android_sdk_version, const NNAPIOpMappingArgs& mapping_args,
    ANeuralNetworksOperationType* nn_op_type) {
  auto add_zero_bias = [mapping_args](int input_id, int filter_id,
                                      int num_elements) -> void {
    // NNAPI requires a bias tensor, so we allocate a new tensor to fill
    // it with zeroes. It is deleted with other tensors in the context
    // during subgraph destructor call.
    int bias_index = -1;
    mapping_args.context->AddTensors(mapping_args.context, 1, &bias_index);
    TfLiteTensor* bias_tensor = &mapping_args.context->tensors[bias_index];
    const auto input_type = mapping_args.context->tensors[input_id].type;
    if (input_type == kTfLiteFloat32) {
      bias_tensor->type = kTfLiteFloat32;
    } else {
      bias_tensor->type = kTfLiteInt32;
    }
    // Create an array with a required bias shape and resize the bias
    // tensor.
    TfLiteIntArray* bias_shape = TfLiteIntArrayCreate(1);
    bias_shape->data[0] = num_elements;
    bias_tensor->allocation_type = kTfLiteDynamic;
    mapping_args.context->ResizeTensor(mapping_args.context, bias_tensor,
                                       bias_shape);
    // Set tensor's values to zeroes and add it using AddVector*, so
    // that the values are copied to NNAPI. We don't use the AddTensor
    // function because it doesn't copy values and the tensor we just
    // created is not in the node->inputs.
    if (input_type == kTfLiteFloat32) {
      memset(bias_tensor->data.f, 0, num_elements * sizeof(float));
      mapping_args.builder->AddVectorFloat32Operand(bias_tensor->data.f,
                                                    num_elements);
    } else {
      memset(bias_tensor->data.i32, 0, num_elements * sizeof(int));
      const TfLiteTensor& input_tensor =
          mapping_args.context->tensors[input_id];
      const TfLiteTensor& filter_tensor =
          mapping_args.context->tensors[filter_id];
      // NNAPI requires bias scale to be a product of an input scale and
      // a filter scale.
      bias_tensor->params.scale =
          input_tensor.params.scale * filter_tensor.params.scale;
      mapping_args.builder->AddVectorInt32Operand(
          bias_tensor->data.i32, num_elements, bias_tensor->params.scale,
          /*zero_point=*/0);
    }
  };
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
      mapping_args.builder->AddScalarInt32Operand(builtin->padding);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_width);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_height);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      // NNAPI supports dilated Conv2D since NNAPI 1.2.
      if (builtin->dilation_width_factor != 1 ||
          builtin->dilation_height_factor != 1) {
        mapping_args.builder->AddScalarBoolOperand(false);  // Use NHWC format
        mapping_args.builder->AddScalarInt32Operand(
            builtin->dilation_width_factor);
        mapping_args.builder->AddScalarInt32Operand(
            builtin->dilation_height_factor);
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
      const bool is_bias_present =
          mapping_args.node->inputs->size == 3 &&
          mapping_args.node->inputs->data[2] != kTfLiteOptionalTensor;
      if (!is_bias_present) {
        const int input_tensor_id =
            mapping_args.node->inputs->data[/*kInputTensor*/ 0];
        const int filter_tensor_id =
            mapping_args.node->inputs->data[/*kWeightsTensor*/ 1];
        const int num_units =
            mapping_args.context->tensors[filter_tensor_id].dims->data[0];
        add_zero_bias(input_tensor_id, filter_tensor_id, num_units);
      }
      auto builtin = reinterpret_cast<TfLiteFullyConnectedParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = ANEURALNETWORKS_FULLY_CONNECTED;
    } break;
    case kTfLiteBuiltinHardSwish: {
      *nn_op_type = ANEURALNETWORKS_HARD_SWISH;
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
      if (mapping_args.node->inputs->size == 1) {
        // if no new_shape tensor, construct the new shape from params.
        auto* params = reinterpret_cast<TfLiteReshapeParams*>(
            mapping_args.node->builtin_data);
        int num_dimensions = params->num_dimensions;
        std::vector<int32_t> output_shape(num_dimensions);
        for (int i = 0; i < num_dimensions; ++i) {
          output_shape[i] = params->shape[i];
        }
        mapping_args.builder->AddVectorInt32Operand(
            output_shape.data(), static_cast<uint32_t>(num_dimensions));
      }
      *nn_op_type = ANEURALNETWORKS_RESHAPE;
    } break;
    case kTfLiteBuiltinResizeBilinear: {
      const int output_id = mapping_args.node->outputs->data[0];
      auto& output = mapping_args.context->tensors[output_id];
      const int output_height = output.dims->data[1];
      const int output_width = output.dims->data[2];
      mapping_args.builder->AddScalarInt32Operand(output_width);
      mapping_args.builder->AddScalarInt32Operand(output_height);
      auto builtin = reinterpret_cast<TfLiteResizeBilinearParams*>(
          mapping_args.node->builtin_data);
      if (builtin->align_corners == true ||
          builtin->half_pixel_centers == true) {
        mapping_args.builder->AddScalarBoolOperand(false);  // Use NHWC format
        mapping_args.builder->AddScalarBoolOperand(builtin->align_corners);
        mapping_args.builder->AddScalarBoolOperand(builtin->half_pixel_centers);
      }
      *nn_op_type = ANEURALNETWORKS_RESIZE_BILINEAR;
    } break;
    case kTfLiteBuiltinResizeNearestNeighbor: {
      const TfLiteTensor& new_shape =
          mapping_args.context->tensors[mapping_args.node->inputs->data[1]];
      // NNAPI uses scalar inputs for height and width.
      mapping_args.builder->AddScalarInt32Operand(new_shape.data.i32[1]);
      mapping_args.builder->AddScalarInt32Operand(new_shape.data.i32[0]);
      mapping_args.builder->AddScalarBoolOperand(false);  // Use NHWC format
      auto builtin = reinterpret_cast<TfLiteResizeNearestNeighborParams*>(
          mapping_args.node->builtin_data);
      if (builtin->align_corners == true ||
          builtin->half_pixel_centers == true) {
        mapping_args.builder->AddScalarBoolOperand(builtin->align_corners);
        mapping_args.builder->AddScalarBoolOperand(builtin->half_pixel_centers);
      }
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
      int input_tensor_flags = 0;
      const int input_tensor_id =
          mapping_args.node->inputs->data[/*kDataInputTensor*/ 2];
      const int weight_tensor_id =
          mapping_args.node->inputs->data[/*kWeightsTensor*/ 1];

      // Transpose convolution doesn't have hybrid variation.
      const bool hybrid_op = false;

      if (android_sdk_version >= kMinSdkVersionForNNAPI13) {
        mapping_args.builder->AddTensorInput(
            input_tensor_id, hybrid_op,
            input_tensor_flags | NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED);

      } else {
        mapping_args.builder->AddTensorInput(
            input_tensor_id, hybrid_op,
            input_tensor_flags | NN_TENSOR_FLAG_INT8_CONVERSION);
      }
      // Transpose convlution uses per-channel quantization with int8 inputs
      // even if the number of channels in quantization parameters is equal to 1
      // (as opposed to conv2d, which uses per-tensor quantization in this
      // case).
      mapping_args.builder->AddTensorInput(
          weight_tensor_id, hybrid_op,
          input_tensor_flags | NN_TENSOR_FLAG_FORCE_PER_CHANNEL);

      const bool is_bias_present =
          mapping_args.node->inputs->size == 4 &&
          mapping_args.node->inputs->data[/*kBiasTensor*/ 3] !=
              kTfLiteOptionalTensor;

      if (is_bias_present) {
        mapping_args.builder->AddTensorInput(
            mapping_args.node->inputs->data[/*kBiasTensor*/ 3], hybrid_op);
      } else {
        const TfLiteTensor& output_shape =
            mapping_args.context->tensors[mapping_args.node->inputs
                                              ->data[/*kOutputShapeTensor*/ 0]];
        const int output_depth = output_shape.data.i32[3];
        add_zero_bias(input_tensor_id, weight_tensor_id, output_depth);
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
            mapping_args.node->outputs->data[0 /*kOutputActivation*/],
            mapping_args.node->inputs->data[1 /*kInputPrevActivation*/]));

        mapping_args.feedback_loops->push_back(std::make_tuple(
            mapping_args.node->outputs->data[1 /*kOutputState*/],
            mapping_args.node->inputs->data[4 /*kInputPrevState*/]));

        // OUTPUTS
        // Setting only the first two since the remaining ones are
        // ignored by NNAPI
        mapping_args.builder->AddTensorOutput(
            mapping_args.node->outputs->data[1 /* kOutputState */], 0);

        mapping_args.builder->AddTensorOutput(
            mapping_args.node->outputs->data[0 /* kOutputActivation */], 0);

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
    case kTfLiteBuiltinLeakyRelu: {
      const auto input_type =
          mapping_args.context->tensors[mapping_args.node->inputs->data[0]]
              .type;
      auto builtin = reinterpret_cast<TfLiteLeakyReluParams*>(
          mapping_args.node->builtin_data);

      TfLiteTensor alpha_tensor;
      alpha_tensor.type = input_type;
      alpha_tensor.allocation_type = kTfLiteDynamic;
      alpha_tensor.dims = TfLiteIntArrayCreate(1);
      alpha_tensor.dims->data[0] = 1;
      alpha_tensor.params.zero_point = 0;

      int new_tensor_index = -1;
      if (input_type == kTfLiteFloat32) {
        alpha_tensor.params.scale = 0;
        std::vector<float> alpha_value = {builtin->alpha};
        mapping_args.builder->AddNewInputConstantTensor(
            ANEURALNETWORKS_TENSOR_FLOAT32, kTfLiteFloat32, alpha_tensor.dims,
            alpha_value, alpha_tensor.params, &new_tensor_index);
      } else if (input_type == kTfLiteInt8 &&
                 android_sdk_version >= kMinSdkVersionForNNAPI13) {
        alpha_tensor.params.scale = builtin->alpha;
        std::vector<int8_t> alpha_value = {1};
        mapping_args.builder->AddNewInputConstantTensor(
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, kTfLiteInt8,
            alpha_tensor.dims, alpha_value, alpha_tensor.params,
            &new_tensor_index);
      } else {
        alpha_tensor.params.scale = builtin->alpha;
        std::vector<uint8_t> alpha_value = {1};
        mapping_args.builder->AddNewInputConstantTensor(
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, kTfLiteUInt8,
            alpha_tensor.dims, alpha_value, alpha_tensor.params,
            &new_tensor_index);
      }

      *nn_op_type = ANEURALNETWORKS_PRELU;
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
      mapping_args.builder->AddScalarInt32Operand(builtin->axis);
      mapping_args.builder->AddTensorInput(mapping_args.node->inputs->data[1],
                                           /* hybrid_op */ false,
                                           /* tensor_flags */ 0);
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
      auto input_index = mapping_args.node->inputs->data[0];
      // NNAPI doesn't support requantization cases but only quantizations
      // from float. Dequantizing our input adding a Dequantize node before
      // this one.
      if (IsQuantized(mapping_args.context->tensors[input_index].type)) {
        mapping_args.builder->AddDequantize(0, input_index, kTfLiteFloat32,
                                            mapping_args.node_index);
      }

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
    case kTfLiteBuiltinDepthToSpace: {
      auto builtin = reinterpret_cast<TfLiteDepthToSpaceParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->block_size);
      *nn_op_type = ANEURALNETWORKS_DEPTH_TO_SPACE;
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
    case kTfLiteBuiltinElu: {
      mapping_args.builder->AddScalarFloat32Operand(1.0);
      *nn_op_type = ANEURALNETWORKS_ELU;
    } break;
    case kTfLiteBuiltinFill: {
      *nn_op_type = ANEURALNETWORKS_FILL;
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

  // Initialize densify map and dequantize map.
  densify_output_to_node_mapping_ = std::vector<int>(context->tensors_size, -1);
  non_const_dequantize_output_to_node_mapping_ =
      std::vector<int>(context->tensors_size, -1);
  const auto delegate_options =
      StatefulNnApiDelegate::GetOptions(params->delegate);
  if (nnapi_->android_sdk_version >= kMinSdkVersionForNNAPI12 &&
      ShouldUseTargetDevices(delegate_options, nnapi_)) {
    TF_LITE_ENSURE_STATUS(GetTargetDevices(context, params->delegate, nnapi_,
                                           nnapi_errno, &nnapi_devices_));

    if (nnapi_devices_.empty()) {
      context->ReportError(
          context, "NNAPI delegate requested but no accelerators available.");
      return kTfLiteError;
    }
  }

  // Mark the handle backed tensors.
  tensor_memory_map_ =
      &StatefulNnApiDelegate::GetTensorMemoryMap(params->delegate);

  if (!nn_model_) {
    ANeuralNetworksModel* model = nullptr;
    RETURN_TFLITE_ERROR_IF_NN_ERROR(context,
                                    nnapi_->ANeuralNetworksModel_create(&model),
                                    "creating NNAPI model", nnapi_errno);
    nn_model_.reset(model);

    TF_LITE_ENSURE_STATUS(BuildGraph(context, delegate_options,
                                     params->input_tensors,
                                     params->output_tensors, nnapi_errno));
  }

  auto* cache = StatefulNnApiDelegate::GetCache(params->delegate);
  if (cache) {
    // Compilation caching is enabled, construct the uint8 token.
    uint64_t token_parts[4];
    // model_token is incorporated into parition_key by TFLite Serialization.
    // NNAPI uses 256-bit key, but we can just tile the unique 64-bit
    // fingerprint from TFLite.
    auto partition_entry = cache->GetEntryForKernel(kNnapiId, context, params);
    token_parts[0] = partition_entry.GetFingerprint();
    token_parts[1] = partition_entry.GetFingerprint();
    token_parts[2] = partition_entry.GetFingerprint();
    token_parts[3] = partition_entry.GetFingerprint();
    // TODO(b/172238515): get token size from header instead of hardcoding.
    // Allocate one extra 'null' byte to avoid bugs with backends that might
    // be doing strlen() on the token ptr.
    std::vector<uint8_t> nnapi_cache_token(33, 0);
    // Copy the token bits.
    uint8_t* p = reinterpret_cast<uint8_t*>(token_parts);
    for (int i = 0; i < 4 * sizeof(uint64_t); i++) {
      nnapi_cache_token[i] = p[i];
    }

    nn_compilation_cache_token_ = nnapi_cache_token;
  }

  initialised_ = true;

  return kTfLiteOk;
}

TfLiteStatus NNAPIDelegateKernel::Prepare(TfLiteContext* context,
                                          TfLiteNode* node, int* nnapi_errno) {
  if (!initialised_) {
    return kTfLiteError;
  }

  const auto delegate_options =
      StatefulNnApiDelegate::GetOptions(node->delegate);
  if (nn_compilation_) {
    return kTfLiteOk;
  }

  ANeuralNetworksCompilation* compilation = nullptr;
  if (!nnapi_devices_.empty()) {
    // Compile for the selected accelerator.
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context,
        nnapi_->ANeuralNetworksCompilation_createForDevices(
            nn_model_.get(), nnapi_devices_.data(), nnapi_devices_.size(),
            &compilation),
        "creating NNAPI model for given devices", nnapi_errno);
  } else {
    // Trying to call ANeuralNetworksCompilation_create when the delegate is
    // constructed from a support library would result in a crash.
    if (nnapi_->ANeuralNetworksCompilation_create != nullptr) {
      RETURN_TFLITE_ERROR_IF_NN_ERROR(context,
                                      nnapi_->ANeuralNetworksCompilation_create(
                                          nn_model_.get(), &compilation),
                                      "creating NNAPI compilation",
                                      nnapi_errno);
    } else {
      TF_LITE_KERNEL_LOG(
          context,
          "Attempted to call ANeuralNetworksCompilation_create from NNAPI "
          "delegate that is constructed from a support library");
      return kTfLiteError;
    }
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
    RETURN_TFLITE_ERROR_IF_NN_ERROR(context, preference_result,
                                    "setting compilation preferences",
                                    nnapi_errno);
  }

  if (!nn_compilation_cache_token_.empty()) {
    const char* cache_dir = delegate_options.cache_dir;
    const int set_caching_result =
        nnapi_->ANeuralNetworksCompilation_setCaching(
            compilation, cache_dir, nn_compilation_cache_token_.data());
    if (set_caching_result != ANEURALNETWORKS_NO_ERROR) {
      nnapi_->ANeuralNetworksCompilation_free(compilation);
      compilation = nullptr;
    }
    RETURN_TFLITE_ERROR_IF_NN_ERROR(context, set_caching_result,
                                    "configuring NNAPI caching", nnapi_errno);
  }
  // Set compilation timeout if applicable.
  if (nnapi_->android_sdk_version >= kMinSdkVersionForNNAPI13) {
    if (delegate_options.max_compilation_timeout_duration_ns > 0) {
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context,
          nnapi_->ANeuralNetworksCompilation_setTimeout(
              compilation,
              delegate_options.max_compilation_timeout_duration_ns),
          "setting compilation timeout", nnapi_errno);
    }
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context,
        nnapi_->ANeuralNetworksCompilation_setPriority(
            compilation, delegate_options.execution_priority),
        "setting compilation priority", nnapi_errno);
  }
  const int finish_result =
      nnapi_->ANeuralNetworksCompilation_finish(compilation);
  if (finish_result != ANEURALNETWORKS_NO_ERROR) {
    nnapi_->ANeuralNetworksCompilation_free(compilation);
    compilation = nullptr;
  }
  RETURN_TFLITE_ERROR_IF_NN_ERROR(context, finish_result,
                                  "completing NNAPI compilation", nnapi_errno);
  nn_compilation_.reset(compilation);

  bool should_use_burst_mode = delegate_options.use_burst_computation;
  // Override should_use_burst_mode to true if the selected NNAPI devices are of
  // NNAPI feature level 5 or higher.
  if (!nnapi_devices_.empty() &&
      target_feature_level_ >= kNNAPIRuntimeFeatureLevel5) {
    should_use_burst_mode = true;
  }
  // Create burst object to be reused across a sequence of executions
  if (should_use_burst_mode &&
      nnapi_->android_sdk_version >= kMinSdkVersionForNNAPI12 &&
      nnapi_->ANeuralNetworksBurst_create) {
    ANeuralNetworksBurst* burst = nullptr;
    const int create_burst_result =
        nnapi_->ANeuralNetworksBurst_create(nn_compilation_.get(), &burst);
    if (create_burst_result != ANEURALNETWORKS_NO_ERROR) {
      nnapi_->ANeuralNetworksBurst_free(burst);
      burst = nullptr;
    }
    RETURN_TFLITE_ERROR_IF_NN_ERROR(context, create_burst_result,
                                    "creating NNAPI burst", nnapi_errno);
    nn_burst_.reset(burst);
  }

  return kTfLiteOk;
}

TfLiteStatus NNAPIDelegateKernel::GetOperationsSupportedByTargetNnApiDevices(
    TfLiteContext* context, std::vector<int>* supported_nodes,
    int* nnapi_errno) {
  if (!nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices) {
    return kTfLiteError;
  }

  const auto nnapi_model_size = nnapi_to_tflite_op_mapping_.size();

  // Determine the list of operations the device actually supports
  std::unique_ptr<bool[]> nnapi_ops_support_flags(new bool[nnapi_model_size]);

  RETURN_TFLITE_ERROR_IF_NN_ERROR(
      context,
      nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices(
          nn_model_.get(), nnapi_devices_.data(), nnapi_devices_.size(),
          nnapi_ops_support_flags.get()),
      "Checking supported operations for devices", nnapi_errno);

  // A TfLite op is supported only if all the associated NNAPI ones are.
  auto tflite_ops_support_status = std::map<int, bool>();
  std::for_each(nodes_.begin(), nodes_.end(),
                [&tflite_ops_support_status](int tflite_node_index) {
                  tflite_ops_support_status[tflite_node_index] = true;
                });
  for (int nnapi_op_index = 0; nnapi_op_index < nnapi_model_size;
       nnapi_op_index++) {
    const auto tflite_op_index = nnapi_to_tflite_op_mapping_[nnapi_op_index];
    tflite_ops_support_status[tflite_op_index] &=
        nnapi_ops_support_flags[nnapi_op_index];
    if (!tflite_ops_support_status[tflite_op_index]) {
      if (std::count(non_const_dequantize_output_to_node_mapping_.begin(),
                     non_const_dequantize_output_to_node_mapping_.end(), -1) <
              non_const_dequantize_output_to_node_mapping_.size() ||
          std::count(densify_output_to_node_mapping_.begin(),
                     densify_output_to_node_mapping_.end(),
                     -1) < densify_output_to_node_mapping_.size()) {
        // Only allow full model delegation for sparse model.
        return kTfLiteOk;
      }
    }
  }

  supported_nodes->clear();
  std::for_each(nodes_.begin(), nodes_.end(),
                [&supported_nodes, &tflite_ops_support_status](int node_index) {
                  if (tflite_ops_support_status[node_index]) {
                    supported_nodes->push_back(node_index);
                  }
                });

  return kTfLiteOk;
}

TfLiteStatus NNAPIDelegateKernel::Invoke(TfLiteContext* context,
                                         TfLiteNode* node, int* nnapi_errno) {
  const bool allow_padding =
      nnapi_->nnapi_runtime_feature_level > kMinSdkVersionForNNAPI13 &&
      nnapi_->ANeuralNetworksExecution_enableInputAndOutputPadding != nullptr;
  const auto delegate_options =
      StatefulNnApiDelegate::GetOptions(node->delegate);

  // Check for conditions where we need to re-create NN Execution object and
  // re-configure the settings and inputs / outputs.
  bool should_reset_execution = false;
  if (nnapi_->nnapi_runtime_feature_level <= kMinSdkVersionForNNAPI13 ||
      delegate_options.allow_dynamic_dimensions) {
    // Must reset execution before Android API 31, or using dynamic dimensions.
    should_reset_execution = true;
  } else {
    // For Android API 31+, check for BufferHandle changes and reset the
    // execution if any.
    std::vector<int> curr_in_tensor_handle_map(context->tensors_size);
    for (int i = 0; i < curr_in_tensor_handle_map.size(); i++) {
      curr_in_tensor_handle_map[i] = context->tensors[i].buffer_handle;
    }
    if (!(tensor_handle_map_ == curr_in_tensor_handle_map)) {
      should_reset_execution = true;
      tensor_handle_map_ = curr_in_tensor_handle_map;
    }
  }
  if (should_reset_execution) {
    ANeuralNetworksExecution* execution = nullptr;
    RETURN_TFLITE_ERROR_IF_NN_ERROR(context,
                                    nnapi_->ANeuralNetworksExecution_create(
                                        nn_compilation_.get(), &execution),
                                    "creating NNAPI execution", nnapi_errno);
    if (nnapi_->nnapi_runtime_feature_level > kMinSdkVersionForNNAPI13) {
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context,
          nnapi_->ANeuralNetworksExecution_setReusable(execution,
                                                       /*reusable=*/true),
          "making execution reusable", nnapi_errno);
    }
    nn_execution_.reset(execution);

    // Allow padding bytes for execution inputs & outputs if applicable.
    if (allow_padding) {
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context,
          nnapi_->ANeuralNetworksExecution_enableInputAndOutputPadding(
              nn_execution_.get(),
              /*enable=*/true),
          "setting allow padding for execution intputs and outputs",
          nnapi_errno);
    }
    // Set compilation timeout if applicable.
    if (nnapi_->android_sdk_version >= kMinSdkVersionForNNAPI13) {
      if (delegate_options.max_execution_timeout_duration_ns > 0) {
        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            context,
            nnapi_->ANeuralNetworksExecution_setTimeout(
                nn_execution_.get(),
                delegate_options.max_execution_timeout_duration_ns),
            "setting execution timeout", nnapi_errno);
      }
      if (delegate_options.max_execution_loop_timeout_duration_ns > 0) {
        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            context,
            nnapi_->ANeuralNetworksExecution_setLoopTimeout(
                nn_execution_.get(),
                delegate_options.max_execution_loop_timeout_duration_ns),
            "setting execution loop timeout", nnapi_errno);
      }
    }
    // Check if the size of input and output memory pool needs to be resized.
    if (delegate_options.allow_dynamic_dimensions) {
      size_t total_input_byte_size = 0;
      // Make the TensorFlow Lite inputs and outputs to ann_indices.
      for (int i : TfLiteIntArrayView(node->inputs)) {
        // Constant tensors are not NNAPI inputs.
        if (i != kTfLiteOptionalTensor &&
            context->tensors[i].allocation_type != kTfLiteMmapRo &&
            // The delegate might not have mapped this input (this can
            // happen if one tensor is split in several ones)
            operand_mapping_.lite_index_to_ann(i) != -1) {
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
                context,
                GetSizeOfType(context, nn_type_conversion, &type_size));
            tensor_size = NumElements(&context->tensors[i]) * type_size;
          }
          total_input_byte_size += tensor_size;
          total_input_byte_size += GetNumPaddingBytes(tensor_size);
        }
      }
      if (total_input_byte_size > nn_input_memory_->get_byte_size()) {
        nn_input_memory_.reset(
            new NNMemory(nnapi_, "input_pool", total_input_byte_size));
      }

      size_t total_output_byte_size = 0;
      for (int i : TfLiteIntArrayView(node->outputs)) {
        if (context->tensors[i].buffer_handle != kTfLiteNullBufferHandle) {
          continue;
        }
        total_output_byte_size += context->tensors[i].bytes;
        total_output_byte_size += GetNumPaddingBytes(context->tensors[i].bytes);
      }
      if (total_output_byte_size > nn_output_memory_->get_byte_size()) {
        nn_output_memory_.reset(
            new NNMemory(nnapi_, "output_pool", total_output_byte_size));
      }
    }
  }
  // Set the input tensor buffers. Note: we access tflite tensors using
  // absolute indices but NN api indices inputs by relative indices.
  int relative_input_index = 0;

  const bool use_int8_asymm_signed =
      target_feature_level_ >= kMinSdkVersionForNNAPI13;

  size_t input_offset = 0;
  for (auto absolute_input_index : TfLiteIntArrayView(node->inputs)) {
    if (absolute_input_index == kTfLiteOptionalTensor) {
      continue;
    }
    ANeuralNetworksOperandType input_nn_operand_type;
    ANeuralNetworksOperandType* input_nn_operand_type_ptr = nullptr;
    TfLiteTensor* tensor = &context->tensors[absolute_input_index];
    TfLiteType ann_type_equivalent =
        operand_mapping_.lite_index_to_ann_type_conversion(
            absolute_input_index);
    if (delegate_options.allow_dynamic_dimensions &&
        HasUnspecifiedDimension(tensor)) {
      input_nn_operand_type =
          ConvertTensorTypeToNNType(tensor, ann_type_equivalent);
      input_nn_operand_type_ptr = &input_nn_operand_type;
    }
    if (tensor->allocation_type != kTfLiteMmapRo) {
      if (tensor->buffer_handle != kTfLiteNullBufferHandle &&
          tensor->buffer_handle < tensor_memory_map_->size()) {
        RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(
            context,
            nnapi_->ANeuralNetworksExecution_setInputFromMemory(
                nn_execution_.get(), relative_input_index,
                input_nn_operand_type_ptr,
                tensor_memory_map_->at(tensor->buffer_handle).memory, 0,
                tensor->bytes),
            "associating NNAPI execution input with a memory object", tensor,
            nnapi_errno);
        relative_input_index++;
        continue;
      }
      int tensor_size = 0;
      int padding_bytes = 0;
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
          if (use_int8_asymm_signed) {
            for (int i = 0; i < num_elements; ++i) {
              reinterpret_cast<int32_t*>(input_ptr)[i] =
                  static_cast<const int32_t>(tensor->data.int8[i]);
            }
          } else {
            for (int i = 0; i < num_elements; ++i) {
              reinterpret_cast<int32_t*>(input_ptr)[i] =
                  static_cast<const int32_t>(tensor->data.int8[i]) + 128;
            }
          }
        } else if (tensor->type == kTfLiteInt64 &&
                   ann_type_equivalent == kTfLiteInt32) {
          // Check that values fit into int32.
          int32_t* input_ptr_i32 = reinterpret_cast<int32_t*>(input_ptr);
          for (int i = 0; i < num_elements; ++i) {
            if (input_ptr_i32[i] < std::numeric_limits<int32_t>::min() ||
                input_ptr_i32[i] > std::numeric_limits<int32_t>::max()) {
              TF_LITE_KERNEL_LOG(context,
                                 "NN API Delegate: int64 value out of bounds "
                                 "for int32 target NNAPI tensor\n");
              return kTfLiteError;
            }
            input_ptr_i32[i] = static_cast<int32_t>(tensor->data.i64[i]);
          }
        } else {
          TF_LITE_KERNEL_LOG(
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
        padding_bytes = GetNumPaddingBytes(tensor_size);
        if (should_reset_execution) {
          RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(
              context,
              nnapi_->ANeuralNetworksExecution_setInputFromMemory(
                  nn_execution_.get(), relative_input_index,
                  input_nn_operand_type_ptr, nn_input_memory_->get_handle(),
                  input_offset, GetNNTensorSize(tensor_size, allow_padding)),
              "associating NNAPI execution input with a memory object", tensor,
              nnapi_errno);
        }
      } else {
        // copy data to pre-allocated shared memory.
        memcpy(nn_input_memory_->get_data_ptr() + input_offset,
               tensor->data.raw, tensor->bytes);
        tensor_size = tensor->bytes;
        padding_bytes = GetNumPaddingBytes(tensor_size);
        if (should_reset_execution) {
          RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(
              context,
              nnapi_->ANeuralNetworksExecution_setInputFromMemory(
                  nn_execution_.get(), relative_input_index,
                  input_nn_operand_type_ptr, nn_input_memory_->get_handle(),
                  input_offset, GetNNTensorSize(tensor_size, allow_padding)),
              "associating NNAPI execution input with a memory object", tensor,
              nnapi_errno);
        }
      }
      input_offset += tensor_size + padding_bytes;
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
    ANeuralNetworksOperandType output_nn_operand_type;
    ANeuralNetworksOperandType* output_nn_operand_type_ptr = nullptr;
    TfLiteTensor* tensor = &context->tensors[output_index];
    if (delegate_options.allow_dynamic_dimensions &&
        HasUnspecifiedDimension(tensor)) {
      TfLiteType ann_type_equivalent =
          operand_mapping_.lite_index_to_ann_type_conversion(output_index);
      output_nn_operand_type =
          ConvertTensorTypeToNNType(tensor, ann_type_equivalent);
      output_nn_operand_type_ptr = &output_nn_operand_type;
    }
    if (tensor->buffer_handle != kTfLiteNullBufferHandle &&
        tensor->buffer_handle < tensor_memory_map_->size() &&
        should_reset_execution) {
      RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(
          context,
          nnapi_->ANeuralNetworksExecution_setOutputFromMemory(
              nn_execution_.get(), relative_output_index,
              output_nn_operand_type_ptr,
              tensor_memory_map_->at(tensor->buffer_handle).memory, 0,
              tensor->bytes),
          "associating NNAPI execution output to a memory object", tensor,
          nnapi_errno);

    } else {
      int padding_bytes = GetNumPaddingBytes(tensor->bytes);
      if (should_reset_execution) {
        RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(
            context,
            nnapi_->ANeuralNetworksExecution_setOutputFromMemory(
                nn_execution_.get(), relative_output_index,
                output_nn_operand_type_ptr, nn_output_memory_->get_handle(),
                output_offset, GetNNTensorSize(tensor->bytes, allow_padding)),
            "associating NNAPI execution output to a memory object", tensor,
            nnapi_errno);
      }
      output_offset += tensor->bytes + padding_bytes;
    }
    relative_output_index++;
  }

  // Set memory for NNAPI state_outputs.
  for (size_t i = 0; i < model_state_tfl_inputs_.size(); i++) {
    int state_tensor_idx = model_state_tfl_inputs_[i];
    TfLiteTensor* tensor = &context->tensors[state_tensor_idx];
    int padding_bytes = GetNumPaddingBytes(tensor->bytes);
    if (should_reset_execution) {
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context,
          nnapi_->ANeuralNetworksExecution_setOutputFromMemory(
              nn_execution_.get(), relative_output_index, nullptr,
              nn_output_memory_->get_handle(), output_offset,
              GetNNTensorSize(tensor->bytes, allow_padding)),
          "associating NNAPI execution state output to a memory object",
          nnapi_errno);
    }
    output_offset += tensor->bytes + padding_bytes;
    relative_output_index++;
  }

  // Invoke ANN in blocking fashion.
  if (nnapi_->android_sdk_version < kMinSdkVersionForNNAPI12) {
    ANeuralNetworksEvent* event = nullptr;
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context,
        nnapi_->ANeuralNetworksExecution_startCompute(nn_execution_.get(),
                                                      &event),
        "starting async computation", nnapi_errno);
    const int wait_result = nnapi_->ANeuralNetworksEvent_wait(event);
    nnapi_->ANeuralNetworksEvent_free(event);
    RETURN_TFLITE_ERROR_IF_NN_ERROR(context, wait_result,
                                    "waiting for async computation completion",
                                    nnapi_errno);
  } else {
    // Use Burst mode by default for NNAPI 1.2+.
    if (nn_burst_) {
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context,
          nnapi_->ANeuralNetworksExecution_burstCompute(nn_execution_.get(),
                                                        nn_burst_.get()),
          "running burst computation", nnapi_errno);
    } else {
      // Use synchronous execution for NNAPI 1.2+ as a fallback.
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context,
          nnapi_->ANeuralNetworksExecution_compute(nn_execution_.get()),
          "running computation", nnapi_errno);
    }
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
    output_offset += GetNumPaddingBytes(tensor->bytes);
  }
  // The state_out of previous invocation need to be copied to state_in of
  // current invocation.
  for (size_t i = 0; i < model_state_tfl_inputs_.size(); i++) {
    int state_tensor_idx = model_state_tfl_inputs_[i];
    TfLiteTensor* tensor = &context->tensors[state_tensor_idx];
    memcpy(tensor->data.raw, nn_output_memory_->get_data_ptr() + output_offset,
           tensor->bytes);
    output_offset += tensor->bytes;
    output_offset += GetNumPaddingBytes(tensor->bytes);
  }

  // copy output of all output tensors in feedback_loops_ into the
  // associated input
  for (auto feedback_loop : feedback_loops_) {
    int output_tensor_idx;
    int input_tensor_idx;
    std::tie(output_tensor_idx, input_tensor_idx) = feedback_loop;
    TfLiteTensor& src = context->tensors[output_tensor_idx];
    TfLiteTensor& dest = context->tensors[input_tensor_idx];

    memcpy(dest.data.raw, src.data.raw, src.bytes);
  }

  return kTfLiteOk;
}

void NNAPIDelegateKernel::AddDequantizeOperatorsWhereNeeded(
    const TfLiteContext* context, int builtin_code, const TfLiteNode* node,
    int tflite_node_index, NNAPIOpBuilder* builder, int* nnapi_errno) {
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
    builder->AddDequantize(i, node->inputs->data[i], type, tflite_node_index);
  }
}

TfLiteStatus NNAPIDelegateKernel::DensifyAndDequantizeConstTensor(
    TfLiteContext* context, int densify_node_id, bool should_dequantize,
    NNAPIOpBuilder& builder) {
  TfLiteNode* densify_node;
  TfLiteRegistration* reg;
  TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
      context, densify_node_id, &densify_node, &reg));
  int sparse_weight_tid = densify_node->inputs->data[0];
  auto input_tensor = context->tensors[sparse_weight_tid];
  auto output_tensor = context->tensors[densify_node->outputs->data[0]];
  if (input_tensor.sparsity == nullptr) {
    return kTfLiteError;
  }
  const int dims_count = output_tensor.dims->size;
  std::vector<int> vector_shape(dims_count);
  for (int i = 0; i < dims_count; i++) {
    vector_shape[i] = output_tensor.dims->data[i];
  }
  size_t dense_size;
  int new_tensor_index = -1;
  switch (input_tensor.type) {
    case kTfLiteFloat32: {
      dense_size = output_tensor.bytes / sizeof(float);
      std::vector<float> output_data(dense_size);
      tflite::internal::sparsity::FormatConverter<float> converter(
          vector_shape, *input_tensor.sparsity);
      converter.SparseToDense(static_cast<const float*>(input_tensor.data.data),
                              dense_size, output_data.data(), context);
      TF_LITE_ENSURE_STATUS(builder.AddNewInputConstantTensor<float>(
          ANEURALNETWORKS_TENSOR_FLOAT32, kTfLiteFloat32, output_tensor.dims,
          output_data, output_tensor.params, &new_tensor_index));
      break;
    }
    case kTfLiteFloat16: {
      dense_size = output_tensor.bytes / sizeof(Eigen::half);
      std::vector<uint16_t> output_data(dense_size);
      Eigen::half* unpacked_fp16_data =
          reinterpret_cast<Eigen::half*>(output_data.data());
      tflite::internal::sparsity::FormatConverter<Eigen::half> converter(
          vector_shape, *input_tensor.sparsity);
      converter.SparseToDense(
          static_cast<const Eigen::half*>(input_tensor.data.data), dense_size,
          unpacked_fp16_data, context);
      if (should_dequantize) {
        // we need to dequantize the fp16 dense tensor
        std::vector<float> float_dense_data(dense_size);
        for (int i = 0; i < dense_size; ++i) {
          float_dense_data[i] = fp16_ieee_to_fp32_value(
              reinterpret_cast<uint16_t*>(output_data.data())[i]);
        }
        TF_LITE_ENSURE_STATUS(builder.AddNewInputConstantTensor<float>(
            ANEURALNETWORKS_TENSOR_FLOAT32, kTfLiteFloat32, output_tensor.dims,
            float_dense_data, output_tensor.params, &new_tensor_index));
      } else {
        TF_LITE_ENSURE_STATUS(builder.AddNewInputConstantTensor<uint16_t>(
            ANEURALNETWORKS_TENSOR_FLOAT16, kTfLiteFloat16, output_tensor.dims,
            output_data, output_tensor.params, &new_tensor_index));
      }
      break;
    }
    case kTfLiteInt8: {
      dense_size = output_tensor.bytes / sizeof(int8_t);
      std::vector<int8_t> output_data(dense_size);
      tflite::internal::sparsity::FormatConverter<int8_t> converter(
          vector_shape, *input_tensor.sparsity);
      converter.SparseToDense(
          static_cast<const int8_t*>(input_tensor.data.data), dense_size,
          output_data.data(), context);
      TF_LITE_ENSURE_STATUS(builder.AddNewInputConstantTensor<int8_t>(
          ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, kTfLiteInt8,
          output_tensor.dims, output_data, output_tensor.params,
          &new_tensor_index));
      break;
    }
    default: {
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus NNAPIDelegateKernel::AddOpsAndTensors(
    TfLiteContext* context, int* nnapi_errno, bool allow_dynamic_dimensions) {
  DequantizeMapping dequantize_mapping;
  // The operand builder allows creating a single op. It is created outside
  // the for loop to avoid reallocating the vectors.
  NNAPIOpBuilder builder(nnapi_, context, &operand_mapping_,
                         &dequantize_mapping, &allocation_memory_mapping_,
                         &nnapi_to_tflite_op_mapping_, nn_model_.get(),
                         nnapi_errno, allow_dynamic_dimensions);
  // If we have target accelerators the target SDK version might be
  // different than the current android version.
  target_feature_level_ = nnapi_->nnapi_runtime_feature_level;
  if (!nnapi_devices_.empty()) {
    TF_LITE_ENSURE_STATUS(GetTargetFeatureLevel(
        context, nnapi_, nnapi_devices_, &target_feature_level_, nnapi_errno));
  }
  // First path, handle const fp16->fp32 dequantize and densify if needed.
  for (auto node_index : nodes_) {
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));
    if (IsDequantizeConstFloat16(context, node, registration)) {
      builder.AddTensorInput(node->inputs->data[0], /*hybrid_op=*/false,
                             NN_TENSOR_FLAG_HALF_TO_FLOAT_CONVERSION);
    }
    if (IsDensifyConstTensor(context, node, registration)) {
      densify_output_to_node_mapping_[node->outputs->data[0]] = node_index;
    }
    if (IsDequantizeNonConstFloat16(context, node, registration)) {
      non_const_dequantize_output_to_node_mapping_[node->outputs->data[0]] =
          node_index;
    }
  }
  // Clear the input and output lists for the dequantize path.
  builder.ClearInputOuputLists();

  // Add other tensors.
  for (auto node_index : nodes_) {
    // Obtain the op and registration.
    TfLiteNode* node;
    TfLiteRegistration* reg;
    TF_LITE_ENSURE_STATUS(
        context->GetNodeAndRegistration(context, node_index, &node, &reg));
    // skip DENSIFY -> DEQUANTIZE as they are handled elsewhere.
    if (IsDensifyConstTensor(context, node, reg) ||
        IsDequantizeNonConstFloat16(context, node, reg)) {
      continue;
    }

    // Delegate PACK by lowering it into CONCAT + RESHAPE.
    if (reg->builtin_code == kTfLiteBuiltinPack) {
      TF_LITE_ENSURE_STATUS(
          builder.TransformPackIntoSupportedOps(node_index, node, reg));
      continue;
    }
    // Delegate SPLIT_V by lowering it into SLICEs.
    if (reg->builtin_code == kTfLiteBuiltinSplitV) {
      TF_LITE_ENSURE_STATUS(
          builder.TransformSplitVIntoSupportedOps(node_index, node, reg));
      continue;
    }
    // Delegate SQUARED_DIFFERENCE by lowering it into SUB + MUL.
    if (reg->builtin_code == kTfLiteBuiltinSquaredDifference) {
      TF_LITE_ENSURE_STATUS(builder.TransformSquaredDifferenceIntoSupportedOps(
          node_index, node, reg));
      continue;
    }
    // Fully quantized full LSTM.
    if (target_feature_level_ >= kMinSdkVersionForNNAPI13 &&
        reg->builtin_code == kTfLiteBuiltinLstm && isLstmFullKernel(node) &&
        context->tensors[node->inputs->data[0]].type == kTfLiteInt8) {
      const auto quant8_full_lstm_op_code = ANEURALNETWORKS_QUANTIZED_LSTM;

      constexpr int kInputTensor = 0;
      constexpr int kInputToInputWeightsTensor = 1;
      constexpr int kRecurrentToInputWeightsTensor = 5;
      constexpr int kInputGateBiasTensor = 12;
      constexpr int kForgetGateBiasTensor = 13;
      constexpr int kCellGateBiasTensor = 14;
      constexpr int kOutputGateBiasTensor = 15;
      constexpr int kProjectionWeightsTensor = 16;
      constexpr int kProjectionBiasTensor = 17;
      constexpr int kPrevOutputTensor = 18;

      // Add input tensors.
      for (int input_pos = 0; input_pos < node->inputs->size; ++input_pos) {
        const auto input_index = node->inputs->data[input_pos];
        if (input_index == kTfLiteOptionalTensor) {
          if (input_pos == kInputToInputWeightsTensor ||
              input_pos == kRecurrentToInputWeightsTensor ||
              input_pos == kProjectionWeightsTensor) {
            TF_LITE_ENSURE_STATUS(builder.AddVectorInt8Operand(nullptr, 0));
          } else if (input_pos == kInputGateBiasTensor ||
                     input_pos == kForgetGateBiasTensor ||
                     input_pos == kCellGateBiasTensor ||
                     input_pos == kOutputGateBiasTensor ||
                     input_pos == kProjectionBiasTensor) {
            TF_LITE_ENSURE_STATUS(builder.AddVectorInt32Operand(nullptr, 0));
          } else {  // cell-to-* and layer norm weights.
            TF_LITE_ENSURE_STATUS(builder.AddVectorInt16Operand(nullptr, 0));
          }
        } else {
          // Only input and previous output use INT8_ASYM_SIGNED.
          int flags =
              (input_pos == kInputTensor || input_pos == kPrevOutputTensor)
                  ? NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED
                  : 0;
          TF_LITE_ENSURE_STATUS(
              builder.AddTensorInput(input_index, /*hybrid_op=*/false, flags));
        }
      }

      // Add clip parameters.
      auto builtin = reinterpret_cast<TfLiteLSTMParams*>(node->builtin_data);
      TF_LITE_ENSURE_STATUS(
          builder.AddScalarFloat32Operand(builtin->cell_clip));
      TF_LITE_ENSURE_STATUS(
          builder.AddScalarFloat32Operand(builtin->proj_clip));

      // Add quantization parameters for intermediate tensors.
      TF_LITE_ENSURE_EQ(context, node->intermediates->size, 5);
      for (int intermediate_pos = 0;
           intermediate_pos < node->intermediates->size; ++intermediate_pos) {
        const auto intermediate_index =
            node->intermediates->data[intermediate_pos];
        const TfLiteTensor& tensor = context->tensors[intermediate_index];
        TfLiteAffineQuantization* quantization_params =
            static_cast<TfLiteAffineQuantization*>(tensor.quantization.params);
        if (intermediate_pos == 4) {
          TF_LITE_ENSURE_STATUS(builder.AddScalarInt32Operand(
              quantization_params->zero_point->data[0]));
        }
        TF_LITE_ENSURE_STATUS(builder.AddScalarFloat32Operand(
            quantization_params->scale->data[0]));
      }

      // Activation state output.
      int ann_index;
      builder.AddStateInt8AsymTensor(
          node->inputs->data[/*kInputActivationStateTensor*/ 18], &ann_index);
      model_state_outputs_.push_back(ann_index);
      model_state_tfl_inputs_.push_back(
          node->inputs->data[/*kInputActivationStateTensor*/ 18]);

      // Cell state output.
      builder.AddStateInt16Tensor(
          node->inputs->data[/*kInputCellStateTensor*/ 19], &ann_index);
      model_state_outputs_.push_back(ann_index);
      model_state_tfl_inputs_.push_back(
          node->inputs->data[/*kInputCellStateTensor*/ 19]);

      // Add output tensors.
      for (int output_pos = 0; output_pos < node->outputs->size; ++output_pos) {
        const auto output_index = node->outputs->data[output_pos];
        TF_LITE_ENSURE_STATUS(builder.AddTensorOutput(
            output_index, NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED));
      }

      builder.FinalizeAddOperation(quant8_full_lstm_op_code, node_index);
      continue;
    }

    const bool hybrid_op = IsHybridOperator(context, reg->builtin_code, node);
    const bool scalar_as_tensor = IsScalarInputSupported(reg->builtin_code);
    const bool need_int8_conversion =
        target_feature_level_ < kMinSdkVersionForNNAPI13 &&
        NeedInt8Conversion(context, reg->builtin_code, node);
    const bool use_int8_asymm_signed =
        target_feature_level_ >= kMinSdkVersionForNNAPI13 && !hybrid_op;

    // skip DEQUANTIZE (fp16 -> fp32) as it is handled elsewhere
    if (IsDequantizeConstFloat16(context, node, reg)) {
      continue;
    }

    int input_tensor_flags = 0;
    if (scalar_as_tensor) {
      input_tensor_flags |= NN_TENSOR_FLAG_SCALAR_AS_TENSOR;
    }
    if (use_int8_asymm_signed) {
      input_tensor_flags |= NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED;
    }

    // On SDK level less than 30, h_swish will be lowered into supported NNAPI
    // operations. Since SDK level 30, h_swish is supported as a single
    // operation.
    if (reg->builtin_code == kTfLiteBuiltinHardSwish &&
        nnapi_->android_sdk_version < kMinSdkVersionForNNAPI13) {
      builder.TransformHardSwishIntoSupportedOps(
          node->inputs->data[0], node->outputs->data[0], need_int8_conversion,
          node_index);
      continue;
    }
    // Map inputs to NN API tensor indices.
    for (int input_pos = 0; input_pos < node->inputs->size; ++input_pos) {
      if (reg->builtin_code == kTfLiteBuiltinTransposeConv) {
        // Everything is added during Map since input tensors
        // have different order.
        continue;
      }
      if (reg->builtin_code == kTfLiteBuiltinFullyConnected &&
          node->inputs->data[input_pos] == kTfLiteOptionalTensor) {
        // skip optional bias and handle it during mapping
        continue;
      }
      const auto input_index = node->inputs->data[input_pos];
      // handle sparse weights for Conv2d
      if (reg->builtin_code == kTfLiteBuiltinConv2d && input_pos == 1) {
        int densify_node_id = -1;
        bool should_dequantize = false;
        int dequantize_node_id =
            non_const_dequantize_output_to_node_mapping_[input_index];
        if (dequantize_node_id != -1) {
          should_dequantize = true;
          // Find densify->dequantize pattern.
          TfLiteNode* dequant_node;
          TfLiteRegistration* reg;
          TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
              context, dequantize_node_id, &dequant_node, &reg));
          densify_node_id =
              densify_output_to_node_mapping_[dequant_node->inputs->data[0]];
        } else {
          densify_node_id = densify_output_to_node_mapping_[input_index];
        }
        if (densify_node_id != -1) {
          TF_LITE_ENSURE_STATUS(DensifyAndDequantizeConstTensor(
              context, densify_node_id, should_dequantize, builder));
          continue;
        }
      }
      if (need_int8_conversion &&
          (input_pos == 0 ||
           reg->builtin_code == kTfLiteBuiltinFullyConnected ||
           reg->builtin_code == kTfLiteBuiltinConv2d ||
           reg->builtin_code == kTfLiteBuiltinDepthwiseConv2d ||
           reg->builtin_code == kTfLiteBuiltinAdd ||
           reg->builtin_code == kTfLiteBuiltinMul ||
           reg->builtin_code == kTfLiteBuiltinSub ||
           reg->builtin_code == kTfLiteBuiltinConcatenation ||
           reg->builtin_code == kTfLiteBuiltinMaximum ||
           reg->builtin_code == kTfLiteBuiltinMinimum ||
           reg->builtin_code == kTfLiteBuiltinLeakyRelu ||
           reg->builtin_code == kTfLiteBuiltinLess ||
           reg->builtin_code == kTfLiteBuiltinLessEqual ||
           reg->builtin_code == kTfLiteBuiltinPrelu ||
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
              if (need_int8_conversion) {
                builder.AddScalarInt32Operand(
                    static_cast<int32_t>(*constant_value.data.int8) + 128);
              } else {
                builder.AddScalarInt32Operand(*constant_value.data.int8);
              }
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
          TF_LITE_ENSURE_STATUS(builder.AddTensorInput(input_index, hybrid_op,
                                                       input_tensor_flags));
        }
      } else if (reg->builtin_code == kTfLiteBuiltinTopkV2 && input_pos > 0) {
        // The K parameter tensor is not handled here but by the functor
        // returned by Map, the input tensor is instead added in
        // the else clause below
        continue;
      } else if (reg->builtin_code == kTfLiteBuiltinGather) {
        // Everything else is added during Map since input tensors
        // have different order.
        if (input_pos == 0) {
          TF_LITE_ENSURE_STATUS(builder.AddTensorInput(input_index, hybrid_op,
                                                       input_tensor_flags));
        }
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
          TF_LITE_ENSURE_STATUS(builder.AddTensorInput(input_index, hybrid_op,
                                                       input_tensor_flags));
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
      } else if (reg->builtin_code == kTfLiteBuiltinMaximum ||
                 reg->builtin_code == kTfLiteBuiltinMinimum) {
        const TfLiteTensor& operand_tensor =
            context->tensors[node->inputs->data[input_pos]];
        if (operand_tensor.dims->size == 0) {
          int tensor_index;

          TF_LITE_ENSURE_EQ(context, operand_tensor.allocation_type,
                            kTfLiteMmapRo);
          switch (operand_tensor.type) {
            case kTfLiteFloat32:
              TF_LITE_ENSURE_STATUS(builder.AddNewInputConstantTensor(
                  ANEURALNETWORKS_TENSOR_FLOAT32, operand_tensor.type, {1},
                  std::vector<float>(1, operand_tensor.data.f[0]),
                  operand_tensor.params, &tensor_index));
              break;
            case kTfLiteUInt8:
              TF_LITE_ENSURE_STATUS(builder.AddNewInputConstantTensor(
                  ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, operand_tensor.type, {1},
                  std::vector<uint8_t>(1, operand_tensor.data.uint8[0]),
                  operand_tensor.params, &tensor_index));
              break;
            case kTfLiteInt8: {
              auto params = operand_tensor.params;
              if (params.scale == 0.0) {
                params.scale = 1.0;
              }

              if (use_int8_asymm_signed) {
                TF_LITE_ENSURE_STATUS(builder.AddNewInputConstantTensor(
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED,
                    operand_tensor.type, {1},
                    std::vector<int8_t>(1, operand_tensor.data.int8[0]), params,
                    &tensor_index));
              } else {
                TF_LITE_ENSURE_STATUS(builder.AddNewInputConstantTensor(
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, operand_tensor.type,
                    {1},
                    std::vector<int8_t>(1, operand_tensor.data.int8[0] + 128),
                    params, &tensor_index));
              }
            } break;
            case kTfLiteInt32:
              TF_LITE_ENSURE_STATUS(builder.AddNewInputConstantTensor(
                  ANEURALNETWORKS_TENSOR_INT32, operand_tensor.type, {1},
                  std::vector<int32_t>(1, operand_tensor.data.i32[0]),
                  operand_tensor.params, &tensor_index));
              break;
            default:
              return kTfLiteError;
          }
        } else {
          TF_LITE_ENSURE_STATUS(builder.AddTensorInput(input_index, hybrid_op,
                                                       input_tensor_flags));
        }
      } else if ((reg->builtin_code == kTfLiteBuiltinReduceAny ||
                  reg->builtin_code == kTfLiteBuiltinReduceMax ||
                  reg->builtin_code == kTfLiteBuiltinReduceMin ||
                  reg->builtin_code == kTfLiteBuiltinReduceProd ||
                  reg->builtin_code == kTfLiteBuiltinSum) &&
                 (input_pos == 1)) {
        // The axis needs, be converted to a tensor if specified as scalar
        const TfLiteTensor& axis_tensor =
            context->tensors[node->inputs->data[input_pos]];
        if (axis_tensor.dims->size == 0) {
          TF_LITE_ENSURE_STATUS(
              builder.AddVectorInt32Operand(axis_tensor.data.i32, 1));
        } else {
          TF_LITE_ENSURE_STATUS(builder.AddTensorInput(input_index, hybrid_op,
                                                       input_tensor_flags));
        }
      } else if (reg->builtin_code == kTfLiteBuiltinFill) {
        if (input_pos == 0) {
          const int dims_id = node->inputs->data[0];
          const TfLiteTensor& dims_tensor = context->tensors[dims_id];
          switch (dims_tensor.type) {
            case kTfLiteInt32:
              TF_LITE_ENSURE_STATUS(
                  builder.AddTensorInput(input_index, hybrid_op));
              break;
            case kTfLiteInt64: {
              // We made sure that dimensions are constant and fit into int32
              // in Map(), so we can safely create a new tensor with casted
              // values.
              const int dims_size = dims_tensor.dims->data[0];
              std::vector<int32_t> dims_int32(dims_size);
              std::copy(dims_tensor.data.i64, dims_tensor.data.i64 + dims_size,
                        dims_int32.begin());
              int new_tensor_index = -1;
              builder.AddNewInputConstantTensor(
                  ANEURALNETWORKS_TENSOR_INT32, kTfLiteInt32, dims_tensor.dims,
                  dims_int32, dims_tensor.params, &new_tensor_index);
            } break;
            default:
              return kTfLiteError;
          }
        } else {
          const int value_id = node->inputs->data[1];
          const TfLiteTensor& value_tensor = context->tensors[value_id];
          switch (value_tensor.type) {
            case kTfLiteFloat32:
              if (value_tensor.allocation_type == kTfLiteMmapRo) {
                TF_LITE_ENSURE_STATUS(
                    builder.AddScalarFloat32Operand(*value_tensor.data.f));
              } else {
                TF_LITE_ENSURE_STATUS(
                    builder.AddSingleValueTensorAsScalarOperand(
                        value_id, ANEURALNETWORKS_FLOAT32));
              }
              break;
            case kTfLiteInt32:
              if (value_tensor.allocation_type == kTfLiteMmapRo) {
                TF_LITE_ENSURE_STATUS(
                    builder.AddScalarInt32Operand(*value_tensor.data.i32));
              } else {
                TF_LITE_ENSURE_STATUS(
                    builder.AddSingleValueTensorAsScalarOperand(
                        value_id, ANEURALNETWORKS_INT32));
              }
              break;
            case kTfLiteInt64:
              if (value_tensor.allocation_type == kTfLiteMmapRo) {
                // Map() function already makes sure const int64 input fits into
                // int32.
                TF_LITE_ENSURE_STATUS(builder.AddScalarInt32Operand(
                    static_cast<int32_t>(*value_tensor.data.i64)));
              } else {
                TF_LITE_ENSURE_STATUS(
                    builder.AddSingleValueTensorAsScalarOperand(
                        value_id, ANEURALNETWORKS_INT32));
              }
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
    // Fails if the Validate function failed
    int nn_op_type;
    TF_LITE_ENSURE_STATUS(
        Map(context, reg->builtin_code, reg->version, target_feature_level_,
            {context, &builder, node, node_index, &model_state_outputs_,
             &model_state_tfl_inputs_, &feedback_loops_, nnapi_errno},
            &nn_op_type));

    // Map outputs to NN API tensor indices.
    int output_tensor_flags = 0;
    if (need_int8_conversion) {
      output_tensor_flags |= NN_TENSOR_FLAG_INT8_CONVERSION;
    }
    if (use_int8_asymm_signed) {
      output_tensor_flags |= NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED;
    }
    // fc_nn_intermediate_output_index is used to indicate whether additional
    // RESHAPE op is needed.
    int fc_nn_intermediate_output_index = -1;
    // mean_nn_intermediate_output_index is used to indicate whether additional
    // re-quantization is needed.
    int mean_nn_intermediate_output_index = -1;
    for (int output_pos = 0; output_pos < node->outputs->size; ++output_pos) {
      auto output_index = node->outputs->data[output_pos];

      // Outputs for  basic LSTM cell are set in the Map function since
      if (reg->builtin_code == kTfLiteBuiltinLstm && isLstmBasicKernel(node)) {
        continue;
      }
      // Handle FC with keep_num_dims==true.
      if (reg->builtin_code == kTfLiteBuiltinFullyConnected &&
          reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data)
              ->keep_num_dims) {
        auto& output_tensor = context->tensors[output_index];

        int num_units = output_tensor.dims->data[output_tensor.dims->size - 1];
        std::vector<uint32_t> output_dims(2);
        output_dims[0] = NumElements(output_tensor.dims) / num_units;
        output_dims[1] = num_units;
        TF_LITE_ENSURE_STATUS(builder.AddIntermediateOutputTensor(
            output_tensor.type, output_dims.size(), output_dims.data(),
            output_tensor.params.scale, output_tensor.params.zero_point,
            &fc_nn_intermediate_output_index));
      } else if (reg->builtin_code == kTfLiteBuiltinMean &&
                 IsMeanWithDifferentInputOutputQuantization(context, node)) {
        // Handle MEAN with different input and output quantization params.
        auto& input_tensor = context->tensors[node->inputs->data[0]];
        auto& output_tensor = context->tensors[output_index];
        TF_LITE_ENSURE_STATUS(builder.AddIntermediateOutputTensor(
            output_tensor.type, output_tensor.dims->size,
            reinterpret_cast<const uint32_t*>(output_tensor.dims->data),
            input_tensor.params.scale, input_tensor.params.zero_point,
            &mean_nn_intermediate_output_index, need_int8_conversion));
      } else {
        TF_LITE_ENSURE_STATUS(
            builder.AddTensorOutput(output_index, output_tensor_flags));
      }
    }

    // Dequantize operators may have to be added in case inputs are to be
    // floating-point.
    AddDequantizeOperatorsWhereNeeded(context, reg->builtin_code, node,
                                      node_index, &builder, nnapi_errno);

    TF_LITE_ENSURE_OK(context_,
                      builder.FinalizeAddOperation(nn_op_type, node_index));
    if (fc_nn_intermediate_output_index > -1) {
      TF_LITE_ENSURE_STATUS(builder.AppendReshape(
          fc_nn_intermediate_output_index, node->outputs->data[0], node_index));
    }
    if (mean_nn_intermediate_output_index > -1) {
      TF_LITE_ENSURE_STATUS(builder.AppendRequantize(
          mean_nn_intermediate_output_index, node->outputs->data[0], node_index,
          output_tensor_flags));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus NNAPIDelegateKernel::BuildGraph(
    TfLiteContext* context,
    const StatefulNnApiDelegate::Options& delegate_options,
    const TfLiteIntArray* input_tensors, const TfLiteIntArray* output_tensors,
    int* nnapi_errno) {
  // Build the ops and tensors.
  TF_LITE_ENSURE_STATUS(AddOpsAndTensors(
      context, nnapi_errno, delegate_options.allow_dynamic_dimensions));
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
      total_input_byte_size += GetNumPaddingBytes(tensor_size);
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
    total_output_byte_size += GetNumPaddingBytes(context->tensors[i].bytes);
  }

  // Add state output tensors as model outputs.
  for (int i = 0; i < model_state_outputs_.size(); i++) {
    outputs.push_back(model_state_outputs_[i]);
    auto tfl_state_idx = model_state_tfl_inputs_[i];
    total_output_byte_size += context->tensors[tfl_state_idx].bytes;
    total_output_byte_size +=
        GetNumPaddingBytes(context->tensors[tfl_state_idx].bytes);
  }

  // Tell ANN to declare inputs/outputs
  RETURN_TFLITE_ERROR_IF_NN_ERROR(
      context,
      nnapi_->ANeuralNetworksModel_identifyInputsAndOutputs(
          nn_model_.get(), inputs.size(), inputs.data(), outputs.size(),
          outputs.data()),
      "identifying model inputs and outputs", nnapi_errno);

  auto allow_fp16 =
      context->allow_fp32_relax_to_fp16 | delegate_options.allow_fp16;
  if (nnapi_->android_sdk_version >= kMinSdkVersionForNNAPI11) {
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context,
        nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16(
            nn_model_.get(), allow_fp16),
        "set relaxed computation mode for fp32 if possible", nnapi_errno);
  }

  RETURN_TFLITE_ERROR_IF_NN_ERROR(
      context, nnapi_->ANeuralNetworksModel_finish(nn_model_.get()),
      "finalizing the model", nnapi_errno);

  // Create shared memory pool for inputs and outputs.
  nn_input_memory_.reset(
      new NNMemory(nnapi_, "input_pool", total_input_byte_size));
  nn_output_memory_.reset(
      new NNMemory(nnapi_, "output_pool", total_output_byte_size));

  return kTfLiteOk;
}

}  // namespace nnapi
}  // namespace delegate

using ::tflite::delegate::nnapi::kMinSdkVersionForNNAPI;
using ::tflite::delegate::nnapi::kMinSdkVersionForNNAPI11;
using ::tflite::delegate::nnapi::kMinSdkVersionForNNAPI12;
using ::tflite::delegate::nnapi::NNAPIDelegateKernel;

StatefulNnApiDelegate::Data::Data(const NnApi* nnapi) : nnapi(nnapi) {}
StatefulNnApiDelegate::Data::Data(std::unique_ptr<const NnApi> nnapi)
    : nnapi(nnapi.get()), owned_nnapi(std::move(nnapi)) {}

StatefulNnApiDelegate::Data::~Data() {
  std::for_each(std::begin(delegate_state_cache),
                std::end(delegate_state_cache),
                [](const std::pair<int, NNAPIDelegateKernel*>& entry) {
                  delete entry.second;
                });
}

void StatefulNnApiDelegate::Data::CacheDelegateKernel(
    const TfLiteDelegateParams* delegate_params,
    NNAPIDelegateKernel* delegate_state) {
  const int cache_key = delegate_params->nodes_to_replace->data[0];
  delegate_state_cache.emplace(cache_key, delegate_state);
}

NNAPIDelegateKernel* StatefulNnApiDelegate::Data::MaybeGetCachedDelegateKernel(
    const TfLiteDelegateParams* delegate_params) {
  const int cache_key = delegate_params->nodes_to_replace->data[0];
  const auto cached_state = delegate_state_cache.find(cache_key);
  if (cached_state != std::end(delegate_state_cache)) {
    auto result = cached_state->second;
    delegate_state_cache.erase(cached_state);
    return result;
  } else {
    return nullptr;
  }
}

void StatefulNnApiDelegate::StatefulNnApiDelegateConstructorImpl(
    const Options& options) {
  if (options.accelerator_name) {
    delegate_data_.accelerator_name = options.accelerator_name;
  }
  if (options.cache_dir) {
    delegate_data_.cache_dir = options.cache_dir;
  }
  if (options.model_token) {
    delegate_data_.model_token = options.model_token;
  }
  delegate_data_.execution_preference = options.execution_preference;
  delegate_data_.disallow_nnapi_cpu = options.disallow_nnapi_cpu;
  delegate_data_.max_number_delegated_partitions =
      options.max_number_delegated_partitions;
  delegate_data_.allow_fp16 = options.allow_fp16;
  delegate_data_.execution_priority = options.execution_priority;
  delegate_data_.max_compilation_timeout_duration_ns =
      options.max_compilation_timeout_duration_ns;
  delegate_data_.max_execution_timeout_duration_ns =
      options.max_execution_timeout_duration_ns;
  delegate_data_.max_execution_loop_timeout_duration_ns =
      options.max_execution_loop_timeout_duration_ns;
  if (delegate_data_.nnapi->android_sdk_version >= kMinSdkVersionForNNAPI11) {
    delegate_data_.allow_dynamic_dimensions = options.allow_dynamic_dimensions;
  }
  delegate_data_.use_burst_computation = options.use_burst_computation;
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Created TensorFlow Lite delegate for NNAPI.");
  Prepare = DoPrepare;
  CopyFromBufferHandle = DoCopyFromBufferHandle;
  CopyToBufferHandle = DoCopyToBufferHandle;
  FreeBufferHandle = DoFreeBufferHandle;
  data_ = &delegate_data_;
  if (delegate_data_.allow_dynamic_dimensions) {
    flags |= kTfLiteDelegateFlagsAllowDynamicTensors;
    flags |= kTfLiteDelegateFlagsRequirePropagatedShapes;
  }
}

StatefulNnApiDelegate::StatefulNnApiDelegate(const NnApi* nnapi)
    : StatefulNnApiDelegate(nnapi, Options()) {}

StatefulNnApiDelegate::StatefulNnApiDelegate(Options options)
    : StatefulNnApiDelegate(NnApiImplementation(), options) {}

StatefulNnApiDelegate::StatefulNnApiDelegate(
    const NnApiSLDriverImplFL5* nnapi_support_library_driver, Options options)
    : TfLiteDelegate(TfLiteDelegateCreate()),
      delegate_data_(
          CreateNnApiFromSupportLibrary(nnapi_support_library_driver)) {
  StatefulNnApiDelegateConstructorImpl(options);
}

StatefulNnApiDelegate::StatefulNnApiDelegate(const NnApi* nnapi,
                                             Options options)
    : TfLiteDelegate(TfLiteDelegateCreate()), delegate_data_(nnapi) {
  StatefulNnApiDelegateConstructorImpl(options);
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
  options.max_number_delegated_partitions =
      delegate_data->max_number_delegated_partitions;
  options.allow_fp16 = delegate_data->allow_fp16;
  options.execution_priority = delegate_data->execution_priority;
  options.max_compilation_timeout_duration_ns =
      delegate_data->max_compilation_timeout_duration_ns;
  options.max_execution_timeout_duration_ns =
      delegate_data->max_execution_timeout_duration_ns;
  options.max_execution_loop_timeout_duration_ns =
      delegate_data->max_execution_loop_timeout_duration_ns;
  options.allow_dynamic_dimensions = delegate_data->allow_dynamic_dimensions;
  options.use_burst_computation = delegate_data->use_burst_computation;
  return options;
}

const std::vector<StatefulNnApiDelegate::MemoryRegistration>&
StatefulNnApiDelegate::GetTensorMemoryMap(TfLiteDelegate* delegate) {
  auto delegate_data = reinterpret_cast<Data*>(delegate->data_);
  return delegate_data->tensor_memory_map;
}

delegates::Serialization* StatefulNnApiDelegate::GetCache(
    TfLiteDelegate* delegate) {
  auto delegate_data = reinterpret_cast<Data*>(delegate->data_);
  return delegate_data->cache.get();
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

// static
TfLiteStatus StatefulNnApiDelegate::GetNodesSupportedByAccelerator(
    TfLiteContext* context, TfLiteDelegate* delegate, const NnApi* nnapi,
    const std::vector<int>& supported_nodes,
    std::vector<int>* device_supported_nodes, int* num_partitions,
    TfLiteDelegateParams** params_array, int* nnapi_errno) {
  auto* delegate_data = static_cast<Data*>(delegate->data_);
  // The first entry in the array is the element count

  auto supported_nodes_int_array = BuildTfLiteIntArray(supported_nodes);
  TF_LITE_ENSURE_STATUS(context->PreviewDelegatePartitioning(
      context, supported_nodes_int_array.get(), params_array, num_partitions));
  // For each partition check if which nodes are actually supported by the
  // target accelerators.
  delegate_data->delegate_state_cache.clear();
  for (int idx = 0; idx < *num_partitions; idx++) {
    const auto& partition_params = (*params_array)[idx];
    std::unique_ptr<NNAPIDelegateKernel> kernel_state(
        new NNAPIDelegateKernel(nnapi));
    TfLiteDelegateParams params_with_delegate = partition_params;
    params_with_delegate.delegate = delegate;
    TF_LITE_ENSURE_STATUS(
        kernel_state->Init(context, &params_with_delegate, nnapi_errno));
    std::vector<int> supported_partition_nodes;
    TF_LITE_ENSURE_STATUS(
        kernel_state->GetOperationsSupportedByTargetNnApiDevices(
            context, &supported_partition_nodes, nnapi_errno));
    device_supported_nodes->insert(device_supported_nodes->end(),
                                   supported_partition_nodes.begin(),
                                   supported_partition_nodes.end());

    bool model_fully_supported = (supported_partition_nodes.size() ==
                                  partition_params.nodes_to_replace->size);
    if (model_fully_supported) {
      delegate_data->CacheDelegateKernel(&partition_params,
                                         kernel_state.release());
    }
  }

  if (device_supported_nodes->size() != supported_nodes.size()) {
    // We changed the set of nodes to delegate this will create a different
    // partitioning layout.
    auto device_sup_nodes_int_array =
        BuildTfLiteIntArray(*device_supported_nodes);
    TF_LITE_ENSURE_STATUS(context->PreviewDelegatePartitioning(
        context, device_sup_nodes_int_array.get(), params_array,
        num_partitions));
  }

  return kTfLiteOk;
}

// static
TfLiteStatus StatefulNnApiDelegate::LimitDelegatedPartitions(
    int max_partitions,
    std::vector<TfLiteDelegateParams> partition_params_array,
    std::vector<int>* nodes_to_delegate) {
  int num_partitions = partition_params_array.size();
  if (max_partitions <= 0 || num_partitions <= max_partitions) {
    return kTfLiteOk;
  }

  int number_delegated_partitions = std::count_if(
      partition_params_array.begin(), partition_params_array.end(),
      [nodes_to_delegate](const TfLiteDelegateParams& partition_params) {
        return std::find(nodes_to_delegate->begin(), nodes_to_delegate->end(),
                         partition_params.nodes_to_replace->data[0]) !=
               nodes_to_delegate->end();
      });

  if (number_delegated_partitions > max_partitions) {
    std::sort(partition_params_array.begin(), partition_params_array.end(),
              [](const TfLiteDelegateParams& left,
                 const TfLiteDelegateParams& right) -> bool {
                // Reverse sort
                return left.nodes_to_replace->size >
                       right.nodes_to_replace->size;
              });

    nodes_to_delegate->clear();

    for (int i = 0; i < max_partitions; i++) {
      const TfLiteDelegateParams& partition_params = partition_params_array[i];

      nodes_to_delegate->insert(nodes_to_delegate->end(),
                                partition_params.nodes_to_replace->data,
                                partition_params.nodes_to_replace->data +
                                    partition_params.nodes_to_replace->size);
    }
  }

  return kTfLiteOk;
}

static std::vector<int> GetSupportedOpsWithFp16WeightRemapping(
    TfLiteContext* context, int target_feature_level,
    bool is_accelerator_specified, int max_number_delegated_partitions) {
  std::vector<int> supported_nodes;
  delegates::IsNodeSupportedFn node_supported_fn =
      [=](TfLiteContext* context, TfLiteNode* node,
          TfLiteRegistration* registration,
          std::string* unsupported_details) -> bool {
    std::vector<delegate::nnapi::NNAPIValidationFailure> map_failures;
    const auto is_supported = NNAPIDelegateKernel::Validate(
        context, registration->builtin_code, registration->version,
        target_feature_level, node, is_accelerator_specified, &map_failures);
    if (!is_supported) {
      if (unsupported_details) {
        for (auto& failure : map_failures) {
          unsupported_details->append(failure.message.c_str());
        }
      }
      return false;
    }
    return true;
  };

  delegates::FP16GraphPartitionHelper partition_helper(context,
                                                       node_supported_fn);
  std::set<std::string> unsupported_nodes_info;
  if (partition_helper.Partition(&unsupported_nodes_info) == kTfLiteOk) {
    // By default, we simply get 1st largest partition as
    // 'max_delegate_partions'
    // is set to 1 by default.
    supported_nodes = partition_helper.GetNodesOfFirstNLargestPartitions(
        max_number_delegated_partitions);
  }
  return supported_nodes;
}

TfLiteStatus StatefulNnApiDelegate::DoPrepare(TfLiteContext* context,
                                              TfLiteDelegate* delegate) {
  auto* delegate_data = static_cast<Data*>(delegate->data_);
  int* nnapi_errno = &(delegate_data->nnapi_errno);
  const NnApi* nnapi = delegate_data->nnapi;

  // Resetting the error code when the delegate is initialized
  // by TFLite. This causes the error to be reset if reusing the same
  // StatefulNnApiDelegate after a failure
  *nnapi_errno = 0;

  // Do not check nodes_ if NN API is unavailable.
  if (nnapi->android_sdk_version < kMinSdkVersionForNNAPI ||
      !nnapi->nnapi_exists) {
    return kTfLiteOk;
  }

  int target_feature_level = nnapi->android_sdk_version;
  const StatefulNnApiDelegate::Options delegate_options =
      StatefulNnApiDelegate::GetOptions(delegate);
  // For NNAPI 1.2+, check if there is any accelerator available.
  // If not, don't delegate to NNAPI's CPU reference implementation unless
  // it has been specified as target accelerator.
  if (nnapi->android_sdk_version >= kMinSdkVersionForNNAPI12) {
    if (ShouldUseTargetDevices(delegate_options, nnapi)) {
      std::vector<ANeuralNetworksDevice*> devices;
      TF_LITE_ENSURE_STATUS(
          GetTargetDevices(context, delegate, nnapi, nnapi_errno, &devices));

      if (devices.empty()) {
        if (delegate_options.accelerator_name) {
          // There was a selected device and it is not available.
          return kTfLiteError;
        } else {
          // Only nnapi-reference is available but was disabled by the delegate
          // options
          return kTfLiteOk;
        }
      }

      TF_LITE_ENSURE_STATUS(GetTargetFeatureLevel(
          context, nnapi, devices, &target_feature_level, nnapi_errno));
    } else {
      // If no accelerator is specified, only use NNAPI if an accelerator is
      // available. Any available accelerator will make the device_count larger
      // than 1. More sophisticated check and allowlisting can be added later.
      uint32_t device_count = 0;
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context, nnapi->ANeuralNetworks_getDeviceCount(&device_count),
          "getting number of NNAPI devices", nnapi_errno);
      if (device_count <= 1) {
        return kTfLiteOk;
      }
    }
  }

  std::vector<int> supported_nodes;
  // We don't care about all nodes_, we only care about ones in the
  // current plan.
  TfLiteIntArray* execution_plan;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &execution_plan));
  // Copy the execution plan and wrap it with unique_ptr.
  std::unique_ptr<TfLiteIntArray, decltype(&TfLiteIntArrayFree)> plan(
      TfLiteIntArrayCopy(execution_plan), TfLiteIntArrayFree);

  // Check for every node if it is supported
  const bool is_accelerator_specified = ShouldUseTargetDevices(
      delegate_options, nnapi, /*exclude_nnapi_reference=*/true);
  std::vector<delegate::nnapi::NNAPIValidationFailure> map_failures;
  // First pass through execution plan to remember mapping of FP16->FP32
  // dequantizations in the graph.
  std::vector<int> fp16_to_fp32(context->tensors_size, -1);
  bool should_prune_fp16_dequantize = false;
  for (int i = 0; i < plan->size; ++i) {
    const int node_id = plan->data[i];
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_id, &node, &registration));
    if (IsDequantizeConstFloat16(context, node, registration)) {
      should_prune_fp16_dequantize = true;
      fp16_to_fp32[node->inputs->data[0]] = node->outputs->data[0];
    }
  }
  if (should_prune_fp16_dequantize) {
    supported_nodes = GetSupportedOpsWithFp16WeightRemapping(
        context, target_feature_level, is_accelerator_specified,
        delegate_options.max_number_delegated_partitions);
  } else {
    for (int node_index : TfLiteIntArrayView(plan.get())) {
      TfLiteNode* node;
      TfLiteRegistration* registration;
      TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
          context, node_index, &node, &registration));
      if (NNAPIDelegateKernel::Validate(
              context, registration->builtin_code, registration->version,
              target_feature_level, node, is_accelerator_specified,
              &map_failures)) {
        supported_nodes.push_back(node_index);
      }
#ifdef NNAPI_VERBOSE_VALIDATION
      for (auto& failure : map_failures) {
        TFLITE_LOG_PROD(
            TFLITE_LOG_WARNING,
            "Operator %s (v%d) refused by NNAPI delegate: %s",
            tflite::EnumNameBuiltinOperator(
                static_cast<BuiltinOperator>(registration->builtin_code)),
            registration->version, failure.message.c_str());
      }
      map_failures.clear();
#endif
    }
  }

  // If there are no delegated nodes, short-circuit node replacement.
  if (supported_nodes.empty()) {
    return kTfLiteOk;
  }

  // NN API Delegate Registration (the pseudo kernel that will invoke NN
  // API node sub sets)
  static const TfLiteRegistration nnapi_delegate_kernel = {
      .init = [](TfLiteContext* context, const char* buffer,
                 size_t length) -> void* {
        const TfLiteDelegateParams* params =
            reinterpret_cast<const TfLiteDelegateParams*>(buffer);

        auto* delegate_data = static_cast<Data*>(params->delegate->data_);
        int* nnapi_errno = &(delegate_data->nnapi_errno);

        NNAPIDelegateKernel* kernel_state =
            delegate_data->MaybeGetCachedDelegateKernel(params);
        if (!kernel_state) {
          kernel_state = new NNAPIDelegateKernel(delegate_data->nnapi);
          kernel_state->Init(context, params, nnapi_errno);
        }

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

  // Initialize caching, if applicable, from Options.
  const char* cache_dir = delegate_options.cache_dir;
  const char* model_token = delegate_options.model_token;
  delegates::SerializationParams params = {model_token, cache_dir};
  if (nnapi->android_sdk_version >= kMinSdkVersionForNNAPI12 && cache_dir &&
      model_token) {
    delegate_data->cache.reset(new delegates::Serialization(params));
  }

  delegates::Serialization* cache_ptr = delegate_data->cache.get();

  if (cache_ptr) {
    // Reuse cached delegation decision if possible.
    std::string accelerator_id = NnApiBackendId(delegate_options);
    TfLiteIntArray* cached_nodes_to_delegate = nullptr;
    if (delegates::GetDelegatedNodes(context, cache_ptr, accelerator_id,
                                     &cached_nodes_to_delegate) == kTfLiteOk) {
      if (cached_nodes_to_delegate->size == 0) return kTfLiteOk;
      auto status = context->ReplaceNodeSubsetsWithDelegateKernels(
          context, nnapi_delegate_kernel, cached_nodes_to_delegate, delegate);
      TfLiteIntArrayFree(cached_nodes_to_delegate);
      return status;
    }
  }

  std::vector<int> nodes_to_delegate;

  int num_partitions;
  TfLiteDelegateParams* params_array;
  if (is_accelerator_specified &&
      nnapi->android_sdk_version >= kMinSdkVersionForNNAPI12) {
    // Filtering out nodes not supported by target accelerators.
    // Cannot query supported operation before NNAPI 1.2
    TF_LITE_ENSURE_STATUS(GetNodesSupportedByAccelerator(
        context, delegate, nnapi, supported_nodes, &nodes_to_delegate,
        &num_partitions, &params_array, nnapi_errno));
  } else {
    nodes_to_delegate = supported_nodes;
    auto supported_nodes_int_array = BuildTfLiteIntArray(supported_nodes);
    TF_LITE_ENSURE_STATUS(context->PreviewDelegatePartitioning(
        context, supported_nodes_int_array.get(), &params_array,
        &num_partitions));
  }

  // FP16GraphPartitionHelper alters the orginal graph by remapping fp32
  // dequantize output to fp16 input. In the case of accelerator backends does
  // not support all the nodes of the fp16 model, We need to restore original
  // graph in order for things to work.
  if (should_prune_fp16_dequantize &&
      supported_nodes.size() != nodes_to_delegate.size()) {
    // Restore original graph
    for (int execution_plan_index = 0; execution_plan_index < plan->size;
         ++execution_plan_index) {
      int node_index = plan->data[execution_plan_index];
      TfLiteNode* node = nullptr;
      TfLiteRegistration* reg = nullptr;
      TF_LITE_ENSURE_STATUS(
          context->GetNodeAndRegistration(context, node_index, &node, &reg));
      if (reg->builtin_code == kTfLiteBuiltinDequantize) continue;

      for (int i = 0; i < node->inputs->size; ++i) {
        const int original_input_idx = node->inputs->data[i];
        if (original_input_idx == kTfLiteOptionalTensor) continue;
        // Use original FP32 input
        if (context->tensors[original_input_idx].type == kTfLiteFloat16 &&
            fp16_to_fp32[original_input_idx] != -1) {
          node->inputs->data[i] = fp16_to_fp32[original_input_idx];
        }
      }
    }
    // Only allow full model delegation for fp16 model.
    return kTfLiteOk;
  }

  TF_LITE_ENSURE_STATUS(
      LimitDelegatedPartitions(delegate_options.max_number_delegated_partitions,
                               std::vector<TfLiteDelegateParams>(
                                   params_array, params_array + num_partitions),
                               &nodes_to_delegate));

  auto nodes_to_delegate_int_array = BuildTfLiteIntArray(nodes_to_delegate);

  if (cache_ptr) {
    // Cache list of nodes to be delegated for later.
    std::string accelerator_id = NnApiBackendId(delegate_options);
    if (delegates::SaveDelegatedNodes(context, cache_ptr, accelerator_id,
                                      nodes_to_delegate_int_array.get()) !=
        kTfLiteOk) {
      // Not a critical error.
      TF_LITE_KERNEL_LOG(context, "Could not save delegated nodes");
    }
  }

  if (nodes_to_delegate_int_array->size == 0) {
    return kTfLiteOk;
  } else {
    // Request TFLite to partition the graph and make kernels
    // for each independent node sub set a new nnapi_delegate_kernel.
    return context->ReplaceNodeSubsetsWithDelegateKernels(
        context, nnapi_delegate_kernel, nodes_to_delegate_int_array.get(),
        delegate);
  }
}

// Returns a singleton NNAPI Delegate that can check for support of ops.
TfLiteDelegate* NnApiDelegate() {
  static StatefulNnApiDelegate* delegate = new StatefulNnApiDelegate();
  return delegate;
}

}  // namespace tflite
