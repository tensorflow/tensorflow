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

#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/util.h"

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif
#if defined __ANDROID__ || defined __unix__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace tflite {
namespace {

// TODO(b/80621585): Consider printing error string, but don't for now to
// minimize binary size.
#define RETURN_TFLITE_ERROR_IF_NN_ERROR(context, code)                        \
  do {                                                                        \
    const auto _code = (code);                                                \
    if (_code != ANEURALNETWORKS_NO_ERROR) {                                  \
      context->ReportError(context, "NN API returned error (%d, line %d).\n", \
                           _code, __LINE__);                                  \
      return kTfLiteError;                                                    \
    }                                                                         \
  } while (0)

namespace {

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

bool IsFloatOperator(const TfLiteContext* context, const TfLiteNode* node) {
  const auto input_type = context->tensors[node->inputs->data[0]].type;
  return IsFloat(input_type);
}

bool IsFloatOrUint8Operator(const TfLiteContext* context,
                            const TfLiteNode* node) {
  const auto input_type = context->tensors[node->inputs->data[0]].type;
  return IsFloatOrUInt8(input_type);
}

// Check if the operation requires explict conversion from int8 to uint8 values.
bool NeedInt8Conversion(const TfLiteContext* context, int builtin_code,
                        const TfLiteNode* node) {
  switch (builtin_code) {
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinDepthwiseConv2d:
    case kTfLiteBuiltinFullyConnected:
    case kTfLiteBuiltinL2Normalization: {
      const int input_id = node->inputs->data[0];
      const TfLiteType input_type = context->tensors[input_id].type;
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
    default:
      return false;
  }
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
    case kTfLiteBuiltinLstm:
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

// When using NN API version 1.0 or 1.1, the condition below must be true for
// quantized versions of the following ops:
// * CONV_2D
// * DEPTHWISE_CONV_2D
// * FULLY_CONNECTED (where filter actually stands for weights)
// The condition is relaxed and no longer required since version 1.2.
bool IsRestrictedScalesCompliant(const TfLiteContext* context,
                                 const TfLiteNode* node) {
  const int input_id = node->inputs->data[0];
  const int filter_id = node->inputs->data[1];
  const int output_id = node->outputs->data[0];
  const float input_scale = context->tensors[input_id].params.scale;
  const float filter_scale = context->tensors[filter_id].params.scale;
  const float output_scale = context->tensors[output_id].params.scale;
  return input_scale * filter_scale < output_scale;
}

constexpr int32_t kMinSdkVersionForNNAPI = 27;
constexpr int32_t kMinSdkVersionForNNAPI11 = 28;
constexpr int32_t kMinSdkVersionForNNAPI12 = 29;
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
ANeuralNetworksDevice* GetDeviceHandle(const char* device_name_ptr) {
  if (!device_name_ptr) return nullptr;
  ANeuralNetworksDevice* device_handle = nullptr;
  std::string device_name(device_name_ptr);
  uint32_t numDevices = 0;
  NnApiImplementation()->ANeuralNetworks_getDeviceCount(&numDevices);

  for (uint32_t i = 0; i < numDevices; i++) {
    ANeuralNetworksDevice* device = nullptr;
    const char* buffer = nullptr;
    NnApiImplementation()->ANeuralNetworks_getDevice(i, &device);
    NnApiImplementation()->ANeuralNetworksDevice_getName(device, &buffer);
    if (device_name == buffer) {
      device_handle = device;
      break;
    }
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

// RAII NN API Model Destructor for use with std::unique_ptr
struct NNFreeModel {
  void operator()(ANeuralNetworksModel* model) {
    NnApiImplementation()->ANeuralNetworksModel_free(model);
  }
};
// RAII NN API Compilation Destructor for use with std::unique_ptr
struct NNFreeCompilation {
  void operator()(ANeuralNetworksCompilation* model) {
    NnApiImplementation()->ANeuralNetworksCompilation_free(model);
  }
};

// RAII NN API Execution Destructor for use with std::unique_ptr
struct NNFreeExecution {
  void operator()(ANeuralNetworksExecution* execution) {
    NnApiImplementation()->ANeuralNetworksExecution_free(execution);
  }
};

// Manage NNAPI shared memory handle
class NNMemory {
 public:
#if defined __ANDROID__ || defined __unix__
  NNMemory(const NnApi* nnapi, const char* name, size_t size) {
    if (name && size > 0) {
      nnapi_ = nnapi;
      byte_size_ = size;
      fd_ = nnapi_->ASharedMemory_create(name, size);
      data_ptr_ = reinterpret_cast<uint8_t*>(
          mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
      nnapi_->ANeuralNetworksMemory_createFromFd(size, PROT_READ | PROT_WRITE,
                                                 fd_, 0, &nn_memory_handle_);
    }
  }
#else
  NNMemory(const NnApi* /*nnapi*/, const char* /*name*/, size_t /*size*/) {}
#endif

  ~NNMemory() {
#if defined __ANDROID__ || defined __unix__
    if (data_ptr_) {
      munmap(data_ptr_, byte_size_);
    }
    if (nn_memory_handle_) {
      nnapi_->ANeuralNetworksMemory_free(nn_memory_handle_);
    }
    if (fd_ > 0) close(fd_);
#endif
  }

  ANeuralNetworksMemory* get_handle() { return nn_memory_handle_; }
  uint8_t* get_data_ptr() { return data_ptr_; }

 private:
#if defined __ANDROID__ || defined __unix__
  const NnApi* nnapi_;
  int fd_ = 0;
  size_t byte_size_ = 0;
#endif
  uint8_t* data_ptr_ = nullptr;
  ANeuralNetworksMemory* nn_memory_handle_ = nullptr;
};  // namespace

// Track tensor indices to NN API tensor indices mapping.
class OperandMapping {
 public:
  // Given a TFLite index return the ANN index. If it doesn't exist
  // return -1.
  int lite_index_to_ann(int index) const {
    if (index < lite_tensor_to_ann_tensor_.size())
      return lite_tensor_to_ann_tensor_[index];
    else
      return -1;
  }

  // NN API uses non tensor operands instead of structs. This creates one
  // and returns the index. It uses a std::vector and resizes it as needed
  // keeping -1 to unmapped values. Intermediate tensors likely will not
  // be mapped.
  int add_new_non_tensor_operand() { return next_ann_tensor_index_++; }

  // Add a new mapping from `tflite_index` and return the NN API tensor index.
  int add_new_ann_tensor_index(int tflite_index) {
    if (tflite_index >= lite_tensor_to_ann_tensor_.size()) {
      lite_tensor_to_ann_tensor_.resize(tflite_index + 1, -1);
    }
    int new_tensor_index = next_ann_tensor_index_++;
    lite_tensor_to_ann_tensor_[tflite_index] = new_tensor_index;
    return new_tensor_index;
  }

  // Given a TFLite index returns a TFLite type to which a tensor must be
  // converted during copying the data to the memory allocated for NN API.
  // kTfLiteNoType means no conversion is needed.
  TfLiteType lite_index_to_ann_type_conversion(int index) const {
    if (index >= 0 && index < index_to_type_conversion_.size())
      return index_to_type_conversion_[index];
    else
      return kTfLiteNoType;
  }

  // Add a new mapping from TFLite index to a type conversion.
  void add_type_conversion(int tflite_index, TfLiteType tflite_type) {
    if (tflite_index >= index_to_type_conversion_.size()) {
      index_to_type_conversion_.resize(tflite_index + 1, kTfLiteNoType);
    }
    index_to_type_conversion_[tflite_index] = tflite_type;
  }

 private:
  // Next index of ann tensor
  int next_ann_tensor_index_ = 0;

  // Mapping from lite index. Use a std::vector for speed and code size
  // rather than a map.
  std::vector<int> lite_tensor_to_ann_tensor_;
  // Mapping from lite index to a type which tensor must be converted to during
  // the copying of the data to the memory allocated for NN API. kTfLiteNoType
  // means no conversion is needed. Use an std::vector for speed and code size
  // rather than a map.
  std::vector<TfLiteType> index_to_type_conversion_;
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
                 ANeuralNetworksModel* nn_model)
      : nnapi_(nnapi),
        context_(context),
        operand_mapping_(tensor_mapping),
        dequantize_mapping_(dequantize_mapping),
        nn_model_(nn_model) {}

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
                                     ANEURALNETWORKS_TENSOR_INT32);
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
          nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type));
      dequantized_ann_index = operand_mapping_->add_new_non_tensor_operand();

      // Add Dequantize operation.
      const uint32_t dequantize_input[1] = {static_cast<uint32_t>(ann_index)};
      const uint32_t dequantize_output[1] = {
          static_cast<uint32_t>(dequantized_ann_index)};
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context_, nnapi_->ANeuralNetworksModel_addOperation(
                        nn_model_, ANEURALNETWORKS_DEQUANTIZE, 1,
                        dequantize_input, 1, dequantize_output));
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
            augmented_outputs_.data()));
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
        nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type));
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
        nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type));
    const int ann_index = operand_mapping_->add_new_non_tensor_operand();
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_, nnapi_->ANeuralNetworksModel_setOperandValue(
                      nn_model_, ann_index, &value, sizeof(T)));
    augmented_inputs_.push_back(ann_index);
    return kTfLiteOk;
  }

  template <typename T>
  TfLiteStatus AddVectorOperand(const T* values, uint32_t num_values,
                                int32_t nn_type) {
    ANeuralNetworksOperandType operand_type{
        .type = nn_type, .dimensionCount = 1, .dimensions = &num_values};

    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type));

    const int ann_index = operand_mapping_->add_new_non_tensor_operand();
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_, nnapi_->ANeuralNetworksModel_setOperandValue(
                      nn_model_, ann_index, values, sizeof(T) * num_values));
    augmented_inputs_.push_back(ann_index);
    return kTfLiteOk;
  }

  TfLiteStatus AddFloat32OutputTensor(uint32_t dimension_count,
                                      const uint32_t* dimension_data,
                                      int* ann_index_out) {
    ANeuralNetworksOperandType operand_type{
        .type = ANEURALNETWORKS_TENSOR_FLOAT32,
        .dimensionCount = dimension_count,
        .dimensions = dimension_data,
    };
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type));
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
        nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type));

    if (nn_type == ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL) {
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context_,
          nnapi_->ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(
              nn_model_, ann_tensor_index, &ann_perchannel_params));
    }
    if (tensor->allocation_type == kTfLiteMmapRo) {
      // TODO(b/80630405): Use NNAPIAllocation.
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context_,
          nnapi_->ANeuralNetworksModel_setOperandValue(
              nn_model_, ann_tensor_index, tensor->data.raw, tensor->bytes));
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

  // The NNAPI model.
  ANeuralNetworksModel* const nn_model_;

  // Inputs and outputs for the current op. These are augmented in the sense
  // that NN API uses operands for all arguments, not just tensors, unlike
  // TensorFlow Lite.
  std::vector<uint32_t> augmented_inputs_;
  std::vector<uint32_t> augmented_outputs_;
};

struct NNAPIOpMappingArgs {
  TfLiteContext* context;
  NNAPIOpBuilder* builder;
  TfLiteNode* node;
  std::vector<int>* model_state_outputs;
  std::vector<int>* model_state_tfl_inputs;
};

// Mapping function simply returning the operation type without adding any
// additional parameter.
template <ANeuralNetworksOperationType OperationType>
ANeuralNetworksOperationType BasicMappingFn(
    const NNAPIOpMappingArgs& mapping_args) {
  return OperationType;
}

// The kernel that represents the node sub set of TF Lite being run on NN API.
class NNAPIDelegateKernel {
 public:
  NNAPIDelegateKernel() { nnapi_ = NnApiImplementation(); }

  typedef ANeuralNetworksOperationType (*MappingFn)(
      const NNAPIOpMappingArgs& mapping_args);

  // Return a function that knows how to translate a node into its operands
  // when called. You can use this function to see if a node is supported
  // (i.e. if the returned MappingFn is null, then the node is not supported).
  static MappingFn Map(const TfLiteContext* context, int builtin_code,
                       int version, int android_sdk_version,
                       const TfLiteNode* node) {
    switch (builtin_code) {
      case kTfLiteBuiltinAdd:
        if (version == 1) {
          if (!IsFloatOrUint8Operator(context, node)) {
            return nullptr;
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteAddParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_ADD;
          };
        }
        break;
      case kTfLiteBuiltinMul:
        if (version == 1) {
          if (!IsFloatOrUint8Operator(context, node)) {
            return nullptr;
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteMulParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_MUL;
          };
        }
        break;
      case kTfLiteBuiltinAveragePool2d:
        if (version == 1) {
          if (!IsFloatOrUint8Operator(context, node)) {
            return nullptr;
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            mapping_args.builder->AddPoolingParams(
                mapping_args.node->builtin_data);
            return ANEURALNETWORKS_AVERAGE_POOL_2D;
          };
        }
        break;
      case kTfLiteBuiltinMaxPool2d:
        if (version == 1) {
          if (!IsFloatOrUint8Operator(context, node)) {
            return nullptr;
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            mapping_args.builder->AddPoolingParams(
                mapping_args.node->builtin_data);
            return ANEURALNETWORKS_MAX_POOL_2D;
          };
        }
        break;
      case kTfLiteBuiltinL2Pool2d:
        if (version == 1) {
          if (!IsFloatOperator(context, node)) {
            return nullptr;
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            mapping_args.builder->AddPoolingParams(
                mapping_args.node->builtin_data);
            return ANEURALNETWORKS_L2_POOL_2D;
          };
        }
        break;
      case kTfLiteBuiltinConv2d:
        if (version <= 2) {
          if ((android_sdk_version < kMinSdkVersionForNNAPI12) &&
              (IsHybridOperator(context, builtin_code, node) ||
               !IsFloatOrUint8Operator(context, node))) {
            // Hybrid operators not supported before NNAPI 1.2.
            return nullptr;
          }
          if (android_sdk_version < kMinSdkVersionForNNAPI12) {
            // Per-channel quantized convolution not supported before NNAPI 1.2.
            const auto& filter_tensor = context->tensors[node->inputs->data[1]];
            if (filter_tensor.quantization.type == kTfLiteAffineQuantization) {
              TfLiteAffineQuantization* quantization_params =
                  static_cast<TfLiteAffineQuantization*>(
                      filter_tensor.quantization.params);
              if (quantization_params->scale->size > 1) {
                return nullptr;
              }
            }
          }
          const auto input_type = context->tensors[node->inputs->data[0]].type;
          if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
              input_type == kTfLiteUInt8 &&
              !IsRestrictedScalesCompliant(context, node)) {
            return nullptr;
          }
          auto builtin =
              reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
          if (node->inputs->size != 3) {
            // TODO(b/132950584): Add support for Conv2D with omitted bias
            return nullptr;
          }
          // NNAPI supports dilated Conv2D since NNAPI 1.2.
          if (builtin->dilation_width_factor != 1 ||
              builtin->dilation_height_factor != 1) {
            if (android_sdk_version < kMinSdkVersionForNNAPI12) {
              return nullptr;
            }
            return [](const NNAPIOpMappingArgs& mapping_args)
                       -> ANeuralNetworksOperationType {
              auto builtin = reinterpret_cast<TfLiteConvParams*>(
                  mapping_args.node->builtin_data);
              mapping_args.builder->AddScalarInt32Operand(builtin->padding);
              mapping_args.builder->AddScalarInt32Operand(
                  builtin->stride_width);
              mapping_args.builder->AddScalarInt32Operand(
                  builtin->stride_height);
              mapping_args.builder->AddScalarInt32Operand(builtin->activation);
              mapping_args.builder->AddScalarBoolOperand(
                  false);  // Use NHWC format
              mapping_args.builder->AddScalarInt32Operand(
                  builtin->dilation_width_factor);
              mapping_args.builder->AddScalarInt32Operand(
                  builtin->dilation_height_factor);
              return ANEURALNETWORKS_CONV_2D;
            };
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteConvParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->padding);
            mapping_args.builder->AddScalarInt32Operand(builtin->stride_width);
            mapping_args.builder->AddScalarInt32Operand(builtin->stride_height);
            mapping_args.builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_CONV_2D;
          };
        }
        break;
      case kTfLiteBuiltinDepthwiseConv2d:
        if (version == 1) {
          if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
              !IsFloatOrUint8Operator(context, node)) {
            return nullptr;
          }
          const auto input_type = context->tensors[node->inputs->data[0]].type;
          if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
              input_type == kTfLiteUInt8 &&
              !IsRestrictedScalesCompliant(context, node)) {
            return nullptr;
          }
          auto builtin =
              reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
          if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
              (builtin->dilation_width_factor != 1 ||
               builtin->dilation_height_factor != 1)) {
            return nullptr;
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteDepthwiseConvParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->padding);
            mapping_args.builder->AddScalarInt32Operand(builtin->stride_width);
            mapping_args.builder->AddScalarInt32Operand(builtin->stride_height);
            mapping_args.builder->AddScalarInt32Operand(
                builtin->depth_multiplier);
            mapping_args.builder->AddScalarInt32Operand(builtin->activation);
            if (builtin->dilation_width_factor != 1 ||
                builtin->dilation_height_factor != 1) {
              mapping_args.builder->AddScalarBoolOperand(
                  false);  // Use NHWC format
              mapping_args.builder->AddScalarInt32Operand(
                  builtin->dilation_width_factor);
              mapping_args.builder->AddScalarInt32Operand(
                  builtin->dilation_height_factor);
            }
            return ANEURALNETWORKS_DEPTHWISE_CONV_2D;
          };
        }
        break;
      case kTfLiteBuiltinFullyConnected:
        if (version == 1) {
          if (node->inputs->size != 3 ||
              node->inputs->data[2] == kOptionalTensor) {
            // TODO(b/132950584): Add support for FullyConnected with no bias.
            return nullptr;
          }
          const auto output_type =
              context->tensors[node->outputs->data[0]].type;
          if (output_type == kTfLiteInt16) {
            return nullptr;
          }
          if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
              (IsHybridOperator(context, builtin_code, node) ||
               !IsFloatOrUint8Operator(context, node))) {
            // Hybrid operators not supported before NNAPI 1.2.
            return nullptr;
          }
          const auto input_type = context->tensors[node->inputs->data[0]].type;
          if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
              input_type == kTfLiteUInt8 &&
              !IsRestrictedScalesCompliant(context, node)) {
            return nullptr;
          }
          auto builtin =
              reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
          if (builtin->keep_num_dims) {
            return nullptr;
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteFullyConnectedParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_FULLY_CONNECTED;
          };
        }
        break;
      case kTfLiteBuiltinSoftmax:
        if (version == 1) {
          const auto& input = context->tensors[node->outputs->data[0]];
          if (input.type != kTfLiteFloat32 && input.type != kTfLiteUInt8) {
            return nullptr;
          }
          const int input_rank = input.dims->size;
          if (input_rank > 4) return nullptr;
          // Before API level 29 only 2D and 4D input tensors were supported.
          if (android_sdk_version < kMinSdkVersionForNNAPI12) {
            if (input_rank != 2 && input_rank != 4) return nullptr;
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteSoftmaxParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarFloat32Operand(builtin->beta);
            // Optional scalar specifying the dimension the activation would be
            // performed on is not added. Default to -1.
            return ANEURALNETWORKS_SOFTMAX;
          };
        }
        break;
      case kTfLiteBuiltinReshape:
        if (version == 1) {
          if (!IsFloatOrUint8Operator(context, node)) {
            return nullptr;
          }
          // The shape input tensor must be constant.
          if ((node->inputs->size < 2) ||
              (context->tensors[node->inputs->data[1]].allocation_type !=
               kTfLiteMmapRo)) {
            return nullptr;
          }
          return BasicMappingFn<ANEURALNETWORKS_RESHAPE>;
        }
        break;
      case kTfLiteBuiltinResizeBilinear:
        if (version == 1) {
          const auto& input = context->tensors[node->inputs->data[0]];
          const auto output_dims =
              context->tensors[node->outputs->data[0]].dims;
          if (input.dims->size != 4) return nullptr;
          if (!IsFloatOrUint8Operator(context, node)) {
            return nullptr;
          }
          // The size input tensor must be constant.
          if ((node->inputs->size < 2) ||
              (context->tensors[node->inputs->data[1]].allocation_type !=
               kTfLiteMmapRo)) {
            return nullptr;
          }
          if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
              output_dims->data[1] != output_dims->data[2]) {
            // Require width == height due to driver differences in NNAPI < 1.2
            return nullptr;
          }
          auto builtin =
              reinterpret_cast<TfLiteResizeBilinearParams*>(node->builtin_data);
          if (builtin->align_corners) {
            // NNAPI does not support align_corners == true.
            return nullptr;
          }
          if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
              input.type != kTfLiteFloat32) {
            // NNAPI 1.0 & 1.1 only supports float input.
            return nullptr;
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            const int output_id = mapping_args.node->outputs->data[0];
            auto& output = mapping_args.context->tensors[output_id];
            const int output_height = output.dims->data[1];
            const int output_width = output.dims->data[2];
            mapping_args.builder->AddScalarInt32Operand(output_width);
            mapping_args.builder->AddScalarInt32Operand(output_height);
            return ANEURALNETWORKS_RESIZE_BILINEAR;
          };
        }
        break;
      case kTfLiteBuiltinSqueeze:
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI11) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteSqueezeParams*>(
                mapping_args.node->builtin_data);
            // Note that we add the squeeze dimensions even if the dimensions
            // were unspecified (empty), as NNAPI requires the operand.
            mapping_args.builder->AddVectorInt32Operand(
                builtin->num_squeeze_dims ? builtin->squeeze_dims : nullptr,
                static_cast<uint32_t>(builtin->num_squeeze_dims));
            return ANEURALNETWORKS_SQUEEZE;
          };
        }
        break;
      case kTfLiteBuiltinUnidirectionalSequenceLstm:
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12) {
          if (IsHybridOperator(context, builtin_code, node)) {
            // Hybrid version of this op is not supported by NN API.
            return nullptr;
          }
          if (node->inputs->size != 20 && node->inputs->size != 24) {
            return nullptr;
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin =
                reinterpret_cast<TfLiteUnidirectionalSequenceLSTMParams*>(
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
                if (input_index != kOptionalTensor) {
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

            return ANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_LSTM;
          };
        }
        break;
      case kTfLiteBuiltinL2Normalization: {
        if (version == 1) {
          if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
              !IsFloatOperator(context, node)) {
            return nullptr;
          }
          auto builtin =
              reinterpret_cast<TfLiteL2NormParams*>(node->builtin_data);
          if (builtin->activation == kTfLiteActNone) {
            return BasicMappingFn<ANEURALNETWORKS_L2_NORMALIZATION>;
          }
        }
        break;
      }
      case kTfLiteBuiltinLocalResponseNormalization:
        if (version == 1) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteLocalResponseNormParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->radius);
            mapping_args.builder->AddScalarFloat32Operand(builtin->bias);
            mapping_args.builder->AddScalarFloat32Operand(builtin->alpha);
            mapping_args.builder->AddScalarFloat32Operand(builtin->beta);
            return ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION;
          };
        }
        break;
      case kTfLiteBuiltinLshProjection:
        if (version == 1) {
          // NNAPI does not support sparse projection correctly (b/111751836).
          if (reinterpret_cast<TfLiteLSHProjectionParams*>(node->builtin_data)
                  ->type == kTfLiteLshProjectionSparse) {
            return nullptr;
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteLSHProjectionParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->type);
            return ANEURALNETWORKS_LSH_PROJECTION;
          };
        }
        break;
      case kTfLiteBuiltinConcatenation:
        if (version == 1 &&
            reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data)
                    ->activation == kTfLiteActNone) {
          if (context->tensors[node->inputs->data[0]].type == kTfLiteUInt8 &&
              android_sdk_version < kMinSdkVersionForNNAPI12) {
            // NNAPI 1.0-1 only supported concatenating quantized tensor of the
            // same scale and offset.
            auto first_param = context->tensors[node->inputs->data[0]].params;
            for (int i = 1; i < node->inputs->size; i++) {
              auto curr_param = context->tensors[node->inputs->data[i]].params;
              if (curr_param.scale != first_param.scale ||
                  curr_param.zero_point != first_param.zero_point) {
                return nullptr;
              }
            }
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteConcatenationParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->axis);
            return ANEURALNETWORKS_CONCATENATION;
          };
        }
        break;
      case kTfLiteBuiltinDequantize:
        if (version == 1 || version == 2) {
          const auto& input = context->tensors[node->inputs->data[0]];
          if (input.type == kTfLiteFloat16) {
            return nullptr;
          }
          const auto zero_point = input.params.zero_point;
          // NN API supports int8 type since version 1.2 but only for symmetric
          // quantization.
          if (input.type == kTfLiteInt8 &&
              (zero_point != 0 ||
               android_sdk_version < kMinSdkVersionForNNAPI12)) {
            return nullptr;
          }
          return BasicMappingFn<ANEURALNETWORKS_DEQUANTIZE>;
        }
        break;
      case kTfLiteBuiltinFloor:
        if (version == 1) {
          return BasicMappingFn<ANEURALNETWORKS_FLOOR>;
        }
        break;
      case kTfLiteBuiltinRelu:
        if (version == 1) {
          if (!IsFloatOrUint8Operator(context, node)) {
            return nullptr;
          }
          return BasicMappingFn<ANEURALNETWORKS_RELU>;
        }
        break;
      case kTfLiteBuiltinReluN1To1:
        if (version == 1) {
          if (!IsFloatOrUint8Operator(context, node)) {
            return nullptr;
          }
          return BasicMappingFn<ANEURALNETWORKS_RELU1>;
        }
        break;
      case kTfLiteBuiltinRelu6:
        if (version == 1) {
          if (!IsFloatOrUint8Operator(context, node)) {
            return nullptr;
          }
          return BasicMappingFn<ANEURALNETWORKS_RELU6>;
        }
        break;
      case kTfLiteBuiltinLogistic:
        if (version == 1) {
          if (!IsFloatOrUint8Operator(context, node)) {
            return nullptr;
          }
          return BasicMappingFn<ANEURALNETWORKS_LOGISTIC>;
        }
        break;
      case kTfLiteBuiltinTanh:
        // TODO(miaowang): add additional checks for the parameters.
        if (version == 1 &&
            context->tensors[node->inputs->data[0]].type == kTfLiteFloat32) {
          // NNAPI only support float tanh.
          return BasicMappingFn<ANEURALNETWORKS_TANH>;
        }
        break;
      case kTfLiteBuiltinSub:
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI11 &&
            context->tensors[node->inputs->data[0]].type == kTfLiteFloat32) {
          // NNAPI only support float sub.
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteSubParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_SUB;
          };
        }
        break;
      case kTfLiteBuiltinDiv:
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI11 &&
            context->tensors[node->inputs->data[0]].type == kTfLiteFloat32) {
          // NNAPI only support float div.
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteDivParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_DIV;
          };
        }
        break;
      case kTfLiteBuiltinPad:
      case kTfLiteBuiltinPadv2: {
        const TfLiteType input_type =
            context->tensors[node->inputs->data[0]].type;
        if (version == 1 &&
            (input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8)) {
          const TfLiteIntArrayView input_shape(
              context->tensors[node->inputs->data[0]].dims);
          if (HasZeroes(input_shape)) {
            // NN API pad ops do not support input tensors with no elements
            return nullptr;
          }
          if (node->inputs->size == 2 &&
              android_sdk_version >= kMinSdkVersionForNNAPI11 &&
              (context->tensors[node->inputs->data[0]].type == kTfLiteFloat32 ||
               android_sdk_version >= kMinSdkVersionForNNAPI12)) {
            // NNAPI does not support specifying the padding value.
            // Before 1.2, NNAPI pads physical zero for quantized tensors, so
            // only delegate float pad to NNAPI. NNAPI 1.2 onwards pads with
            // zero-point, so delegate quantized pad as well.
            return BasicMappingFn<ANEURALNETWORKS_PAD>;
          } else if (node->inputs->size == 3 &&
                     android_sdk_version >= kMinSdkVersionForNNAPI12) {
            const int constant_value_id = node->inputs->data[2];
            if (constant_value_id == kOptionalTensor) {
              return BasicMappingFn<ANEURALNETWORKS_PAD>;
            }
            return BasicMappingFn<ANEURALNETWORKS_PAD_V2>;
          }
        }
      } break;
      case kTfLiteBuiltinUnidirectionalSequenceRnn:
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12) {
          if (IsHybridOperator(context, builtin_code, node)) {
            // Hybrid version of this op is not supported by NN API.
            return nullptr;
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteSequenceRNNParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->activation);
            mapping_args.builder->AddScalarInt32Operand(builtin->time_major);
            return ANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_RNN;
          };
        }
        break;
      case kTfLiteBuiltinSpaceToBatchNd:
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI11) {
          return BasicMappingFn<ANEURALNETWORKS_SPACE_TO_BATCH_ND>;
        }
        break;
      case kTfLiteBuiltinStridedSlice:
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI11) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteStridedSliceParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->begin_mask);
            mapping_args.builder->AddScalarInt32Operand(builtin->end_mask);
            mapping_args.builder->AddScalarInt32Operand(
                builtin->shrink_axis_mask);
            return ANEURALNETWORKS_STRIDED_SLICE;
          };
        }
        break;
      case kTfLiteBuiltinTranspose:
        // Note that the permutation input tensor value dictates the output
        // dimensions.
        // TODO(b/110888333): Support dynamically-sized tensors in delegates.
        if ((version == 1) &&
            (android_sdk_version >= kMinSdkVersionForNNAPI11) &&
            (node->inputs->size > 1) &&
            (context->tensors[node->inputs->data[1]].allocation_type ==
             kTfLiteMmapRo)) {
          return BasicMappingFn<ANEURALNETWORKS_TRANSPOSE>;
        }
        break;
      case kTfLiteBuiltinAbs:
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12) {
          return BasicMappingFn<ANEURALNETWORKS_ABS>;
        }
        break;
      case kTfLiteBuiltinExp:
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12) {
          return BasicMappingFn<ANEURALNETWORKS_EXP>;
        }
        break;
      case kTfLiteBuiltinLog:
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12) {
          return BasicMappingFn<ANEURALNETWORKS_LOG>;
        }
        break;
      case kTfLiteBuiltinRsqrt:
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12) {
          return BasicMappingFn<ANEURALNETWORKS_RSQRT>;
        }
        break;
      case kTfLiteBuiltinPow:
        // NN API only supports float inputs to this op.
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            context->tensors[node->inputs->data[0]].type == kTfLiteFloat32) {
          return BasicMappingFn<ANEURALNETWORKS_POW>;
        }
        break;
      case kTfLiteBuiltinSin:
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12) {
          return BasicMappingFn<ANEURALNETWORKS_SIN>;
        }
        break;
      case kTfLiteBuiltinSqrt:
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12) {
          return BasicMappingFn<ANEURALNETWORKS_SQRT>;
        }
        break;
      case kTfLiteBuiltinRnn:
        // NNAPI only support float32 weights.
        if (version == 1 && node->inputs->size == 5 &&
            context->tensors[node->inputs->data[/*kWeightsTensor*/ 1]].type ==
                kTfLiteFloat32) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            // NNAPI need both state_in and state_out.
            int ann_index;
            mapping_args.builder->AddStateFloat32Tensor(
                mapping_args.node->inputs->data[/*kHiddenStateTensor*/ 4],
                &ann_index);
            mapping_args.model_state_outputs->push_back(ann_index);
            mapping_args.model_state_tfl_inputs->push_back(
                mapping_args.node->inputs->data[/*kHiddenStateTensor*/ 4]);
            auto builtin = reinterpret_cast<TfLiteRNNParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_RNN;
          };
        }
        break;
      case kTfLiteBuiltinSvdf:
        // NNAPI only support float32 weights.
        // Only delegate to NNAPI 1.1, as SVDF does not support rank > 1 on 1.0.
        if (version == 1 && node->inputs->size == 5 &&
            android_sdk_version >= kMinSdkVersionForNNAPI11 &&
            context->tensors[node->inputs->data[/*kWeightsFeatureTensor*/ 1]]
                    .type == kTfLiteFloat32) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            // NNAPI need both state_in and state_out.
            int ann_index;
            mapping_args.builder->AddStateFloat32Tensor(
                mapping_args.node->inputs
                    ->data[/*kInputActivationStateTensor*/ 4],
                &ann_index);
            mapping_args.model_state_outputs->push_back(ann_index);
            mapping_args.model_state_tfl_inputs->push_back(
                mapping_args.node->inputs
                    ->data[/*kInputActivationStateTensor*/ 4]);

            auto builtin = reinterpret_cast<TfLiteSVDFParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->rank);
            mapping_args.builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_SVDF;
          };
        }
        break;
      case kTfLiteBuiltinLstm:
        // TODO(miaowang): add loggings to indicate why the op is rejected.
        if (version == 1) {
          if (android_sdk_version < kMinSdkVersionForNNAPI11) {
            // Only delegate to NNAPI 1.1+, as 1.0 has a bug for optional
            // tensors which would affect LSTM.
            return nullptr;
          }
          if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
              IsHybridOperator(context, builtin_code, node)) {
            // Hybrid operators not supported before NNAPI 1.2.
            return nullptr;
          }
          // TODO(levp): name the constants for number of inputs in LSTM kernel.
          if (node->inputs->size != 20 && node->inputs->size != 24) {
            return nullptr;
          }
          if (node->inputs->size == 24 &&
              android_sdk_version < kMinSdkVersionForNNAPI12) {
            // LSTM with layer norm introduced in API level 29
            return nullptr;
          }
          const TfLiteType weight_type =
              context
                  ->tensors[node->inputs
                                ->data[/*kInputToOutputWeightsTensor*/ 4]]
                  .type;
          if (weight_type != kTfLiteFloat32 && weight_type != kTfLiteUInt8) {
            return nullptr;
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
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
                mapping_args.node->inputs
                    ->data[/*kInputActivationStateTensor*/ 18],
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
                if (input_index != kOptionalTensor) {
                  mapping_args.builder->AddTensorInput(input_index, hybrid_op);
                } else {
                  mapping_args.builder->AddVectorFloat32Operand(nullptr, 0);
                }
              }
            }

            return ANEURALNETWORKS_LSTM;
          };
        }
        break;
      case kTfLiteBuiltinMean:
        // NNAPI does not support generating a scalar as output for MEAN.
        if (version == 1 &&
            ((android_sdk_version >= kMinSdkVersionForNNAPI11 &&
              context->tensors[node->inputs->data[0]].type == kTfLiteFloat32) ||
             (android_sdk_version >= kMinSdkVersionForNNAPI12 &&
              context->tensors[node->inputs->data[0]].type == kTfLiteUInt8)) &&
            context->tensors[node->outputs->data[0]].dims->size > 0) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteReducerParams*>(
                mapping_args.node->builtin_data);
            int32_t keep_dims = 0;
            if (builtin->keep_dims) keep_dims = 1;
            mapping_args.builder->AddScalarInt32Operand(keep_dims);
            return ANEURALNETWORKS_MEAN;
          };
        }
        break;
      case kTfLiteBuiltinEmbeddingLookup:
        // NNAPI only support float32 values.
        if (version == 1 &&
            context->tensors[node->inputs->data[1]].type == kTfLiteFloat32) {
          return BasicMappingFn<ANEURALNETWORKS_EMBEDDING_LOOKUP>;
        }
        break;
      case kTfLiteBuiltinHashtableLookup:
        // NNAPI only support float32 output.
        if (version == 1 &&
            context->tensors[node->outputs->data[0]].type == kTfLiteFloat32) {
          return BasicMappingFn<ANEURALNETWORKS_HASHTABLE_LOOKUP>;
        }
        break;
      case kTfLiteBuiltinMaximum: {
        const auto input_type = context->tensors[node->inputs->data[0]].type;
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            (input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
             input_type == kTfLiteInt32)) {
          return BasicMappingFn<ANEURALNETWORKS_MAXIMUM>;
        }
      } break;
      case kTfLiteBuiltinMinimum: {
        const auto input_type = context->tensors[node->inputs->data[0]].type;
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            (input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
             input_type == kTfLiteInt32)) {
          return BasicMappingFn<ANEURALNETWORKS_MINIMUM>;
        }
      } break;
      case kTfLiteBuiltinPrelu:
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12) {
          if (!IsFloatOrUint8Operator(context, node)) {
            return nullptr;
          }
          return BasicMappingFn<ANEURALNETWORKS_PRELU>;
        }
        break;
      case kTfLiteBuiltinTile: {
        // NN API doesn't support int64 and boolean inputs to this op
        const auto input_type = context->tensors[node->inputs->data[0]].type;
        const auto multipliers_type =
            context->tensors[node->inputs->data[1]].type;
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            (input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
             input_type == kTfLiteInt32) &&
            (multipliers_type == kTfLiteInt32)) {
          return BasicMappingFn<ANEURALNETWORKS_TILE>;
        }
      } break;
      case kTfLiteBuiltinLogicalOr: {
        const auto input_type = context->tensors[node->inputs->data[0]].type;
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            input_type == kTfLiteBool) {
          return BasicMappingFn<ANEURALNETWORKS_LOGICAL_OR>;
        }
      } break;
      case kTfLiteBuiltinLogicalAnd: {
        const auto input_type = context->tensors[node->inputs->data[0]].type;
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            input_type == kTfLiteBool) {
          return BasicMappingFn<ANEURALNETWORKS_LOGICAL_AND>;
        }
      } break;
      case kTfLiteBuiltinLogicalNot: {
        const auto input_type = context->tensors[node->inputs->data[0]].type;
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            input_type == kTfLiteBool) {
          return BasicMappingFn<ANEURALNETWORKS_LOGICAL_NOT>;
        }
      } break;
      case kTfLiteBuiltinLess: {
        const auto input_type = context->tensors[node->inputs->data[0]].type;
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            (input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
             input_type == kTfLiteBool || input_type == kTfLiteInt32)) {
          return BasicMappingFn<ANEURALNETWORKS_LESS>;
        }
      } break;
      case kTfLiteBuiltinLessEqual: {
        const auto input_type = context->tensors[node->inputs->data[0]].type;
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            (input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
             input_type == kTfLiteBool || input_type == kTfLiteInt32)) {
          return BasicMappingFn<ANEURALNETWORKS_LESS_EQUAL>;
        }
      } break;
      case kTfLiteBuiltinGreater: {
        const auto input_type = context->tensors[node->inputs->data[0]].type;
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            (input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
             input_type == kTfLiteBool || input_type == kTfLiteInt32)) {
          return BasicMappingFn<ANEURALNETWORKS_GREATER>;
        }
      } break;
      case kTfLiteBuiltinGreaterEqual: {
        const auto input_type = context->tensors[node->inputs->data[0]].type;
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            (input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
             input_type == kTfLiteBool || input_type == kTfLiteInt32)) {
          return BasicMappingFn<ANEURALNETWORKS_GREATER_EQUAL>;
        }
      } break;
      case kTfLiteBuiltinEqual: {
        const auto input_type = context->tensors[node->inputs->data[0]].type;
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            (input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
             input_type == kTfLiteBool || input_type == kTfLiteInt32)) {
          return BasicMappingFn<ANEURALNETWORKS_EQUAL>;
        }
      } break;
      case kTfLiteBuiltinNotEqual: {
        const auto input_type = context->tensors[node->inputs->data[0]].type;
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            (input_type == kTfLiteFloat32 || input_type == kTfLiteUInt8 ||
             input_type == kTfLiteBool || input_type == kTfLiteInt32)) {
          return BasicMappingFn<ANEURALNETWORKS_NOT_EQUAL>;
        }
      } break;
      case kTfLiteBuiltinNeg: {
        const auto input_type = context->tensors[node->inputs->data[0]].type;
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            (input_type == kTfLiteFloat32 || input_type == kTfLiteInt32)) {
          return BasicMappingFn<ANEURALNETWORKS_NEG>;
        }
      } break;
      case kTfLiteBuiltinTopkV2: {
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12) {
          const auto& input = context->tensors[node->outputs->data[0]];
          const auto& k_param = context->tensors[node->outputs->data[1]];
          if ((input.type == kTfLiteFloat32 || input.type == kTfLiteInt32 ||
               input.type == kTfLiteInt8) &&
              (k_param.type == kTfLiteInt32 &&
               k_param.allocation_type == kTfLiteMmapRo)) {
            return [](const NNAPIOpMappingArgs& mapping_args)
                       -> ANeuralNetworksOperationType {
              const TfLiteTensor& k_param =
                  mapping_args.context
                      ->tensors[mapping_args.node->inputs->data[1]];
              mapping_args.builder->AddScalarInt32Operand(*k_param.data.i32);
              return ANEURALNETWORKS_TOPK_V2;
            };
          } else {
            return nullptr;
          }
        }
      } break;
      case kTfLiteBuiltinSelect: {
        const auto value_type = context->tensors[node->inputs->data[1]].type;
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            (value_type == kTfLiteFloat32 || value_type == kTfLiteUInt8 ||
             value_type == kTfLiteInt32)) {
          TfLiteIntArray* condition_shape =
              context->tensors[node->inputs->data[0]].dims;
          TfLiteIntArray* input_shape =
              context->tensors[node->inputs->data[1]].dims;
          // The Android Q-variant of select does not support broadcasting.
          if (!TfLiteIntArrayEqual(condition_shape, input_shape)) {
            return nullptr;
          }
          return BasicMappingFn<ANEURALNETWORKS_SELECT>;
        }
      } break;
      case kTfLiteBuiltinGather: {
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12) {
          const auto& input = context->tensors[node->inputs->data[0]];
          const auto& positions = context->tensors[node->inputs->data[1]];

          auto is_supported_input_type = [](const TfLiteTensor& t) {
            return (t.type == kTfLiteFloat32 || t.type == kTfLiteFloat16 ||
                    t.type == kTfLiteInt32 || t.type == kTfLiteUInt8);
          };

          if (!is_supported_input_type(input) ||
              !is_supported_input_type(positions)) {
            return nullptr;
          }

          // 0-dimension args are not supported by NNAPI.
          if (positions.dims->size == 0) {
            return nullptr;
          }

          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteGatherParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddTensorInput(
                mapping_args.node->inputs->data[0],
                /* hybrid_op */ false,
                /* scalar_as_tensor */ false);

            mapping_args.builder->AddScalarInt32Operand(builtin->axis);

            mapping_args.builder->AddTensorInput(
                mapping_args.node->inputs->data[1],
                /* hybrid_op */ false,
                /* scalar_as_tensor */ false);

            return ANEURALNETWORKS_GATHER;
          };
        }
      } break;
      case kTfLiteBuiltinBidirectionalSequenceLstm:
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12) {
          if (IsHybridOperator(context, builtin_code, node)) {
            // Hybrid version of this op is not supported by NN API.
            return nullptr;
          }
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin =
                reinterpret_cast<TfLiteBidirectionalSequenceLSTMParams*>(
                    mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->activation);
            mapping_args.builder->AddScalarFloat32Operand(builtin->cell_clip);
            mapping_args.builder->AddScalarFloat32Operand(builtin->proj_clip);
            mapping_args.builder->AddScalarBoolOperand(builtin->merge_outputs);
            mapping_args.builder->AddScalarBoolOperand(builtin->time_major);
            // TF Lite doesn't support layer normalization in bidirectional
            // sequence LSTM, so we insert optional tensors for NNAPI
            for (int i = 0; i < 8; ++i) {
              mapping_args.builder->AddVectorFloat32Operand(nullptr, 0);
            }
            return ANEURALNETWORKS_BIDIRECTIONAL_SEQUENCE_LSTM;
          };
        }
        break;
      case kTfLiteBuiltinExpandDims: {
        const auto input_type = context->tensors[node->inputs->data[0]].type;
        const auto axis = context->tensors[node->inputs->data[1]];
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI12 &&
            (input_type == kTfLiteFloat16 || input_type == kTfLiteFloat32 ||
             input_type == kTfLiteInt32 || input_type == kTfLiteUInt8) &&
            // TFLite supports axis also as int64 but NNAPI only int32
            (axis.type == kTfLiteInt32 &&
             axis.allocation_type == kTfLiteMmapRo)) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            const TfLiteTensor& axis_param =
                mapping_args.context
                    ->tensors[mapping_args.node->inputs->data[1]];
            mapping_args.builder->AddScalarInt32Operand(*axis_param.data.i32);
            return ANEURALNETWORKS_EXPAND_DIMS;
          };
        }
      } break;
      default:
        // All other operators are not mapped.
        return nullptr;
    }
    return nullptr;
  }

  // Initialize the kernel (a NN model).
  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) {
    for (auto node_index : TfLiteIntArrayView(params->nodes_to_replace)) {
      nodes_.push_back(node_index);
    }

    const auto delegate_options =
        StatefulNnApiDelegate::GetOptions(params->delegate);
    const char* device_name_ptr = delegate_options.accelerator_name;
    // user specified an acclelerator to use.
    if (nnapi_->android_sdk_version >= kMinSdkVersionForNNAPI12 &&
        device_name_ptr != nullptr) {
      nnapi_device_ = GetDeviceHandle(device_name_ptr);
      if (nnapi_device_ == nullptr) {
        context->ReportError(context,
                             "Could not find the specified accelerator: %s.",
                             device_name_ptr);
        return kTfLiteError;
      }
    }

    // Mark the handle backed tensors.
    tensor_memory_map_ =
        &StatefulNnApiDelegate::GetTensorMemoryMap(params->delegate);

    if (!nn_model_) {
      ANeuralNetworksModel* model = nullptr;
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context, nnapi_->ANeuralNetworksModel_create(&model));
      nn_model_.reset(model);

      TF_LITE_ENSURE_STATUS(
          BuildGraph(context, params->input_tensors, params->output_tensors));
    }

    if (!nn_compilation_) {
      ANeuralNetworksCompilation* compilation = nullptr;
      if (nnapi_device_ != nullptr) {
        // Compile for the selected accelerator.
        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            context, nnapi_->ANeuralNetworksCompilation_createForDevices(
                         nn_model_.get(), &nnapi_device_, 1, &compilation));
      } else {
        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            context, nnapi_->ANeuralNetworksCompilation_create(nn_model_.get(),
                                                               &compilation));
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
        RETURN_TFLITE_ERROR_IF_NN_ERROR(context, preference_result);
      }

      const char* cache_dir = delegate_options.cache_dir;
      const char* model_token = delegate_options.model_token;
      if (nnapi_->android_sdk_version >= kMinSdkVersionForNNAPI12 &&
          cache_dir && model_token) {
        // Compilation caching could be enabled, try construct the uint8 token.
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
        RETURN_TFLITE_ERROR_IF_NN_ERROR(context, set_caching_result);
      }
      const int finish_result =
          nnapi_->ANeuralNetworksCompilation_finish(compilation);
      if (finish_result != ANEURALNETWORKS_NO_ERROR) {
        nnapi_->ANeuralNetworksCompilation_free(compilation);
        compilation = nullptr;
      }
      RETURN_TFLITE_ERROR_IF_NN_ERROR(context, finish_result);
      nn_compilation_.reset(compilation);
    }
    return kTfLiteOk;
  }

  TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
    ANeuralNetworksExecution* execution = nullptr;
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context, nnapi_->ANeuralNetworksExecution_create(nn_compilation_.get(),
                                                         &execution));
    std::unique_ptr<ANeuralNetworksExecution, NNFreeExecution>
        execution_unique_ptr(execution);

    // Set the input tensor buffers. Note: we access tflite tensors using
    // absolute indices but NN api indices inputs by relative indices.
    int relative_input_index = 0;

    size_t input_offset = 0;
    for (auto absolute_input_index : TfLiteIntArrayView(node->inputs)) {
      if (absolute_input_index == kOptionalTensor) {
        continue;
      }
      TfLiteTensor* tensor = &context->tensors[absolute_input_index];
      // TODO(miaowang): make sure the delegation works with dequantized weights
      // as intermediate tensors.
      if (tensor->allocation_type != kTfLiteMmapRo) {
        if (tensor->buffer_handle != kTfLiteNullBufferHandle &&
            tensor->buffer_handle < tensor_memory_map_->size()) {
          RETURN_TFLITE_ERROR_IF_NN_ERROR(
              context, nnapi_->ANeuralNetworksExecution_setInputFromMemory(
                           execution, relative_input_index, nullptr,
                           tensor_memory_map_->at(tensor->buffer_handle).memory,
                           0, tensor->bytes));
          relative_input_index++;
          continue;
        }
        TfLiteType ann_type_equivalent =
            operand_mapping_.lite_index_to_ann_type_conversion(
                absolute_input_index);
        int tensor_size = 0;
        if (ann_type_equivalent != kTfLiteNoType) {
          if (tensor->type == kTfLiteUInt8 &&
              ann_type_equivalent == kTfLiteInt32) {
            for (int i = 0; i < NumElements(tensor); ++i) {
              reinterpret_cast<int32_t*>(nn_input_memory_->get_data_ptr() +
                                         input_offset)[i] =
                  static_cast<const int32_t>(tensor->data.raw_const[i]);
            }
          } else if (tensor->type == kTfLiteInt8 &&
                     ann_type_equivalent == kTfLiteUInt8) {
            // Explicitly convert int8 values to uint8 values.
            uint8_t* input_ptr = reinterpret_cast<uint8_t*>(
                nn_input_memory_->get_data_ptr() + input_offset);
            for (int i = 0; i < NumElements(tensor); ++i) {
              input_ptr[i] = static_cast<const uint8_t>(
                  static_cast<int32_t>(tensor->data.int8[i]) + 128);
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
                  nn_input_memory_->get_handle(), input_offset, tensor_size));
        } else {
          // copy data to pre-allocated shared memory.
          memcpy(nn_input_memory_->get_data_ptr() + input_offset,
                 tensor->data.raw, tensor->bytes);
          RETURN_TFLITE_ERROR_IF_NN_ERROR(
              context,
              nnapi_->ANeuralNetworksExecution_setInputFromMemory(
                  execution, relative_input_index, nullptr,
                  nn_input_memory_->get_handle(), input_offset, tensor->bytes));
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
      TfLiteTensor* tensor = &context->tensors[output_index];
      if (tensor->buffer_handle != kTfLiteNullBufferHandle &&
          tensor->buffer_handle < tensor_memory_map_->size()) {
        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            context, nnapi_->ANeuralNetworksExecution_setOutputFromMemory(
                         execution, relative_output_index, nullptr,
                         tensor_memory_map_->at(tensor->buffer_handle).memory,
                         0, tensor->bytes));

      } else {
        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            context,
            nnapi_->ANeuralNetworksExecution_setOutputFromMemory(
                execution, relative_output_index, nullptr,
                nn_output_memory_->get_handle(), output_offset, tensor->bytes));
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
          context, nnapi_->ANeuralNetworksExecution_setOutput(
                       execution, relative_output_index, nullptr,
                       tensor->data.raw, tensor->bytes));
      relative_output_index++;
    }
    // Invoke ANN in blocking fashion.
    if (nnapi_->android_sdk_version < kMinSdkVersionForNNAPI12) {
      ANeuralNetworksEvent* event = nullptr;
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context,
          nnapi_->ANeuralNetworksExecution_startCompute(execution, &event));
      const int wait_result = nnapi_->ANeuralNetworksEvent_wait(event);
      nnapi_->ANeuralNetworksEvent_free(event);
      RETURN_TFLITE_ERROR_IF_NN_ERROR(context, wait_result);
    } else {
      // Use synchronous execution for NNAPI 1.2+.
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context, nnapi_->ANeuralNetworksExecution_compute(execution));
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
        for (int i = 0; i < NumElements(tensor); ++i) {
          output_ptr[i] =
              static_cast<uint8_t>(static_cast<int32_t>(output_ptr[i]) - 128);
        }
      }
      memcpy(tensor->data.raw,
             nn_output_memory_->get_data_ptr() + output_offset, tensor->bytes);
      output_offset += tensor->bytes;
      output_offset += getNumPaddingBytes(tensor->bytes);
    }

    return kTfLiteOk;
  }

 private:
  // Access to NNApi.
  const NnApi* nnapi_;
  // ANN device handle.
  ANeuralNetworksDevice* nnapi_device_ = nullptr;
  // ANN API state.
  std::unique_ptr<ANeuralNetworksModel, NNFreeModel> nn_model_;
  std::unique_ptr<ANeuralNetworksCompilation, NNFreeCompilation>
      nn_compilation_;
  // Node indices that this delegate is responsible for. Indices here
  // indexes into the nodes array in the TfLiteContext.
  std::vector<int> nodes_;
  // Track indices we use
  OperandMapping operand_mapping_;
  // Track memory map
  const std::vector<StatefulNnApiDelegate::MemoryRegistration>*
      tensor_memory_map_;
  std::vector<int> model_state_outputs_;
  std::vector<int> model_state_tfl_inputs_;

  std::unique_ptr<NNMemory> nn_input_memory_;
  std::unique_ptr<NNMemory> nn_output_memory_;

  void AddDequantizeOperatorsWhereNeeded(const TfLiteContext* context,
                                         int builtin_code,
                                         const TfLiteNode* node,
                                         NNAPIOpBuilder* builder) {
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

  TfLiteStatus AddOpsAndTensors(TfLiteContext* context) {
    DequantizeMapping dequantize_mapping;
    // The operand builder allows creating a single op. It is created outside
    // the for loop to avoid reallocating the vectors.
    NNAPIOpBuilder builder(nnapi_, context, &operand_mapping_,
                           &dequantize_mapping, nn_model_.get());
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

      // Map inputs to NN API tensor indices.
      for (int input_pos = 0; input_pos < node->inputs->size; ++input_pos) {
        const auto input_index = node->inputs->data[input_pos];
        if (need_int8_conversion &&
            (input_pos == 0 ||
             reg->builtin_code == kTfLiteBuiltinFullyConnected)) {
          // Only selected inputs require int8 conversion.
          TF_LITE_ENSURE_STATUS(builder.AddTensorInput(
              input_index, hybrid_op,
              input_tensor_flags | NN_TENSOR_FLAG_INT8_CONVERSION));
          continue;
        }
        if (reg->builtin_code == kTfLiteBuiltinLstm && input_pos >= 20) {
          // Skip layer normalization weights. They are added in the Map
          // function (after all the other inputs added there) since layer
          // normalization weights are the last four inputs of the LSTM op in
          // NNAPI.
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
          if (input_index == kOptionalTensor) {
            TF_LITE_ENSURE_STATUS(builder.AddVectorFloat32Operand(nullptr, 0));
            continue;
          }
        }
        // Pad and Padv2 have an optional parameter for a pad value which has
        // to be converted to a scalar type in NN API.
        if ((reg->builtin_code == kTfLiteBuiltinPadv2 ||
             reg->builtin_code == kTfLiteBuiltinPad) &&
            node->inputs->size == 3 && input_pos == 2) {
          const int constant_value_id = node->inputs->data[2];
          if (constant_value_id == kOptionalTensor) {
            continue;
          }
          const TfLiteTensor constant_value =
              context->tensors[constant_value_id];

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
            default:
              context->ReportError(
                  context, "Unsupported type of pad value for pad_v2\n");
              return kTfLiteError;
          }
          continue;
        }

        if (input_index == kOptionalTensor &&
            (reg->builtin_code == kTfLiteBuiltinLstm ||
             reg->builtin_code == kTfLiteBuiltinSvdf ||
             reg->builtin_code == kTfLiteBuiltinBidirectionalSequenceLstm)) {
          // properly handle the optional tensor for LSTM and SVDF.
          // currently only support float32.
          // TODO(miaowang): make sure this is also able to handle quantized
          // tensor when supported by NNAPI.
          TF_LITE_ENSURE_STATUS(builder.AddVectorFloat32Operand(nullptr, 0));
        } else if (reg->builtin_code == kTfLiteBuiltinResizeBilinear) {
          if (input_pos == 0) {
            // Only the first input tensor is added. The second one,
            // specifying the output height and width, is not added and
            // instead the height and width will be added individually as
            // scalars by the mapping function returned by Map().
            TF_LITE_ENSURE_STATUS(
                builder.AddTensorInput(input_index, hybrid_op));
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
        } else {
          TF_LITE_ENSURE_STATUS(builder.AddTensorInput(input_index, hybrid_op,
                                                       input_tensor_flags));
        }
      }
      // Get op type and operands
      int nn_op_type = Map(
          context, reg->builtin_code, reg->version, nnapi_->android_sdk_version,
          node)({context, &builder, node, &model_state_outputs_,
                 &model_state_tfl_inputs_});
      // Map outputs to NN API tensor indices.
      int output_tensor_flags = 0;
      if (need_int8_conversion) {
        output_tensor_flags |= NN_TENSOR_FLAG_INT8_CONVERSION;
      }
      for (auto output_index : TfLiteIntArrayView(node->outputs)) {
        TF_LITE_ENSURE_STATUS(
            builder.AddTensorOutput(output_index, output_tensor_flags));
      }

      // Dequantize operators may have to be added in case inputs are to be
      // floating-point.
      AddDequantizeOperatorsWhereNeeded(context, reg->builtin_code, node,
                                        &builder);

      builder.FinalizeAddOperation(nn_op_type);
    }
    return kTfLiteOk;
  }

  TfLiteStatus BuildGraph(TfLiteContext* context,
                          const TfLiteIntArray* input_tensors,
                          const TfLiteIntArray* output_tensors) {
    // Build the ops and tensors.
    TF_LITE_ENSURE_STATUS(AddOpsAndTensors(context));
    // Map input and output tensor indices to ANN
    std::vector<uint32_t> inputs;
    inputs.reserve(input_tensors->size);
    std::vector<uint32_t> outputs;
    outputs.reserve(output_tensors->size);

    size_t total_input_byte_size = 0;
    // Make the TensorFlow Lite inputs and outputs to ann_indices.
    for (int i : TfLiteIntArrayView(input_tensors)) {
      // Constant tensors are not NNAPI inputs.
      if (i != kOptionalTensor &&
          context->tensors[i].allocation_type != kTfLiteMmapRo) {
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
      outputs.push_back(operand_mapping_.lite_index_to_ann(i));
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
        context, nnapi_->ANeuralNetworksModel_identifyInputsAndOutputs(
                     nn_model_.get(), inputs.size(), inputs.data(),
                     outputs.size(), outputs.data()));

    // Set relaxed computation mode for fp32 if possible.
    if (nnapi_->android_sdk_version >= kMinSdkVersionForNNAPI11) {
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context,
          nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16(
              nn_model_.get(), context->allow_fp32_relax_to_fp16));
    }

    // Finalize the model
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context, nnapi_->ANeuralNetworksModel_finish(nn_model_.get()));

    // Create shared memory pool for inputs and outputs.
    nn_input_memory_.reset(
        new NNMemory(nnapi_, "input_pool", total_input_byte_size));
    nn_output_memory_.reset(
        new NNMemory(nnapi_, "output_pool", total_output_byte_size));

    return kTfLiteOk;
  }
};

}  // namespace

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

TfLiteStatus StatefulNnApiDelegate::DoPrepare(TfLiteContext* context,
                                              TfLiteDelegate* delegate) {
  // Do not check nodes_ if NN API is unavailable.
  const NnApi* nnapi = NnApiImplementation();
  if (nnapi->android_sdk_version < kMinSdkVersionForNNAPI ||
      !nnapi->nnapi_exists) {
    return kTfLiteOk;
  }
  // For NNAPI 1.2+, check if there is any accelerator available.
  // If not, don't delegate to NNAPI's CPU reference implementation.
  if (nnapi->android_sdk_version >= kMinSdkVersionForNNAPI12) {
    // Check if user specified an acclelerator to use.
    const char* device_name_ptr = GetOptions(delegate).accelerator_name;
    if (device_name_ptr) {
      if (!GetDeviceHandle(device_name_ptr)) {
        // If the selected accelerator cannot be found, NNAPI will not be used.
        context->ReportError(context,
                             "Could not find the specified accelerator: %s.",
                             device_name_ptr);
        return kTfLiteOk;
      }
    } else {
      // If no accelerator is specified, only use NNAPI if an accelerator is
      // available. Any available accelerator will make the device_count larger
      // than 1. More sophisticated check and whitelisting can be added later.
      uint32_t device_count = 0;
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context, nnapi->ANeuralNetworks_getDeviceCount(&device_count));
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
  // TODO(b/80625235): Fix this to do more careful checking of versioning.
  for (int node_index : TfLiteIntArrayView(plan)) {
    TfLiteNode* node;
    TfLiteRegistration* registration;
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));
    if (NNAPIDelegateKernel::Map(context, registration->builtin_code,
                                 registration->version, android_sdk_version,
                                 node)) {
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
        NNAPIDelegateKernel* kernel_state = new NNAPIDelegateKernel;
        kernel_state->Init(context, params);
        return kernel_state;
      },

      .free = [](TfLiteContext* context, void* buffer) -> void {
        delete reinterpret_cast<NNAPIDelegateKernel*>(buffer);
      },

      .prepare = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        // Since the underlying resize happened ahead of delegation
        // worked. This does nothing.
        return kTfLiteOk;
      },

      .invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        NNAPIDelegateKernel* state =
            reinterpret_cast<NNAPIDelegateKernel*>(node->user_data);
        return state->Invoke(context, node);
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
