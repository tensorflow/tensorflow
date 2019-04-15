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
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

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
    nnapi_ = nnapi;
    byte_size_ = size;
    fd_ = nnapi_->ASharedMemory_create(name, size);
    data_ptr_ = reinterpret_cast<uint8_t*>(
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    nnapi_->ANeuralNetworksMemory_createFromFd(size, PROT_READ | PROT_WRITE,
                                               fd_, 0, &nn_memory_handle_);
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

 private:
  // Next index of ann tensor
  int next_ann_tensor_index_ = 0;

  // Mapping from lite index. Use a std::vector for speed and code size
  // rather than a map.
  std::vector<int> lite_tensor_to_ann_tensor_;
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

  TfLiteStatus AddTensorInput(int tensor_index, bool hybrid_op) {
    return AddTensor(tensor_index, hybrid_op, &augmented_inputs_);
  }

  TfLiteStatus AddTensorOutput(int tensor_index) {
    return AddTensor(tensor_index, /*hybrid_op=*/false, &augmented_outputs_);
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
          dequantized_type, static_cast<uint32_t>(tensor.dims->size),
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

 private:
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
                         std::vector<uint32_t>* indices) {
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
        nn_type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
        scale = tensor->params.scale;
        zeroPoint = tensor->params.zero_point;
        if (scale == 0) {
          // TENSOR_QUANT8_ASYMM with zero scale is not valid in NNAPI.
          scale = 1;
        }
        break;
      case kTfLiteInt8:
        nn_type = ANEURALNETWORKS_TENSOR_QUANT8_SYMM;
        scale = tensor->params.scale;
        break;
      case kTfLiteInt32:
        nn_type = ANEURALNETWORKS_TENSOR_INT32;
        scale = tensor->params.scale;
        zeroPoint = tensor->params.zero_point;
        break;
      default:
        context_->ReportError(context_, "Logic error in NN API Delegate.\n");
        return kTfLiteError;
    }

    ANeuralNetworksOperandType operand_type{
        nn_type, static_cast<uint32_t>(tensor->dims->size),
        reinterpret_cast<uint32_t*>(tensor->dims->data), scale, zeroPoint};
    RETURN_TFLITE_ERROR_IF_NN_ERROR(
        context_,
        nnapi_->ANeuralNetworksModel_addOperand(nn_model_, &operand_type));

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
  // (i.e. that MappingFn is not nullptr).
  static MappingFn Map(const TfLiteContext* context, int builtin_code,
                       int version, int android_sdk_version,
                       const TfLiteNode* node) {
    switch (builtin_code) {
      case kTfLiteBuiltinAdd:
        if (version == 1) {
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
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            mapping_args.builder->AddPoolingParams(
                mapping_args.node->builtin_data);
            return ANEURALNETWORKS_L2_POOL_2D;
          };
        }
        break;
      case kTfLiteBuiltinConv2d:
        if (version == 1) {
          if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
              IsHybridOperator(context, builtin_code, node)) {
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
              reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
          if (builtin->dilation_width_factor != 1 ||
              builtin->dilation_height_factor != 1 || node->inputs->size != 3) {
            // NNAPI does not support dilated Conv2D.
            return nullptr;
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
          const auto input_type = context->tensors[node->inputs->data[0]].type;
          if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
              input_type == kTfLiteUInt8 &&
              !IsRestrictedScalesCompliant(context, node)) {
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
            return ANEURALNETWORKS_DEPTHWISE_CONV_2D;
          };
        }
        break;
      case kTfLiteBuiltinFullyConnected:
        if (version == 1) {
          if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
              IsHybridOperator(context, builtin_code, node)) {
            // Hybrid operators not supported before NNAPI 1.2.
            return nullptr;
          }
          const auto input_type = context->tensors[node->inputs->data[0]].type;
          if (android_sdk_version < kMinSdkVersionForNNAPI12 &&
              input_type == kTfLiteUInt8 &&
              !IsRestrictedScalesCompliant(context, node)) {
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
        if (version == 1 && node->inputs->size == 2) {
          return BasicMappingFn<ANEURALNETWORKS_RESHAPE>;
        }
        break;
      case kTfLiteBuiltinResizeBilinear:
        if (version == 1) {
          if (android_sdk_version < kMinSdkVersionForNNAPI12) {
            // Some NNAPI 1.1 drivers don't support this operator properly.
            return nullptr;
          }
          const auto& input = context->tensors[node->inputs->data[0]];
          if (input.dims->size != 4) return nullptr;
          if (input.type != kTfLiteFloat32 && input.type != kTfLiteUInt8) {
            return nullptr;
          }

          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            const int output_id = mapping_args.node->outputs->data[0];
            auto& output = mapping_args.context->tensors[output_id];
            const int output_height = output.dims->data[1];
            const int output_width = output.dims->data[2];
            // TfLiteResizeBilinearParams's |align_corners| is ignored.
            mapping_args.builder->AddScalarInt32Operand(output_height);
            mapping_args.builder->AddScalarInt32Operand(output_width);
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
      case kTfLiteBuiltinL2Normalization: {
        auto builtin =
            reinterpret_cast<TfLiteL2NormParams*>(node->builtin_data);
        if (builtin->activation == kTfLiteActNone) {
          return BasicMappingFn<ANEURALNETWORKS_L2_NORMALIZATION>;
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
          return BasicMappingFn<ANEURALNETWORKS_RELU>;
        }
        break;
      case kTfLiteBuiltinReluN1To1:
        if (version == 1) {
          return BasicMappingFn<ANEURALNETWORKS_RELU1>;
        }
        break;
      case kTfLiteBuiltinRelu6:
        if (version == 1) {
          return BasicMappingFn<ANEURALNETWORKS_RELU6>;
        }
        break;
      case kTfLiteBuiltinLogistic:
        if (version == 1) {
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
        if (version == 1 && node->inputs->size == 2 &&
            (android_sdk_version >= kMinSdkVersionForNNAPI11) &&
            (context->tensors[node->inputs->data[0]].type == kTfLiteFloat32 ||
             android_sdk_version >= kMinSdkVersionForNNAPI12)) {
          // NNAPI does not support specifying the padding value.
          // Before 1.2, NNAPI pads physical zero for quantized tensors, so only
          // delegate float pad to NNAPI. NNAPI 1.2 onwards pads with
          // zero-point, so delegate quantized pad as well.
          return BasicMappingFn<ANEURALNETWORKS_PAD>;
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
        if (version == 1 && android_sdk_version >= kMinSdkVersionForNNAPI11 &&
            context->tensors[node->inputs->data[0]].type == kTfLiteFloat32 &&
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
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context, nnapi_->ANeuralNetworksCompilation_create(nn_model_.get(),
                                                             &compilation));
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
        // copy data to pre-allocated shared memory.
        memcpy(nn_input_memory_->get_data_ptr() + input_offset,
               tensor->data.raw, tensor->bytes);
        RETURN_TFLITE_ERROR_IF_NN_ERROR(
            context,
            nnapi_->ANeuralNetworksExecution_setInputFromMemory(
                execution, relative_input_index, nullptr,
                nn_input_memory_->get_handle(), input_offset, tensor->bytes));
        input_offset += tensor->bytes;
        relative_input_index++;
      }
    }

    // Set the output tensor buffers.
    int relative_output_index = 0;
    size_t output_offset = 0;
    for (auto output_index : TfLiteIntArrayView(node->outputs)) {
      TfLiteTensor* tensor = &context->tensors[output_index];
      RETURN_TFLITE_ERROR_IF_NN_ERROR(
          context,
          nnapi_->ANeuralNetworksExecution_setOutputFromMemory(
              execution, relative_output_index, nullptr,
              nn_output_memory_->get_handle(), output_offset, tensor->bytes));
      output_offset += tensor->bytes;
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
      memcpy(tensor->data.raw,
             nn_output_memory_->get_data_ptr() + output_offset, tensor->bytes);
      output_offset += tensor->bytes;
    }

    return kTfLiteOk;
  }

 private:
  // Access to NNApi.
  const NnApi* nnapi_;
  // ANN API state.
  std::unique_ptr<ANeuralNetworksModel, NNFreeModel> nn_model_;
  std::unique_ptr<ANeuralNetworksCompilation, NNFreeCompilation>
      nn_compilation_;
  // Node indices that this delegate is responsible for. Indices here
  // indexes into the nodes array in the TfLiteContext.
  std::vector<int> nodes_;
  // Track indices we use
  OperandMapping operand_mapping_;

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
      if (type != kTfLiteUInt8) continue;

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

      // Map inputs to NN API tensor indices.
      int num_added_inputs = 0;
      for (auto input_index : TfLiteIntArrayView(node->inputs)) {
        if (reg->builtin_code == kTfLiteBuiltinLstm && num_added_inputs >= 20) {
          // Skip layer normalization weights. They are added in the Map
          // function (after all the other inputs added there) since layer
          // normalization weights are the last four inputs of the LSTM op in
          // NNAPI.
          continue;
        }
        if (input_index == kOptionalTensor &&
            (reg->builtin_code == kTfLiteBuiltinLstm ||
             reg->builtin_code == kTfLiteBuiltinSvdf)) {
          // properly handle the optional tensor for LSTM and SVDF.
          // currently only support float32.
          // TODO(miaowang): make sure this is also able to handle quantized
          // tensor when supported by NNAPI.
          TF_LITE_ENSURE_STATUS(builder.AddVectorFloat32Operand(nullptr, 0));
        } else if (reg->builtin_code == kTfLiteBuiltinResizeBilinear) {
          if (num_added_inputs == 0) {
            // Only the first input tensor is added. The second one, specifying
            // the output height and width, is not added and instead the height
            // and width will be added individually as scalars by the mapping
            // function returned by Map().
            TF_LITE_ENSURE_STATUS(
                builder.AddTensorInput(input_index, hybrid_op));
          }
        } else {
          TF_LITE_ENSURE_STATUS(builder.AddTensorInput(input_index, hybrid_op));
        }
        ++num_added_inputs;
      }
      // Get op type and operands
      int nn_op_type = Map(
          context, reg->builtin_code, reg->version, nnapi_->android_sdk_version,
          node)({context, &builder, node, &model_state_outputs_,
                 &model_state_tfl_inputs_});
      // Map outputs to NN API tensor indices.
      for (auto output_index : TfLiteIntArrayView(node->outputs)) {
        TF_LITE_ENSURE_STATUS(builder.AddTensorOutput(output_index));
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
        total_input_byte_size += context->tensors[i].bytes;
      }
    }

    size_t total_output_byte_size = 0;
    for (int i : TfLiteIntArrayView(output_tensors)) {
      outputs.push_back(operand_mapping_.lite_index_to_ann(i));
      total_output_byte_size += context->tensors[i].bytes;
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

// Return a NN API Delegate struct that can check for support of ops.
TfLiteDelegate* NnApiDelegate() {
  static TfLiteDelegate delegate = {
      .data_ = nullptr,
      .Prepare = [](TfLiteContext* context,
                    TfLiteDelegate* delegate) -> TfLiteStatus {
        // Do not check nodes_ if NN API is unavailable.
        const NnApi* nnapi = NnApiImplementation();
        if (nnapi->android_sdk_version < kMinSdkVersionForNNAPI ||
            !nnapi->nnapi_exists) {
          return kTfLiteOk;
        }
        // For NNAPI 1.2+, check if there is any accelerator available.
        // If not, don't delegate to NNAPI's CPU reference implementation.
        if (nnapi->android_sdk_version >= kMinSdkVersionForNNAPI12) {
          uint32_t device_count = 0;
          RETURN_TFLITE_ERROR_IF_NN_ERROR(
              context, nnapi->ANeuralNetworks_getDeviceCount(&device_count));
          // Any available accelerator will make the device_count larger than 1.
          // More sophisticated check and whitelisting can be added later.
          if (device_count <= 1) {
            return kTfLiteOk;
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
                                       registration->version,
                                       android_sdk_version, node)) {
            supported_nodes.push_back(node_index);
          }
        }
        // First element in vector must be the number of actual nodes.
        supported_nodes[0] = supported_nodes.size() - 1;

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

            .prepare = [](TfLiteContext* context,
                          TfLiteNode* node) -> TfLiteStatus {
              // Since the underlying resize happened ahead of delegation
              // worked. This does nothing.
              return kTfLiteOk;
            },

            .invoke = [](TfLiteContext* context,
                         TfLiteNode* node) -> TfLiteStatus {
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
            reinterpret_cast<TfLiteIntArray*>(supported_nodes.data()),
            delegate);
      },

      .CopyFromBufferHandle = nullptr,
      .CopyToBufferHandle = nullptr,
      .FreeBufferHandle = nullptr,
      .flags = kTfLiteDelegateFlagsNone,
  };

  return &delegate;
}

}  // namespace tflite
