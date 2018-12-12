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
#include "tensorflow/lite/nnapi/NeuralNetworksShim.h"

#ifdef __ANDROID__
#include <sys/mman.h>
#include <sys/system_properties.h>
#include <unistd.h>
#endif

namespace tflite {
namespace {

// TODO(b/80621585): Consider printing error string, but don't for now to
// minimize binary size.
#define CHECK_NN(context, code)                                           \
  if (code != ANEURALNETWORKS_NO_ERROR) {                                 \
    context->ReportError(context, "NN API returned error (%d).\n", code); \
    return kTfLiteError;                                                  \
  }

namespace {
int32_t GetAndroidSdkVersion() {
#ifdef __ANDROID__
  const char* sdkProp = "ro.build.version.sdk";
  char sdkVersion[PROP_VALUE_MAX];
  int length = __system_property_get(sdkProp, sdkVersion);
  if (length != 0) {
    for (int i = 0; i < length; ++i) {
      int digit = sdkVersion[i] - '0';
      if (digit < 0 || digit > 9) {
        // Non-numeric SDK version, assume it's higher then expected;
        return std::numeric_limits<int32_t>::max();
      }
    }
    return atoi(sdkVersion);
  }
#endif  // __ANDROID__
  return 0;
}

constexpr int32_t kMinSdkVersionForNNAPI = 27;
constexpr int32_t kMinSdkVersionForNNAPI11 = 28;
static const int32_t kAndroidSdkVersion = GetAndroidSdkVersion();

}  // namespace

// RAII NN API Model Destructor for use with std::unique_ptr
struct NNFreeModel {
  void operator()(ANeuralNetworksModel* model) {
    ANeuralNetworksModel_free(model);
  }
};
// RAII NN API Compilation Destructor for use with std::unique_ptr
struct NNFreeCompilation {
  void operator()(ANeuralNetworksCompilation* model) {
    ANeuralNetworksCompilation_free(model);
  }
};

// Manage NNAPI shared memory handle
class NNMemory {
 public:
  NNMemory(const char* name, size_t size) {
#ifdef __ANDROID__
    byte_size_ = size;
    fd_ = ASharedMemory_create(name, size);
    data_ptr_ = reinterpret_cast<uint8_t*>(
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    ANeuralNetworksMemory_createFromFd(size, PROT_READ | PROT_WRITE, fd_, 0,
                                       &nn_memory_handle_);
#endif
  }

  ~NNMemory() {
#ifdef __ANDROID__
    if (data_ptr_) {
      munmap(data_ptr_, byte_size_);
    }
    if (nn_memory_handle_) {
      ANeuralNetworksMemory_free(nn_memory_handle_);
    }
    if (fd_ > 0) close(fd_);
#endif
  }

  ANeuralNetworksMemory* get_handle() { return nn_memory_handle_; }
  uint8_t* get_data_ptr() { return data_ptr_; }

 private:
#ifdef __ANDROID__
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

// Abstract builder for building an op in the NN API graph. This handles
// the disparity between TFLite and NN API operand types. NN API has singular
// operands for both tensors and parameters, and TFLite separates the two.
class NNAPIOpBuilder {
 public:
  NNAPIOpBuilder(TfLiteContext* context, OperandMapping* tensor_mapping,
                 ANeuralNetworksModel* nn_model)
      : context_(context),
        operand_mapping_(tensor_mapping),
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

  TfLiteStatus AddTensorInput(int tensor_index) {
    int ann_index;
    TF_LITE_ENSURE_STATUS(AddTensor(tensor_index, &ann_index));
    augmented_inputs_.push_back(ann_index);
    return kTfLiteOk;
  }

  TfLiteStatus AddTensorOutput(int tensor_index) {
    int ann_index;
    TF_LITE_ENSURE_STATUS(AddTensor(tensor_index, &ann_index));
    augmented_outputs_.push_back(ann_index);
    return kTfLiteOk;
  }

  TfLiteStatus AddAdditionalFloat32OutputTensor(uint32_t dimension_count) {
    std::vector<uint32_t> dims(dimension_count, 0);
    ANeuralNetworksOperandType operand_type{
        .type = ANEURALNETWORKS_TENSOR_FLOAT32,
        .dimensionCount = dimension_count,
        .dimensions = dims.data()};
    CHECK_NN(context_,
             ANeuralNetworksModel_addOperand(nn_model_, &operand_type));
    int ann_operand = operand_mapping_->add_new_non_tensor_operand();
    augmented_outputs_.push_back(ann_operand);
    return kTfLiteOk;
  }

  TfLiteStatus AddStateFloat32Tensor(int tensor_index,
                                     int* ann_tensor_index_out) {
    TfLiteTensor* tensor = &context_->tensors[tensor_index];
    int ann_index = operand_mapping_->add_new_non_tensor_operand();

    ANeuralNetworksOperandType operand_type{
        ANEURALNETWORKS_TENSOR_FLOAT32,
        static_cast<uint32_t>(tensor->dims->size),
        reinterpret_cast<uint32_t*>(tensor->dims->data), tensor->params.scale,
        tensor->params.zero_point};
    CHECK_NN(context_,
             ANeuralNetworksModel_addOperand(nn_model_, &operand_type));
    augmented_outputs_.push_back(ann_index);

    *ann_tensor_index_out = ann_index;
    return kTfLiteOk;
  }

  // Adds a new NN API tensor that shadows the TF Lite tensor `tensor_index`.
  // This returns the NN API tensor index corresponding to the created tensor.
  // If another caller previously created a NN API tensor for `tensor_index`
  // then the existing one is returned.
  TfLiteStatus AddTensor(int tensor_index, int* ann_tensor_index_out) {
    int ann_tensor_index = operand_mapping_->lite_index_to_ann(tensor_index);
    if (ann_tensor_index != -1) {
      *ann_tensor_index_out = ann_tensor_index;
      return kTfLiteOk;
    }
    // Allocate a new tensor index
    ann_tensor_index = operand_mapping_->add_new_ann_tensor_index(tensor_index);

    // Parameters needed for new type.
    int32_t nn_type = 0;
    float scale = 0.0f;
    int32_t zeroPoint = 0;
    TfLiteTensor* tensor = &context_->tensors[tensor_index];
    switch (tensor->type) {
      case kTfLiteNoType:
        // Tensors added during initialization of Ops don't have a type yet and
        // should not be registered with the NNAPI.
        *ann_tensor_index_out = -1;
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
    CHECK_NN(context_,
             ANeuralNetworksModel_addOperand(nn_model_, &operand_type));

    if (tensor->allocation_type == kTfLiteMmapRo) {
      // TODO(b/80630405): Use NNAPIAllocation.
      CHECK_NN(context_, ANeuralNetworksModel_setOperandValue(
                             nn_model_, ann_tensor_index, tensor->data.raw,
                             tensor->bytes));
    }

    *ann_tensor_index_out = ann_tensor_index;
    return kTfLiteOk;
  }

  // Finish emitting the op (of type `type`) into the NN API.
  TfLiteStatus FinalizeAddOperation(ANeuralNetworksOperationType type) {
    // Actually add a NN API operation
    CHECK_NN(context_, ANeuralNetworksModel_addOperation(
                           nn_model_, type,
                           static_cast<uint32_t>(augmented_inputs_.size()),
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
    CHECK_NN(context_,
             ANeuralNetworksModel_addOperand(nn_model_, &operand_type));
    int ann_operand = operand_mapping_->add_new_non_tensor_operand();
    CHECK_NN(context_, ANeuralNetworksModel_setOperandValue(
                           nn_model_, ann_operand, &value, sizeof(T)));
    augmented_inputs_.push_back(ann_operand);
    return kTfLiteOk;
  }

  template <typename T>
  TfLiteStatus AddVectorOperand(const T* values, uint32_t num_values,
                                int32_t nn_type) {
    ANeuralNetworksOperandType operand_type{
        .type = nn_type, .dimensionCount = 1, .dimensions = &num_values};
    CHECK_NN(context_,
             ANeuralNetworksModel_addOperand(nn_model_, &operand_type));
    int ann_operand = operand_mapping_->add_new_non_tensor_operand();
    CHECK_NN(context_,
             ANeuralNetworksModel_setOperandValue(
                 nn_model_, ann_operand, values, sizeof(T) * num_values));
    augmented_inputs_.push_back(ann_operand);
    return kTfLiteOk;
  }

  // TfLiteContext for error handling. Must be named context for macros to
  // work.
  TfLiteContext* context_;

  // Tracks relationship between indices
  OperandMapping* operand_mapping_;

  // The model
  ANeuralNetworksModel* nn_model_;

  // Inputs and outputs for the current op. These are augmented in the sense
  // that NN API uses operands for all arguments, not just tensors, unlike
  // TensorFlow lite.
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

// The kernel that represents the node sub set of TF Lite being run on NN API.
class NNAPIDelegateKernel {
 public:
  NNAPIDelegateKernel() = default;

  typedef ANeuralNetworksOperationType (*MappingFn)(
      const NNAPIOpMappingArgs& mapping_args);

  // Return a function that knows how to translate a node into its operands
  // when called. You can use this function to see if a node is supported
  // (i.e. that MappingFn is not nullptr).
  MappingFn Map(TfLiteContext* context, int builtin_code, int version,
                TfLiteNode* node) {
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
        } else {
          return nullptr;
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
        } else {
          return nullptr;
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
        } else {
          return nullptr;
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
        } else {
          return nullptr;
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
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinConv2d:
        if (version == 1) {
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
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinDepthwiseConv2d:
        if (version == 1) {
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
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinFullyConnected:
        if (version == 1) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteFullyConnectedParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_FULLY_CONNECTED;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinSoftmax:
        if (version == 1) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteSoftmaxParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarFloat32Operand(builtin->beta);
            return ANEURALNETWORKS_SOFTMAX;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinReshape:
        if (version == 1 && node->inputs->size == 2) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            return ANEURALNETWORKS_RESHAPE;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinSqueeze:
        if (version == 1 && kAndroidSdkVersion >= kMinSdkVersionForNNAPI11) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteSqueezeParams*>(
                mapping_args.node->builtin_data);
            // Note that we add the squeeze dimensions even if the dimensions
            // were unspecified (empty), as NNAPI requires the operand.
            mapping_args.builder->AddVectorInt32Operand(
                builtin->squeeze_dims,
                static_cast<uint32_t>(builtin->num_squeeze_dims));
            return ANEURALNETWORKS_SQUEEZE;
          };
        } else {
          return nullptr;
        }
      case kTfLiteBuiltinL2Normalization: {
        auto builtin =
            reinterpret_cast<TfLiteL2NormParams*>(node->builtin_data);
        if (builtin->activation != kTfLiteActNone) {
          // NNAPI does not support activations
          return nullptr;
        }
        return [](const NNAPIOpMappingArgs& mapping_args)
                   -> ANeuralNetworksOperationType {
          return ANEURALNETWORKS_L2_NORMALIZATION;
        };
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
        } else {
          // TODO(miaowang): clean-up code and return early in the unsupported
          // case.
          return nullptr;
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
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinConcatenation:
        if (version == 1 &&
            reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data)
                    ->activation == kTfLiteActNone) {
          if (context->tensors[node->inputs->data[0]].type == kTfLiteUInt8) {
            // NNAPI only support concatenating quantized tensor of the same
            // scale and offset.
            auto first_param = context->tensors[node->inputs->data[0]].params;
            for (int i = 0; i < node->inputs->size; i++) {
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
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinDequantize:
        if (version == 1) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            return ANEURALNETWORKS_DEQUANTIZE;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinFloor:
        if (version == 1) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            return ANEURALNETWORKS_FLOOR;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinCeil:
        if (version == 1) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            return ANEURALNETWORKS_CEIL;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinRelu:
        if (version == 1) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            return ANEURALNETWORKS_RELU;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinReluN1To1:
        if (version == 1) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            return ANEURALNETWORKS_RELU1;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinRelu6:
        if (version == 1) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            return ANEURALNETWORKS_RELU6;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinLogistic:
        if (version == 1) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            return ANEURALNETWORKS_LOGISTIC;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinTanh:
        // TODO(miaowang): add additional checks for the parameters.
        if (version == 1 &&
            context->tensors[node->inputs->data[0]].type == kTfLiteFloat32) {
          // NNAPI only support float tanh.
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            return ANEURALNETWORKS_TANH;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinSub:
        if (version == 1 && kAndroidSdkVersion >= kMinSdkVersionForNNAPI11 &&
            context->tensors[node->inputs->data[0]].type == kTfLiteFloat32) {
          // NNAPI only support float sub.
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteSubParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_SUB;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinDiv:
        if (version == 1 && kAndroidSdkVersion >= kMinSdkVersionForNNAPI11 &&
            context->tensors[node->inputs->data[0]].type == kTfLiteFloat32) {
          // NNAPI only support float div.
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteDivParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_DIV;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinPad:
        if (version == 1 && kAndroidSdkVersion >= kMinSdkVersionForNNAPI11 &&
            node->inputs->size == 2 &&
            context->tensors[node->inputs->data[0]].type == kTfLiteFloat32) {
          // NNAPI does not support specifying the padding value.
          // NNAPI pads physical zero for quantized tensors, so only delegate
          // float pad to NNAPI.
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            return ANEURALNETWORKS_PAD;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinSpaceToBatchNd:
        if (version == 1 && kAndroidSdkVersion >= kMinSdkVersionForNNAPI11) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            return ANEURALNETWORKS_SPACE_TO_BATCH_ND;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinStridedSlice:
        if (version == 1 && kAndroidSdkVersion >= kMinSdkVersionForNNAPI11) {
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
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinTranspose:
        // Note that the permutation input tensor value dictates the output
        // dimensions.
        // TODO(b/110888333): Support dynamically-sized tensors in delegates.
        if ((version == 1) &&
            (kAndroidSdkVersion >= kMinSdkVersionForNNAPI11) &&
            (node->inputs->size > 1) &&
            (context->tensors[node->inputs->data[1]].allocation_type ==
             kTfLiteMmapRo)) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            return ANEURALNETWORKS_TRANSPOSE;
          };
        } else {
          return nullptr;
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
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinSvdf:
        // NNAPI only support float32 weights.
        if (version == 1 && node->inputs->size == 5 &&
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
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinLstm:
        // NNAPI only support float32 weights.
        // TODO(miaowang): add loggings to indicate why the op is rejected.
        if (version == 1 && node->inputs->size == 20 &&
            context->tensors[node->inputs
                                 ->data[/*kInputToOutputWeightsTensor*/ 4]]
                    .type == kTfLiteFloat32) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteLSTMParams*>(
                mapping_args.node->builtin_data);
            mapping_args.builder->AddScalarInt32Operand(builtin->activation);
            mapping_args.builder->AddScalarFloat32Operand(builtin->cell_clip);
            mapping_args.builder->AddScalarFloat32Operand(builtin->proj_clip);

            // Current NNAPI implementation requires the sratch_buffer as
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

            return ANEURALNETWORKS_LSTM;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinMean:
        // NNAPI does not support generating a scalar as output for MEAN.
        if (version == 1 && kAndroidSdkVersion >= kMinSdkVersionForNNAPI11 &&
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
        } else {
          return nullptr;
        }
      case kTfLiteBuiltinEmbeddingLookup:
        // NNAPI only support float32 values.
        if (version == 1 &&
            context->tensors[node->inputs->data[1]].type == kTfLiteFloat32) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            return ANEURALNETWORKS_EMBEDDING_LOOKUP;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinHashtableLookup:
        // NNAPI only support float32 output.
        if (version == 1 &&
            context->tensors[node->outputs->data[0]].type == kTfLiteFloat32) {
          return [](const NNAPIOpMappingArgs& mapping_args)
                     -> ANeuralNetworksOperationType {
            return ANEURALNETWORKS_HASHTABLE_LOOKUP;
          };
        } else {
          return nullptr;
        }
        break;
      default:
        return nullptr;
    }
  }

  // Initialize the kernel (a NN model).
  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) {
    for (auto node_index : TfLiteIntArrayView(params->nodes_to_replace)) {
      nodes_.push_back(node_index);
    }

    if (!nn_model_) {
      ANeuralNetworksModel* model;
      CHECK_NN(context, ANeuralNetworksModel_create(&model));
      nn_model_.reset(model);

      TF_LITE_ENSURE_STATUS(
          BuildGraph(context, params->input_tensors, params->output_tensors));
    }

    if (!nn_compilation_) {
      ANeuralNetworksCompilation* compilation;
      CHECK_NN(context, ANeuralNetworksCompilation_create(nn_model_.get(),
                                                          &compilation));
      CHECK_NN(context, ANeuralNetworksCompilation_finish(compilation));
      nn_compilation_.reset(compilation);
    }
    return kTfLiteOk;
  }

  TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
    ANeuralNetworksExecution* execution = nullptr;
    CHECK_NN(context, ANeuralNetworksExecution_create(nn_compilation_.get(),
                                                      &execution));

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
        CHECK_NN(context, ANeuralNetworksExecution_setInputFromMemory(
                              execution, relative_input_index, nullptr,
                              nn_input_memory_->get_handle(), input_offset,
                              tensor->bytes));
        input_offset += tensor->bytes;
        relative_input_index++;
      }
    }

    // Set the output tensor buffers.
    int relative_output_index = 0;
    size_t output_offset = 0;
    for (auto output_index : TfLiteIntArrayView(node->outputs)) {
      TfLiteTensor* tensor = &context->tensors[output_index];
      CHECK_NN(context, ANeuralNetworksExecution_setOutputFromMemory(
                            execution, relative_output_index, nullptr,
                            nn_output_memory_->get_handle(), output_offset,
                            tensor->bytes));
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
      CHECK_NN(context, ANeuralNetworksExecution_setOutput(
                            execution, relative_output_index, nullptr,
                            tensor->data.raw, tensor->bytes));
      relative_output_index++;
    }
    // Invoke ANN in blocking fashion.
    ANeuralNetworksEvent* event = nullptr;
    CHECK_NN(context, ANeuralNetworksExecution_startCompute(execution, &event));
    CHECK_NN(context, ANeuralNetworksEvent_wait(event));
    ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);

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

  TfLiteStatus AddOpsAndTensors(TfLiteContext* context) {
    // The operand builder allows creating a single op. We create it at this
    // reduced power position rather than in the for loop to avoid reallocating
    // the vectors.
    NNAPIOpBuilder builder(context, &operand_mapping_, nn_model_.get());
    // Add Tensors
    // allocate outside to avoid realloc
    for (auto node_index : nodes_) {
      // Obtain the op and registration.
      TfLiteNode* node;
      TfLiteRegistration* reg;
      context->GetNodeAndRegistration(context, node_index, &node, &reg);
      // Map inputs to NN API tensor indices.
      for (auto input_index : TfLiteIntArrayView(node->inputs)) {
        if (input_index == kOptionalTensor &&
            (reg->builtin_code == kTfLiteBuiltinLstm ||
             reg->builtin_code == kTfLiteBuiltinSvdf)) {
          // properly handle the optional tensor for LSTM and SVDF.
          // currently only support float32.
          // TODO(miaowang): make sure this is also able to handle quantized
          // tensor when supported by NNAPI.
          TF_LITE_ENSURE_STATUS(builder.AddVectorFloat32Operand(nullptr, 0));
        } else {
          TF_LITE_ENSURE_STATUS(builder.AddTensorInput(input_index));
        }
      }
      // Get op type and operands
      int nn_op_type = Map(context, reg->builtin_code, reg->version, node)(
          {context, &builder, node, &model_state_outputs_,
           &model_state_tfl_inputs_});
      // Map outputs to NN API tensor indices.
      for (auto output_index : TfLiteIntArrayView(node->outputs)) {
        TF_LITE_ENSURE_STATUS(builder.AddTensorOutput(output_index));
      }

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
    // Make the TensorFlow lite inputs and outputs to ann_indices.
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

    // Add state output tensors as model inputs
    for (int i : model_state_outputs_) {
      outputs.push_back(i);
    }

    // Tell ANN to declare inputs/outputs
    CHECK_NN(context, ANeuralNetworksModel_identifyInputsAndOutputs(
                          nn_model_.get(), inputs.size(), inputs.data(),
                          outputs.size(), outputs.data()));

    // Set relaxed computation mode for fp32 if possible.
    if (kAndroidSdkVersion >= kMinSdkVersionForNNAPI11) {
      CHECK_NN(context,
               ANeuralNetworksModel_relaxComputationFloat32toFloat16(
                   nn_model_.get(), context->allow_fp32_relax_to_fp16));
    }

    // Finalize the model
    CHECK_NN(context, ANeuralNetworksModel_finish(nn_model_.get()));

    // Create shared memory pool for inputs and outputs.
    nn_input_memory_.reset(new NNMemory("input_pool", total_input_byte_size));
    nn_output_memory_.reset(
        new NNMemory("output_pool", total_output_byte_size));

    return kTfLiteOk;
  }
};

}  // namespace

// Return a NN API Delegate struct that can check for support of ops.
TfLiteDelegate* NnApiDelegate() {
  static TfLiteDelegate delegate = {
      .data_ = nullptr,
      .flags = kTfLiteDelegateFlagsNone,
      .Prepare = [](TfLiteContext* context,
                    TfLiteDelegate* delegate) -> TfLiteStatus {
        // Do not check nodes_ if NN API is unavailable.
        if (kAndroidSdkVersion < kMinSdkVersionForNNAPI || !NNAPIExists()) {
          return kTfLiteOk;
        }

        std::vector<int> supported_nodes(1);
        // We don't care about all nodes_, we only care about ones in the
        // current plan.
        TfLiteIntArray* plan;
        TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));
        int total_supported_nodes = 0;

        // Check for every node if it is supported
        // TODO(b/80625235): Fix this to do more careful checking of versioning.
        for (int node_index : TfLiteIntArrayView(plan)) {
          TfLiteNode* node;
          TfLiteRegistration* registration;
          TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
              context, node_index, &node, &registration));
          NNAPIDelegateKernel dummy_kernel;
          if (dummy_kernel.Map(context, registration->builtin_code,
                               registration->version, node)) {
            supported_nodes.push_back(node_index);
          }
          total_supported_nodes += 1;
        }
        // Put the size at the beginning of the array.
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

            .builtin_code = kTfLiteBuiltinDelegate,
        };

        // Request TFLite to partition the graph and make kernels
        // for each independent node sub set a new nnapi_delegate_kernel.
        context->ReplaceNodeSubsetsWithDelegateKernels(
            context, nnapi_delegate_kernel,
            reinterpret_cast<TfLiteIntArray*>(supported_nodes.data()),
            delegate);
        return kTfLiteOk;
      }};

  return &delegate;
}

}  // namespace tflite
