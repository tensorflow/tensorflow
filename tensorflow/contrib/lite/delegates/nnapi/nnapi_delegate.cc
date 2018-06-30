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

#include "tensorflow/contrib/lite/allocation.h"
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/builtin_ops.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/context_util.h"
#include "tensorflow/contrib/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/nnapi/NeuralNetworksShim.h"

#ifdef __ANDROID__
#include <sys/system_properties.h>
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

// The kernel that represents the subgraph of TF Lite being run on NN API.
class NNAPIDelegateKernel {
 public:
  NNAPIDelegateKernel() = default;

  typedef ANeuralNetworksOperationType (*MappingFn)(TfLiteContext*,
                                                    NNAPIOpBuilder* builder,
                                                    TfLiteNode* node);

  // Return a function that knows how to translate a node into its operands
  // when called. You can use this function to see if a node is supported
  // (i.e. that MappingFn is not nullptr).
  MappingFn Map(TfLiteContext* context, int builtin_code, int version,
                TfLiteNode* node) {
    switch (builtin_code) {
      case kTfLiteBuiltinAdd:
        if (version == 1) {
          return [](TfLiteContext* context, NNAPIOpBuilder* builder,
                    TfLiteNode* node) -> ANeuralNetworksOperationType {
            auto builtin =
                reinterpret_cast<TfLiteAddParams*>(node->builtin_data);
            builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_ADD;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinMul:
        if (version == 1) {
          return [](TfLiteContext* context, NNAPIOpBuilder* builder,
                    TfLiteNode* node) -> ANeuralNetworksOperationType {
            auto builtin =
                reinterpret_cast<TfLiteMulParams*>(node->builtin_data);
            builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_MUL;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinAveragePool2d:
        if (version == 1) {
          return [](TfLiteContext* context, NNAPIOpBuilder* builder,
                    TfLiteNode* node) -> ANeuralNetworksOperationType {
            builder->AddPoolingParams(node->builtin_data);
            return ANEURALNETWORKS_AVERAGE_POOL_2D;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinMaxPool2d:
        if (version == 1) {
          return [](TfLiteContext* context, NNAPIOpBuilder* builder,
                    TfLiteNode* node) -> ANeuralNetworksOperationType {
            builder->AddPoolingParams(node->builtin_data);
            return ANEURALNETWORKS_MAX_POOL_2D;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinL2Pool2d:
        if (version == 1) {
          return [](TfLiteContext* context, NNAPIOpBuilder* builder,
                    TfLiteNode* node) -> ANeuralNetworksOperationType {
            builder->AddPoolingParams(node->builtin_data);
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
          return [](TfLiteContext* context, NNAPIOpBuilder* builder,
                    TfLiteNode* node) -> ANeuralNetworksOperationType {
            auto builtin =
                reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
            builder->AddScalarInt32Operand(builtin->padding);
            builder->AddScalarInt32Operand(builtin->stride_width);
            builder->AddScalarInt32Operand(builtin->stride_height);
            builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_CONV_2D;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinDepthwiseConv2d:
        if (version == 1) {
          return [](TfLiteContext* context, NNAPIOpBuilder* builder,
                    TfLiteNode* node) -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteDepthwiseConvParams*>(
                node->builtin_data);
            builder->AddScalarInt32Operand(builtin->padding);
            builder->AddScalarInt32Operand(builtin->stride_width);
            builder->AddScalarInt32Operand(builtin->stride_height);
            builder->AddScalarInt32Operand(builtin->depth_multiplier);
            builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_DEPTHWISE_CONV_2D;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinFullyConnected:
        if (version == 1) {
          return [](TfLiteContext* context, NNAPIOpBuilder* builder,
                    TfLiteNode* node) -> ANeuralNetworksOperationType {
            auto builtin = reinterpret_cast<TfLiteFullyConnectedParams*>(
                node->builtin_data);
            builder->AddScalarInt32Operand(builtin->activation);
            return ANEURALNETWORKS_FULLY_CONNECTED;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinSoftmax:
        if (version == 1) {
          return [](TfLiteContext* context, NNAPIOpBuilder* builder,
                    TfLiteNode* node) -> ANeuralNetworksOperationType {
            auto builtin =
                reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);
            builder->AddScalarFloat32Operand(builtin->beta);
            return ANEURALNETWORKS_SOFTMAX;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinReshape:
        if (version == 1) {
          return [](TfLiteContext* context, NNAPIOpBuilder* builder,
                    TfLiteNode* node) -> ANeuralNetworksOperationType {
            return ANEURALNETWORKS_RESHAPE;
          };
        } else {
          return nullptr;
        }
        break;
      case kTfLiteBuiltinSqueeze:
        // Squeeze requires NNAPI1.1.
        if (version == 1 && kAndroidSdkVersion >= kMinSdkVersionForNNAPI11) {
          return [](TfLiteContext* context, NNAPIOpBuilder* builder,
                    TfLiteNode* node) -> ANeuralNetworksOperationType {
            auto builtin =
                reinterpret_cast<TfLiteSqueezeParams*>(node->builtin_data);
            // Note that we add the squeeze dimensions even if the dimensions
            // were unspecified (empty), as NNAPI requires the operand.
            builder->AddVectorInt32Operand(
                builtin->squeeze_dims,
                static_cast<uint32_t>(builtin->num_squeeze_dims));
            return ANEURALNETWORKS_SQUEEZE;
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
    for (auto absolute_input_index : TfLiteIntArrayView(node->inputs)) {
      TfLiteTensor* tensor = &context->tensors[absolute_input_index];
      // TODO(miaowang): make sure the delegation works with dequantized weights
      // as intermediate tensors.
      if (tensor->allocation_type != kTfLiteMmapRo) {
        CHECK_NN(context, ANeuralNetworksExecution_setInput(
                              execution, relative_input_index, nullptr,
                              tensor->data.raw, tensor->bytes));
        relative_input_index++;
      }
    }

    // Set the output tensor buffers.
    int relative_output_index = 0;
    for (auto output_index : TfLiteIntArrayView(node->outputs)) {
      TfLiteTensor* tensor = &context->tensors[output_index];
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
        TF_LITE_ENSURE_STATUS(builder.AddTensorInput(input_index));
      }
      // Get op type and operands
      int nn_op_type = Map(context, reg->builtin_code, reg->version, node)(
          context, &builder, node);
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
    // Make the TensorFlow lite inputs and outputs to ann_indices.
    for (int i : TfLiteIntArrayView(input_tensors)) {
      // Constant tensors are not NNAPI inputs.
      if (context->tensors[i].allocation_type != kTfLiteMmapRo) {
        inputs.push_back(operand_mapping_.lite_index_to_ann(i));
      }
    }
    for (int i : TfLiteIntArrayView(output_tensors))
      outputs.push_back(operand_mapping_.lite_index_to_ann(i));
    // Tell ANN to declare inputs/outputs
    CHECK_NN(context, ANeuralNetworksModel_identifyInputsAndOutputs(
                          nn_model_.get(), inputs.size(), inputs.data(),
                          outputs.size(), outputs.data()));
    // Finalize the model
    CHECK_NN(context, ANeuralNetworksModel_finish(nn_model_.get()));

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
        // API subgraphs)
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
        // for each independent subgraph a new nnapi_delegate_kernel.
        context->ReplaceSubgraphsWithDelegateKernels(
            context, nnapi_delegate_kernel,
            reinterpret_cast<TfLiteIntArray*>(supported_nodes.data()),
            delegate);
        return kTfLiteOk;
      }};

  return &delegate;
}

}  // namespace tflite
