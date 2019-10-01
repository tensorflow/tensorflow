/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/armnn/delegate_kernel.h"

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/minimal_logging.h"

#include "armnn/Network.hpp"
#include "armnn/Runtime.hpp"

#include "tensorflow/lite/delegates/armnn/builder.h"
#include "tensorflow/lite/delegates/armnn/delegate.h"
#include "tensorflow/lite/delegates/armnn/validator.h"

// ArmNN namespace alias
namespace armnn_core = ::armnn;

namespace tflite {
namespace delegate {
namespace arm {
namespace {
// Validation context object
struct OpValidationContext {
  bool is_valid;
  std::vector<std::string>* validation_failures;
};

// Container used as a proxy to an underlying memory
template <typename T>
class ProxyContainer {
 public:
  ProxyContainer(size_t size, T* data) : size_(size), data_(data) {}

  size_t size() const { return size_; }
  T* data() { return data_; }
  const T* data() const { return data_; }

 private:
  size_t size_;
  T* data_;
};
}  // namespace

ArmNNDelegateKernel::ArmNNDelegateKernel()
    : nodes_(),
      runtime_(nullptr, &armnn::IRuntime::Destroy),
      network_(nullptr, &armnn::INetwork::Destroy),
      executable_network_(-1),
      inputBindings_(),
      outputBindings_() {}

bool ArmNNDelegateKernel::Validate(const TfLiteContext* context,
                                   int builtin_code, int version,
                                   const TfLiteNode* node,
                                   const armnn::ILayerSupport* validator,
                                   std::vector<std::string>* map_failures) {
  // Failures tracking construct
  OpValidationContext val_ctx{false, map_failures};

  // Extract backend operator validator
  if (validator == nullptr) {
    return false;
  }
  const auto& lsupport = *validator;

  // Validate against the type of the node
  switch (builtin_code) {
    case kTfLiteBuiltinAveragePool2d:
    case kTfLiteBuiltinMaxPool2d:
    case kTfLiteBuiltinL2Pool2d: {
      val_ctx.is_valid = OperationValidator::IsPooling2dSupported(
          context, node, builtin_code, version, lsupport, map_failures);
      break;
    }
    case kTfLiteBuiltinConv2d: {
      val_ctx.is_valid = OperationValidator::IsConvolution2dSupported(
          context, node, version, lsupport, map_failures);
      break;
    }
    case kTfLiteBuiltinDepthwiseConv2d: {
      val_ctx.is_valid = OperationValidator::IsDepthwiseConvolutionSupported(
          context, node, version, lsupport, map_failures);
      break;
    }
    case kTfLiteBuiltinSoftmax: {
      val_ctx.is_valid = OperationValidator::IsSoftmaxSupported(
          context, node, version, lsupport, map_failures);
      break;
    }
    case kTfLiteBuiltinSqueeze: {
      val_ctx.is_valid = OperationValidator::IsSqueezeSupported(
          context, node, version, lsupport, map_failures);
      break;
    }
  }
  return val_ctx.is_valid;
}

TfLiteStatus ArmNNDelegateKernel::Init(TfLiteContext* context,
                                       const TfLiteDelegateParams* params) {
  TfLiteStatus status = kTfLiteOk;

  // Keep track of nodes to replace
  for (auto node_index : TfLiteIntArrayView(params->nodes_to_replace)) {
    nodes_.push_back(node_index);
  }

  // Extract delegate options
  const auto delegate_options = ArmNNDelegate::GetOptions(params->delegate);

  // Setup runtime and model
  try {
    // Create runtime to execute the model
    if (!runtime_) {
      runtime_ = armnn::Runtime::Create(armnn::IRuntime::CreationOptions());
    }

    // Create final network
    if (executable_network_ < 0) {
      // Build network
      network_ = armnn::Network::Create();
      TF_LITE_ENSURE_STATUS(
          BuildNetwork(context, params->input_tensors, params->output_tensors));

      // Compile for given backend
      auto preferred_backends = {
          armnn::BackendId(delegate_options.backend_name)};
      auto network_opt = armnn::Optimize(*network_, preferred_backends,
                                         runtime_->GetDeviceSpec());

      // Release initial un-optimized network
      network_.reset(nullptr);

      // Load network to Runtime
      runtime_->LoadNetwork(executable_network_, std::move(network_opt));
    }
  } catch (const armnn::Exception& e) {
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO, e.what());
    status = kTfLiteError;
  }

  return status;
}

TfLiteStatus ArmNNDelegateKernel::Prepare(TfLiteContext* context,
                                          TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus ArmNNDelegateKernel::Invoke(TfLiteContext* context,
                                         TfLiteNode* node) {
  // Set inputs
  armnn::InputTensors inputWorkload;
  size_t input_offset = 0;
  for (auto inputIdx : TfLiteIntArrayView(node->inputs)) {
    TfLiteTensor* tensor = &context->tensors[inputIdx];

    // ReadOnly memory is used for ConstTensor (e.g. weights, bias etc)
    if (tensor->allocation_type != kTfLiteMmapRo) {
      const armnn::BindingPointInfo& inputBinding =
          inputBindings_[input_offset];

      ProxyContainer<char> inputData(tensor->bytes, tensor->data.raw);
      armnn::ConstTensor inputTensor(inputBinding.second, inputData.data());
      inputWorkload.push_back(std::make_pair(inputBinding.first, inputTensor));

      ++input_offset;
    }
  }

  // Set outputs
  armnn::OutputTensors outputWorkload;
  size_t output_offset = 0;
  for (auto outputIdx : TfLiteIntArrayView(node->outputs)) {
    const armnn::BindingPointInfo& outputBinding =
        outputBindings_[output_offset];
    TfLiteTensor* tensor = &context->tensors[outputIdx];

    ProxyContainer<char> outputData(tensor->bytes, tensor->data.raw);
    armnn::Tensor outputTensor(outputBinding.second, outputData.data());
    outputWorkload.push_back(std::make_pair(outputBinding.first, outputTensor));

    ++output_offset;
  }

  // Run graph
  auto s = runtime_->EnqueueWorkload(executable_network_, inputWorkload,
                                     outputWorkload);

  return (s == armnn::Status::Success) ? kTfLiteOk : kTfLiteError;
}

// Build the ArmNN network
TfLiteStatus ArmNNDelegateKernel::BuildNetwork(
    TfLiteContext* context, const TfLiteIntArray* input_tensors,
    const TfLiteIntArray* output_tensors) {
  ArmNNBuilder builder(context, network_.get());

  // Add operators
  for (auto node_index : nodes_) {
    // Obtain the op and registration.
    TfLiteNode* node;
    TfLiteRegistration* reg;
    TF_LITE_ENSURE_STATUS(
        context->GetNodeAndRegistration(context, node_index, &node, &reg));

    switch (reg->builtin_code) {
      case kTfLiteBuiltinConv2d: {
        TF_LITE_ENSURE_STATUS(
            builder.AddConvolution2dLayer(node, reg->version));
      } break;
      case kTfLiteBuiltinDepthwiseConv2d: {
        TF_LITE_ENSURE_STATUS(
            builder.AddDepthwiseConvolution2dLayer(node, reg->version));
      } break;
      case kTfLiteBuiltinAveragePool2d:
      case kTfLiteBuiltinL2Pool2d:
      case kTfLiteBuiltinMaxPool2d: {
        TF_LITE_ENSURE_STATUS(
            builder.AddPool2dLayer(node, reg->builtin_code, reg->version));
      } break;
      case kTfLiteBuiltinSoftmax: {
        TF_LITE_ENSURE_STATUS(builder.AddSoftmaxLayer(node, reg->version));
      } break;
      case kTfLiteBuiltinSqueeze: {
        TF_LITE_ENSURE_STATUS(builder.AddSqueezeLayer(node, reg->version));
      } break;
      default:
        // All other operators are not mapped.
        return kTfLiteError;
    }
  }

  // Add inputs
  TF_LITE_ENSURE_STATUS(builder.AddInputs(input_tensors, inputBindings_));

  // Add output
  TF_LITE_ENSURE_STATUS(builder.AddOutputs(output_tensors, outputBindings_));

  // Connect everything together
  TF_LITE_ENSURE_STATUS(builder.Connect());

  return kTfLiteOk;
}
}  // namespace arm
}  // namespace delegate
}  // namespace tflite
