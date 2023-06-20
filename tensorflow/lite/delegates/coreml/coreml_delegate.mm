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
#include "tensorflow/lite/delegates/coreml/coreml_delegate.h"

#include <string.h>
#include <sys/utsname.h>
#include <limits>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/coreml/builders/op_validator.h"
#include "tensorflow/lite/delegates/coreml/builders/util.h"
#include "tensorflow/lite/delegates/coreml/coreml_delegate_kernel.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace {
constexpr int kMinNodesPerCoreMlDelegate = 2;

using delegates::coreml::CoreMlDelegateKernel;

bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration, const TfLiteNode* node,
                               TfLiteContext* context, const TfLiteCoreMlDelegateOptions* options) {
  if (@available(iOS 11.0, *)) {
  } else {
    return false;
  }

  // For most ops, only version 1 is supported.
  if (registration->version > 1) {
    switch (registration->builtin_code) {
      case kTfLiteBuiltinDepthwiseConv2d:
        if (registration->version > 2) return false;
        break;
      // FullyConnected without bias is supported starting from version 6.
      case kTfLiteBuiltinFullyConnected:
        if (registration->version > 6) return false;
        break;
      default:
        return false;
    }
  }

  // The model should not be full-integer quantized. For ops supported by Core ML delegate,
  // Testing if the first input is float is sufficient to filter full-integer quantized ops.
  int input_tensor_index = 0;
  // TransposeConv input: (output_shape, filters, input)
  if (registration->builtin_code == kTfLiteBuiltinTransposeConv) {
    input_tensor_index = 2;
  }
  if (GetInput(context, node, input_tensor_index)->type != kTfLiteFloat32) {
    return false;
  }

  // TODO(b/149179044): Add extra validation if this is not sufficient.

  // TODO(karimnossier): Refactor this function.
  // TODO(karimnosseir): Add
  // 1) Checks for versioning.
  // 2) Checks for input constraints.
  // Follow the ordering of TfLiteBuiltinOperator enum.
  switch (registration->builtin_code) {
    case kTfLiteBuiltinAdd: {
      return node->builtin_data != nullptr &&
             delegates::coreml::IsBinaryOpSupported(registration, node, context);
    }
    case kTfLiteBuiltinAveragePool2d: {
      const auto* params = reinterpret_cast<const TfLitePoolParams*>(node->builtin_data);
      return params != nullptr && params->activation == kTfLiteActNone;
    }
    case kTfLiteBuiltinConcatenation: {
      return delegates::coreml::IsConcatenationOpSupported(registration, node, context);
    }
    case kTfLiteBuiltinConv2d: {
      return delegates::coreml::IsConvolutionOpSupported(registration, node, context);
    }
    case kTfLiteBuiltinDepthwiseConv2d: {
      return delegates::coreml::IsDepthwiseConvolutionOpSupported(registration, node, context);
    }
    case kTfLiteBuiltinFullyConnected: {
      return delegates::coreml::IsFullyConnectedOpSupported(registration, node, context);
    }
    case kTfLiteBuiltinHardSwish: {
      return true;
    }
    case kTfLiteBuiltinLogistic: {
      return true;
    }
    case kTfLiteBuiltinMaxPool2d: {
      const auto* params = reinterpret_cast<const TfLitePoolParams*>(node->builtin_data);
      return params != nullptr && params->activation == kTfLiteActNone;
    }
    case kTfLiteBuiltinMirrorPad: {
      return delegates::coreml::IsMirrorPadOpSupported(registration, node, context);
    }
    case kTfLiteBuiltinMean: {
      return delegates::coreml::IsMeanOpSupported(registration, node, context);
    }
    case kTfLiteBuiltinMul: {
      return node->builtin_data != nullptr &&
             delegates::coreml::IsBinaryOpSupported(registration, node, context);
    }
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinPadv2: {
      return delegates::coreml::IsPadOpSupported(registration, node, context);
    }
    case kTfLiteBuiltinRelu: {
      return true;
    }
    case kTfLiteBuiltinReluN1To1: {
      return true;
    }
    case kTfLiteBuiltinRelu6: {
      return true;
    }
    case kTfLiteBuiltinReshape: {
      return delegates::coreml::IsReshapeOpSupported(registration, node, context,
                                                     options->coreml_version);
    }
    case kTfLiteBuiltinResizeBilinear: {
      return delegates::coreml::IsResizeBilinearOpSupported(registration, node, context);
    }
    case kTfLiteBuiltinSoftmax: {
      // Only supports when beta is 1.0 for now.
      const auto* softmax_params = reinterpret_cast<const TfLiteSoftmaxParams*>(node->builtin_data);
      return softmax_params != nullptr && softmax_params->beta == 1.0;
    }
    case kTfLiteBuiltinTanh: {
      return true;
    }
    case kTfLiteBuiltinTransposeConv: {
      return delegates::coreml::IsTransposeConvolutionOpSupported(registration, node, context);
    }
    default:
      return false;
  }
  return false;
}

class CoreMlDelegate : public TfLiteDelegate {
 public:
  explicit CoreMlDelegate(const TfLiteCoreMlDelegateOptions* params)
      : params_(params != nullptr ? *params : TfLiteCoreMlDelegateOptions()) {
    {
      if (@available(iOS 13.0, *)) {
        if (params_.coreml_version != 2 && params_.coreml_version != 3) {
          NSLog(@"coreml_version must be 2 or 3. Setting to 3.");
          params_.coreml_version = 3;
        }
      } else if (@available(iOS 12.0, *)) {
        if (params_.coreml_version != 2) {
          NSLog(@"coreml_version must be 2 - using Core ML version 2.");
          params_.coreml_version = 2;
        }
      }
      if (params_.max_delegated_partitions <= 0) {
        params_.max_delegated_partitions = std::numeric_limits<int>::max();
      }
      if (params_.min_nodes_per_partition <= 0) {
        params_.min_nodes_per_partition = kMinNodesPerCoreMlDelegate;
      }
#ifdef TFLITE_DEBUG_DELEGATE
      if (params_.first_delegate_node_index < 0) {
        params_.first_delegate_node_index = 0;
      }
      if (params->last_delegate_node_index <= 0) {
        params_.last_delegate_node_index = std::numeric_limits<int>::max();
      }
#endif
    }
  }

  TfLiteCoreMlDelegateOptions* params() { return &params_; }

  bool VerifyDelegate() { return true; }

 private:
  TfLiteCoreMlDelegateOptions params_;
};

TfLiteRegistration GetCoreMlKernelRegistration() {
  // This is the registration for the Delegate Node that gets added to
  // the TFLite graph instead of the subGraph it replaces it.
  // It is treated as an OP node. But in our case
  // Init will initialize the delegate
  // Invoke will run the delegate graph.
  // Prepare for prearing the delegate.
  // Free for any cleaning needed by the delegate.
  TfLiteRegistration kernel_registration{};
  kernel_registration.profiling_string = nullptr;
  kernel_registration.builtin_code = kTfLiteBuiltinDelegate;
  kernel_registration.custom_name = "TfLiteCoreMlDelegate";
  kernel_registration.free = [](TfLiteContext* context, void* buffer) -> void {
    delete reinterpret_cast<CoreMlDelegateKernel*>(buffer);
  };
  kernel_registration.init = [](TfLiteContext* context, const char* buffer,
                                size_t length) -> void* {
    const auto* params = reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    const auto* coreml_options = (reinterpret_cast<CoreMlDelegate*>(params->delegate))->params();
    CoreMlDelegateKernel* coreml_kernel = new CoreMlDelegateKernel(coreml_options->coreml_version);
    if (coreml_kernel->Init(context, params) != kTfLiteOk) {
      delete coreml_kernel;
      return nullptr;
    }
    return coreml_kernel;
  };
  kernel_registration.invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
    CoreMlDelegateKernel* kernel = reinterpret_cast<CoreMlDelegateKernel*>(node->user_data);
    if (!kernel) {
      TF_LITE_KERNEL_LOG(context, "CoreMl Kernel was not initialized");
      return kTfLiteError;
    }
    return kernel->Invoke(context, node);
  };
  kernel_registration.prepare = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
    CoreMlDelegateKernel* kernel = reinterpret_cast<CoreMlDelegateKernel*>(node->user_data);
    if (kernel == nullptr) {
      TF_LITE_KERNEL_LOG(context, "CoreMl Kernel was not initialized");
      return kTfLiteError;
    }
    return kernel->Prepare(context, node);
  };

  return kernel_registration;
}

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  const auto* params = reinterpret_cast<TfLiteCoreMlDelegateOptions*>(delegate->data_);

  delegates::IsNodeSupportedFn node_supported_fn = [=](TfLiteContext* context, TfLiteNode* node,
                                                       TfLiteRegistration* registration,
                                                       std::string* unsupported_details) -> bool {
    return IsNodeSupportedByDelegate(registration, node, context, params);
  };

  delegates::FP16GraphPartitionHelper partition_helper(context, node_supported_fn);
#ifndef TFLITE_DEBUG_DELEGATE
  TF_LITE_ENSURE_STATUS(partition_helper.Partition(nullptr));
#else
  TF_LITE_ENSURE_STATUS(partition_helper.Partition(nullptr, params->first_delegate_node_index,
                                                   params->last_delegate_node_index));
#endif

  std::vector<int> delegated_nodes = partition_helper.GetNodesOfFirstNLargestPartitions(
      params->max_delegated_partitions, params->min_nodes_per_partition);
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "CoreML delegate: %d nodes delegated out of %d nodes, "
                  "with %d partitions.\n",
                  delegated_nodes.size(), partition_helper.num_total_nodes(),
                  partition_helper.num_partitions());
  return context->ReplaceNodeSubsetsWithDelegateKernels(
      context, GetCoreMlKernelRegistration(), BuildTfLiteArray(delegated_nodes).get(), delegate);
}

TfLiteDelegate* CreateCoreMlDelegate(const TfLiteCoreMlDelegateOptions* options) {
  TfLiteDelegate* delegate = new CoreMlDelegate(options);
  if (!static_cast<CoreMlDelegate*>(delegate)->VerifyDelegate()) {
    delete delegate;
    return nullptr;
  }

  delegate->data_ = static_cast<tflite::CoreMlDelegate*>(delegate)->params();
  delegate->flags = kTfLiteDelegateFlagsNone;
  delegate->Prepare = &DelegatePrepare;
  delegate->CopyFromBufferHandle = nullptr;
  delegate->CopyToBufferHandle = nullptr;
  delegate->FreeBufferHandle = nullptr;

  return delegate;
}
}  // namespace
}  // namespace tflite

namespace {
// utsname.machine has device identifier. For example, identifier for iPhone Xs is "iPhone11,2".
// Since Neural Engine is only available for use on A12 and later, major device version in the
// identifier is checked for these models:
// A12: iPhone XS (11,2), iPad Mini - 5th Gen (11,1)
// A12X: iPad Pro - 3rd Gen (8,1)
// For more information, see https://www.theiphonewiki.com/wiki/Models
bool IsNeuralEngineAvailable() {
  struct utsname system_info;
  uname(&system_info);

  if (strncmp("iPad", system_info.machine, 4) == 0) {
    const int major_version = atoi(system_info.machine + 4);
    return major_version >= 8;  // There are no device between iPad 8 and 11.
  } else if (strncmp("iPhone", system_info.machine, 6) == 0) {
    const int major_version = atoi(system_info.machine + 6);
    return major_version >= 11;
  }
  return false;
}

}  // namespace

TfLiteDelegate* TfLiteCoreMlDelegateCreate(const TfLiteCoreMlDelegateOptions* options) {
  if (@available(iOS 12.0, *)) {
    if (options->enabled_devices == TfLiteCoreMlDelegateDevicesWithNeuralEngine &&
        !IsNeuralEngineAvailable()) {
      NSLog(@"This device does not have Neural Engine, so Core ML delegate will not be enabled. "
             "If you want to run Core ML delegate anyway, set enabled_devices option to "
             "TfLiteCoreMlDelegateAllDevices (or enabledDevices to .allDevices in Swift).");
      return nullptr;
    }
    return tflite::CreateCoreMlDelegate(options);
  } else {
    NSLog(@"Core ML delegate is not supported in this iOS version. "
           "Minimum required iOS version is 12.0.");
    return nullptr;
  }
}

void TfLiteCoreMlDelegateDelete(TfLiteDelegate* delegate) { delete delegate; }
