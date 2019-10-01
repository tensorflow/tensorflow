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
#ifndef TENSORFLOW_LITE_DELEGATES_ARMNN_DELEGATE_KERNEL_H_
#define TENSORFLOW_LITE_DELEGATES_ARMNN_DELEGATE_KERNEL_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/c/c_api_internal.h"

#include "armnn/BackendId.hpp"
#include "armnn/ILayerSupport.hpp"
#include "armnn/INetwork.hpp"
#include "armnn/IRuntime.hpp"

namespace tflite {
namespace delegate {
namespace arm {

// The kernel that represents the node sub set of TF Lite being run on ArmNN.
class ArmNNDelegateKernel {
 public:
  ArmNNDelegateKernel();
  ~ArmNNDelegateKernel() = default;

  // Returns true if the node can be accelerated with ArmNN.
  static bool Validate(const TfLiteContext* context, int builtin_code,
                       int version, const TfLiteNode* node,
                       // Layer validator
                       const armnn::ILayerSupport* validator,
                       // Collects lists of failures collected during
                       // the validation of the possibility of accelerating
                       // the given node
                       std::vector<std::string>* map_failures = nullptr);

  // Initialize the kernel (a NN model).
  TfLiteStatus Init(TfLiteContext* context, const TfLiteDelegateParams* params);
  // Prepares the kernel (a NN model).
  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node);
  // Invokes/Runs the kernel (a NN model).
  TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node);

 private:
  // Build the ArmNN Network
  TfLiteStatus BuildNetwork(TfLiteContext* context,
                            const TfLiteIntArray* input_tensors,
                            const TfLiteIntArray* output_tensors);

 private:
  // Node indices that this delegate is responsible for. Indices here
  // indexes into the nodes array in the TfLiteContext.
  std::vector<int> nodes_;
  // ArmNN Runtime used to execute compiled networks
  armnn::IRuntimePtr runtime_;
  // ArmNN Network
  armnn::INetworkPtr network_;
  // Optimized ArmNN Network
  armnn::NetworkId executable_network_;
  // Binding information for inputs/outputs
  std::vector<armnn::BindingPointInfo> inputBindings_;
  std::vector<armnn::BindingPointInfo> outputBindings_;
};
}  // namespace arm
}  // namespace delegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_ARMNN_DELEGATE_KERNEL_H_
