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
#ifndef TENSORFLOW_LITE_DELEGATES_ARMNN_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_ARMNN_BUILDER_H_

#include <string>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"

#include "armnn/INetwork.hpp"
#include "armnn/Tensor.hpp"

namespace tflite {
namespace delegate {
namespace arm {

// Network Builder for ArmNN
class ArmNNBuilder {
 public:
  // Constructor
  ArmNNBuilder(TfLiteContext* context, armnn::INetwork* network);

  // Add a convolution layer to network
  TfLiteStatus AddConvolution2dLayer(const TfLiteNode* node, int version);
  // Add a depthwise convolution layer to network
  TfLiteStatus AddDepthwiseConvolution2dLayer(const TfLiteNode* node,
                                              int version);
  // Add a pool layer to network
  TfLiteStatus AddPool2dLayer(const TfLiteNode* node, int builtin_code,
                              int version);
  // Add a softmax layer to network
  TfLiteStatus AddSoftmaxLayer(const TfLiteNode* node, int version);
  // Add a squeeze layer to network
  TfLiteStatus AddSqueezeLayer(const TfLiteNode* node, int version);

  // Add inputs to graph
  TfLiteStatus AddInputs(const TfLiteIntArray* inputs,
                         std::vector<armnn::BindingPointInfo>& inputBindings);
  // Add outputs to graph
  TfLiteStatus AddOutputs(const TfLiteIntArray* outputs,
                          std::vector<armnn::BindingPointInfo>& outputBindings);
  // Connect nodes in graph
  TfLiteStatus Connect();

 private:
  bool RegisterProducerOfTensor(size_t tensorIndex, armnn::IOutputSlot* slot);
  void RegisterConsumerOfTensor(size_t tensorIndex, armnn::IInputSlot* slot);
  bool RegisterInputSlots(armnn::IConnectableLayer* layer,
                          const std::vector<unsigned int>& tensorIndexes);
  bool RegisterOutputSlots(armnn::IConnectableLayer* layer,
                           const std::vector<unsigned int>& tensorIndexes);
  armnn::IConnectableLayer* AddFusedActivationLayer(
      armnn::IConnectableLayer* prevLayer, unsigned int outputSlot,
      TfLiteFusedActivation activationType);
  bool GenerateConnections(
      armnn::IConnectableLayer* layer, const TfLiteNode* node,
      TfLiteFusedActivation activationType = kTfLiteActNone);

 private:
  TfLiteContext* context_;
  armnn::INetwork* network_;

  /// A mapping that indicates the connections of each output slot
  /// to other input slots
  struct TensorSlots {
    armnn::IOutputSlot* outputSlot;
    std::vector<armnn::IInputSlot*> inputSlots;

    TensorSlots() : outputSlot(nullptr) {}
  };
  using TensorConnections = std::vector<TensorSlots>;
  /// Connections for tensors in each subgraph
  /// The first index is the subgraph ID, the second index is the tensor ID
  TensorConnections graphConnections_;
};
}  // namespace arm
}  // namespace delegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_ARMNN_BUILDER_H_
