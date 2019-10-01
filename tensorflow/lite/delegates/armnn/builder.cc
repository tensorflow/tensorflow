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
#include "tensorflow/lite/delegates/armnn/builder.h"

#include <memory>

#include "tensorflow/lite/delegates/armnn/descriptor_helpers.h"
#include "tensorflow/lite/delegates/armnn/macros.h"
#include "tensorflow/lite/delegates/armnn/utils.h"

#include "armnn/INetwork.hpp"
#include "armnn/Optional.hpp"
#include "armnn/Types.hpp"
#include "armnn/TypesUtils.hpp"
#include "armnnUtils/Permute.hpp"

namespace tflite {
namespace delegate {
namespace arm {
namespace {
// Generates a layer binding id
armnn::LayerBindingId GenerateLayerBindingId(size_t tensorIndex) {
  return static_cast<armnn::LayerBindingId>((tensorIndex));
}

// DataStorage stores data until they are transferred to the network
struct DataStorage {
 public:
  DataStorage() : f32_(nullptr), u8_(nullptr), s32_(nullptr) {}
  // Convenience constructors
  DataStorage(std::unique_ptr<float[]>&& data)
      : f32_(std::move(data)), u8_(nullptr), s32_(nullptr) {}
  DataStorage(std::unique_ptr<uint8_t[]>&& data)
      : f32_(nullptr), u8_(std::move(data)), s32_(nullptr) {}
  DataStorage(std::unique_ptr<int32_t[]>&& data)
      : f32_(nullptr), u8_(nullptr), s32_(std::move(data)) {}

 private:
  std::unique_ptr<float[]> f32_;
  std::unique_ptr<uint8_t[]> u8_;
  std::unique_ptr<int32_t[]> s32_;
};

template <typename T>
std::pair<armnn::ConstTensor, std::unique_ptr<T[]>> CreateConstTensorImpl(
    TfLiteTensor* tensor, armnn::TensorInfo& tensorInfo,
    armnn::Optional<armnn::PermutationVector&> permutationVector) {
  std::unique_ptr<T[]> data(new T[tensorInfo.GetNumElements()]);
  if (permutationVector.has_value() &&
      permutationVector.value().GetSize() > 0) {
    tensorInfo = armnnUtils::Permuted(tensorInfo, permutationVector.value());
    armnnUtils::Permute(tensorInfo.GetShape(), permutationVector.value(),
                        reinterpret_cast<const T*>(tensor->data.raw),
                        data.get(), sizeof(T));
  } else {
    ::memcpy(data.get(), tensor->data.raw, tensorInfo.GetNumBytes());
  }
  return std::make_pair(armnn::ConstTensor(tensorInfo, data.get()),
                        std::move(data));
}

template <typename T>
std::pair<armnn::ConstTensor, DataStorage> CreateConstTensorAndStoreData(
    TfLiteTensor* tensor, armnn::TensorInfo& tensorInfo,
    armnn::Optional<armnn::PermutationVector&> permutationVector) {
  auto constData =
      CreateConstTensorImpl<T>(tensor, tensorInfo, permutationVector);
  DataStorage storage(std::move(constData.second));
  return std::make_pair(constData.first, std::move(storage));
}

std::pair<armnn::ConstTensor, DataStorage> CreateConstTensor(
    TfLiteTensor* tensor, armnn::TensorInfo& tensorInfo,
    armnn::Optional<armnn::PermutationVector&> permutationVector) {
  switch (tensorInfo.GetDataType()) {
    case armnn::DataType::Float32:
      return CreateConstTensorAndStoreData<float>(tensor, tensorInfo,
                                                  permutationVector);
    case armnn::DataType::QuantisedAsymm8:
      return CreateConstTensorAndStoreData<uint8_t>(tensor, tensorInfo,
                                                    permutationVector);
    case armnn::DataType::Signed32:
      return CreateConstTensorAndStoreData<int32_t>(tensor, tensorInfo,
                                                    permutationVector);
    default: {
      return std::make_pair(armnn::ConstTensor(tensorInfo, nullptr),
                            DataStorage());
    }
  }
}
}  // namespace

ArmNNBuilder::ArmNNBuilder(TfLiteContext* context, armnn::INetwork* network)
    : context_(context), network_(network), graphConnections_() {
  graphConnections_.resize(context_->tensors_size);
}

TfLiteStatus ArmNNBuilder::AddConvolution2dLayer(const TfLiteNode* node,
                                                 int version) {
  std::vector<armnn::TensorInfo> inputsInfo;
  std::vector<armnn::TensorInfo> outputsInfo;
  GetInputsInfo(context_, node, inputsInfo);
  GetOutputsInfo(context_, node, outputsInfo);

  const auto& input = inputsInfo[0];
  auto& filter = inputsInfo[1];
  auto& output = outputsInfo[0];
  const bool hasBias = (inputsInfo.size() > 2);

  const auto builtin = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
  armnn::Convolution2dDescriptor desc;
  ToConv2dDescriptor(builtin, version, input.GetShape(), filter.GetShape(),
                     hasBias, desc);

  auto filterTensorAndData =
      CreateConstTensor(&context_->tensors[node->inputs->data[1]], filter,
                        armnn::EmptyOptional());

  armnn::IConnectableLayer* layer = nullptr;
  if (hasBias) {
    auto biasTensorAndData =
        CreateConstTensor(&context_->tensors[node->inputs->data[2]],
                          inputsInfo[2], armnn::EmptyOptional());
    layer = network_->AddConvolution2dLayer(
        desc, filterTensorAndData.first,
        armnn::Optional<armnn::ConstTensor>(biasTensorAndData.first));
  } else {
    layer = network_->AddConvolution2dLayer(desc, filterTensorAndData.first,
                                            armnn::EmptyOptional());
  }
  RETURN_TFLITE_ERROR_IF(layer == nullptr);

  layer->GetOutputSlot(0).SetTensorInfo(output);
  RETURN_TFLITE_ERROR_ON_FALSE(
      GenerateConnections(layer, node, builtin->activation));

  return kTfLiteOk;
}

TfLiteStatus ArmNNBuilder::AddDepthwiseConvolution2dLayer(
    const TfLiteNode* node, int version) {
  std::vector<armnn::TensorInfo> inputsInfo;
  std::vector<armnn::TensorInfo> outputsInfo;

  GetInputsInfo(context_, node, inputsInfo);
  GetOutputsInfo(context_, node, outputsInfo);

  const auto& input = inputsInfo[0];
  auto& filter = inputsInfo[1];
  auto& output = outputsInfo[0];
  const bool hasBias = (inputsInfo.size() > 2);

  const auto builtin =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  armnn::DepthwiseConvolution2dDescriptor desc;
  ToDepthwiseConvDescriptor(builtin, version, input.GetShape(),
                            filter.GetShape(), hasBias, desc);

  // Reshape weights as [ H, W, I, M ]
  armnn::TensorInfo filterTensorInfo = filter;
  filterTensorInfo.SetShape(
      {filterTensorInfo.GetShape()[1], filterTensorInfo.GetShape()[2],
       input.GetShape()[3],
       filterTensorInfo.GetShape()[3] / input.GetShape()[3]});

  // Mappings from TensorflowLite filter tensors to the ArmNN filter tensors
  // (ArmNN weights have to be [M, I, H, W])
  armnn::PermutationVector permutationVector{
      2, 3, 1, 0};  // [H, W, I, M] -> [M, I, H, W]

  auto filterTensorAndData =
      CreateConstTensor(&context_->tensors[node->inputs->data[1]],
                        filterTensorInfo, permutationVector);

  armnn::IConnectableLayer* layer = nullptr;
  if (hasBias) {
    auto biasTensorAndData =
        CreateConstTensor(&context_->tensors[node->inputs->data[2]],
                          inputsInfo[2], armnn::EmptyOptional());
    layer = network_->AddDepthwiseConvolution2dLayer(
        desc, filterTensorAndData.first,
        armnn::Optional<armnn::ConstTensor>(biasTensorAndData.first));
  } else {
    layer = network_->AddDepthwiseConvolution2dLayer(
        desc, filterTensorAndData.first, armnn::EmptyOptional());
  }
  RETURN_TFLITE_ERROR_IF(layer == nullptr);

  layer->GetOutputSlot(0).SetTensorInfo(output);
  RETURN_TFLITE_ERROR_ON_FALSE(
      GenerateConnections(layer, node, builtin->activation));

  return kTfLiteOk;
}

TfLiteStatus ArmNNBuilder::AddPool2dLayer(const TfLiteNode* node,
                                          int builtin_code, int version) {
  std::vector<armnn::TensorInfo> inputsInfo;
  std::vector<armnn::TensorInfo> outputsInfo;

  GetInputsInfo(context_, node, inputsInfo);
  GetOutputsInfo(context_, node, outputsInfo);

  const auto& input = inputsInfo[0];
  auto& output = outputsInfo[0];

  const auto builtin = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  armnn::Pooling2dDescriptor desc;
  ToPool2dDescriptor(builtin, builtin_code, version, input.GetShape(), desc);

  armnn::IConnectableLayer* layer = network_->AddPooling2dLayer(desc);
  RETURN_TFLITE_ERROR_IF(layer == nullptr);

  layer->GetOutputSlot(0).SetTensorInfo(output);
  RETURN_TFLITE_ERROR_ON_FALSE(
      GenerateConnections(layer, node, builtin->activation));

  return kTfLiteOk;
}

TfLiteStatus ArmNNBuilder::AddSoftmaxLayer(const TfLiteNode* node,
                                           int version) {
  std::vector<armnn::TensorInfo> inputsInfo;
  std::vector<armnn::TensorInfo> outputsInfo;

  GetInputsInfo(context_, node, inputsInfo);
  GetOutputsInfo(context_, node, outputsInfo);

  const auto builtin =
      reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);
  const auto& input = inputsInfo[0];
  const auto& output = outputsInfo[0];

  armnn::SoftmaxDescriptor desc;
  ToSoftmaxDescriptor(builtin, version, desc);

  armnn::IConnectableLayer* layer = network_->AddSoftmaxLayer(desc);
  RETURN_TFLITE_ERROR_IF(layer == nullptr);

  layer->GetOutputSlot(0).SetTensorInfo(output);
  RETURN_TFLITE_ERROR_ON_FALSE(GenerateConnections(layer, node));

  return kTfLiteOk;
}

TfLiteStatus ArmNNBuilder::AddSqueezeLayer(const TfLiteNode* node,
                                           int version) {
  std::vector<armnn::TensorInfo> inputsInfo;
  std::vector<armnn::TensorInfo> outputsInfo;

  GetInputsInfo(context_, node, inputsInfo);
  GetOutputsInfo(context_, node, outputsInfo);

  const auto builtin =
      reinterpret_cast<TfLiteSqueezeParams*>(node->builtin_data);
  const auto& input = inputsInfo[0];
  auto& output = outputsInfo[0];

  armnn::ReshapeDescriptor desc;
  ToSqueezeDescriptor(builtin, version, input.GetShape(), desc);

  armnn::IConnectableLayer* layer = network_->AddReshapeLayer(desc);
  RETURN_TFLITE_ERROR_IF(layer == nullptr);

  layer->GetOutputSlot(0).SetTensorInfo(output);
  RETURN_TFLITE_ERROR_ON_FALSE(GenerateConnections(layer, node));

  return kTfLiteOk;
}

TfLiteStatus ArmNNBuilder::AddInputs(
    const TfLiteIntArray* inputs,
    std::vector<armnn::BindingPointInfo>& inputBindings) {
  const size_t numInputs = inputs->size;
  for (unsigned int i = 0; i < numInputs; ++i) {
    const int32_t tensorId = inputs->data[i];
    const TfLiteTensor tensor = context_->tensors[tensorId];

    auto bindingId = GenerateLayerBindingId(tensorId);
    armnn::IConnectableLayer* layer = network_->AddInputLayer(bindingId);

    auto tensorInfo = ToTensorInfo(&tensor);
    layer->GetOutputSlot(0).SetTensorInfo(tensorInfo.value());

    RegisterOutputSlots(layer, {static_cast<uint32_t>(tensorId)});

    // Do not create bindings for constant inputs
    if (tensor.allocation_type != kTfLiteMmapRo) {
      inputBindings.push_back(std::make_pair(bindingId, tensorInfo.value()));
    }
  }

  return kTfLiteOk;
}

TfLiteStatus ArmNNBuilder::AddOutputs(
    const TfLiteIntArray* outputs,
    std::vector<armnn::BindingPointInfo>& outputBindings) {
  const size_t numOutputs = outputs->size;
  for (unsigned int i = 0; i < numOutputs; ++i) {
    const int32_t tensorId = outputs->data[i];
    const TfLiteTensor tensor = context_->tensors[tensorId];

    auto bindingId = GenerateLayerBindingId(tensorId);
    armnn::IConnectableLayer* layer = network_->AddOutputLayer(bindingId);

    RegisterInputSlots(layer, {static_cast<uint32_t>(tensorId)});

    auto tensorInfo = ToTensorInfo(&tensor);
    outputBindings.push_back(std::make_pair(bindingId, tensorInfo.value()));
  }

  return kTfLiteOk;
}

TfLiteStatus ArmNNBuilder::Connect() {
  // Establish the connections from the layer outputs to respective inputs
  for (size_t tensorIndex = 0; tensorIndex < graphConnections_.size();
       ++tensorIndex) {
    if (graphConnections_[tensorIndex].outputSlot != nullptr) {
      for (size_t inputSlotIdx = 0;
           inputSlotIdx < graphConnections_[tensorIndex].inputSlots.size();
           ++inputSlotIdx) {
        graphConnections_[tensorIndex].outputSlot->Connect(
            *(graphConnections_[tensorIndex].inputSlots[inputSlotIdx]));
      }
    }
  }
  return kTfLiteOk;
}

bool ArmNNBuilder::RegisterProducerOfTensor(size_t tensorIndex,
                                            armnn::IOutputSlot* slot) {
  TensorSlots& tensorSlots = graphConnections_[tensorIndex];
  // Assume there is only one producer for that tensor
  RETURN_FALSE_IF(tensorSlots.outputSlot != nullptr);
  tensorSlots.outputSlot = slot;
  return true;
}

void ArmNNBuilder::RegisterConsumerOfTensor(size_t tensorIndex,
                                            armnn::IInputSlot* slot) {
  TensorSlots& tensorSlots = graphConnections_[tensorIndex];
  tensorSlots.inputSlots.push_back(slot);
}

bool ArmNNBuilder::RegisterInputSlots(
    armnn::IConnectableLayer* layer,
    const std::vector<unsigned int>& tensorIndexes) {
  RETURN_FALSE_IF(layer == nullptr);
  RETURN_FALSE_IF(tensorIndexes.size() != layer->GetNumInputSlots());
  for (unsigned int slotIndex = 0; slotIndex < layer->GetNumInputSlots();
       ++slotIndex) {
    unsigned int tensorIndex = tensorIndexes[slotIndex];
    armnn::IInputSlot* slot = &(layer->GetInputSlot(slotIndex));
    RegisterConsumerOfTensor(tensorIndex, slot);
  }
  return true;
}

bool ArmNNBuilder::RegisterOutputSlots(
    armnn::IConnectableLayer* layer,
    const std::vector<unsigned int>& tensorIndexes) {
  RETURN_FALSE_IF(layer == nullptr);
  RETURN_FALSE_IF(tensorIndexes.size() != layer->GetNumOutputSlots());
  for (unsigned int slotIndex = 0; slotIndex < layer->GetNumOutputSlots();
       ++slotIndex) {
    unsigned int tensorIndex = tensorIndexes[slotIndex];
    armnn::IOutputSlot* slot = &(layer->GetOutputSlot(slotIndex));
    RegisterProducerOfTensor(tensorIndex, slot);
  }
  return true;
}

bool ArmNNBuilder::GenerateConnections(armnn::IConnectableLayer* layer,
                                       const TfLiteNode* node,
                                       TfLiteFusedActivation activationType) {
  // register the input connection slots for the layer, connections are made
  // after all layers have been created
  // only the tensors for the inputs are relevant, exclude the const tensors
  std::vector<unsigned int> inputTensorIndexes;
  RETURN_ON_FALSE(AsUnsignedVector(node->inputs->data, node->inputs->size,
                                   inputTensorIndexes));
  RegisterInputSlots(layer, {inputTensorIndexes[0]});

  // Add fused activation if needed
  layer = AddFusedActivationLayer(layer, 0, activationType);
  RETURN_FALSE_IF(layer == nullptr);

  // register the output connection slots for the layer, connections are made
  // after all layers have been created
  std::vector<unsigned int> outputTensorIndexes;
  RETURN_ON_FALSE(AsUnsignedVector(node->outputs->data, node->outputs->size,
                                   outputTensorIndexes));
  RegisterOutputSlots(layer, {outputTensorIndexes[0]});

  return true;
}

armnn::IConnectableLayer* ArmNNBuilder::AddFusedActivationLayer(
    armnn::IConnectableLayer* prevLayer, unsigned int outputSlot,
    TfLiteFusedActivation activationType) {
  armnn::ActivationDescriptor activationDesc;

  switch (activationType) {
    case kTfLiteActNone: {
      // this is a no-op: return previous layer
      return prevLayer;
    }
    case kTfLiteActRelu: {
      activationDesc.m_Function = armnn::ActivationFunction::ReLu;
      break;
    }
    case kTfLiteActRelu1: {
      activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
      activationDesc.m_A = 1.0f;
      activationDesc.m_B = 0.0f;
      break;
    }
    case kTfLiteActRelu6: {
      activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
      activationDesc.m_A = 6.0f;
      activationDesc.m_B = 0.0f;
      break;
    }
    case kTfLiteActTanh: {
      activationDesc.m_Function = armnn::ActivationFunction::TanH;
      activationDesc.m_A = 1.0f;
      activationDesc.m_B = 1.0f;
      break;
    }
    case kTfLiteActSigmoid: {
      activationDesc.m_Function = armnn::ActivationFunction::Sigmoid;
      break;
    }

    // I only put these here as a reminder what others we could support
    case kTfLiteActSignBit:
    default: { return nullptr; }
  }

  armnn::IConnectableLayer* activationLayer =
      network_->AddActivationLayer(activationDesc);

  auto& prevOutputSlot = prevLayer->GetOutputSlot(outputSlot);
  prevOutputSlot.Connect(activationLayer->GetInputSlot(0));
  activationLayer->GetOutputSlot(0).SetTensorInfo(
      prevOutputSlot.GetTensorInfo());
  return activationLayer;
}
}  // namespace arm
}  // namespace delegate
}  // namespace tflite
