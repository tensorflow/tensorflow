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
#include "tensorflow/lite/experimental/delegates/coreml/builders/op_builder.h"

#include "external/coremltools/mlmodel/format/NeuralNetwork.pb.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/experimental/delegates/coreml/builders/op_factory.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace coreml {
OpBuilder* GraphBuilder::AddBuilder(int builtin_code, const TfLiteNode* node) {
  // Follow the ordering of TfLiteBuiltinOperator enum.
  switch (builtin_code) {
    case kTfLiteBuiltinAdd:
      return AddBuilder(CreateAddOpBuilder, node);
    case kTfLiteBuiltinAveragePool2d:
      return AddBuilder(CreateAveragePool2dOpBuilder, node);
    case kTfLiteBuiltinConcatenation:
      return AddBuilder(CreateConcatenationOpBuilder, node);
    case kTfLiteBuiltinConv2d:
      return AddBuilder(CreateConvolutionOpBuilder, node);
    case kTfLiteBuiltinDepthwiseConv2d:
      return AddBuilder(CreateDepthwiseConvolutionOpBuilder, node);
    case kTfLiteBuiltinFullyConnected:
      return AddBuilder(CreateFullyConnectedOpBuilder, node);
    case kTfLiteBuiltinLogistic:
      return AddBuilder(CreateLogisticOpBuilder, node);
    case kTfLiteBuiltinMaxPool2d:
      return AddBuilder(CreateMaxPool2dOpBuilder, node);
    case kTfLiteBuiltinMul:
      return AddBuilder(CreateMulOpBuilder, node);
    case kTfLiteBuiltinRelu:
      return AddBuilder(CreateReluOpBuilder, node);
    case kTfLiteBuiltinReluN1To1:
      return AddBuilder(CreateReluN1To1OpBuilder, node);
    case kTfLiteBuiltinRelu6:
      return AddBuilder(CreateRelu6OpBuilder, node);
    case kTfLiteBuiltinReshape:
      return AddBuilder(CreateReshapeOpBuilder, node);
    case kTfLiteBuiltinResizeBilinear:
      return AddBuilder(CreateResizeBilinearOpBuilder, node);
    case kTfLiteBuiltinSoftmax:
      return AddBuilder(CreateSoftmaxOpBuilder, node);
    case kTfLiteBuiltinTanh:
      return AddBuilder(CreateTanhOpBuilder, node);
    case kTfLiteBuiltinTransposeConv:
      return AddBuilder(CreateTransposeConvolutionOpBuilder, node);
    case kTfLiteBuiltinHardSwish:
      return AddBuilder(CreateHardSwishOpBuilder, node);
    default:
      return nullptr;
  }
}

OpBuilder* GraphBuilder::AddBuilder(
    const std::function<OpBuilder*(GraphBuilder*)>& builder,
    const TfLiteNode* node) {
  if (builder == nullptr) {
    fprintf(stderr, "builder should be set.\n");
    return nullptr;
  }
  OpBuilder* op = builder(this);

  builders_.emplace_back(op);
  op->SetNodeID(builders_.size());
  if (node != nullptr) {
    op->SetBuiltinData(node->builtin_data);
    op->SetTfLiteNode(node);
  }
  return builders_.back().get();
}

CoreML::Specification::Model* GraphBuilder::BuildModel() {
  CoreML::Specification::Model* model = new CoreML::Specification::Model();
  if (coreml_version_ == 2) {  // Core ML 2, iOS >= 12.0
    model->set_specificationversion(3);
  } else if (coreml_version_ == 3) {  // Core ML 3, iOS >= 13.0
    model->set_specificationversion(4);
    model->mutable_neuralnetwork()->set_arrayinputshapemapping(
        CoreML::Specification::EXACT_ARRAY_MAPPING);
  } else {
    fprintf(stderr, "Unsupported Core ML version: %d\n", coreml_version_);
    return nullptr;
  }
  auto* neural_network = model->mutable_neuralnetwork();
  for (auto& builder : builders_) {
    CoreML::Specification::NeuralNetworkLayer* layer = builder->Build();
    if (layer == nullptr) {
      fprintf(stderr, "Null layer returned from builder: %s\n",
              builder->DebugName());
      continue;
    }
    neural_network->mutable_layers()->AddAllocated(layer);
  }
  return model;
}

void GraphBuilder::AddTensorWithID(int tf_tensor_id,
                                   const TensorID& tensor_id) {
  if (tensors_.size() <= tf_tensor_id) {
    tensors_.resize(tf_tensor_id + 1);
    used_tensor_.resize(tf_tensor_id + 1);
  }
  tensors_[tf_tensor_id] = tensor_id;
}

std::string GraphBuilder::GetTensorName(int tensor_id) {
  return GetTensorID(tensor_id).ToString();
}

const TensorID GraphBuilder::GetTensorID(int tensor_id) {
  if (!HasTensor(tensor_id)) {
    // TODO(karimnosseir): Double check if this happened, if we are
    // adding in execution order it shouldn't happen.
    fprintf(stderr, "index out of range...!!! Requested index %d , size %d\n",
            tensor_id, static_cast<int>(tensors_.size()));
    // Return invalid ID.
    return TensorID(-1, -1);
  }
  used_tensor_[tensor_id] = true;
  return tensors_[tensor_id];
}

bool GraphBuilder::HasTensor(int tflite_tensor_index) {
  if (tensors_.size() <= tflite_tensor_index) {
    return false;
  }
  return tensors_[tflite_tensor_index].NodeID() != -1;
}

bool GraphBuilder::IsTensorUsed(int tflite_tensor_index) {
  if (!HasTensor(tflite_tensor_index)) return false;
  return used_tensor_[tflite_tensor_index];
}

void OpBuilder::AddInput(const std::string& input_name) {
  if (layer_ == nullptr) {
    layer_.reset(new CoreML::Specification::NeuralNetworkLayer);
  }
  *layer_->mutable_input()->Add() = input_name;
}

void OpBuilder::AddInput(const TensorID& input_id) {
  AddInput(input_id.ToString());
}

void OpBuilder::AddInput(int tf_input_id) {
  AddInput(graph_builder_->GetTensorName(tf_input_id));
}

TensorID OpBuilder::AddOutput() {
  auto tensor_id = TensorID(GetID(), num_outputs_++);
  *layer_->mutable_output()->Add() = tensor_id.ToString();
  return tensor_id;
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
