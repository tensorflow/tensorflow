/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/delegates/coreml/builders/fully_connected_op_builder.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/delegates/coreml/builders/activation_layer_builder.h"
#include "tensorflow/lite/experimental/delegates/coreml/builders/op_factory.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace coreml {
const char* FullyConnectedOpBuilder::DebugName() {
  if (!str_debug_name_[0])
    GetDebugName("FullyConnectedOpBuilder", node_id_, str_debug_name_);
  return str_debug_name_;
}

void FullyConnectedOpBuilder::SetWeights(TfLiteTensor* weights) {
  weights_ = weights;
}

void FullyConnectedOpBuilder::SetBias(TfLiteTensor* bias) { bias_ = bias; }

CoreML::Specification::NeuralNetworkLayer* FullyConnectedOpBuilder::Build() {
  if (layer_ == nullptr) {
    layer_.reset(new CoreML::Specification::NeuralNetworkLayer);
  }
  layer_->set_name(DebugName());

  FillCoreMLWeights();
  FillCoreMLBias();

  return layer_.release();
}

void FullyConnectedOpBuilder::FillCoreMLWeights() {
  layer_->mutable_innerproduct()->set_inputchannels(weights_->dims->data[1]);
  layer_->mutable_innerproduct()->set_outputchannels(weights_->dims->data[0]);
  if (weights_->type == kTfLiteFloat32) {
    const float* weights_data = GetTensorData<float>(weights_);
    std::copy(weights_data, weights_data + NumElements(weights_),
              google::protobuf::RepeatedFieldBackInserter(layer_->mutable_innerproduct()
                                                    ->mutable_weights()
                                                    ->mutable_floatvalue()));
  } else if (weights_->type == kTfLiteFloat16) {
    // float16value has type of bytes (std::string)
    layer_->mutable_innerproduct()
        ->mutable_weights()
        ->mutable_float16value()
        ->assign(weights_->data.raw, weights_->bytes);
  }
}

void FullyConnectedOpBuilder::FillCoreMLBias() {
  if (bias_ != nullptr) {
    layer_->mutable_innerproduct()->set_hasbias(true);
    if (bias_->type == kTfLiteFloat32) {
      const float* bias_data = GetTensorData<float>(bias_);
      std::copy(bias_data, bias_data + NumElements(bias_),
                google::protobuf::RepeatedFieldBackInserter(layer_->mutable_innerproduct()
                                                      ->mutable_bias()
                                                      ->mutable_floatvalue()));
    } else if (bias_->type == kTfLiteFloat16) {
      // float16value has type of bytes (std::string)
      layer_->mutable_innerproduct()
          ->mutable_bias()
          ->mutable_float16value()
          ->assign(bias_->data.raw, bias_->bytes);
    }
  }
}

TfLiteStatus FullyConnectedOpBuilder::PopulateSubgraph(TfLiteContext* context) {
  const auto* fc_params =
      reinterpret_cast<const TfLiteFullyConnectedParams*>(builtin_data_);
  TfLiteFusedActivation activation = fc_params->activation;

  if (activation == kTfLiteActNone) {
    builder_output_ = AddOutput();
  } else {
    ActivationLayerBuilder* activation_builder =
        reinterpret_cast<ActivationLayerBuilder*>(
            graph_builder_->AddBuilder(CreateActivationLayerBuilder, nullptr));
    activation_builder->SetActivation(activation);
    activation_builder->AddInput(AddOutput());
    activation_builder->PopulateSubgraph(context);
    builder_output_ = activation_builder->GetOutput(context);
  }
  return kTfLiteOk;
}

TfLiteStatus FullyConnectedOpBuilder::RegisterInputs(
    const TfLiteIntArray* inputs, TfLiteContext* context) {
  const int kInput = 0;
  const int kWeights = 1;
  const int kBias = 2;
  AddInput(inputs->data[kInput]);
  SetWeights(&context->tensors[inputs->data[kWeights]]);
  if (inputs->size > 2) {
    SetBias(&context->tensors[inputs->data[kBias]]);
  }
  return kTfLiteOk;
}

TfLiteStatus FullyConnectedOpBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of outputs!.");
    return kTfLiteError;
  }
  TensorID output_tensor = GetOutput(context);
  if (output_tensor.NodeID() == -1) {
    TF_LITE_KERNEL_LOG(context, "Failed to build output tensor.");
    return kTfLiteError;
  }
  graph_builder_->AddTensorWithID(outputs->data[0], output_tensor);
  return kTfLiteOk;
}

OpBuilder* CreateFullyConnectedOpBuilder(GraphBuilder* graph_builder) {
  return new FullyConnectedOpBuilder(graph_builder);
}

bool IsFloatType(TfLiteType type) {
  return type == kTfLiteFloat32 || type == kTfLiteFloat16;
}

bool IsFullyConnectedOpSupported(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) {
  if (node->builtin_data == nullptr) return false;
  const auto* fc_params =
      reinterpret_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);
  const int kInput = 0;
  const int kWeights = 1;
  const int kBias = 2;

  if (fc_params->weights_format != kTfLiteFullyConnectedWeightsFormatDefault) {
    return false;
  }
  const TfLiteTensor* input = GetInput(context, node, kInput);
  const TfLiteTensor* weights = GetInput(context, node, kWeights);

  if (!IsFloatType(input->type)) {
    return false;
  }
  if (!IsFloatType(weights->type) || !IsConstantTensor(weights)) {
    return false;
  }
  // Core ML 2 only supports single-batch fully connected layer, thus dimensions
  // except the last one should be 1.
  if (input->dims->data[input->dims->size - 1] != NumElements(input)) {
    return false;
  }

  if (node->inputs->size > 2) {
    const TfLiteTensor* bias = GetInput(context, node, kBias);
    if (!IsFloatType(bias->type) || !IsConstantTensor(bias)) {
      return false;
    }
  }

  TfLiteFusedActivation activation = fc_params->activation;
  if (activation == kTfLiteActSignBit) {
    return false;
  }
  return true;
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
