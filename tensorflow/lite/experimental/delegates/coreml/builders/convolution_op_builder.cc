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
#include "tensorflow/lite/experimental/delegates/coreml/builders/convolution_op_builder.h"

#include "google/protobuf/repeated_field.h"
#include "external/coremltools/mlmodel/format/NeuralNetwork.pb.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/delegates/coreml/builders/activation_layer_builder.h"
#include "tensorflow/lite/experimental/delegates/coreml/builders/op_factory.h"
#include "tensorflow/lite/experimental/delegates/coreml/builders/op_validator.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace coreml {
const char* ConvolutionOpBuilder::DebugName() {
  if (!str_debug_name_[0])
    GetDebugName("ConvolutionOpBuilder", node_id_, str_debug_name_);
  return str_debug_name_;
}

void ConvolutionOpBuilder::SetWeights(TfLiteTensor* weights) {
  weights_ = weights;
}

void ConvolutionOpBuilder::SetBias(TfLiteTensor* bias) { bias_ = bias; }

CoreML::Specification::NeuralNetworkLayer* ConvolutionOpBuilder::Build() {
  if (layer_ == nullptr) {
    layer_.reset(new CoreML::Specification::NeuralNetworkLayer);
  }
  layer_->set_name(DebugName());

  int stride_height;
  int stride_width;
  int dilation_height;
  int dilation_width;
  TfLitePadding padding;

  if (is_depthwise_) {
    const auto* depthwise_conv_params =
        reinterpret_cast<const TfLiteDepthwiseConvParams*>(builtin_data_);
    stride_height = depthwise_conv_params->stride_height;
    stride_width = depthwise_conv_params->stride_width;
    dilation_height = depthwise_conv_params->dilation_height_factor;
    dilation_width = depthwise_conv_params->dilation_width_factor;
    padding = depthwise_conv_params->padding;

    // n_groups = kernel_channel / depth_multiplier
    layer_->mutable_convolution()->set_ngroups(
        weights_->dims->data[3] / depthwise_conv_params->depth_multiplier);
  } else {
    const auto* conv_params =
        reinterpret_cast<const TfLiteConvParams*>(builtin_data_);
    stride_height = conv_params->stride_height;
    stride_width = conv_params->stride_width;
    dilation_height = conv_params->dilation_height_factor;
    dilation_width = conv_params->dilation_width_factor;
    padding = conv_params->padding;

    layer_->mutable_convolution()->set_ngroups(1);
  }

  // If not set, it will default to (1,1)
  if (stride_height) {
    layer_->mutable_convolution()->add_stride(stride_height);
    layer_->mutable_convolution()->add_stride(stride_width);
  }

  layer_->mutable_convolution()->add_dilationfactor(dilation_height);
  layer_->mutable_convolution()->add_dilationfactor(dilation_width);

  switch (padding) {
    case kTfLitePaddingSame:
      layer_->mutable_convolution()->mutable_same();
      break;
    case kTfLitePaddingValid:
      layer_->mutable_convolution()->mutable_valid();
      break;
    case kTfLitePaddingUnknown:
      fprintf(stderr, "Padding is unknown.\n");
      break;
  }

  FillCoreMLWeights();
  FillCoreMLBias();
  // TODO(taeheej): add output shape when deconvolution == true

  return layer_.release();
}

void ConvolutionOpBuilder::FillCoreMLWeights() {
  if (is_depthwise_) {
    layer_->mutable_convolution()->set_kernelchannels(1);
    layer_->mutable_convolution()->set_outputchannels(weights_->dims->data[3]);
  } else {
    layer_->mutable_convolution()->set_kernelchannels(weights_->dims->data[3]);
    layer_->mutable_convolution()->set_outputchannels(weights_->dims->data[0]);
  }
  layer_->mutable_convolution()->add_kernelsize(weights_->dims->data[1]);
  layer_->mutable_convolution()->add_kernelsize(weights_->dims->data[2]);

  TransposeKernelWeights();  // Should be called after CoreML shape is set.
}

void ConvolutionOpBuilder::TransposeKernelWeights() {
  RuntimeShape tfl_shape(4, weights_->dims->data);
  // CoreML kernel has shape of (C_out, C_in, H, W)
  RuntimeShape coreml_shape(
      {static_cast<int>(layer_->convolution().outputchannels()),
       static_cast<int>(layer_->convolution().kernelchannels()),
       static_cast<int>(layer_->convolution().kernelsize()[0]),
       static_cast<int>(layer_->convolution().kernelsize()[1])});

  TransposeParams params;

  if (is_depthwise_) {
    // DepthwiseConv2D: TFL kernel has shape of (1, H, W, C_out),
    // and CoreML kernel has shape of (C_out, 1, H, W)
    params = {/*perm_count=*/4, /*perm=*/{3, 0, 1, 2}};
  } else {
    // Conv2D: TFL kernel has shape of (C_out, H, W, C_in),
    // and CoreML kernel has shape of (C_out, C_in, H, W)
    params = {/*perm_count=*/4, /*perm=*/{0, 3, 1, 2}};
  }

  auto* coreml_weights =
      layer_->mutable_convolution()->mutable_weights()->mutable_floatvalue();
  coreml_weights->Resize(NumElements(weights_), 0);

  optimized_ops::Transpose<float>(params, tfl_shape, weights_->data.f,
                                  coreml_shape, coreml_weights->mutable_data());
}

void ConvolutionOpBuilder::FillCoreMLBias() {
  if (bias_ != nullptr) {
    layer_->mutable_convolution()->set_hasbias(true);
    std::copy(bias_->data.f, bias_->data.f + NumElements(bias_->dims),
              google::protobuf::RepeatedFieldBackInserter(layer_->mutable_convolution()
                                                    ->mutable_bias()
                                                    ->mutable_floatvalue()));
  }
}

TfLiteStatus ConvolutionOpBuilder::PopulateSubgraph(TfLiteContext* context) {
  TfLiteFusedActivation activation;
  if (is_depthwise_) {
    const auto* depthwise_conv_params =
        reinterpret_cast<const TfLiteDepthwiseConvParams*>(builtin_data_);
    activation = depthwise_conv_params->activation;
  } else {
    const auto* conv_params =
        reinterpret_cast<const TfLiteConvParams*>(builtin_data_);
    activation = conv_params->activation;
  }

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

TfLiteStatus ConvolutionOpBuilder::RegisterInputs(const TfLiteIntArray* inputs,
                                                  TfLiteContext* context) {
  if (inputs->size != 2 && inputs->size != 3) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of inputs!.");
    return kTfLiteError;
  }
  AddInput(inputs->data[0]);
  SetWeights(&context->tensors[inputs->data[1]]);
  if (inputs->size > 2) {
    SetBias(&context->tensors[inputs->data[2]]);
  }
  return kTfLiteOk;
}

TfLiteStatus ConvolutionOpBuilder::RegisterOutputs(
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

OpBuilder* CreateConvolutionOpBuilder(GraphBuilder* graph_builder) {
  return new ConvolutionOpBuilder(graph_builder, /*is_depthwise=*/false);
}

OpBuilder* CreateDepthwiseConvolutionOpBuilder(GraphBuilder* graph_builder) {
  return new ConvolutionOpBuilder(graph_builder, /*is_depthwise=*/true);
}

bool IsConvolutionOpSupported(const TfLiteRegistration* registration,
                              const TfLiteNode* node, TfLiteContext* context) {
  if (node->builtin_data == nullptr) return false;
  const int kWeightTensor = 1;
  const int kBiasTensor = 2;

  TfLiteFusedActivation activation;
  int dilation_width_factor;
  int dilation_height_factor;

  if (registration->builtin_code == kTfLiteBuiltinConv2d) {
    const auto* conv_params =
        reinterpret_cast<const TfLiteConvParams*>(node->builtin_data);
    activation = conv_params->activation;
    dilation_width_factor = conv_params->dilation_width_factor;
    dilation_height_factor = conv_params->dilation_height_factor;
  } else if (registration->builtin_code == kTfLiteBuiltinDepthwiseConv2d) {
    const auto* depthwise_conv_params =
        reinterpret_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);
    activation = depthwise_conv_params->activation;
    dilation_width_factor = depthwise_conv_params->dilation_width_factor;
    dilation_height_factor = depthwise_conv_params->dilation_height_factor;
  } else {
    TF_LITE_KERNEL_LOG(context,
                       "Invalid op: op must be Conv2D or DepthwiseConv2D.");

    return false;
  }

  if (activation == kTfLiteActSignBit) {
    TF_LITE_KERNEL_LOG(context, "Unsupported activation.");
    return false;
  }

  const TfLiteTensor* weights = GetInput(context, node, kWeightTensor);
  const int max_kernel_size = 16384;
  if (!IsConstantTensor(weights)) {
    TF_LITE_KERNEL_LOG(context, "Weight tensor should be constant.");
    return false;
  }
  if (weights->dims->data[1] > max_kernel_size ||
      weights->dims->data[2] > max_kernel_size) {
    TF_LITE_KERNEL_LOG(
        context, "Core ML supports filters with width and height less than %d.",
        max_kernel_size);
    return false;
  }
  if (node->inputs->size >= kBiasTensor &&
      !IsConstantTensor(GetInput(context, node, kBiasTensor))) {
    TF_LITE_KERNEL_LOG(context, "Bias tensor should be constant.");
    return false;
  }

  return true;
}

bool IsDepthwiseConvolutionOpSupported(const TfLiteRegistration* registration,
                                       const TfLiteNode* node,
                                       TfLiteContext* context) {
  return IsConvolutionOpSupported(registration, node, context);
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
