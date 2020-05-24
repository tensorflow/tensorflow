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
#include "tensorflow/lite/experimental/delegates/coreml/builders/pooling_layer_builder.h"

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/delegates/coreml/builders/op_factory.h"

namespace tflite {
namespace delegates {
namespace coreml {

const char* PoolingLayerBuilder::DebugName() {
  if (str_debug_name_[0]) return str_debug_name_;
  switch (pooling_type_) {
    case kTfLiteBuiltinAveragePool2d:
      GetDebugName("PoolingLayerBuilder (AVERAGE)", node_id_, str_debug_name_);
      break;

    case kTfLiteBuiltinMaxPool2d:
      GetDebugName("PoolingLayerBuilder (MAX)", node_id_, str_debug_name_);
      break;
    case kTfLiteBuiltinL2Pool2d:
      GetDebugName("PoolingLayerBuilder (L2, unsupported)",
                   node_id_, str_debug_name_);
      break;
    default:
      GetDebugName("PoolingLayerBuilder (ERROR)", node_id_, str_debug_name_);
  }
  return str_debug_name_;
}

CoreML::Specification::NeuralNetworkLayer* PoolingLayerBuilder::Build() {
  if (layer_ == nullptr) {
    layer_.reset(new CoreML::Specification::NeuralNetworkLayer);
  }
  layer_->set_name(DebugName());
  const TfLitePoolParams* params =
      reinterpret_cast<const TfLitePoolParams*>(builtin_data_);
  auto* pooling_params = layer_->mutable_pooling();
  pooling_params->mutable_stride()->Add(params->stride_height);
  pooling_params->mutable_stride()->Add(params->stride_width);
  pooling_params->mutable_kernelsize()->Add(params->filter_height);
  pooling_params->mutable_kernelsize()->Add(params->filter_width);

  if (params->padding == kTfLitePaddingSame) {
    pooling_params->mutable_same();
  } else {
    pooling_params->mutable_valid();
  }

  switch (pooling_type_) {
    case kTfLiteBuiltinAveragePool2d:
      pooling_params->set_type(
          CoreML::Specification::PoolingLayerParams::AVERAGE);
      pooling_params->set_avgpoolexcludepadding(true);
      break;
    case kTfLiteBuiltinMaxPool2d:
      pooling_params->set_type(CoreML::Specification::PoolingLayerParams::MAX);
      break;
    case kTfLiteBuiltinL2Pool2d:
      // TODO(b/145873272) implement L2 pooling
      // NOLINTNEXTLINE: minimize absl usage
      fprintf(stderr, "L2 pooling is not supported yet.\n");
      return nullptr;
    default:
      // NOLINTNEXTLINE: minimize absl usage
      fprintf(stderr, "Unexpected pooling type.\n");  // Should not reach here.
      return nullptr;
  }

  // TODO(b/145582958): Add padding values.
  // TODO(b/145582958): Handle fused activation function.
  return layer_.release();
}

TfLiteStatus PoolingLayerBuilder::RegisterInputs(const TfLiteIntArray* inputs,
                                                 TfLiteContext* context) {
  if (inputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of inputs to Pooling!.");
    return kTfLiteError;
  }
  AddInput(inputs->data[0]);
  return kTfLiteOk;
}

TfLiteStatus PoolingLayerBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                                  TfLiteContext* context) {
  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of outputs to Pooling!.");
    return kTfLiteError;
  }
  graph_builder_->AddTensorWithID(outputs->data[0], GetOutput(context));
  return kTfLiteOk;
}

OpBuilder* CreateAveragePool2dOpBuilder(GraphBuilder* graph_builder) {
  return new PoolingLayerBuilder(graph_builder, kTfLiteBuiltinAveragePool2d);
}

OpBuilder* CreateMaxPool2dOpBuilder(GraphBuilder* graph_builder) {
  return new PoolingLayerBuilder(graph_builder, kTfLiteBuiltinMaxPool2d);
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
