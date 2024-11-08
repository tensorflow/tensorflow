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
#include "tensorflow/lite/delegates/coreml/builders/pooling_layer_builder.h"

#include <cstdio>
#include <string>
#include <vector>

#include "mlmodel/format/NeuralNetwork.pb.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/coreml/builders/op_factory.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace coreml {

const std::string& PoolingLayerBuilder::DebugName() {
  if (!debug_name_.empty()) return debug_name_;
  switch (pooling_type_) {
    case kTfLiteBuiltinAveragePool2d:
      SetDebugName("PoolingLayerBuilder (AVERAGE)", node_id_);
      break;
    case kTfLiteBuiltinMaxPool2d:
      SetDebugName("PoolingLayerBuilder (MAX)", node_id_);
      break;
    case kTfLiteBuiltinL2Pool2d:
      SetDebugName("PoolingLayerBuilder (L2, unsupported)", node_id_);
      break;
    case kTfLiteBuiltinMean:
      SetDebugName("PoolingLayerBuilder (MEAN)", node_id_);
      break;
    default:
      SetDebugName("PoolingLayerBuilder (ERROR)", node_id_);
  }
  return debug_name_;
}

CoreML::Specification::NeuralNetworkLayer* PoolingLayerBuilder::Build() {
  layer_->set_name(DebugName());
  auto* pooling_params = layer_->mutable_pooling();

  if (pooling_type_ == kTfLiteBuiltinMean) {
    pooling_params->set_type(
        CoreML::Specification::PoolingLayerParams::AVERAGE);
    pooling_params->set_globalpooling(true);
    return layer_.release();
  }

  const TfLitePoolParams* params =
      reinterpret_cast<const TfLitePoolParams*>(builtin_data_);
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
  if (pooling_type_ == kTfLiteBuiltinMean) {
    if (inputs->size != 2) {
      TF_LITE_KERNEL_LOG(context, "Wrong # of inputs to Mean!.");
      return kTfLiteError;
    }
  } else if (inputs->size != 1) {
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

OpBuilder* CreateMeanOpBuilder(GraphBuilder* graph_builder) {
  return new PoolingLayerBuilder(graph_builder, kTfLiteBuiltinMean);
}

// Only supports averaging over H and W dimensions, as
bool IsMeanOpSupported(const TfLiteRegistration* registration,
                       const TfLiteNode* node, TfLiteContext* context) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteTensor* axis = GetInput(context, node, 1);
  const auto* params =
      reinterpret_cast<TfLiteReducerParams*>(node->builtin_data);

  if (!params->keep_dims) {
    TF_LITE_KERNEL_LOG(context, "keep_dims should be true for Mean op.");
    return false;
  }
  if (input->dims->size != 4) {
    TF_LITE_KERNEL_LOG(context, "Mean op is only supported for 4D input.");
    return false;
  }
  const int* axis_data = GetTensorData<int>(axis);
  std::vector<bool> axis_mask = {false, true, true, false};
  for (int i = 0; i < axis->dims->data[0]; ++i) {
    if (!axis_mask[(axis_data[i] + 4) % 4]) {
      TF_LITE_KERNEL_LOG(context,
                         "Mean op should reduce for H and W dimensions.");
      return false;
    }
  }
  return true;
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
