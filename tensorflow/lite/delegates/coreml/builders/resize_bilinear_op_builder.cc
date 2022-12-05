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
#include "tensorflow/lite/delegates/coreml/builders/resize_bilinear_op_builder.h"

#include <cstdint>
#include <memory>
#include <string>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/coreml/builders/op_factory.h"
#include "tensorflow/lite/delegates/coreml/builders/op_validator.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace coreml {

const std::string& ResizeBilinearOpBuilder::DebugName() {
  if (!debug_name_.empty()) return debug_name_;
  SetDebugName("ResizeBilinearOpBuilder", node_id_);
  return debug_name_;
}

CoreML::Specification::NeuralNetworkLayer* ResizeBilinearOpBuilder::Build() {
  if (layer_ == nullptr) {
    layer_ = std::make_unique<CoreML::Specification::NeuralNetworkLayer>();
  }
  layer_->set_name(DebugName());
  const TfLiteResizeBilinearParams* params =
      reinterpret_cast<const TfLiteResizeBilinearParams*>(builtin_data_);

  layer_->mutable_resizebilinear()->mutable_targetsize()->Add(height_);
  layer_->mutable_resizebilinear()->mutable_targetsize()->Add(width_);

  // align_corners makes last sampling position to be aligned with last index of
  // input. This is the same behavior as STRICT_ALIGN_ENDPOINTS_MODE in Core ML
  // sampling mode. When not set, the sampling positions are the same as
  // UPSAMPLE_MODE. (indices are in [0, (input_size-1)/output_size])
  if (params->align_corners) {
    layer_->mutable_resizebilinear()->mutable_mode()->set_samplingmethod(
        CoreML::Specification::SamplingMode::STRICT_ALIGN_ENDPOINTS_MODE);
  } else {
    layer_->mutable_resizebilinear()->mutable_mode()->set_samplingmethod(
        CoreML::Specification::SamplingMode::UPSAMPLE_MODE);
  }
  return layer_.release();
}

TfLiteStatus ResizeBilinearOpBuilder::RegisterInputs(
    const TfLiteIntArray* inputs, TfLiteContext* context) {
  if (inputs->size != 2) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of inputs to ResizeBilinear!.");
    return kTfLiteError;
  }
  AddInput(inputs->data[0]);
  TfLiteTensor* size = &context->tensors[inputs->data[1]];
  height_ = GetTensorData<int32_t>(size)[0];
  width_ = GetTensorData<int32_t>(size)[1];
  return kTfLiteOk;
}

TfLiteStatus ResizeBilinearOpBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of outputs to ResizeBilinear!.");
    return kTfLiteError;
  }
  graph_builder_->AddTensorWithID(outputs->data[0], GetOutput(context));
  return kTfLiteOk;
}

OpBuilder* CreateResizeBilinearOpBuilder(GraphBuilder* graph_builder) {
  return new ResizeBilinearOpBuilder(graph_builder);
}

bool IsResizeBilinearOpSupported(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) {
  if (node->builtin_data == nullptr) {
    return false;
  }
  const int kOutputSize = 1;
  if (!IsConstantTensor(GetInput(context, node, kOutputSize))) {
    TF_LITE_KERNEL_LOG(context,
                       "Output size of ResizeBilinear should be constant.");
    return false;
  }
  return true;
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
