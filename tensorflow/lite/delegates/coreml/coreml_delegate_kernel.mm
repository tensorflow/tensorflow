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
#include "tensorflow/lite/delegates/coreml/coreml_delegate_kernel.h"

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#import "tensorflow/lite/delegates/coreml/coreml_executor.h"

namespace tflite {
namespace delegates {
namespace coreml {
namespace {
// TODO(karimnosseir): Move to util library
TfLiteStatus GetDims(int* batch_size, int* height_size, int* width_size, int* depth_size,
                     const TfLiteIntArray* dims) {
  if (dims == nullptr || dims->size > 4) {
    return kTfLiteError;
  }
  int* dim[] = {batch_size, height_size, width_size, depth_size};
  for (int i = 0; i < 4; ++i) *(dim[i]) = 1;
  for (int i = 4 - dims->size; i < 4; ++i) {
    *dim[i] = dims->data[i - (4 - dims->size)];
  }
  return kTfLiteOk;
}

void TransposeToCHW(const float* hwc, float* chw, const TfLiteIntArray* hwc_dims) {
  int batch_size, height_size, width_size, depth_size;
  GetDims(&batch_size, &height_size, &width_size, &depth_size, hwc_dims);
  RuntimeShape hwc_shape({batch_size, height_size, width_size, depth_size});
  RuntimeShape chw_shape({batch_size, depth_size, height_size, width_size});
  TransposeParams params = {/*perm_count=*/4, /*perm=*/{0, 3, 1, 2}};
  optimized_ops::Transpose<float>(params, hwc_shape, hwc, chw_shape, chw);
}

void TransposeToHWC(const float* chw, float* hwc, const TfLiteIntArray* hwc_dims) {
  int batch_size, height_size, width_size, depth_size;
  GetDims(&batch_size, &height_size, &width_size, &depth_size, hwc_dims);
  RuntimeShape hwc_shape({batch_size, height_size, width_size, depth_size});
  RuntimeShape chw_shape({batch_size, depth_size, height_size, width_size});
  TransposeParams params = {/*perm_count=*/4, /*perm=*/{0, 2, 3, 1}};
  optimized_ops::Transpose<float>(params, chw_shape, chw, hwc_shape, hwc);
}
}  // namespace

TfLiteStatus CoreMlDelegateKernel::Init(TfLiteContext* context,
                                        const TfLiteDelegateParams* delegate_params) {
  if (@available(iOS 12.0, *)) {
    executor_ = [[::CoreMlExecutor alloc] init];
    TF_LITE_ENSURE_STATUS(BuildModel(context, delegate_params));
    // Serialize the model protocol buffer and compile it.
    if (model_ == nullptr) {
      TF_LITE_KERNEL_LOG(context, "Failed to createModel");
      return kTfLiteError;
    }
    NSURL* model_url = [executor_ saveModel:model_.get()];
    model_.reset();
    if (![executor_ build:model_url]) {
      TF_LITE_KERNEL_LOG(context, "Failed to Compile and save Model.");
      return kTfLiteError;
    }
    return kTfLiteOk;
  } else {
    TF_LITE_KERNEL_LOG(context, "Minimum required iOS version is 12.0.");
    return kTfLiteError;
  }
}

void CoreMlDelegateKernel::AddInputTensors(const TfLiteIntArray* input_tensors,
                                           TfLiteContext* context) {
  int num_inputs = 0;
  for (int i = 0; i < input_tensors->size; ++i) {
    const int tensor_id = input_tensors->data[i];
    const auto& tensor = context->tensors[tensor_id];
    builder_->AddTensorWithID(tensor_id, delegates::coreml::TensorID(0, num_inputs++));
  }
}

void CoreMlDelegateKernel::AddOutputTensors(const TfLiteIntArray* output_tensors,
                                            TfLiteContext* context) {
  auto* model_description = model_->mutable_description();
  for (int i = 0; i < output_tensors->size; ++i) {
    const int tensor_id = output_tensors->data[i];
    const auto& tensor = context->tensors[tensor_id];

    auto* output = model_description->mutable_output()->Add();
    output->set_name(builder_->GetTensorName(tensor_id));
    auto* multi_array = output->mutable_type()->mutable_multiarraytype();
    int batch_size, height_size, width_size, depth_size;
    GetDims(&batch_size, &height_size, &width_size, &depth_size, tensor.dims);
    multi_array->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);
    if (coreml_version_ >= 3) {
      multi_array->mutable_shape()->Add(batch_size);
    }
    multi_array->mutable_shape()->Add(depth_size);
    multi_array->mutable_shape()->Add(height_size);
    multi_array->mutable_shape()->Add(width_size);
  }
}

TfLiteStatus CoreMlDelegateKernel::BuildModel(TfLiteContext* context,
                                              const TfLiteDelegateParams* delegate_params) {
  TfLiteNode* node;
  TfLiteRegistration* reg;
  builder_.reset(new delegates::coreml::GraphBuilder(coreml_version_));
  // Add Inputs
  AddInputTensors(delegate_params->input_tensors, context);
  // Build all ops.
  for (int node_index : TfLiteIntArrayView(delegate_params->nodes_to_replace)) {
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(context, node_index, &node, &reg));
    auto* op_builder = builder_->AddBuilder(reg->builtin_code, node);
    if (op_builder == nullptr) {
      TF_LITE_KERNEL_LOG(context, "Failed to build node %d.", node_index);
      return kTfLiteError;
    }
    if (op_builder->RegisterInputs(node->inputs, context) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context, "Failed to add inputs for node %d.", node_index);
      return kTfLiteError;
    }
    if (op_builder->PopulateSubgraph(context) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context, "Failed to add sub-builders for node %d.", node_index);
      return kTfLiteError;
    }
    if (op_builder->RegisterOutputs(node->outputs, context) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context, "Failed to add outputs for node %d.", node_index);
      return kTfLiteError;
    }
  }
  model_.reset(builder_->BuildModel());
  if (model_ == nullptr) {
    TF_LITE_KERNEL_LOG(context, "Failed to build Model");
    return kTfLiteError;
  }
  AddOutputTensors(delegate_params->output_tensors, context);
  auto* model_description = model_->mutable_description();
  for (int i = 0; i < delegate_params->input_tensors->size; ++i) {
    const int tensor_id = delegate_params->input_tensors->data[i];
    if (builder_->IsTensorUsed(tensor_id)) {
      const auto& tensor = context->tensors[tensor_id];
      auto* input = model_description->mutable_input()->Add();
      input->set_name(builder_->GetTensorName(tensor_id));
      // TODO(karimnosseir): Handle different types ?
      auto* multi_array = input->mutable_type()->mutable_multiarraytype();
      int batch_size, height_size, width_size, depth_size;
      GetDims(&batch_size, &height_size, &width_size, &depth_size, tensor.dims);
      multi_array->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);
      if (coreml_version_ >= 3) {
        multi_array->mutable_shape()->Add(batch_size);
      }
      multi_array->mutable_shape()->Add(depth_size);
      multi_array->mutable_shape()->Add(height_size);
      multi_array->mutable_shape()->Add(width_size);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus CoreMlDelegateKernel::Prepare(TfLiteContext* context, TfLiteNode* node) {
  for (int tensor_index : TfLiteIntArrayView(node->inputs)) {
    if (builder_->IsTensorUsed(tensor_index)) {
      input_tensor_ids_.push_back(tensor_index);
    }
  }

  inputs_.reserve(input_tensor_ids_.size());
  for (int tensor_index : input_tensor_ids_) {
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    const int input_size = NumElements(tensor);
    int batch_size, height_size, width_size, depth_size;
    GetDims(&batch_size, &height_size, &width_size, &depth_size, tensor->dims);

    std::vector<int> input_shape = {depth_size, height_size, width_size};
    if (coreml_version_ >= 3) {
      input_shape.insert(input_shape.begin(), batch_size);
    }
    inputs_.push_back(
        {std::vector<float>(input_size), builder_->GetTensorName(tensor_index), input_shape});
  }

  outputs_.reserve(node->outputs->size);
  for (int tensor_index : TfLiteIntArrayView(node->outputs)) {
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    const int output_size = NumElements(tensor);

    outputs_.push_back({std::vector<float>(output_size), builder_->GetTensorName(tensor_index)});
  }

  return kTfLiteOk;
}

TfLiteStatus CoreMlDelegateKernel::Invoke(TfLiteContext* context, TfLiteNode* node) {
  if (@available(iOS 11.0, *)) {
    TfLiteIntArrayView node_inputs(node->inputs);
    for (int i = 0; i < input_tensor_ids_.size(); ++i) {
      const int tensor_id = input_tensor_ids_[i];
      TfLiteTensor* tensor = &context->tensors[tensor_id];
      // Transpose input to CHW.
      // TODO(b/143992544): try adding transpose op for inputs.
      TransposeToCHW(tensor->data.f, inputs_[i].data.data(), tensor->dims);
    }

    if (![executor_ invokeWithInputs:inputs_ outputs:outputs_]) {
      return kTfLiteError;
    }
    for (int i = 0; i < node->outputs->size; ++i) {
      TfLiteTensor* output_tensor = GetOutput(context, node, i);
      TransposeToHWC(outputs_[i].data.data(), output_tensor->data.f, output_tensor->dims);
    }
    return kTfLiteOk;
  } else {
    TF_LITE_KERNEL_LOG(context, "Minimum required iOS version is 11.0.");
    return kTfLiteError;
  }
}

CoreMlDelegateKernel::~CoreMlDelegateKernel() { [executor_ cleanup]; }

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
