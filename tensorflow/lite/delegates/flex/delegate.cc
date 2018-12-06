/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/flex/delegate.h"

#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/flex/buffer_map.h"
#include "tensorflow/lite/delegates/flex/kernel.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace flex {
namespace delegate {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  // Get the nodes in the current execution plan. Interpreter owns this array.
  TfLiteIntArray* plan;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));

  // Add all custom ops starting with "Flex" to list of supported nodes.
  std::vector<int> supported_nodes;
  for (int node_index : TfLiteIntArrayView(plan)) {
    TfLiteNode* node;
    TfLiteRegistration* registration;
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));

    if (IsFlexOp(registration->custom_name)) {
      supported_nodes.push_back(node_index);
    }
  }

  // Request TFLite to partition the graph and make kernels for each independent
  // node sub set.
  TfLiteIntArray* size_and_nodes =
      ConvertVectorToTfLiteIntArray(supported_nodes);
  context->ReplaceNodeSubsetsWithDelegateKernels(context, GetKernel(),
                                                 size_and_nodes, delegate);
  TfLiteIntArrayFree(size_and_nodes);
  return kTfLiteOk;
}

TfLiteStatus CopyFromBufferHandle(TfLiteContext* context,
                                  TfLiteDelegate* delegate,
                                  TfLiteBufferHandle buffer_handle,
                                  TfLiteTensor* output) {
  BufferMap* buffer_map =
      reinterpret_cast<DelegateData*>(delegate->data_)->GetBufferMap(context);

  if (!buffer_map->HasTensor(buffer_handle)) {
    context->ReportError(context, "Invalid tensor index %d.", buffer_handle);
    return kTfLiteError;
  }

  tensorflow::Tensor t = buffer_map->GetTensor(buffer_handle);

  if (output->type == kTfLiteString) {
    if (t.dtype() != tensorflow::DT_STRING) {
      context->ReportError(context,
                           "Inconsistent type for TF string tensor index %d.",
                           buffer_handle);
      return kTfLiteError;
    }
    DynamicBuffer dynamic_buffer;

    auto tf_data = t.flat<string>();
    for (int i = 0; i < t.NumElements(); ++i) {
      dynamic_buffer.AddString(tf_data(i).data(), tf_data(i).size());
    }

    dynamic_buffer.WriteToTensor(output, /*new_shape=*/nullptr);
    return kTfLiteOk;
  }

  tensorflow::StringPiece t_data = t.tensor_data();

  if (output->bytes != t_data.size()) {
    context->ReportError(context,
                         absl::StrCat("The given ", output->bytes,
                                      " bytes are not enough to store "
                                      "TensorFlow's aligned buffer of size ",
                                      t_data.size(), " bytes.")
                             .c_str());
    return kTfLiteError;
  }

  memcpy(output->data.raw, t_data.data(), t_data.size());
  return kTfLiteOk;
}

}  // namespace delegate
}  // namespace flex

// Corresponding weak declaration found in lite/model.cc.
std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
AcquireFlexDelegate() {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      tflite::FlexDelegate::Create().release(), [](TfLiteDelegate* delegate) {
        delete reinterpret_cast<tflite::FlexDelegate*>(delegate);
      });
}

std::unique_ptr<FlexDelegate> FlexDelegate::Create() {
  std::unique_ptr<flex::DelegateData> delegate_data;
  if (!flex::DelegateData::Create(&delegate_data).ok()) {
    fprintf(stderr, "Unable to initialize TensorFlow context.\n");
    return nullptr;
  }

  return std::unique_ptr<FlexDelegate>(
      new FlexDelegate(std::move(delegate_data)));
}

FlexDelegate::FlexDelegate(std::unique_ptr<flex::DelegateData> delegate_data)
    : TfLiteDelegate(TfLiteDelegateCreate()),
      delegate_data_(std::move(delegate_data)) {
  data_ = delegate_data_.get();
  Prepare = &flex::delegate::Prepare;
  CopyFromBufferHandle = &flex::delegate::CopyFromBufferHandle;
  flags = kTfLiteDelegateFlagsAllowDynamicTensors;
}

FlexDelegate::~FlexDelegate() {}

}  // namespace tflite
