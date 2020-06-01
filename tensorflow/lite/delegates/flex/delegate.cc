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

#include <memory>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/flex/buffer_map.h"
#include "tensorflow/lite/delegates/flex/kernel.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {

// Corresponding weak declaration found in lite/interpreter_builder.cc.
TfLiteDelegateUniquePtr AcquireFlexDelegate() {
  return tflite::FlexDelegate::Create();
}

TfLiteDelegateUniquePtr FlexDelegate::Create(
    std::unique_ptr<FlexDelegate> base_delegate) {
  TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO,
                       "Created TensorFlow Lite delegate for select TF ops.");
  if (base_delegate == nullptr) {
    base_delegate.reset(new FlexDelegate());
  }
  auto flex_delegate = TfLiteDelegateFactory::Create(std::move(base_delegate));
  flex_delegate->CopyFromBufferHandle =
      [](TfLiteContext* context, TfLiteDelegate* delegate,
         TfLiteBufferHandle buffer_handle,
         TfLiteTensor* tensor) -> TfLiteStatus {
    return reinterpret_cast<FlexDelegate*>(delegate->data_)
        ->CopyFromBufferHandle(context, buffer_handle, tensor);
  };
  flex_delegate->flags |= kTfLiteDelegateFlagsAllowDynamicTensors;
  return flex_delegate;
}

TfLiteStatus FlexDelegate::Initialize(TfLiteContext* context) {
  // If the TensorFlow Lite thread count is explicitly configured, use it,
  // otherwise rely on the default TensorFlow threading behavior.
  tensorflow::SessionOptions session_options;
  if (context->recommended_num_threads > 0) {
    session_options.config.set_intra_op_parallelism_threads(
        context->recommended_num_threads);
  }

  auto status = delegate_data_.Prepare(session_options);
  if (!status.ok()) {
    context->ReportError(context, "Failed to initialize TensorFlow context: %s",
                         status.error_message().c_str());
    return kTfLiteError;
  }

  return kTfLiteOk;
}

const char* FlexDelegate::name() const {
  static constexpr char kName[] = "TfLiteFlexDelegate";
  return kName;
}

bool FlexDelegate::IsNodeSupportedByDelegate(
    const TfLiteRegistration* registration, const TfLiteNode* node,
    TfLiteContext* context) const {
  return IsFlexOp(registration->custom_name);
}

std::unique_ptr<SimpleDelegateKernelInterface>
FlexDelegate::CreateDelegateKernelInterface() {
  return std::unique_ptr<SimpleDelegateKernelInterface>(
      new tflite::flex::DelegateKernel());
}

TfLiteStatus FlexDelegate::CopyFromBufferHandle(
    TfLiteContext* context, TfLiteBufferHandle buffer_handle,
    TfLiteTensor* output) {
  flex::BufferMap* buffer_map = delegate_data_.GetBufferMap(context);

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

    auto tf_data = t.flat<tensorflow::tstring>();
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

}  // namespace tflite
