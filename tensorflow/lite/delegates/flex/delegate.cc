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
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/macros.h"
#include "tensorflow/lite/delegates/flex/buffer_map.h"
#include "tensorflow/lite/delegates/flex/kernel.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {

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
  reinterpret_cast<FlexDelegate*>(flex_delegate->data_)->base_delegate_ =
      flex_delegate.get();
  return flex_delegate;
}

TfLiteStatus FlexDelegate::Initialize(TfLiteContext* context) {
  // If the TensorFlow Lite thread count is explicitly configured, use it,
  // otherwise rely on the default TensorFlow threading behavior.
  tensorflow::SessionOptions session_options;
  // We don't run multiple ops at the same time, so prefer using
  // 1 thread for inter-op parallelism.
  // Negative value means all are done on the caller thread.
  session_options.config.set_inter_op_parallelism_threads(-1);
  if (context->recommended_num_threads > 0) {
    session_options.config.set_intra_op_parallelism_threads(
        context->recommended_num_threads);
  }

  auto status = delegate_data_.Prepare(
      session_options, reinterpret_cast<Subgraph*>(context->impl_),
      base_delegate_);
  if (!status.ok()) {
    TF_LITE_KERNEL_LOG(context, "Failed to initialize TensorFlow context: %s",
                       status.error_message().c_str());
    return kTfLiteError;
  }

  // Initializes the cancellation manager.
  if (!cancellation_manager_) {
    cancellation_manager_ =
        absl::make_unique<tensorflow::CancellationManager>();
    delegate_data_.SetCancellationManager(cancellation_manager_.get());
  }

  return kTfLiteOk;
}

const char* FlexDelegate::Name() const {
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
    TF_LITE_KERNEL_LOG(context, "Invalid tensor index %d.", buffer_handle);
    return kTfLiteError;
  }

  tensorflow::Tensor t = buffer_map->GetTensor(buffer_handle);

  if (output->type == kTfLiteString) {
    if (t.dtype() != tensorflow::DT_STRING) {
      TF_LITE_KERNEL_LOG(context,
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

  // TODO(b/179094265): This is an experimental implementation, subject to
  // change. This can be re-implemented with life cycle management mechanism
  // like reference counting.
  // When copying resource and variant tensors from Flex delegate to TensorFlow
  // Lite tensors, the CopyFromBufferHandle method of the Flex delegate is
  // invoked and it will store the `data` field of the given TensorFlow Lite
  // tensor and pass the TensorFlow Lite tensor pointer. Copying the `data`
  // field will act as passing pointers between TensorFlow Lite tensors.
  //
  // The life cycle of the pointer will be managed by the reference counting in
  // the TensorFlow world and the pointer will be freed when all the buffer
  // maps, who own it, are gone.
  if (flex::IsResourceOrVariant(output)) {
    const size_t required_bytes = sizeof(tensorflow::Tensor**);
    const tensorflow::Tensor** tf_tensor_ptr =
        reinterpret_cast<const tensorflow::Tensor**>(malloc(required_bytes));
    *tf_tensor_ptr = buffer_map->GetTensorPtr(buffer_handle);

    TfLiteTensorDataFree(output);
    output->data.raw = reinterpret_cast<char*>(tf_tensor_ptr);
    output->bytes = required_bytes;
    output->data_is_stale = true;
    return kTfLiteOk;
  }

  tensorflow::StringPiece t_data = t.tensor_data();

  if (output->bytes != t_data.size()) {
    TF_LITE_KERNEL_LOG(context,
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

void FlexDelegate::Cancel() { cancellation_manager_->StartCancel(); }

bool FlexDelegate::HasCancelled(void* data) {
  if (data == nullptr) {
    return false;
  }

  auto* flex_delegate = static_cast<FlexDelegate*>(data);
  return flex_delegate->cancellation_manager_->IsCancelled();
}

}  // namespace tflite
