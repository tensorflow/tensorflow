/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/shim/tflite_tensor_view.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/type_to_tflitetype.h"

// Creates a case statement for the switch() clause given the dtype
#define CASE_FOR_DTYPE_GIVEN_CPP_DTYPE(TFLITE_DTYPE, CPP_DTYPE) \
  case TFLITE_DTYPE: {                                          \
    using DType = typename CPP_DTYPE;                           \
    return TfLiteTensorView(wrapped_tensor, DType());           \
  }

#define CASE_FOR_DTYPE(TFLITE_DTYPE) \
  CASE_FOR_DTYPE_GIVEN_CPP_DTYPE(    \
      TFLITE_DTYPE, ::tflite::TfLiteTypeToType<TFLITE_DTYPE>::Type)

namespace tflite {
namespace shim {

TfLiteTensorView::TfLiteTensorView(::TfLiteTensor *wrapped_tensor,
                                   const ::tensorflow::tstring &dtype)
    : TensorView(absl::Span<int>(wrapped_tensor->dims->data,
                                 wrapped_tensor->dims->size),
                 nullptr, 0, dtype),
      wrapped_tensor_(wrapped_tensor),
      const_wrapped_tensor_(wrapped_tensor) {
  InitForStringDType();
}

TfLiteTensorView::TfLiteTensorView(const ::TfLiteTensor *wrapped_tensor,
                                   const ::tensorflow::tstring &dtype)
    : TensorView(absl::Span<int>(wrapped_tensor->dims->data,
                                 wrapped_tensor->dims->size),
                 nullptr, 0, dtype),
      const_wrapped_tensor_(wrapped_tensor) {
  InitForStringDType();
}

TfLiteTensorView::TfLiteTensorView(TfLiteTensorView &&o) noexcept
    : TensorView(std::move(o)),
      wrapped_tensor_(o.wrapped_tensor_),
      const_wrapped_tensor_(o.const_wrapped_tensor_),
      str_vec_(std::move(o.str_vec_)) {
  if (absl::holds_alternative<absl::Span<::tensorflow::tstring>>(data_)) {
    InitForStringDType();
  }
}

TfLiteTensorView::TfLiteTensorView(const TfLiteTensorView &o)
    : TensorView(o),
      wrapped_tensor_(o.wrapped_tensor_),
      const_wrapped_tensor_(o.const_wrapped_tensor_),
      str_vec_(o.str_vec_) {
  if (absl::holds_alternative<absl::Span<::tensorflow::tstring>>(data_)) {
    InitForStringDType();
  }
}

TfLiteTensorView &TfLiteTensorView::operator=(TfLiteTensorView &&o) noexcept {
  wrapped_tensor_ = o.wrapped_tensor_;
  const_wrapped_tensor_ = o.const_wrapped_tensor_;
  str_vec_ = std::move(o.str_vec_);
  TensorView::operator=(std::move(o));
  if (absl::holds_alternative<absl::Span<::tensorflow::tstring>>(data_)) {
    InitForStringDType();
  }
  return *this;
}

TfLiteTensorView &TfLiteTensorView::operator=(const TfLiteTensorView &o) {
  if (&o == this) return *this;
  TensorView::operator=(o);
  wrapped_tensor_ = o.wrapped_tensor_;
  const_wrapped_tensor_ = o.const_wrapped_tensor_;
  str_vec_ = o.str_vec_;
  if (absl::holds_alternative<absl::Span<::tensorflow::tstring>>(data_)) {
    InitForStringDType();
  }
  return *this;
}

void TfLiteTensorView::InitForStringDType() {
  if (str_vec_ == nullptr) {
    str_vec_ = std::make_shared<StringBuffer>(this);
  }
  data_ = absl::Span<::tensorflow::tstring>(str_vec_->buffer);
}

TfLiteTensorView::StringBuffer::StringBuffer(TfLiteTensorView *t_view)
    : wrapped_tensor(t_view->wrapped_tensor_) {
  buffer.resize(NumElements(t_view->shape_));
  // Read the TfLite string into the buffer
  const auto const_wrapped_tensor = t_view->const_wrapped_tensor_;
  std::size_t str_count;
  if (const_wrapped_tensor->data.raw == nullptr)
    str_count = 0;
  else
    str_count = ::tflite::GetStringCount(const_wrapped_tensor);
  for (int i = 0; i < str_count; ++i) {
    const auto str_ref = ::tflite::GetString(const_wrapped_tensor, i);
    buffer[i].assign_as_view(str_ref.str, str_ref.len);
  }
}

TfLiteTensorView::StringBuffer::~StringBuffer() {
  if (wrapped_tensor == nullptr) return;
  tflite::DynamicBuffer buf;
  for (const auto &s : buffer) buf.AddString(s.data(), s.length());
  buf.WriteToTensor(wrapped_tensor, /*new_shape=*/nullptr);
}

template <typename TfLiteTensorType>
absl::StatusOr<
    typename MatchConstNess<TfLiteTensorType, TfLiteTensorView>::Type>
TfLiteTensorViewTemplatizedNew(TfLiteTensorType *wrapped_tensor) {
  switch (wrapped_tensor->type) {
    CASE_FOR_DTYPE(kTfLiteBool);
    CASE_FOR_DTYPE(kTfLiteUInt8);
    CASE_FOR_DTYPE(kTfLiteUInt64);
    CASE_FOR_DTYPE(kTfLiteInt8);
    CASE_FOR_DTYPE(kTfLiteInt16);
    CASE_FOR_DTYPE(kTfLiteInt32);
    CASE_FOR_DTYPE(kTfLiteInt64);
    CASE_FOR_DTYPE(kTfLiteFloat32);
    CASE_FOR_DTYPE(kTfLiteFloat64);
    // The DType for kTfLiteString is slightly different as we need to use
    // tensorflow::tstring rather than std::string
    CASE_FOR_DTYPE_GIVEN_CPP_DTYPE(kTfLiteString, ::tensorflow::tstring);
    default: {
      return absl::UnimplementedError(
          absl::StrCat("Unsupported dtype: ", wrapped_tensor->type));
    }
  }
}

template <>
absl::StatusOr<TfLiteTensorView> TensorView::New<::TfLiteTensor>(
    ::TfLiteTensor *wrapped_tensor) {
  return TfLiteTensorViewTemplatizedNew(wrapped_tensor);
}

template <>
absl::StatusOr<const TfLiteTensorView> TensorView::New<const ::TfLiteTensor>(
    const ::TfLiteTensor *wrapped_tensor) {
  return TfLiteTensorViewTemplatizedNew(wrapped_tensor);
}

}  // namespace shim
}  // namespace tflite
