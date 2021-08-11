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
#include "tensorflow/lite/kernels/shim/tf_tensor_view.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/framework/types.pb.h"

// Creates a case statement for the switch() clause given the dtype
#define CASE_FOR_DTYPE_GIVEN_CPP_DTYPE(TF_DTYPE, CPP_DTYPE) \
  case TF_DTYPE: {                                          \
    using DType = typename CPP_DTYPE;                       \
    return TfTensorView(wrapped_tensor, DType());           \
  }

#define CASE_FOR_DTYPE(TF_DTYPE)           \
  CASE_FOR_DTYPE_GIVEN_CPP_DTYPE(TF_DTYPE, \
                                 ::tensorflow::EnumToDataType<TF_DTYPE>::Type)

namespace tflite {
namespace shim {

// ctors

TfTensorView::TfTensorView(TfTensorView &&o) noexcept
    : TensorView(std::move(o)), shape_data_(std::move(o.shape_data_)) {
  shape_ = absl::Span<int>(shape_data_);
}

TfTensorView::TfTensorView(const TfTensorView &o)
    : TensorView(o), shape_data_(o.shape_data_) {
  shape_ = absl::Span<int>(shape_data_);
}

TfTensorView &TfTensorView::operator=(TfTensorView &&o) noexcept {
  shape_data_ = std::move(o.shape_data_);
  TensorView::operator=(std::move(o));
  shape_ = absl::Span<int>(shape_data_);
  return *this;
}

TfTensorView &TfTensorView::operator=(const TfTensorView &o) {
  if (&o == this) return *this;
  TensorView::operator=(o);
  shape_data_ = o.shape_data_;
  shape_ = absl::Span<int>(shape_data_);
  return *this;
}

template <typename TfTensorType>
absl::StatusOr<typename MatchConstNess<TfTensorType, TfTensorView>::Type>
TfTensorViewTemplatizedNew(TfTensorType *wrapped_tensor) {
  switch (wrapped_tensor->dtype()) {
    CASE_FOR_DTYPE(::tensorflow::DT_BOOL);
    CASE_FOR_DTYPE(::tensorflow::DT_UINT8);
    CASE_FOR_DTYPE(::tensorflow::DT_UINT64);
    CASE_FOR_DTYPE(::tensorflow::DT_INT8);
    CASE_FOR_DTYPE(::tensorflow::DT_INT16);
    CASE_FOR_DTYPE(::tensorflow::DT_INT32);
    // Map DT_INT64 to int64_t instead of int64 to have a single int64 datatype.
    CASE_FOR_DTYPE_GIVEN_CPP_DTYPE(::tensorflow::DT_INT64, std::int64_t);
    CASE_FOR_DTYPE(::tensorflow::DT_FLOAT);
    CASE_FOR_DTYPE(::tensorflow::DT_DOUBLE);
    CASE_FOR_DTYPE(::tensorflow::DT_STRING);
    default: {
      return absl::UnimplementedError(
          absl::StrCat("Unsupported data type: ", wrapped_tensor->dtype()));
    }
  }
}

template <>
absl::StatusOr<TfTensorView> TensorView::New<::tensorflow::Tensor>(
    ::tensorflow::Tensor *wrapped_tensor) {
  return TfTensorViewTemplatizedNew(wrapped_tensor);
}

template <>
absl::StatusOr<const TfTensorView> TensorView::New<const ::tensorflow::Tensor>(
    const ::tensorflow::Tensor *wrapped_tensor) {
  return TfTensorViewTemplatizedNew(wrapped_tensor);
}

}  // namespace shim
}  // namespace tflite
