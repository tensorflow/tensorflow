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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TF_TENSOR_VIEW_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TF_TENSOR_VIEW_H_

#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"

namespace tflite {
namespace shim {

// A view over TF Tensor without taking ownership. It can be either mutable or
// immutable.
class TfTensorView : public TensorView {
 public:
  // Move constructor
  TfTensorView(TfTensorView &&o) noexcept;
  // Copy constructor
  TfTensorView(const TfTensorView &o);
  // Move assignment operator
  TfTensorView &operator=(TfTensorView &&o) noexcept;
  // Copy assignment operator
  TfTensorView &operator=(const TfTensorView &);

 protected:
  // Templated constructor. Since it's not possible to specify the template
  // argument directly we place a dummy argument of that type so compiler can
  // deduce the right template parameter
  template <typename DType>
  TfTensorView(const ::tensorflow::Tensor *wrapped_tensor, const DType &dtype);

  // Let the factory implementation use private constructors
  template <typename TfTensorType>
  friend absl::StatusOr<
      typename MatchConstNess<TfTensorType, TfTensorView>::Type>
  TfTensorViewTemplatizedNew(TfTensorType *wrapped_tensor);

  // Stores the shape read from the TensorShape object
  std::vector<int> shape_data_;
};

// Map ::tensorflow::Tensor -> TfTensorView
template <>
struct TensorViewSubType<::tensorflow::Tensor> {
  using Type = TfTensorView;
};

// Map const ::tensorflow::Tensor -> const TfTensorView
template <>
struct TensorViewSubType<const ::tensorflow::Tensor> {
  using Type = const TfTensorView;
};

// Specialization of New() factory
template <>
absl::StatusOr<TfTensorView> TensorView::New<::tensorflow::Tensor>(
    ::tensorflow::Tensor *wrapped_tensor);

// Specialization of New() factory
template <>
absl::StatusOr<const TfTensorView> TensorView::New<const ::tensorflow::Tensor>(
    const ::tensorflow::Tensor *wrapped_tensor);

/////////////////////// Implementation
///////////////////////

// Templated ctor
template <typename DType>
TfTensorView::TfTensorView(const ::tensorflow::Tensor *wrapped_tensor,
                           const DType &dtype)
    : TensorView({}, wrapped_tensor->data(),
                 wrapped_tensor->tensor_data().size(), dtype) {
  shape_data_.resize(wrapped_tensor->shape().dims());
  for (int dim = 0; dim < wrapped_tensor->shape().dims(); ++dim) {
    shape_data_[dim] = wrapped_tensor->shape().dim_size(dim);
  }
  shape_ = absl::Span<int>(shape_data_);
}

}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TF_TENSOR_VIEW_H_
