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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_TENSOR_VIEW_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_TENSOR_VIEW_H_

#include <cstring>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"
#include "tensorflow/lite/type_to_tflitetype.h"

namespace tflite {
namespace shim {

// A view over a TFLite tensor without taking ownership. It can either be
// mutable or immutable.
class TfLiteTensorView : public TensorView {
 public:
  // Move constructor
  TfLiteTensorView(TfLiteTensorView &&o) noexcept;
  // Copy constructor
  TfLiteTensorView(const TfLiteTensorView &o);
  // Move assignment operator
  TfLiteTensorView &operator=(TfLiteTensorView &&o) noexcept;
  // Copy assignment operator
  TfLiteTensorView &operator=(const TfLiteTensorView &);

 protected:
  // Templated constructor. Since it's not possible to specify the template
  // argument directly we place a dummy argument of that type so compiler can
  // deduce the right template parameter
  template <typename DType>
  TfLiteTensorView(::TfLiteTensor *wrapped_tensor, const DType &dtype)
      : TensorView(absl::Span<int>(wrapped_tensor->dims->data,
                                   wrapped_tensor->dims->size),
                   wrapped_tensor->data.raw, wrapped_tensor->bytes, dtype),
        wrapped_tensor_(wrapped_tensor),
        const_wrapped_tensor_(wrapped_tensor) {}

  // Specialization for string. (this take precedence over the above template)
  TfLiteTensorView(::TfLiteTensor *wrapped_tensor,
                   const ::tensorflow::tstring &dtype);

  // Templated constructor for const input.
  template <typename DType>
  TfLiteTensorView(const ::TfLiteTensor *wrapped_tensor, const DType &dtype)
      : TensorView(absl::Span<int>(wrapped_tensor->dims->data,
                                   wrapped_tensor->dims->size),
                   wrapped_tensor->data.raw, wrapped_tensor->bytes, dtype),
        const_wrapped_tensor_(wrapped_tensor) {}

  // Specialization for const string. (this take precedence over the above
  // template)
  TfLiteTensorView(const ::TfLiteTensor *wrapped_tensor,
                   const ::tensorflow::tstring &dtype);

  // Let the factory implementation use private constructors
  template <typename TfLiteTensorType>
  friend absl::StatusOr<
      typename MatchConstNess<TfLiteTensorType, TfLiteTensorView>::Type>
  TfLiteTensorViewTemplatizedNew(TfLiteTensorType *wrapped_tensor);

  struct StringBuffer {
    explicit StringBuffer(TfLiteTensorView *t_view);
    ~StringBuffer();

    // A vector of string as the intermediate shared buffer between
    // TensorViews
    std::vector<::tensorflow::tstring> buffer;
    // The TFLite tensor to which the contents of the buffer is flushed in
    // dtor
    ::TfLiteTensor *wrapped_tensor = nullptr;
  };

  // Initialize the data_ field for string tensors
  void InitForStringDType();

  // The wrapped TFLiteTensor
  ::TfLiteTensor *wrapped_tensor_ = nullptr;
  // A const version of the wrapped TFLiteTensor used when the input is const
  const ::TfLiteTensor *const_wrapped_tensor_ = nullptr;
  // A temporary buffer used to expose TfLite strings tensor as Span<tstring>.
  // This buffer will be flushed and serialized back to the underlying TfLite
  // string tensor once all the TensorViews over that tensor are destructed.
  std::shared_ptr<StringBuffer> str_vec_ = nullptr;
};

// Mapping of ::TfLiteTensor -> TfLiteTensorView
template <>
struct TensorViewSubType<::TfLiteTensor> {
  using Type = TfLiteTensorView;
};

// Mapping of const ::TfLiteTensor -> const TfLiteTensorView
template <>
struct TensorViewSubType<const ::TfLiteTensor> {
  using Type = const TfLiteTensorView;
};

// Specialization for TensorView::New()
template <>
absl::StatusOr<TfLiteTensorView> TensorView::New<::TfLiteTensor>(
    ::TfLiteTensor *wrapped_tensor);

// Specialization for TensorView::New()
template <>
absl::StatusOr<const TfLiteTensorView> TensorView::New<const ::TfLiteTensor>(
    const ::TfLiteTensor *wrapped_tensor);

}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_TENSOR_VIEW_H_
