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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TENSOR_VIEW_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TENSOR_VIEW_H_

#include <variant>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tstring.h"

namespace tflite {
namespace shim {

// A type deduction template which is specialized for TF and TFLite.
// That is it maps
//   ::tensorflow::Tensor -> tflite::shim::TfTensorView
//   ::TfLiteTensor -> tflite::shim::TfLiteTensorView
template <typename W>
struct TensorViewSubType {};

// Common denominator for ::tflite::TfLiteTensor and ::tensorflow::Tensor.
// It is a "view" over the underlying tensor without taking ownership.
// Objects of this class can also mutate the underlying tensor depending on
// whether the underlying tensor is "const" qualified or not.
//
// Movable and copyable.
// It can be instantiated with the New() factory function. eg.
//   TfTensorView t           = TensorView::New(&tf_tensor);
//   const TfTensorView t     = TensorView::New(&const_tf_tensor);
//   TfLiteTensorView t       = TensorView::New(&tflite_tensor);
//   const TfLiteTensorView t = TensorView::New(&const_tflite_tensor);
class TensorView {
 protected:
  // Union over all data types
  using DataVariantType =
      std::variant<absl::Span<bool>, absl::Span<int8_t>, absl::Span<uint8_t>,
                   absl::Span<int16_t>, absl::Span<uint16_t>,
                   absl::Span<int32_t>, absl::Span<uint32_t>,
                   absl::Span<int64_t>, absl::Span<uint64_t>, absl::Span<float>,
                   absl::Span<double>, absl::Span<::tensorflow::tstring>>;

  // An interface while provides convenient row-major indexing over the
  // underlying tensor.
  // Example usage:
  //
  //   // A scalar view
  //   const TensorView t_float
  //   float val = t_float.AsScalar<float>();
  //
  //   // A vector view
  //   const TensorView t_int;
  //   auto t_int_vec = t_int.As<int32_t, /*RANK=*/ 1>();
  //   int sum = t_int_vec(0) + t_int_vec(1);
  //
  //   // A matrix view
  //   TensorView t_str;
  //   auto t_str_mat = t_str.As<tensorflow::tstring, /*RANK=*/ 2>();
  //   t_str_mat(0, 0) = "abc";
  //   t_str_mat(2, 3) = "def";
  template <typename DType, int RANK>
  class Tensor {
   public:
    explicit Tensor(TensorView *t)
        : data_(t->Data<DType>()), shape_(t->Shape()) {
      DCHECK_EQ(RANK, shape_.size());
      ComputeRowSizes();
    }

    explicit Tensor(const TensorView *t)
        : data_(t->Data<DType>()), shape_(t->Shape()) {
      DCHECK_EQ(RANK, shape_.size());
      ComputeRowSizes();
    }

    // indexing operator
    template <typename... IndexTypes>
    inline DType &operator()(IndexTypes... indices) {
      const auto idx = RowMajorIndex(std::array<int, RANK>{{indices...}});
      return data_[idx];
    }

    // const indexing operator
    template <typename... IndexTypes>
    inline const DType &operator()(IndexTypes... indices) const {
      const auto idx = RowMajorIndex(std::array<int, RANK>{{indices...}});
      return data_.at(idx);
    }

    // Pointer accessor
    typename absl::Span<DType>::pointer Ptr() { return data_.data(); }
    constexpr typename absl::Span<DType>::const_pointer Ptr() const {
      return data_.data();
    }

    // Size of the given dimension
    inline int Dim(int dim_i) const {
      DCHECK(RANK > 0 && dim_i < RANK) << "dim: " << dim_i << " rank:" << RANK;
      // Handle negative indices
      if (dim_i < 0) dim_i = ((dim_i % RANK) + RANK) % RANK;
      return shape_[dim_i];
    }

    // The tensor's rank: number of dimensions
    /*[[nodiscard]]*/ constexpr std::size_t Rank() const { return RANK; }

   private:
    // Computes the row-major index
    inline std::size_t RowMajorIndex(
        const std::array<int, RANK> &indices) const {
      std::size_t ret = 0;
      for (int i = 0; i < RANK; ++i) ret += indices[i] * row_sizes_[i];
      return ret;
    }

    // Pre computes row sizes to convert multi dim indices into a row major
    // index
    void ComputeRowSizes() {
      // Precompute row sizes for row major index computation
      if (RANK > 0) {
        row_sizes_[RANK - 1] = 1;
        for (int i = RANK - 2; i >= 0; --i) {
          row_sizes_[i] = row_sizes_[i + 1] * shape_[i + 1];
        }
      }
    }

    absl::Span<DType> data_;
    const absl::Span<int> shape_;
    std::size_t row_sizes_[RANK]{};
  };

 public:
  // Factory which gets specialized for different wrapped tensor types.
  template <typename W>
  static absl::StatusOr<typename TensorViewSubType<W>::Type> New(
      W *wrapped_tensor);

 protected:
  // Move constructor
  TensorView(TensorView &&o) = default;
  // Copy constructor
  TensorView(const TensorView &o) = default;
  // Move assignment operator
  TensorView &operator=(TensorView &&o) = default;
  // Copy assignment operator
  TensorView &operator=(const TensorView &) = default;

 public:
  // Dtor
  virtual ~TensorView() = default;

  // Accessors

  // Shape
  absl::Span<int> Shape() { return shape_; }
  /*[[nodiscard]]*/ const absl::Span<int> Shape() const { return shape_; }

  // Data
  template <typename DType>
  absl::Span<DType> &Data() {
    return std::get<absl::Span<DType>>(data_);
  }
  template <typename DType>
  constexpr absl::Span<DType> Data() const {
    return std::get<absl::Span<DType>>(data_);
  }

  // Reads the tensor given the dtype and its rank and provides an indexing
  // operator.
  template <typename DType, int RANK>
  Tensor<DType, RANK> As() {
    return Tensor<DType, RANK>(this);
  }

  // Const version of As()
  template <typename DType, int RANK>
  const Tensor<DType, RANK> As() const {
    return Tensor<DType, RANK>(this);
  }

  // Read the given tensor as a scalar or return error if it isn't
  template <typename DType>
  DType &AsScalar();

  template <typename DType>
  const DType &AsScalar() const;

 protected:
  // Templated constructor. Since it's not possible to specify the template
  // argument directly we place a dummy argument of that type so compiler
  // can deduce the right template parameter
  template <typename DType>
  TensorView(const absl::Span<int> shape, void *data,
             const std::size_t data_size, const DType &)
      : shape_(shape),
        data_(absl::Span<DType>(reinterpret_cast<DType *>(data),
                                data_size / sizeof(DType))) {}

  // Return the total number of elements given the shape.
  static constexpr std::size_t NumElements(const absl::Span<int> shape) {
    std::size_t ret = 1;
    for (const auto dim : shape) ret *= dim;
    return ret;
  }

  // Tensor shape
  // Note: using int rather than size_t to avoid conversion to from TfLite shape
  absl::Span<int> shape_;
  // Tensor data
  DataVariantType data_;
};

// Add or remove const qualifier to O based on whether it is in I.
// For example
//   MatchConstNess<const TfLiteTensor, TensorView>::Type == const TensorView
//   MatchConstNess<TfLiteTensor, TensorView>::Type == TensorView
//   MatchConstNess<TfLiteTensor, const TensorView>::Type == TensorView
template <typename I, typename O>
struct MatchConstNess {
  using Type = std::conditional_t<std::is_const<I>::value, std::add_const_t<O>,
                                  std::remove_const_t<O>>;
};

///////////////////////////// Implementation

template <typename DType>
DType &TensorView::AsScalar() {
  DCHECK_EQ(shape_.size(), 0) << "Tensor is not a scalar";
  return Data<DType>()[0];
}

template <typename DType>
const DType &TensorView::AsScalar() const {
  DCHECK_EQ(shape_.size(), 0) << "Tensor is not a scalar";
  return Data<DType>().at(0);
}

}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TENSOR_VIEW_H_
