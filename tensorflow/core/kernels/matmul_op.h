/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_MATMUL_OP_H_
#define TENSORFLOW_CORE_KERNELS_MATMUL_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/hash/hash.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

namespace tensorflow {
namespace functor {

// Helpers to define tensor<T> needed by MatMul op.
template <typename T>
struct MatMulTypes {
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned>
      out_type;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                           Eigen::Aligned>
      in_type;
};

template <typename Device, typename In0, typename In1, typename Out,
          typename DimPair>
void MatMul(const Device& d, Out out, In0 in0, In1 in1,
            const DimPair& dim_pair) {
  out.device(d) = in0.contract(in1, dim_pair);
}

template <typename Device, typename T>
struct MatMulFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, typename MatMulTypes<T>::out_type out,
      typename MatMulTypes<T>::in_type in0,
      typename MatMulTypes<T>::in_type in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair);
};

}  // end namespace functor

#if GOOGLE_CUDA
// Encapsulate all the shape information that is used in matmul operations.
class MatmulParameters {
 public:
  MatmulParameters(bool transa, bool transb, uint64 m, uint64 n, uint64 k,
                   DataType dtype, int device_id)
      : transa_(transa),
        transb_(transb),
        m_(m),
        n_(n),
        k_(k),
        dtype_(dtype),
        device_id_(device_id) {
    hash_code_ = transa;
    hash_code_ = Hash64Combine(hash_code_, transb);
    hash_code_ = Hash64Combine(hash_code_, m);
    hash_code_ = Hash64Combine(hash_code_, n);
    hash_code_ = Hash64Combine(hash_code_, k);
    hash_code_ = Hash64Combine(hash_code_, dtype);
    hash_code_ = Hash64Combine(hash_code_, device_id);
  }
  bool operator==(const MatmulParameters& other) const {
    return this->get_data_as_tuple() == other.get_data_as_tuple();
  }

  bool operator!=(const MatmulParameters& other) const {
    return !(*this == other);
  }
  uint64 hash() const { return hash_code_; }

  string ToString() const {
    // clang-format off
    return strings::StrCat(
        transa_, ", ", transb_, ", ",
        m_, ", ", n_, ", ", k_,
        dtype_, ", ", device_id_);
    // clang-format on
  }

 private:
  typedef std::tuple<bool, bool, int64, int64, int64, DataType, int>
      ParameterDataType;

  ParameterDataType get_data_as_tuple() const {
    return std::make_tuple(transa_, transb_, m_, n_, k_, dtype_, device_id_);
  }

  bool transa_;
  bool transb_;
  uint64 m_;
  uint64 n_;
  uint64 k_;
  DataType dtype_;
  int device_id_;
  uint64 hash_code_;
};

typedef Eigen::GpuDevice GPUDevice;

#endif  // GOOGLE_CUDA

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MATMUL_OP_H_
