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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_TYPES_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_TYPES_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// Helper to define Tensor types given that the scalar is of type T.
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
struct TTypes {
  // Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Tensor;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned>
      ConstTensor;

  // Unaligned Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType> >
      UnalignedTensor;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType> >
      UnalignedConstTensor;

  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, int>,
                           Eigen::Aligned>
      Tensor32Bit;

  // Scalar tensor (implemented as a rank-0 tensor) of scalar type T.
  typedef Eigen::TensorMap<
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType>,
      Eigen::Aligned>
      Scalar;
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<const T, Eigen::Sizes<>,
                                                  Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      ConstScalar;

  // Unaligned Scalar tensor of scalar type T.
  typedef Eigen::TensorMap<
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, Eigen::RowMajor, IndexType> >
      UnalignedScalar;
  typedef Eigen::TensorMap<Eigen::TensorFixedSize<const T, Eigen::Sizes<>,
                                                  Eigen::RowMajor, IndexType> >
      UnalignedConstScalar;

  // Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Flat;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned>
      ConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Vec;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned>
      ConstVec;

  // Unaligned Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType> >
      UnalignedFlat;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType> >
      UnalignedConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType> >
      UnalignedVec;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType> >
      UnalignedConstVec;

  // Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Matrix;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>, Eigen::Aligned>
      ConstMatrix;

  // Unaligned Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType> >
      UnalignedMatrix;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType> >
      UnalignedConstMatrix;
};

typedef typename TTypes<float, 1>::Tensor32Bit::Index Index32;

template <typename Index, int NumDims>
bool SafeFor32BitIndexing(const Eigen::DSizes<Index, NumDims>& in) {
  for (int i = 0; i < NumDims; ++i) {
    if (in[i] > std::numeric_limits<Index32>::max()) return false;
  }
  return true;
}

template <typename Index, size_t NumDims>
bool SafeFor32BitIndexing(const Eigen::array<Index, NumDims>& in) {
  for (size_t i = 0; i < NumDims; ++i) {
    if (in[i] > std::numeric_limits<Index32>::max()) return false;
  }
  return true;
}

template <typename TensorType,
          typename Enable = typename TTypes<
              typename TensorType::Scalar, TensorType::NumIndices>::Tensor32Bit>
bool SafeFor32BitIndexing(TensorType in) {
  return in.size() <= std::numeric_limits<Index32>::max();
}

template <typename Index, int NumDims>
Eigen::DSizes<Index32, NumDims> To32Bit(
    const Eigen::DSizes<Index, NumDims>& in) {
  DCHECK(SafeFor32BitIndexing(in));
  Eigen::DSizes<Index32, NumDims> out;
  for (int i = 0; i < NumDims; ++i) {
    out[i] = static_cast<Index32>(in[i]);
  }
  return out;
}

template <typename Index, size_t NumDims>
Eigen::array<Index32, NumDims> To32Bit(const Eigen::array<Index, NumDims>& in) {
  DCHECK(SafeFor32BitIndexing(in));
  Eigen::array<Index32, NumDims> out;
  for (size_t i = 0; i < NumDims; ++i) {
    out[i] = static_cast<Index32>(in[i]);
  }
  return out;
}

template <typename TensorType>
typename TTypes<typename TensorType::Scalar,
                TensorType::NumIndices>::Tensor32Bit
To32Bit(TensorType in) {
  typedef typename TTypes<typename TensorType::Scalar,
                          TensorType::NumIndices>::Tensor32Bit RetType;
  DCHECK(SafeFor32BitIndexing(in));
  return RetType(in.data(), To32Bit(in.dimensions()));
}

namespace internal {

template <typename Device>
struct MaybeWith32BitIndexingImpl {
  template <typename Func, typename... Args>
  void operator()(Func func, Args&&... args) const {
    func(std::forward<Args>(args)...);
  }
};

template <>
struct MaybeWith32BitIndexingImpl<Eigen::GpuDevice> {
  template <typename Func, typename... Args>
  void operator()(Func func, Args&&... args) const {
    auto all = [](const auto&... bool_vals) {
      for (bool b : {bool_vals...}) {
        if (!b) return false;
      }
      return true;
    };
    if (all(SafeFor32BitIndexing(std::forward<Args>(args))...)) {
      func(To32Bit(std::forward<Args>(args))...);
    } else {
      func(std::forward<Args>(args)...);
    }
  }
};

}  // namespace internal

template <typename Device, typename Func, typename... Args>
void MaybeWith32BitIndexing(Func func, Args&&... args) {
  return internal::MaybeWith32BitIndexingImpl<Device>()(
      func, std::forward<Args>(args)...);
}

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_TYPES_H_
