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

#ifndef TENSORFLOW_CORE_KERNELS_TRANSPOSE_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_TRANSPOSE_FUNCTOR_H_

#include <numeric>
#include <string>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
// Transpose tensor 'in' into tensor 'out' according to dimension
// permutation 'perm'.
//
// REQUIRES: in.dtype() == out->dtype()
// REQUIRES: in.dims() == out->dims()
// REQUIRES: in.dims() == perm.size()
// REQUIRES: in.dim_size(perm[i]) == out->dim_size(i)
template <typename Device>
Status DoTranspose(const Device& device, const Tensor& in,
                   const gtl::ArraySlice<int32> perm, Tensor* out);

// Conjugate and transpose tensor 'in' into tensor 'out' according to dimension
// permutation 'perm'.
//
// REQUIRES: in.dtype() == out->dtype()
// REQUIRES: in.dims() == out->dims()
// REQUIRES: in.dims() == perm.size()
// REQUIRES: in.dim_size(perm[i]) == out->dim_size(i)
template <typename Device>
Status DoConjugateTranspose(const Device& device, const Tensor& in,
                            const gtl::ArraySlice<int32> perm, Tensor* out);

// Convenience versions of DoTranspose that only swap the last (inner) two
// dimensions.
template <typename Device>
Status DoMatrixTranspose(const Device& device, const Tensor& in, Tensor* out);

// Convenience versions of DoConjugateTranspose that only swap the last (inner)
// two dimensions.
template <typename Device>
Status DoConjugateMatrixTranspose(const Device& device, const Tensor& in,
                                  Tensor* out);

// Primary device specific functor to be specialized for each device and type.
template <typename Device, typename T, bool conjugate = false>
struct Transpose {
  static void run(const Device& d, const Tensor& in,
                  const gtl::ArraySlice<int32> perm, Tensor* out);
};

// Implementation details.
namespace internal {

typedef gtl::InlinedVector<int64_t, 8> TransposeDimsVec;
typedef gtl::InlinedVector<int32, 8> TransposePermsVec;

// Helper function that takes a tensor shape, a permutation, combines the
// neighboring shapes if their indices in the permutation are consecutive.
// The function outputs the combined shape and new permutation.
// Example: Tensor shape {2, 3, 4, 5, 120} and permutation {0, 4, 1, 2, 3} will
// produce new shape {2, 60, 120} and new permutation {0, 2, 1}.
inline void ReduceTransposeDimensions(const TensorShape& shape,
                                      gtl::ArraySlice<int32> perm,
                                      TransposePermsVec* new_perm,
                                      TransposeDimsVec* new_dims) {
  CHECK_EQ(shape.dims(), perm.size());
  if (shape.dims() == 1) {
    // If input dimension is already 1, no need to reduce dimension.
    new_perm->resize(1);
    (*new_perm)[0] = perm[0];
    (*new_dims)[0] = shape.dim_size(0);
    return;
  }
  TransposePermsVec new_dim_position(shape.dims(), -1);
  TransposeDimsVec combined_dims(shape.dims(), 0);
  int cur_head = perm[0];
  new_dim_position[cur_head] = 0;
  combined_dims[0] = shape.dim_size(cur_head);
  int dim_idx = 0;
  for (int perm_idx = 1; perm_idx < shape.dims(); ++perm_idx) {
    // If two indices in permutation are consecutive numbers, combine their
    // dimensions.
    if (cur_head + 1 == perm[perm_idx]) {
      cur_head = perm[perm_idx];
      combined_dims[dim_idx] *= shape.dim_size(cur_head);
    } else {
      // Else start a new dimension.
      cur_head = perm[perm_idx];
      dim_idx++;
      new_dim_position[cur_head] = dim_idx;
      combined_dims[dim_idx] = shape.dim_size(cur_head);
    }
  }
  // Compact the new permutations and dimension sizes.
  new_perm->resize(dim_idx + 1);
  new_dims->resize(dim_idx + 1);
  dim_idx = 0;
  for (int i = 0; i < new_dim_position.size(); ++i) {
    if (new_dim_position[i] >= 0) {
      int new_perm_idx = new_dim_position[i];
      (*new_perm)[dim_idx] = new_perm_idx;
      (*new_dims)[dim_idx] = combined_dims[new_perm_idx];
      dim_idx++;
    }
  }
}

// If all non-singleton dimensions remain in ascending order, the shuffled
// singletons can be transposed by a reshape, saving a memory allocation & copy.
// |permutation| must be a permutation of {0, .., input_shape.dims() - 1}.
// That is, for all i, 0 <= perm[i] < input_shape.dims().
// In practice, this is checked in TransposeOp::Compute prior to calling this
// function, and the function sits here to facilitate unit testing.
inline bool NonSingletonDimensionsAlign(const TensorShape& input_shape,
                                        const std::vector<int32>& permutation) {
  int last_nonsingleton_perm_dim = -1;
  for (int perm_dim : permutation) {
    if (input_shape.dim_size(perm_dim) == 1) {
      continue;
    }
    if (perm_dim < last_nonsingleton_perm_dim) {
      return false;
    }
    last_nonsingleton_perm_dim = perm_dim;
  }
  return true;
}

// Uses Eigen to transpose.
template <typename Device, typename T, int NDIMS>
void TransposeUsingEigen(const Device& d, const Tensor& in,
                         const gtl::ArraySlice<int32> perm, bool conjugate,
                         Tensor* out) {
  Eigen::array<int, NDIMS> p;
  for (int i = 0; i < NDIMS; ++i) p[i] = perm[i];
  auto x = typename TTypes<T, NDIMS>::ConstTensor(
      reinterpret_cast<const T*>(in.tensor_data().data()),
      in.shape().AsEigenDSizes<NDIMS>());
  auto y = typename TTypes<T, NDIMS>::Tensor(
      reinterpret_cast<T*>(const_cast<char*>(out->tensor_data().data())),
      out->shape().AsEigenDSizes<NDIMS>());
  if (conjugate) {
    y.device(d) = x.conjugate().shuffle(p);
  } else {
    y.device(d) = x.shuffle(p);
  }
}

template <typename Device>
Status DoTransposeImpl(const Device& d, const Tensor& in,
                       const gtl::ArraySlice<int32> perm, bool conjugate,
                       Tensor* out) {
  CHECK_EQ(in.dims(), out->dims());
  CHECK_EQ(in.dims(), perm.size());
  CHECK_EQ(in.dtype(), out->dtype());
  switch (in.dtype()) {
    case DT_BOOL:
    case DT_INT8:
    case DT_QINT8:
    case DT_QUINT8:
    case DT_UINT8:
      Transpose<Device, uint8>::run(d, in, perm, out);
      break;

    case DT_BFLOAT16:
    case DT_HALF:
    case DT_INT16:
    case DT_QINT16:
    case DT_QUINT16:
    case DT_UINT16:
      Transpose<Device, uint16>::run(d, in, perm, out);
      break;

    case DT_FLOAT:
    case DT_INT32:
    case DT_QINT32:
    case DT_UINT32:
      Transpose<Device, uint32>::run(d, in, perm, out);
      break;

    case DT_DOUBLE:
    case DT_INT64:
    case DT_UINT64:
      Transpose<Device, uint64>::run(d, in, perm, out);
      break;

    case DT_COMPLEX64:
      if (conjugate) {
#if defined(__ANDROID__) and !defined(__clang__)
        // Workaround for GCC compiler bug in Android toolchain.
        return errors::Unimplemented(
            "Conjugate transpose of complex64 not supported for GCC on "
            "Android.");
#else
        Transpose<Device, complex64, /*conjugate=*/true>::run(d, in, perm, out);
#endif
      } else {
        Transpose<Device, uint64>::run(d, in, perm, out);
      }
      break;

    case DT_COMPLEX128:
      if (conjugate) {
        Transpose<Device, complex128, /*conjugate=*/true>::run(d, in, perm,
                                                               out);
      } else {
        Transpose<Device, complex128, /*conjugate=*/false>::run(d, in, perm,
                                                                out);
      }
      break;

    case DT_STRING:
      Transpose<Device, tstring>::run(d, in, perm, out);
      break;

    default:
      return errors::Unimplemented("Unsupported dtype on CPU: ", in.dtype());
  }
  return Status::OK();
}

template <typename Device>
inline Status DoMatrixTransposeImpl(const Device& device, const Tensor& in,
                                    bool conjugate, Tensor* out) {
  const int ndims = in.dims();
  if (ndims == 0) return Status::OK();
  TransposePermsVec perm(ndims);
  std::iota(perm.begin(), perm.end(), 0);
  std::swap(perm[ndims - 2], perm[ndims - 1]);
  return DoTransposeImpl(device, in, perm, conjugate, out);
}

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TRANSPOSE_FUNCTOR_H_
