/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/transpose_functor.h"

namespace tensorflow {
namespace internal {

template <typename Device, typename T>
void TransposeSimple(const Device& d, const Tensor& in,
                     const gtl::ArraySlice<int32> perm, Tensor* out) {
  const int ndims = in.dims();
  gtl::InlinedVector<int64, 8> in_strides(ndims);
  ComputeStride(in.shape(), in_strides.data());
  gtl::InlinedVector<int64, 8> out_strides(ndims);
  ComputeStride(out->shape(), out_strides.data());
  const int64 nelem = in.NumElements();
  const T* p = reinterpret_cast<const T*>(in.tensor_data().data());
  T* q = reinterpret_cast<T*>(const_cast<char*>((out->tensor_data().data())));

  // TODO(zhifengc): Shard by range.
  // TODO(zhifengc): Avoids the division.
  for (int64 o_idx = 0; o_idx < nelem; ++o_idx) {
    int64 i_idx = 0;
    int64 t = o_idx;
    for (int i = 0; i < ndims; ++i) {
      i_idx += (t / out_strides[i]) * in_strides[perm[i]];
      t = t % out_strides[i];
    }
    q[o_idx] = p[i_idx];
  }
}

template <typename Device, typename T, int NDIMS>
void TransposeUsingEigen(const Device& d, const Tensor& in,
                         const gtl::ArraySlice<int32> perm, Tensor* out) {
  Eigen::array<int, NDIMS> p;
  for (int i = 0; i < NDIMS; ++i) p[i] = perm[i];
  auto x = typename TTypes<T, NDIMS>::ConstTensor(
      reinterpret_cast<const T*>(in.tensor_data().data()),
      in.shape().AsEigenDSizes<NDIMS>());
  auto y = typename TTypes<T, NDIMS>::Tensor(
      reinterpret_cast<T*>(const_cast<char*>(out->tensor_data().data())),
      out->shape().AsEigenDSizes<NDIMS>());
  y.device(d) = x.shuffle(p);
}

}  // end namespace internal

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
struct Transpose<CPUDevice, T> {
  static void run(const CPUDevice& d, const Tensor& in,
                  const gtl::ArraySlice<int32> perm, Tensor* out) {
    switch (in.dims()) {
      case 2:
        internal::TransposeUsingEigen<CPUDevice, T, 2>(d, in, perm, out);
        break;
      case 3:
        internal::TransposeUsingEigen<CPUDevice, T, 3>(d, in, perm, out);
        break;
      case 4:
        internal::TransposeUsingEigen<CPUDevice, T, 4>(d, in, perm, out);
        break;
      case 5:
        internal::TransposeUsingEigen<CPUDevice, T, 5>(d, in, perm, out);
        break;
      default:
        internal::TransposeSimple<CPUDevice, T>(d, in, perm, out);
        break;
    }
  }
};

// TODO(yangzihao): Merge this code with its GPU counterpart to reduce code
// duplication.
template <>
Status DoTranspose<CPUDevice>(const CPUDevice& d, const Tensor& in,
                              const gtl::ArraySlice<int32> perm, Tensor* out) {
  typedef CPUDevice Device;
  CHECK_GE(in.dims(), 2);
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
      Transpose<Device, uint32>::run(d, in, perm, out);
      break;

    case DT_COMPLEX64:
    case DT_DOUBLE:
    case DT_INT64:
      Transpose<Device, uint64>::run(d, in, perm, out);
      break;

    case DT_COMPLEX128:
      Transpose<Device, complex128>::run(d, in, perm, out);
      break;

    case DT_STRING:
      Transpose<Device, string>::run(d, in, perm, out);
      break;

    default:
      return errors::Unimplemented("Unsupported dtype on CPU: ", in.dtype());
  }
  return Status::OK();
}

#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;

template <typename T>
struct internal::Transpose<SYCLDevice, T> {
  static void run(const SYCLDevice& d, const Tensor& in,
                  const gtl::ArraySlice<int32> perm, Tensor* out) {
    // Should add a specialized implementation for SYCLDevice here.
  }
};

template <>
Status DoTranspose<SYCLDevice>(const SYCLDevice& d, const Tensor& in,
                           const gtl::ArraySlice<int32> perm, Tensor* out) {
  CHECK_GE(in.dims(), 2);
  CHECK_EQ(in.dims(), out->dims());
  CHECK_EQ(in.dims(), perm.size());
  CHECK_EQ(in.dtype(), out->dtype());
  switch (in.dtype()) {
    case DT_FLOAT:
    case DT_DOUBLE:
    case DT_INT32:
      internal::Transpose<SYCLDevice, uint32>::run(d, in, perm, out);
      break;

    default:
      return errors::Unimplemented("Unsupported dtype on SYCL: ", in.dtype());
  }
  return Status::OK();
}
#endif // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
