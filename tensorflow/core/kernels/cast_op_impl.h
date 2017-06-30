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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_CAST_OP_IMPL_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_CAST_OP_IMPL_H_

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/cast_op.h"

namespace tensorflow {

namespace functor {

template <typename O, typename I>
struct CastFunctor<Eigen::ThreadPoolDevice, O, I> {
  void operator()(const Eigen::ThreadPoolDevice& d, typename TTypes<O>::Flat o,
                  typename TTypes<I>::ConstFlat i) {
    o.device(d) = i.template cast<O>();
  }
};

#ifdef TENSORFLOW_USE_SYCL
template <typename O, typename I>
struct CastFunctor<Eigen::SyclDevice, O, I> {
  void operator()(const Eigen::SyclDevice& d, typename TTypes<O>::Flat o,
                  typename TTypes<I>::ConstFlat i) {
    o.device(d) = i.template cast<O>();
  }
};
#endif // TENSORFLOW_USE_SYCL

}  // namespace functor

#define CURRY_TYPES3_NO_HALF(FN, arg0, arg1)   \
  FN(arg0, arg1, bool);                        \
  FN(arg0, arg1, uint8);                       \
  FN(arg0, arg1, int8);                        \
  FN(arg0, arg1, uint16);                      \
  FN(arg0, arg1, int16);                       \
  FN(arg0, arg1, int32);                       \
  FN(arg0, arg1, int64);                       \
  FN(arg0, arg1, float);                       \
  FN(arg0, arg1, double);                      \
  FN(arg0, arg1, std::complex<float>);         \
  FN(arg0, arg1, std::complex<double>)

#define CURRY_TYPES3(FN, arg0, arg1)           \
  CURRY_TYPES3_NO_HALF(FN, arg0, arg1)         \
  FN(arg0, arg1, Eigen::half);

#define CAST_CASE(DEVICE, IN, OUT)                                         \
  if (DataTypeToEnum<OUT>::value == dst_dtype) {                           \
    return [](OpKernelContext* ctx, const Tensor& inp, Tensor* out) {      \
      functor::CastFunctor<DEVICE, OUT, IN> func;                          \
      func(ctx->eigen_device<DEVICE>(), out->flat<OUT>(), inp.flat<IN>()); \
    };                                                                     \
  }

// The functions below are implemented in the cast_op_impl_*.cc files.
std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetCpuCastFromBool(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetCpuCastFromUint8(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetCpuCastFromInt8(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetCpuCastFromUint16(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetCpuCastFromInt16(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetCpuCastFromInt32(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetCpuCastFromInt64(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetCpuCastFromHalf(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetCpuCastFromFloat(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetCpuCastFromDouble(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetCpuCastFromComplex64(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetCpuCastFromComplex128(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetCpuCastFromBfloat(DataType dst_dtype);

#if GOOGLE_CUDA
// Same, for GPU.
std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetGpuCastFromBool(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetGpuCastFromUint8(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetGpuCastFromInt8(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetGpuCastFromUint16(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetGpuCastFromInt16(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetGpuCastFromInt32(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetGpuCastFromInt64(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetGpuCastFromHalf(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetGpuCastFromFloat(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetGpuCastFromDouble(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetGpuCastFromComplex64(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetGpuCastFromComplex128(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetGpuCastFromBfloat(DataType dst_dtype);

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetSyclCastFromBool(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetSyclCastFromUint8(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetSyclCastFromUint16(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetSyclCastFromInt16(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetSyclCastFromInt32(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetSyclCastFromInt64(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetSyclCastFromFloat(DataType dst_dtype);

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetSyclCastFromDouble(DataType dst_dtype);
#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_CAST_OP_IMPL_H_
