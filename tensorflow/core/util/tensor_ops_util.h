/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_UTIL_TENSOR_OPS_UTIL_H_
#define TENSORFLOW_CORE_UTIL_TENSOR_OPS_UTIL_H_

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
Status ZerosLikeTensor(OpKernelContext* ctx, const Tensor& x, Tensor* out) {
  AllocatorAttributes attr;
  if (x.dtype() == DT_VARIANT) {
    attr.set_on_host(true);
  }
  TF_RETURN_IF_ERROR(ctx->allocate_temp(x.dtype(), x.shape(), out, attr));

  switch (out->dtype()) {
#define DTYPE_CASE(dtype)                                       \
  case DataTypeToEnum<dtype>::value:                            \
    /* TODO(skyewm): use SetZeroFunctor like in ZerosLikeOp? */ \
    out->flat<dtype>().device(ctx->eigen_device<Device>()) =    \
        out->flat<dtype>().constant(dtype(0));                  \
    break;

    TF_CALL_POD_TYPES(DTYPE_CASE)
#undef DTYPE_CASE

    case DT_INVALID: {
      *out = Tensor(DT_INVALID);
      break;
    }
    case DataTypeToEnum<Variant>::value: {
      Variant* out_variant = out->scalar<Variant>().data();
      TF_RETURN_IF_ERROR(
          UnaryOpVariant<Device>(ctx, ZEROS_LIKE_VARIANT_UNARY_OP,
                                 x.scalar<Variant>()(), out_variant));
      break;
    }
    default:
      return errors::InvalidArgument(
          "Trying to compute zeros_like for unsupported dtype ",
          DataTypeString(out->dtype()));
  }
  return Status::OK();
}

template <typename Device>
Status BinaryAddTensors(OpKernelContext* ctx, const Tensor& a, const Tensor& b,
                        Tensor* out) {
  if (a.dtype() == DT_INVALID) {
    *out = b;
    return Status::OK();
  }
  if (b.dtype() == DT_INVALID) {
    *out = a;
    return Status::OK();
  }
  if (a.dtype() != b.dtype()) {
    return errors::InvalidArgument(
        "Trying to add two tensors with incompatible element types. ",
        "One is ", DataTypeString(a.dtype()), " and the other is ",
        DataTypeString(b.dtype()));
  }
  if (a.shape() != b.shape()) {
    // TODO(apassos) support broadcasting additions here?
    return errors::InvalidArgument(
        "Trying to add two tensors with incompatible element shapes. ",
        "One is ", a.shape().DebugString(), " and the other is ",
        b.shape().DebugString());
  }

  AllocatorAttributes attr;
  if (a.dtype() == DT_VARIANT) {
    attr.set_on_host(true);
  }
  TF_RETURN_IF_ERROR(ctx->allocate_temp(a.dtype(), a.shape(), out, attr));

  switch (out->dtype()) {
#define DTYPE_CASE(dtype)                                    \
  case DataTypeToEnum<dtype>::value:                         \
    out->flat<dtype>().device(ctx->eigen_device<Device>()) = \
        a.flat<dtype>() + b.flat<dtype>();                   \
    break;

    TF_CALL_NUMBER_TYPES(DTYPE_CASE)
#undef DTYPE_CASE

    case DataTypeToEnum<Variant>::value: {
      Variant* out_variant = out->scalar<Variant>().data();
      TF_RETURN_IF_ERROR(BinaryOpVariants<Device>(
          ctx, ADD_VARIANT_BINARY_OP, a.scalar<Variant>()(),
          b.scalar<Variant>()(), out_variant));
      break;
    }
    default:
      return errors::InvalidArgument("Trying to add unsupported dtype ",
                                     out->dtype());
  }
  return Status::OK();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_TENSOR_OPS_UTIL_H_
