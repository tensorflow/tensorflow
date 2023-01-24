/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc, ../ops/math_ops.cc ../ops/nn_ops.cc.

#ifdef INTEL_MKL

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {

// This file contains temporary registrations for some deprecated MKL
// operations. The kernel registered for all these ops simply raises an error.
// Implementation for these ops have been removed.
// TODO(intel-tf): The registration will be removed in next major release.

namespace {
class RaiseDeprecatedMklOpError : public OpKernel {
 public:
  explicit RaiseDeprecatedMklOpError(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES(context, false,
                errors::InvalidArgument("Op has been deprecated"));
  }

  void Compute(OpKernelContext* context) override {}
};
}  // namespace

// Deprecated MklAddN op
#define REGISTER_MKL_CPU(T)                                    \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklAddN")                                         \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      RaiseDeprecatedMklOpError);
TF_CALL_float(REGISTER_MKL_CPU);
TF_CALL_bfloat16(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU

// Deprecated element-wise ops:
//   MklAdd, MklAddV2, MklSub, MklMul,
//   MklMaximum, and MklSquaredDifference
#pragma push_macro("REGISTER")
#undef REGISTER
#define REGISTER(OP, D, N, F, T)                               \
  REGISTER_KERNEL_BUILDER(                                     \
      Name(N)                                                  \
          .Device(DEVICE_##D)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      OP);
REGISTER6(RaiseDeprecatedMklOpError, CPU, "_MklAdd", functor::add, float,
          Eigen::half, double, int32, int64, bfloat16);
REGISTER6(RaiseDeprecatedMklOpError, CPU, "_MklAddV2", functor::add, float,
          Eigen::half, double, int32, int64, bfloat16);
REGISTER8(RaiseDeprecatedMklOpError, CPU, "_MklSub", functor::sub, float,
          Eigen::half, double, int32, int64, complex64, complex128, bfloat16);
REGISTER6(RaiseDeprecatedMklOpError, CPU, "_MklMul", functor::mul, float,
          Eigen::half, double, uint8, int32, bfloat16);
REGISTER6(RaiseDeprecatedMklOpError, CPU, "_MklMaximum", functor::maximum,
          float, Eigen::half, double, int32, int64, bfloat16);
REGISTER6(RaiseDeprecatedMklOpError, CPU, "_MklSquaredDifference",
          functor::squared_difference, float, Eigen::half, double, int32, int64,
          bfloat16);
#undef REGISTER
#pragma pop_macro("REGISTER")

// Deprecated MklIdentity op
#define REGISTER_MKL_CPU(T)                                    \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklIdentity")                                     \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      RaiseDeprecatedMklOpError);
TF_CALL_float(REGISTER_MKL_CPU);
TF_CALL_bfloat16(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU

// Deprecated MklInputConversion op
#define REGISTER_MKL_CPU(T)                                    \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklInputConversion")                              \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      RaiseDeprecatedMklOpError);
TF_CALL_float(REGISTER_MKL_CPU);
TF_CALL_bfloat16(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU

// Deprecated MklLRN and MklLRNGrad ops
#define REGISTER_MKL_CPU(T)                                    \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklLRN")                                          \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      RaiseDeprecatedMklOpError);                              \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklLRNGrad")                                      \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      RaiseDeprecatedMklOpError);
TF_CALL_float(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU

// Deprecated MklReshape op
#define REGISTER_MKL_CPU(T)                                    \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklReshape")                                      \
          .Device(DEVICE_CPU)                                  \
          .HostMemory("shape")                                 \
          .TypeConstraint<T>("T")                              \
          .TypeConstraint("Tshape", {DT_INT32, DT_INT64})      \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      RaiseDeprecatedMklOpError);
TF_CALL_float(REGISTER_MKL_CPU);
TF_CALL_bfloat16(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU

// Deprecated MklSlice op
#define REGISTER_MKL_CPU(type)                                 \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklSlice")                                        \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .HostMemory("begin")                                 \
          .HostMemory("size")                                  \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      RaiseDeprecatedMklOpError);
TF_CALL_float(REGISTER_MKL_CPU);
TF_CALL_bfloat16(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU

// Deprecated MklToTf op
#define REGISTER_MKL_CPU(T)                                    \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklToTf")                                         \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      RaiseDeprecatedMklOpError);
TF_CALL_NUMBER_TYPES(REGISTER_MKL_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU

}  // namespace tensorflow
#endif  // INTEL_MKL
