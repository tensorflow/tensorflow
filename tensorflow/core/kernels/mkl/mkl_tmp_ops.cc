/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/nn_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

// This file contains temporary registrations for some of the Eigen CPU backend
// operators for Float32, BFloat16 and Half types. The kernels registered for
// all these ops simply raise an error. We do this so that MKL graph pass can
// rewrite these ops into corresponding MKL ops. Without such registrations,
// Placer component in TensorFlow fails because Eigen CPU backend does not
// support these ops for the above types.

namespace {
template <typename T>
class RaiseError : public OpKernel {
 public:
  explicit RaiseError(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES(context, false,
                absl::InvalidArgumentError(absl::StrCat(
                    "Op does not support ", typeid(T).name(), " inputs")));
  }

  void Compute(OpKernelContext* context) override {}
};
}  // namespace

#define REGISTER_CPU_CONV_2D(T)                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_FusedConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RaiseError<T>);
TF_CALL_bfloat16(REGISTER_CPU_CONV_2D);
TF_CALL_half(REGISTER_CPU_CONV_2D);
#undef REGISTER_CPU_CONV_2D

#define REGISTER_CPU_CONV_3D(T)                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_FusedConv3D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RaiseError<T>);
TF_CALL_float(REGISTER_CPU_CONV_3D);
TF_CALL_bfloat16(REGISTER_CPU_CONV_3D);
TF_CALL_half(REGISTER_CPU_CONV_3D);
#undef REGISTER_CPU_CONV_3D

#define REGISTER_CPU_MATMUL(T)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_FusedMatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RaiseError<T>);
TF_CALL_bfloat16(REGISTER_CPU_MATMUL);
#undef REGISTER_CPU_MATMUL

}  // namespace tensorflow
