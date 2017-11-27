/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Common kernel registrations for XLA devices.

#ifndef TENSORFLOW_COMPILER_JIT_XLA_DEVICE_OPS_H_
#define TENSORFLOW_COMPILER_JIT_XLA_DEVICE_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/cast_op.h"
#include "tensorflow/core/kernels/constant_op.h"
#include "tensorflow/core/kernels/control_flow_ops.h"
#include "tensorflow/core/kernels/identity_op.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/kernels/sendrecv_ops.h"
#include "tensorflow/core/kernels/variable_ops.h"

namespace tensorflow {

// Dummy OpKernel, used for kernels assigned to an XLA device that should be
// compiled. Should never be called at runtime since such ops should be
// rewritten to a _XlaLaunch op. If it is called, it means the placer placed an
// operator on an XLA device but the compiler did not compile it.
class XlaDeviceDummyOp : public OpKernel {
 public:
  explicit XlaDeviceDummyOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;
};

#define REGISTER_XLA_LAUNCH_KERNEL(DEVICE, KERNEL, TYPES) \
  REGISTER_KERNEL_BUILDER(Name("_XlaLaunch")              \
                              .Device(DEVICE)             \
                              .HostMemory("constants")    \
                              .HostMemory("resources"),   \
                          KERNEL);

#define REGISTER_XLA_DEVICE_KERNELS(DEVICE, TYPES)                             \
  REGISTER_KERNEL_BUILDER(Name("_Send").Device(DEVICE), SendOp);               \
  REGISTER_KERNEL_BUILDER(Name("_Recv").Device(DEVICE), RecvOp);               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_HostSend").Device(DEVICE).HostMemory("tensor"), SendOp);          \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_HostRecv").Device(DEVICE).HostMemory("tensor"), RecvOp);          \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_HostCast").Device(DEVICE).HostMemory("x").HostMemory("y"),        \
      CpuCastOp);                                                              \
  REGISTER_KERNEL_BUILDER(Name("NoOp").Device(DEVICE), NoOp);                  \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Const").Device(DEVICE).TypeConstraint("dtype", TYPES),             \
      ConstantOp);                                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Identity").Device(DEVICE).TypeConstraint("T", TYPES), IdentityOp); \
  REGISTER_KERNEL_BUILDER(Name("Placeholder").Device(DEVICE), PlaceholderOp);  \
  REGISTER_KERNEL_BUILDER(Name("PlaceholderV2").Device(DEVICE),                \
                          PlaceholderOp);                                      \
                                                                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("VarHandleOp").Device(DEVICE).HostMemory("resource"),               \
      ResourceHandleOp<Var>);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_DEVICE_OPS_H_
