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

#include "tensorflow/core/kernels/host_constant_op.h"

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

_HostConstantOp::_HostConstantOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), tensor_(ctx->output_type(0)) {
  const TensorProto* proto = nullptr;
  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
  OP_REQUIRES_OK(
      ctx, ctx->device()->MakeTensorFromProto(*proto, alloc_attr, &tensor_));
  OP_REQUIRES(
      ctx, ctx->output_type(0) == tensor_.dtype(),
      errors::InvalidArgument("Type mismatch between value (",
                              DataTypeString(tensor_.dtype()), ") and dtype (",
                              DataTypeString(ctx->output_type(0)), ")"));
}

void _HostConstantOp::Compute(OpKernelContext* ctx) {
  ctx->set_output(0, tensor_);
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Const")
                            .Device(DEVICE_GPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("dtype"),
                        _HostConstantOp);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


// HostConst: forced to generate output on the host.
REGISTER_KERNEL_BUILDER(Name("HostConst").Device(DEVICE_CPU), _HostConstantOp);
REGISTER_KERNEL_BUILDER(
    Name("HostConst").Device(DEVICE_DEFAULT).HostMemory("output"),
    _HostConstantOp);

}  // end namespace tensorflow

