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

#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"

namespace gpu = perftools::gputools;

namespace tensorflow {

IpuSummaryOp::IpuSummaryOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {}

void IpuSummaryOp::Compute(OpKernelContext* ctx) {
  Tensor* output_tensor = nullptr;
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output("out", TensorShape({}), &output_tensor));
  auto output_flat = output_tensor->flat<string>();
  output_flat(0) = "POPLAR COMPILATION OUTPUT";
}

IpuSummaryOp::~IpuSummaryOp() {}

REGISTER_KERNEL_BUILDER(Name("IpuSummary").Device(DEVICE_CPU),
    IpuSummaryOp);

}  // namespace tensorflow
