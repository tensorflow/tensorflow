/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_CWISE_OP_GPU_BASE_H_
#define TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_CWISE_OP_GPU_BASE_H_

#include <memory>
#include <string>

#include "absl/strings/ascii.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {
class MlirGeneratedUnaryOp : public OpKernel {
 public:
  MlirGeneratedUnaryOp(OpKernelConstruction* ctx, std::string name,
                       absl::Span<const uint8_t> cubin_data)
      : OpKernel(ctx), name_(name), cubin_data_(cubin_data) {}

  void Compute(OpKernelContext* ctx) override;

 private:
  std::string name_;
  absl::Span<const uint8_t> cubin_data_;
  std::unique_ptr<se::KernelBase> kernel_;
  absl::Mutex mu_;
};

#define GENERATE_OP_KERNEL_BASE(kernel_name)                               \
  class MlirGenerated##kernel_name##Op : public MlirGeneratedUnaryOp {     \
   public:                                                                 \
    MlirGenerated##kernel_name##Op(OpKernelConstruction* ctx,              \
                                   absl::Span<const uint8_t> cubin_data)   \
        : MlirGeneratedUnaryOp(ctx, #kernel_name "_kernel", cubin_data) {} \
  };

#define GENERATE_OP_KERNEL_FOR(kernel_name, data_type)    \
  class MlirGenerated##kernel_name##data_type##Op         \
      : public MlirGenerated##kernel_name##Op {           \
   public:                                                \
    explicit MlirGenerated##kernel_name##data_type##Op(   \
        OpKernelConstruction* ctx)                        \
        : MlirGenerated##kernel_name                      \
          ##Op(ctx, k##kernel_name##data_type##Kernel) {} \
  };

#define REGISTER_AND_GENERATE_KERNEL(kernel_name, data_type, native_data_type) \
  namespace {                                                                  \
  GENERATE_OP_KERNEL_FOR(kernel_name, data_type)                               \
  }                                                                            \
  REGISTER_KERNEL_BUILDER(Name(#kernel_name)                                   \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<native_data_type>("T"),          \
                          MlirGenerated##kernel_name##data_type##Op);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_CWISE_OP_GPU_BASE_H_
