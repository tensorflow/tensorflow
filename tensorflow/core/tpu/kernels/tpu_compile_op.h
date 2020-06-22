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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_H_

#include <memory>

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace tpu {
// Forward declaration.
class TpuCompileOpKernelImpl;
}  // namespace tpu

// The TPUCompile operator compiles a Tensorflow function into a
// TPU executable to be run by TPUExecute.
//
class TpuCompileOp : public OpKernel {
 public:
  explicit TpuCompileOp(OpKernelConstruction* ctx);
  ~TpuCompileOp() override = default;

  void Compute(OpKernelContext* ctx) override;

 private:
  std::unique_ptr<tpu::TpuCompileOpKernelImpl> impl_;

  DISALLOW_COPY_AND_ASSIGN(TpuCompileOp);
};

// The TPUCompile operator compiles a MLIR module into a
// TPU executable to be run by TPUExecute.
//
class TpuCompileMlirOp : public OpKernel {
 public:
  explicit TpuCompileMlirOp(OpKernelConstruction* ctx);
  ~TpuCompileMlirOp() override = default;

  void Compute(OpKernelContext* ctx) override;

 private:
  std::unique_ptr<tpu::TpuCompileOpKernelImpl> impl_;

  DISALLOW_COPY_AND_ASSIGN(TpuCompileMlirOp);
};

class TpuCompileSucceededAssertOp : public OpKernel {
 public:
  explicit TpuCompileSucceededAssertOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  ~TpuCompileSucceededAssertOp() override = default;

  void Compute(OpKernelContext* ctx) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(TpuCompileSucceededAssertOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_H_
