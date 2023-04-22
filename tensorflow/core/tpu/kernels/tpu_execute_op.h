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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_EXECUTE_OP_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_EXECUTE_OP_H_

#include <memory>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// Op that executes a precompiled TPU computation.
class TPUExecuteOp : public AsyncOpKernel {
 public:
  explicit TPUExecuteOp(OpKernelConstruction* context);
  ~TPUExecuteOp() override;

  AsyncOpKernel* AsAsync() override;

  void Compute(OpKernelContext* context) override;
  void ComputeAsync(OpKernelContext* context, DoneCallback done) override;

 protected:
  // Used by TPUExecuteAndUpdateVariablesOp to set the fused variable reads and
  // updates indices in the XLA computation. The two vectors must have the same
  // size, and a pair of read index and write index represents a variable's
  // input to the program and its updated value from the program. If the
  // variable is not updated, use -1 as the output index.
  std::vector<int> fused_device_var_reads_in_computation_inputs_;
  std::vector<int> fused_device_var_updates_in_computation_outputs_;

 private:
  Status DoWork(OpKernelContext* context);

  TF_DISALLOW_COPY_AND_ASSIGN(TPUExecuteOp);
};

// A variant of TPUExecuteOp that contains fused device variable reads and
// updates.
class TPUExecuteAndUpdateVariablesOp : public TPUExecuteOp {
 public:
  explicit TPUExecuteAndUpdateVariablesOp(OpKernelConstruction* context);
  ~TPUExecuteAndUpdateVariablesOp() override = default;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TPUExecuteAndUpdateVariablesOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_EXECUTE_OP_H_
