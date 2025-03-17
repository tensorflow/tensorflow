/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_RESHARD_VARIABLES_OP_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_RESHARD_VARIABLES_OP_H_

#include <memory>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_common.pb.h"

namespace tensorflow {

// Op that changes the sharding state for a set of on-device variables. The
// sharding state is represented as the key of the compilation that generated
// the sharding/unsharding programs along with the main program. The op checks
// if the current sharding state matches the desired one, and if not, uses the
// sharding/unsharding programs to transform the variables to the desired state.
class TPUReshardVariablesOpKernel : public AsyncOpKernel {
 public:
  explicit TPUReshardVariablesOpKernel(OpKernelConstruction* context);
  ~TPUReshardVariablesOpKernel() override;

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override;

 private:
  absl::Status DoWork(OpKernelContext* context);
  absl::Status DoTpuExecute(OpKernelContext* context, const Tensor& format_key,
                            tpu::CompilationCacheFetchTarget fetch_target);

  int64_t num_vars_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_RESHARD_VARIABLES_OP_H_
