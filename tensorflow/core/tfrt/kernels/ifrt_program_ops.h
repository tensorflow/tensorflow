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

#ifndef TENSORFLOW_CORE_TFRT_KERNELS_IFRT_PROGRAM_OPS_H_
#define TENSORFLOW_CORE_TFRT_KERNELS_IFRT_PROGRAM_OPS_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable.h"

namespace tensorflow {
namespace tfrt_stub {

// TensorFlow op that calls a Ifrt program registered in `ProgramRegistry`.
class IfrtCallOp : public tensorflow::OpKernel {
 public:
  explicit IfrtCallOp(tensorflow::OpKernelConstruction* ctx);

  IfrtCallOp(const IfrtCallOp& other) = delete;
  IfrtCallOp& operator=(const IfrtCallOp& other) = delete;

  void Compute(tensorflow::OpKernelContext* ctx) override;

 private:
  // Op attributes.
  int64_t program_id_;

  std::vector<std::string> variable_names_;
  std::vector<int> variable_arg_indices_;

  // Ifrt program to be called. Cached after the first call.
  absl::once_flag init_once_;
  tensorflow::ifrt_serving::IfrtServingExecutable* executable_;  // Not owned.
};

}  // namespace tfrt_stub
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_TFRT_KERNELS_IFRT_PROGRAM_OPS_H_
