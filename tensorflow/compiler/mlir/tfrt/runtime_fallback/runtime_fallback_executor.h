/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_RUNTIME_FALLBACK_RUNTIME_FALLBACK_EXECUTOR_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_RUNTIME_FALLBACK_RUNTIME_FALLBACK_EXECUTOR_H_

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tensorflow {

class RuntimeFallbackExecutor {
 public:
  explicit RuntimeFallbackExecutor(int64_t num_threads);

  // Prepare() needs to be called once before calling Execute(). It sets up all
  // things necessary to execute the given 'mlir_input' with the fallback to
  // tensorflow.
  void Prepare(llvm::StringRef mlir_input);

  // Execute() can be called several times after the call to Prepare() (e.g. for
  // benchmarking).
  llvm::SmallVector<Tensor> Execute(llvm::StringRef function_name,
                                    llvm::ArrayRef<Tensor> arguments);

 private:
  void RunTfrtInitializer();

  std::unique_ptr<thread::ThreadPoolInterface> intra_op_;
  std::unique_ptr<tfrt::HostContext> host_context_;
  tfrt::ResourceContext resource_context_;
  std::unique_ptr<tfrt::ExecutionContext> exec_ctx_;
  tfrt::BefBuffer bef_buffer_;
  tfrt::RCReference<tfrt::BEFFile> bef_file_;
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_MLIR_TFRT_RUNTIME_FALLBACK_RUNTIME_FALLBACK_EXECUTOR_H_
