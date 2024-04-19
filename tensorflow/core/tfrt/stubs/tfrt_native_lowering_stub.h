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
#ifndef TENSORFLOW_CORE_TFRT_STUBS_TFRT_NATIVE_LOWERING_STUB_H_
#define TENSORFLOW_CORE_TFRT_STUBS_TFRT_NATIVE_LOWERING_STUB_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/core/tfrt/graph_executor/executable_context.h"
#include "tensorflow/core/tfrt/graph_executor/sync_resource_state.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/executable.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace tfrt {

// The tfrt native lowering stub that provides interface for internal and OSS
// with different impls.
class TfrtNativeLoweringStub {
 public:
  virtual ~TfrtNativeLoweringStub() = default;
  virtual void AddSyncContext(
      mlrt::ExecutionContext& execution_context, HostContext& host_context,
      tensorflow::tfrt_stub::SyncResourceState* sync_state) {}
  virtual void AddNativeLoweringPasses(mlir::OpPassManager* pass_manager) {}
  virtual absl::StatusOr<
      std::shared_ptr<tensorflow::tfrt_stub::ExecutableContext>>
  BuildExecutableContext(mlir::ModuleOp module,
                         const mlrt::KernelRegistry& kernel_registry) {
    return absl::UnimplementedError("");
  }
};

void RegisterTfrtNativeLoweringStub(
    std::unique_ptr<TfrtNativeLoweringStub> stub);

void AddSyncContext(mlrt::ExecutionContext& execution_context,
                    tfrt::HostContext& host_context,
                    tensorflow::tfrt_stub::SyncResourceState* sync_state);

void AddNativeLoweringPasses(mlir::OpPassManager* pass_manager);

absl::StatusOr<std::shared_ptr<tensorflow::tfrt_stub::ExecutableContext>>
BuildExecutableContext(mlir::ModuleOp module,
                       const mlrt::KernelRegistry& kernel_registry);
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_STUBS_TFRT_NATIVE_LOWERING_STUB_H_
