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
#include "tensorflow/core/tfrt/stubs/tfrt_native_lowering_stub.h"

#include <memory>
#include <utility>

#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime

namespace tfrt {

using ExecutableContext = tensorflow::tfrt_stub::ExecutableContext;

namespace {

class TfrtNativeLoweringStubRegistry {
 public:
  TfrtNativeLoweringStubRegistry()
      : stub_(std::make_unique<TfrtNativeLoweringStub>()) {}

  void Register(std::unique_ptr<TfrtNativeLoweringStub> stub) {
    stub_ = std::move(stub);
  }

  TfrtNativeLoweringStub& Get() { return *stub_; }

 private:
  std::unique_ptr<TfrtNativeLoweringStub> stub_;
};

TfrtNativeLoweringStubRegistry& GetTfrtNativeLoweringStubRegistry() {
  static auto* const registry = new TfrtNativeLoweringStubRegistry;
  return *registry;
}

}  // namespace

void RegisterTfrtNativeLoweringStub(
    std::unique_ptr<TfrtNativeLoweringStub> stub) {
  GetTfrtNativeLoweringStubRegistry().Register(std::move(stub));
}

void AddSyncContext(mlrt::ExecutionContext& execution_context,
                    tfrt::HostContext& host_context,
                    tensorflow::tfrt_stub::SyncResourceState* sync_state) {
  GetTfrtNativeLoweringStubRegistry().Get().AddSyncContext(
      execution_context, host_context, sync_state);
}

void AddNativeLoweringPasses(mlir::OpPassManager* pass_manager) {
  GetTfrtNativeLoweringStubRegistry().Get().AddNativeLoweringPasses(
      pass_manager);
}

absl::StatusOr<std::shared_ptr<ExecutableContext>> BuildExecutableContext(
    mlir::ModuleOp module, const mlrt::KernelRegistry& kernel_registry) {
  return GetTfrtNativeLoweringStubRegistry().Get().BuildExecutableContext(
      module, kernel_registry);
}

}  // namespace tfrt
