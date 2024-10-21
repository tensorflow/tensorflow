/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/pjrt/plugin/stablehlo_reference/executable.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/transforms/Passes.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/plugin/stablehlo_reference/buffer.h"
#include "xla/pjrt/plugin/stablehlo_reference/logging.h"
#include "xla/service/computation_placer.h"
#include "tsl/platform/statusor.h"

#define DEBUG_TYPE "stablehlo-pjrt"

namespace mlir::stablehlo {

#define UNIMPLEMENTED(name) \
  xla::Unimplemented("MlirPjrtBuffer::" #name " is not implemented")

using xla::DeviceAssignment;
using xla::PjRtBuffer;
using xla::PjRtClient;
using xla::PjRtDevice;
using xla::PjRtFuture;
using xla::PjRtLoadedExecutable;
using xla::PjRtMemorySpace;

mlir::OwningOpRef<ModuleOp> cloneIntoContext(ModuleOp module,
                                             MLIRContext& context) {
  // Clone the module into the context. MHLO->StableHLO just in case.
  PassManager pm(module->getContext());
  pm.addPass(mhlo::createHloLegalizeToStablehloPass());
  if (failed(pm.run(module))) {
    LOG(ERROR) << "Failed to convert MHLO to StableHLO";
    return nullptr;
  }

  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  mlir::BytecodeWriterConfig config;
  mlir::OwningOpRef<mlir::ModuleOp> cloned = module.clone();
  if (mlir::failed(mlir::writeBytecodeToFile(*cloned, os, config))) {
    LOG(ERROR) << "Failed to write bytecode to string\n";
    return nullptr;
  }
  return *parseStablehloModule(bytecode, context);
}

LogicalResult decomposeChloToStablehlo(ModuleOp module) {
  PassManager pm(module->getContext());
  stablehlo_ext::createChloLegalizeToStablehloPipeline(pm);
  if (failed(pm.run(module))) {
    return module->emitError() << "Failed to recompose CHLO";
  }
  return success();
}

std::optional<Operation*> getUnsupportedOp(ModuleOp module) {
  std::optional<Operation*> unsupported_op(std::nullopt);
  module.walk([&unsupported_op](Operation* op) {
    auto cc = llvm::dyn_cast<stablehlo::CustomCallOp>(op);
    if (cc) {
      unsupported_op = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return unsupported_op;
}

LogicalResult runHardwareIndependentOptimizations(ModuleOp module) {
  PassManager pm(module->getContext());
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloAggressiveFolderPass());
  pm.addNestedPass<func::FuncOp>(
      stablehlo::createStablehloAggressiveSimplificationPass());
  if (failed(pm.run(module))) {
    return module->emitError()
           << "Failed to run hardware independent optimizations";
  }
  return success();
}

class MlirLoadedExecutable : public PjRtLoadedExecutable {
 public:
  MlirLoadedExecutable(ModuleOp module, DeviceAssignment assignment,
                       absl::Span<PjRtDevice* const> devices,
                       PjRtClient* client)
      : PjRtLoadedExecutable(),
        name_("MlirLoadedExecutable"),
        assignment_(assignment),
        devices_(client->devices()),
        client_(client),
        context_(),
        module_(cloneIntoContext(module, context_)) {
    TRACE_ME_MEMBER;
  }

  static absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      ModuleOp module, DeviceAssignment assignment,
      absl::Span<PjRtDevice* const> devices, PjRtClient* client) {
    TRACE_ME;
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(module->getContext());
    if (failed(decomposeChloToStablehlo(module))) {
      return diagnostic_handler.ConsumeStatus();
    }
    if (auto unsupported_op = getUnsupportedOp(module)) {
      LOG(ERROR) << "Unsupported op: " << ToString(*unsupported_op)
                 << "\nFound in module: " << ToString(module);
      return absl::UnimplementedError(
          absl::StrCat("Unsupported op: ", ToString(*unsupported_op)));
    }

    // Simplify the graph using available HWI passes.
    if (failed(runHardwareIndependentOptimizations(module))) {
      return diagnostic_handler.ConsumeStatus();
    }

    auto executable = std::make_unique<MlirLoadedExecutable>(module, assignment,
                                                             devices, client);

    return executable;
  }

  PjRtClient* client() const override {
    TRACE_ME_MEMBER;
    return client_;
  }

  const DeviceAssignment& device_assignment() const override {
    TRACE_ME_MEMBER;
    return assignment_;
  }

  absl::Span<const PjRtLoadedExecutable::LogicalDeviceIds>
  addressable_device_logical_ids() const override {
    TRACE_ME_MEMBER;
    LOG_UNIMPLEMENTED(addressable_device_logical_ids);
    return {};
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    TRACE_ME_MEMBER;
    return devices_;
  }

  // Helper function to get default mem from device.
  PjRtMemorySpace* get_default_memory_space() const {
    TRACE_ME_MEMBER;
    return devices_[0]->default_memory_space().value_or(nullptr);
  }

  absl::StatusOr<PjRtLoadedExecutable::Result> ExecuteWithReferenceInterpreter(
      absl::Span<PjRtBuffer* const> argument_handles, ModuleOp module,
      PjRtDevice* device, bool fill_future) {
    TRACE_ME_MEMBER;
    SmallVector<mlir::DenseElementsAttr> inputs;
    for (auto* arg : argument_handles) {
      TF_ASSIGN_OR_RETURN(auto mlirArg, GetAttributeFromBuffer(arg));
      auto mlirArgInModuleContext =
          CloneIntoContext(mlirArg, *module->getContext());
      inputs.push_back(mlirArgInModuleContext);
    }
    LOG(INFO) << "EvalModule:\n" << ToString(module) << "\n";
    LOG(INFO) << "Inputs: " << ToString(inputs) << "\n";
    InterpreterConfiguration config;
    FailureOr<SmallVector<DenseElementsAttr>> result =
        evalModule(module, inputs, config);
    if (failed(result)) {
      return absl::InternalError("Failed to execute module");
    }
    LOG(INFO) << "Results: " << ToString(result.value()) << "\n";

    // Naive memory space selection, only using CPU global memory.
    PjRtMemorySpace* memory_space =
        device->default_memory_space().value_or(nullptr);
    std::vector<std::unique_ptr<PjRtBuffer>> buffer_results;
    for (auto res : result.value()) {
      buffer_results.push_back(
          CreateMlirBufferFromAttribute(res, memory_space));
    }

    std::optional<PjRtFuture<>> future;
    if (fill_future) {
      // Synchronous! To make async, this would need to return a future that is
      // ready when the computation is done.
      future = PjRtFuture<>(absl::OkStatus());
    }
    return PjRtLoadedExecutable::Result{future, std::move(buffer_results)};
  }

  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const xla::ExecuteOptions& options,
      std::optional<std::vector<PjRtFuture<>>>& returned_futures) override {
    TRACE_ME_MEMBER;
    if (argument_handles.size() != 1) {
      // One arg handle per device.
      return absl::InvalidArgumentError(
          "MlirLoadedExecutable::Execute only supports a single argument "
          "vector");
    }

    // Single device, synchronous, can always use 0.
    PjRtDevice* device = devices_[0];
    bool fill_future = returned_futures.has_value();
    TF_ASSIGN_OR_RETURN(
        PjRtLoadedExecutable::Result result,
        ExecuteWithReferenceInterpreter(argument_handles[0], module_.get(),
                                        device, fill_future));
    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results;
    results.push_back(std::move(result.buffers));
    if (returned_futures.has_value()) {
      returned_futures->push_back(std::move(result.future.value()));
    }
    return results;
  }

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const xla::ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override {
    TRACE_ME_MEMBER;
    // Synchronous! To make async, have the device make a buffer with a ready
    // future that is ready when the computation is done / buffer is ready.
    TF_ASSIGN_OR_RETURN(
        PjRtLoadedExecutable::Result result,
        ExecuteWithReferenceInterpreter(argument_handles, module_.get(), device,
                                        fill_future));
    if (returned_future.has_value() && fill_future) {
      returned_future = std::move(result.future);
    }
    return std::move(result.buffers);
  }

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const xla::ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override {
    TRACE_ME_MEMBER;
    // Synchronous! To make async, have the device make a buffer with a ready
    // future that is ready when the computation is done / buffer is ready.
    TF_ASSIGN_OR_RETURN(
        PjRtLoadedExecutable::Result result,
        ExecuteWithReferenceInterpreter(argument_handles, module_.get(), device,
                                        fill_future));
    if (returned_future.has_value() && fill_future) {
      returned_future = std::move(result.future);
    }
    return std::move(result.buffers);
  }

  void Delete() override {
    TRACE_ME_MEMBER;
    module_.release();
    module_ = nullptr;
  }
  bool IsDeleted() override {
    TRACE_ME_MEMBER;
    return !module_;
  }

  // PjRtExecutable API.
  int num_replicas() const override {
    TRACE_ME_MEMBER;
    return assignment_.replica_count();
  }
  int num_partitions() const override {
    TRACE_ME_MEMBER;
    return assignment_.computation_count();
  }
  int64_t SizeOfGeneratedCodeInBytes() const override {
    // No generated code.. so just return 1.
    TRACE_ME_MEMBER;
    return 1;
  }
  absl::string_view name() const override {
    TRACE_ME_MEMBER;
    return name_;
  }

  absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>> GetHloModules()
      const override {
    // TODO: This shouldn't be needed for an MLIR plugin, its only used in the
    // JAX layer for determining output sharding, which exists on the mlir
    // module.
    TRACE_ME_MEMBER;
    auto moduleClone = llvm::cast<ModuleOp>(module_.get()->clone());
    TF_ASSIGN_OR_RETURN(auto hlo_module,
                        xla::ConvertStablehloToHlo(moduleClone));
    return std::vector<std::shared_ptr<xla::HloModule>>{std::move(hlo_module)};
  }

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override {
    TRACE_ME_MEMBER;
    return UNIMPLEMENTED(GetOutputMemoryKinds);
  }

  absl::StatusOr<std::string> FingerprintExecutable() const override {
    TRACE_ME_MEMBER;
    return UNIMPLEMENTED(FingerprintExecutable);
  }

 private:
  std::string name_;
  DeviceAssignment assignment_;
  absl::Span<PjRtDevice* const> devices_;
  PjRtClient* client_;

  // MLIR
  MLIRContext context_;
  mlir::OwningOpRef<ModuleOp> module_;
};

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
StablehloReferenceCompile(mlir::ModuleOp module, DeviceAssignment assignment,
                          PjRtClient* client) {
  TRACE_ME;
  return MlirLoadedExecutable::Compile(module, assignment, client->devices(),
                                       client);
}

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
StablehloReferenceCompile(xla::XlaComputation const& computation,
                          xla::DeviceAssignment assignment,
                          xla::PjRtClient* client) {
  TRACE_ME;
  MLIRContext context;
  TF_ASSIGN_OR_RETURN(auto module,
                      ConvertHloToStablehlo(context, &computation.proto()));
  return StablehloReferenceCompile(module.get(), assignment, client);
}

}  // namespace mlir::stablehlo
