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
#include "tensorflow/compiler/mlir/lite/experimental/tac/tac_module.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {
// TODO(b/177376459): We should make this configureable.
void AddExportTFLPass(mlir::OpPassManager* pass_manager, bool enable_inliner) {
  if (enable_inliner) pass_manager->addPass(mlir::createInlinerPass());
  pass_manager->addPass(mlir::createSymbolDCEPass());
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());
  pass_manager->addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
}
}  // namespace

// TODO(b/177376459): We should make this configureable.
void TacModule::AddTACPass(mlir::OpPassManager* pass_manager,
                           llvm::ArrayRef<std::string> device_specs) {
  pass_manager->addPass(mlir::TFL::tac::CreateTargetAnnotationPass(this));
  pass_manager->addPass(mlir::TFL::tac::CreateRaiseTargetSubgraphsPass());
  pass_manager->addPass(mlir::TFL::tac::CreateFoldConstantsToSubgraphPass(
      /*fold_all_constants=*/false));
  pass_manager->addPass(
      mlir::TFL::tac::CreateAlternativeSubgraphPass(device_specs));
  if (options_.legalize_to_tflite_ops) {
    // After we creat the alternative subgraph, we can still do canonicalization
    // legalization & other optimizations as long as we're not inlining the
    // function.
    // And in fact, we probably need to do the proper legalization, for the
    // compute cost to work. (in case we added some TF ops)
    pass_manager->addPass(mlir::TFL::CreatePrepareTFPass(
        /*unfold_batch_matmul=*/true,
        /*allow_bf16_and_f16_type_legalization=*/false));
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::createCanonicalizerPass());
    pass_manager->addPass(
        mlir::TFL::CreateLegalizeTFPass(/*run_tfl_runtime_verification=*/true));
    pass_manager->addPass(mlir::TFL::CreateOptimizePass());
  }

  pass_manager->addPass(mlir::TFL::tac::CreateComputeCostPass());
  pass_manager->addPass(mlir::TFL::tac::CreatePickSubgraphsPass());
  // After this pass, we may consider add a pass to merge small functions into
  // large functions (and maybe other metadata as well).
}

const tac::TargetHardware* TacModule::GetTargetHardware(
    const std::string& hardware_name) const {
  for (auto& hardware : backends_) {
    if (GetHardwareName(hardware.get()) == hardware_name) return hardware.get();
  }
  return nullptr;
}

absl::Status TacModule::RunTacPasses(mlir::ModuleOp* module, bool debug_mode) {
  mlir::PassManager pm((*module)->getName(),
                       mlir::OpPassManager::Nesting::Implicit);
  AddTACPass(&pm, options_.hardware_backends);
  if (!debug_mode) {
    AddExportTFLPass(&pm, options_.enable_inliner);
  }

  mlir::StatusScopedDiagnosticHandler statusHandler(module->getContext(),
                                                    /*propagate=*/true);
  if (failed(pm.run(*module))) {
    return absl::InternalError("conversion error");
  }
  return absl::OkStatus();
}

std::vector<std::unique_ptr<tac::TargetHardware>>
TacModule::InstantiateBackends() {
  std::vector<std::unique_ptr<tac::TargetHardware>> backends;
  for (const auto& hardware_name : options_.hardware_backends) {
    auto factory = tac::GetTargetHardwareFactory(hardware_name);
    backends.emplace_back(factory());
    backends.back()->Init();
  }
  return backends;
}

absl::Status TacModule::Run() {
  // Construct all backends.
  backends_ = InstantiateBackends();
  const_backends_.resize(backends_.size());
  for (const auto& backend : backends_)
    const_backends_.emplace_back(backend.get());

  if (!importer_) {
    return absl::Status(absl::StatusCode::kFailedPrecondition,
                        "Null Importer provided");
  }
  if (!exporter_) {
    return absl::Status(absl::StatusCode::kFailedPrecondition,
                        "Null Exporter provided");
  }

  auto module_status = importer_->Import();
  if (!module_status.ok()) {
    return module_status.status();
  }
  auto module = module_status->get();
  auto* context = module->getContext();
  context->appendDialectRegistry(registry_);
  context->loadAllAvailableDialects();

  // Run TAC passes.
  auto status = RunTacPasses(&module, options_.debug_mode);

  if (!status.ok()) {
    return status;
  }

  return exporter_->Export(module);
}

void TacModule::RegisterExtraDialects(mlir::DialectRegistry& registry) {
  registry.appendTo(registry_);
}
}  // namespace tac
}  // namespace TFL
}  // namespace mlir
