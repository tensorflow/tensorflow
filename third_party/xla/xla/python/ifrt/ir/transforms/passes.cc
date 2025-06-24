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

#include "xla/python/ifrt/ir/transforms/passes.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.pb.h"
#include "xla/python/ifrt/ir/version.h"

namespace xla {
namespace ifrt {

void CreateIfrtToOutlinedAtomProgramsPipeline(
    mlir::OpPassManager& pm,
    const IfrtToOutlinedAtomProgramsPipelineOptions& options) {
  // Passes that verify the correctness of the module.
  pm.addPass(CreateSpmdExpandableInterfaceVerificationPass(
      {{mlir::mhlo::MhloDialect::getDialectNamespace().str(),
        mlir::stablehlo::StablehloDialect::getDialectNamespace().str(),
        mlir::sdy::SdyDialect::getDialectNamespace().str()}}));
  pm.addNestedPass<mlir::func::FuncOp>(CreateIfrtVerifyDonationPass());

  pm.addPass(CreateIfrtOutlineAtomProgramToModulePass());

  if (!options.propagate_shardings) {
    pm.addPass(CreateIfrtVerifyShardingSpecifiedPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        xla::ifrt::CreateIfrtMergeReshardsPass());
    // We can split ifrt.Reshard to ifrt.CopyArrays because all the shardings
    // are specified.
    pm.addPass(CreateIfrtReshardToCopyArraysPass());
  }
}

void CreateIfrtPopulateAtomProgramMetadataPipeline(mlir::OpPassManager& pm) {
  pm.addPass(CreateIfrtPopulateAtomProgramMetadataPass());
  pm.addPass(CreateIfrtDuplicatedCalleeEliminationPass());
  pm.addPass(mlir::createSymbolDCEPass());
}

void CreateIfrtCompileXlaPreprocessingPipeline(mlir::OpPassManager& pm) {
  pm.addPass(CreateIfrtLowerAtomProgramMetadataToXlaPass());
  pm.addPass(CreateIfrtRemoveIfrtAttrsPass());
}

void CreateIfrtToVersionedPipeline(mlir::OpPassManager& pm,
                                   std::string ifrt_target_version,
                                   std::string vhlo_target_version,
                                   IfrtIrProgramProto& ifrt_ir_program) {
  pm.addPass(CreateIfrtRemoveAttrsFromOtherDialectsPass());
  pm.addPass(CreateIfrtAtomProgramsToVhloPass(
      ifrt_ir_program.mutable_atom_programs(), std::move(vhlo_target_version)));
  pm.addPass(createIfrtLegalizeToVifrtPass());
  // Run symbol DCE to remove atom programs that have been legalized to VHLO.
  pm.addPass(mlir::createSymbolDCEPass());
}

void CreateIfrtFromVersionedPipeline(
    mlir::OpPassManager& pm, const IfrtIrProgramProto& ifrt_ir_program) {
  // Converts from given VIFRT version to the current VIFRT version.
  pm.addPass(
      CreateVifrtToVersionPass({Version::getCurrentVersion().toString()}));
  // Deserializes atom programs (including VHLO serialized version to VHLO
  // current conversion), and inserts them to the IFRT IR program ModuleOp.
  pm.addPass(
      CreateIfrtAtomProgramsFromVhloPass(ifrt_ir_program.atom_programs()));
  // Converts VIFRT current to IFRT.
  pm.addPass(createVifrtLegalizeToIfrtPass());
}

void RegisterIfrtPassesAndPipelines(
    std::shared_ptr<AtomProgramCompiler> compiler,
    std::shared_ptr<
        absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
        compile_options_overrides,
    std::shared_ptr<AtomExecutableMap> atom_executable_map,
    std::shared_ptr<AtomExecutableMap> bound_executable_map) {
  registerIfrtIrPasses();
  RegisterIfrtCompileAtomProgramPass(compiler, compile_options_overrides,
                                     atom_executable_map);
  RegisterIfrtCompileAndPropagateShardingsPass(
      compiler, compile_options_overrides, atom_executable_map);
  RegisterIfrtVerifyBoundExternalLoadedExecutablePass(bound_executable_map);
  mlir::PassPipelineRegistration<IfrtToOutlinedAtomProgramsPipelineOptions>(
      "ifrt-to-outlined-atom-programs-pipeline",
      "Runs passes that do not require compilation-time information",
      CreateIfrtToOutlinedAtomProgramsPipeline);
  mlir::PassPipelineRegistration<>(
      "ifrt-populate-atom-program-metadata-pipeline",
      "Run passes to populate atom program metadata with IFRT info",
      CreateIfrtPopulateAtomProgramMetadataPipeline);
  mlir::PassPipelineRegistration<>(
      "ifrt-compile-xla-preprocessing-pipeline",
      "Run passes to lower an IFRT XLA program for XLA compilation",
      CreateIfrtCompileXlaPreprocessingPipeline);
}

}  // namespace ifrt
}  // namespace xla
