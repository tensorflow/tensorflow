/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/compilation/passes.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "xla/python/ifrt/ir/compilation/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace xla {
namespace ifrt {

namespace {

std::string GetDotDumpDir(std::string dot_graph_dump_to) {
  if (dot_graph_dump_to == "sponge") {
    if (!tsl::io::GetTestUndeclaredOutputsDir(&dot_graph_dump_to)) {
      // Compile option `dot_graph_dump_to=sponge` is specified outside of a
      // test. Ignore the value.
      return "";
    }
  }
  return dot_graph_dump_to;
}

IfrtToDotPassOptions GetIfrtToDotPassOptions(
    const IfrtIRCompileOptions& compile_options) {
  return IfrtToDotPassOptions{
      /*dot_graph_dump_to=*/GetDotDumpDir(compile_options.dot_graph_dump_to),
      /*dot_graph_min_executable_peak_memory_bytes=*/
      compile_options.dot_graph_min_executable_peak_memory_bytes,
      /*dot_graph_min_executable_flops=*/
      compile_options.dot_graph_min_executable_flops,
      /*dot_graph_min_per_device_transfer_size_bytes=*/
      compile_options.dot_graph_min_per_device_transfer_size_bytes};
}

}  // namespace

void createIfrtCompileXlaPreprocessingPipeline(
    mlir::OpPassManager& pm,
    std::shared_ptr<IfrtIRCompileOptions> compile_options) {
  pm.addPass(createIfrtLowerAtomProgramMetadataToXlaPass(
      {/*compile_options=*/compile_options}));
  pm.addPass(createIfrtRemoveIfrtAttrsPass());
}

absl::Status createOutlinedAtomProgramsToCompiledPipeline(
    mlir::OpPassManager& pm, std::shared_ptr<AtomProgramCompiler> compiler,
    const OutlinedAtomProgramsToCompiledPipelineOptions& options,
    std::shared_ptr<IfrtIRCompileOptions> compile_options,
    std::shared_ptr<AtomExecutableFutureMap> atom_executable_future_map,
    std::shared_ptr<AtomExecutableMap> bound_executable_map) {
  IfrtToDotPassOptions ifrt_to_dot_pass_options =
      GetIfrtToDotPassOptions(*compile_options);
  pm.addPass(createIfrtVerifyDeviceTypeConsistencyPass(
      {/*platform_names=*/llvm::to_vector(options.platform_names)}));
  pm.addPass(createIfrtLowerMpmdReshardToCallPass());
  pm.addPass(createIfrtPrecompileAtomProgramPreprocessingPass(
      {/*platform_names=*/llvm::to_vector(options.platform_names),
       /*compile_options=*/compile_options}));
  pm.addPass(createIfrtCompileAtomProgramPass(
      std::move(compiler), compile_options->compile_options_overrides,
      atom_executable_future_map));
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addPass(createIfrtVerifyBoundExternalLoadedExecutablePass(
      std::move(bound_executable_map)));

  if (!ifrt_to_dot_pass_options.dot_graph_dump_to.empty()) {
    ABSL_RETURN_IF_ERROR(tsl::Env::Default()->RecursivelyCreateDir(
        ifrt_to_dot_pass_options.dot_graph_dump_to));
    pm.addPass(createIfrtToDotPass(std::move(ifrt_to_dot_pass_options),
                                   atom_executable_future_map));
  }
  return absl::OkStatus();
}

void registerIfrtCompilationPassesAndPipelines(
    std::shared_ptr<AtomProgramCompiler> compiler,
    std::shared_ptr<IfrtIRCompileOptions> compile_options,
    std::shared_ptr<AtomExecutableFutureMap> atom_executable_future_map,
    std::shared_ptr<AtomExecutableMap> bound_executable_map) {
  registerIfrtIrCompilationPasses();
  registerIfrtCompileAtomProgramPass(compiler,
                                     compile_options->compile_options_overrides,
                                     atom_executable_future_map);
  registerIfrtVerifyBoundExternalLoadedExecutablePass(bound_executable_map);
  registerIfrtToDotPass(GetIfrtToDotPassOptions(*compile_options),
                        atom_executable_future_map);
  mlir::PassPipelineRegistration<>(
      "ifrt-compile-xla-preprocessing-pipeline",
      "Run passes to lower an IFRT XLA program for XLA compilation",
      [compile_options](mlir::OpPassManager& pm) mutable {
        createIfrtCompileXlaPreprocessingPipeline(pm, compile_options);
      });
  // Do not move to lambda captures because the pass pipeline registration is
  // invoked for each module in a test file.
  mlir::PassPipelineRegistration<OutlinedAtomProgramsToCompiledPipelineOptions>(
      "ifrt-outlined-atom-programs-to-compiled-pipeline",
      "Runs passes to compile IFRT IR programs with outlined atom programs",
      [compiler, compile_options, atom_executable_future_map,
       bound_executable_map](
          mlir::OpPassManager& pm,
          const OutlinedAtomProgramsToCompiledPipelineOptions&
              options) mutable {
        CHECK_OK(createOutlinedAtomProgramsToCompiledPipeline(
            pm, compiler, options, compile_options, atom_executable_future_map,
            bound_executable_map));
      });
}

}  // namespace ifrt
}  // namespace xla
