/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <stdlib.h>
#include <fstream>

#include "tensorflow/compiler/plugin/executor/compiler.h"
#include "tensorflow/compiler/plugin/executor/executable.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/inliner.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/strcat.h"

namespace xla {
namespace executorplugin {

namespace se = ::perftools::gputools;
namespace sep = ::perftools::gputools::executorplugin;

/*
 * Run optimization passes on the module.  The graph is transformed by
 * each pass in the optimization pipeline.  The service subdirectory
 * contains useful optimization passes.
 */
Status ExecutorCompiler::RunHloOptimization(HloModule* hlo_module) {
  HloPassPipeline pipeline("Executor");
  pipeline.AddPass<Inliner>();
  pipeline.AddPass<HloSubcomputationUnification>();
  pipeline.AddPass<HloCSE>(false);

  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(
      false, [](const Shape&, const Shape&) { return false; });
  pipeline.AddPass<ReshapeMover>();
  pipeline.AddPass<HloConstantFolding>();
  pipeline.AddPass<HloCSE>(true);

  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<FlattenCallGraph>();
  return pipeline.Run(hlo_module).status();
}

StatusOr<std::unique_ptr<Executable>> ExecutorCompiler::Compile(
        std::unique_ptr<HloModule> hlo_module,
        se::StreamExecutor* stream_exec) {
  TF_RET_CHECK(stream_exec != nullptr);

  VLOG(1) << "Generate graph " << hlo_module->name();

  TF_RETURN_IF_ERROR(RunHloOptimization(hlo_module.get()));

  // Typically you would visit the HLO graph, building up a compiled equivalent
  // In this case we are using an Hlo evaluator at execution time, so we don't
  // need to compile anything

  // Create executable from only the Hlo module
  std::unique_ptr<Executable> executable;
  executable.reset(new ExecutorExecutable(std::move(hlo_module)));

  return std::move(executable);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> ExecutorCompiler::Compile(
        std::vector<std::unique_ptr<HloModule>> hlo_modules,
        std::vector<se::StreamExecutor*> stream_execs) {

  return tensorflow::errors::Unimplemented(
      "Compilation of multiple HLO modules is not supported on Executor.");
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
ExecutorCompiler::CompileAheadOfTime(
    std::vector<std::unique_ptr<HloModule>> hlo_modules,
    const AotCompilationOptions& aot_options) {

  return tensorflow::errors::InvalidArgument(
      "AOT compilation not supported on Executor");
}

se::Platform::Id ExecutorCompiler::PlatformId() const {
  return sep::kExecutorPlatformId;
}

HloCostAnalysis::ShapeSizeFunction
ExecutorCompiler::ShapeSizeBytesFunction() const {
  return ExecutorExecutable::ShapeSizeBytes;
}

REGISTER_MODULE_INITIALIZER(executor_compiler, {
  xla::Compiler::RegisterCompilerFactory(sep::kExecutorPlatformId, []() {
    return xla::MakeUnique<xla::executorplugin::ExecutorCompiler>();
  });
});

}  // namespace executorplugin
}  // namespace xla
