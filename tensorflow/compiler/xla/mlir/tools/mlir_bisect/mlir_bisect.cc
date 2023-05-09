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

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/InitAllPasses.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Tools/ParseUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_dialect.h"
#include "tensorflow/compiler/xla/mlir/tools/mlir_bisect/bisect_lib.h"
#include "tensorflow/compiler/xla/mlir/tools/mlir_bisect/test_passes.h"
#include "tensorflow/compiler/xla/mlir/tools/mlir_replay/public/execution_trace_utils.h"
#include "tensorflow/compiler/xla/mlir_hlo/deallocation/IR/deallocation_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/gml_st/IR/gml_st_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/gml_st/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/gml_st/transforms/test_passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/register.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/thlo/IR/thlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/thlo/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/tools/mlir_interpreter/framework/interpreter.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/init_main.h"

struct Options {
  llvm::cl::opt<std::string> input_filename{llvm::cl::Positional,
                                            llvm::cl::desc("<input file>"),
                                            llvm::cl::init("-")};
  llvm::cl::opt<std::string> hlo_snapshot{
      "hlo-snapshot",
      llvm::cl::desc(
          "If set, get argument values from the given snapshot. If not set, "
          "the input function must not have any arguments."),
      llvm::cl::init("")};
  llvm::cl::opt<std::string> debug_strategy{
      "debug-strategy",
      llvm::cl::desc("If set, print all reductions for the given strategy and "
                     "exit. For testing."),
      llvm::cl::init("")};
  llvm::cl::opt<std::string> expected_error{
      "expected-error",
      llvm::cl::desc("If set, expect the given error message after applying "
                     "the pass instead of a successful execution."),
      llvm::cl::init("")};
  llvm::cl::opt<int64_t> max_steps_per_run{
      "max-steps-per-run",
      llvm::cl::desc("Maximum number of steps to execute for each attempt."),
      llvm::cl::init(100000)};
  mlir::PassPipelineCLParser pass_pipeline{"", "Passes to run"};
  llvm::cl::opt<bool> canonicalize{
      "enable-canonicalization",
      llvm::cl::desc("If set, canonicalize candidates before trying them. Set "
                     "to false if you're bisecting --canonicalize."),
      llvm::cl::init(true)};
};

namespace mlir {
namespace bisect {
namespace {

OwningOpRef<ModuleOp> ParseMlirInput(llvm::StringRef inputFilename,
                                     MLIRContext* context) {
  std::string error_message;
  auto file = mlir::openInputFile(inputFilename, &error_message);
  if (!file) {
    llvm::errs() << error_message << "\n";
    return {};
  }

  auto source_mgr = std::make_shared<llvm::SourceMgr>();
  source_mgr->AddNewSourceBuffer(std::move(file), SMLoc());
  return {
      llvm::cast<ModuleOp>(parseSourceFileForTool(source_mgr, context,
                                                  /*insertImplicitModule=*/true)
                               .release())};
}

LogicalResult RunPipeline(ModuleOp module, const Options& options) {
  if (!options.pass_pipeline.hasAnyOccurrences()) {
    return mlir::success();
  }

  auto error_handler = [&](const Twine& msg) {
    llvm::errs() << msg << "\n";
    return failure();
  };
  PassManager pm(module.getContext());
  if (failed(options.pass_pipeline.addToPipeline(pm, error_handler)) ||
      failed(pm.run(module))) {
    llvm::errs() << "pipeline failed\n";
    return failure();
  }
  return success();
}

LogicalResult Run(ModuleOp module, interpreter::ExecutionTrace* trace,
                  const Options& options) {
  SymbolTable symbol_table{module};
  interpreter::ExecutionTraceListener tracer(trace);
  interpreter::InterpreterOptions interpreter_options;
  interpreter_options.listener = &tracer;
  interpreter_options.maxSteps = options.max_steps_per_run;
  auto results_before_pass = interpreter::runInterpreter(
      symbol_table, llvm::cast<func::FuncOp>(symbol_table.lookup("main")), {},
      interpreter_options);

  if (!succeeded(results_before_pass)) {
    llvm::errs() << "Interpreter failed\n";
    return failure();
  }

  if (!options.debug_strategy.empty()) {
    return success();
  }

  OwningOpRef<ModuleOp> clone(module.clone());
  if (!succeeded(RunPipeline(*clone, options))) {
    return failure();
  }

  SymbolTable symbol_table_after{*clone};
  interpreter_options.listener = nullptr;
  bool found_expected_error = false;
  if (!options.expected_error.empty()) {
    auto original_handler = interpreter_options.errorHandler;
    interpreter_options.errorHandler = [&](llvm::StringRef failure) {
      found_expected_error |=
          failure.find(options.expected_error) != std::string::npos;
      original_handler(failure);
    };
  }

  auto results_after_pass = interpreter::runInterpreter(
      symbol_table_after,
      llvm::cast<func::FuncOp>(symbol_table_after.lookup("main")), {},
      interpreter_options);

  if (!succeeded(results_after_pass)) {
    if (found_expected_error) {
      return success();
    }
    llvm::errs() << "Interpreter failed\n";
    return failure();
  } else if (!options.expected_error.empty()) {
    llvm::errs() << "Expected error not seen\n";
    return failure();
  }

  // If the results are the same, the bug is no longer present.
  if (*results_before_pass == *results_after_pass) {
    return failure();
  }

  llvm::errs() << "results before:\n";
  for (auto& result : *results_before_pass) {
    llvm::errs() << "  " << result.toString() << "\n";
  }
  llvm::errs() << "\nresults after:\n";
  for (auto& result : *results_after_pass) {
    llvm::errs() << "  " << result.toString() << "\n";
  }

  return success();
}

LogicalResult Canonicalize(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createCanonicalizerPass());
  return pm.run(module.getOperation());
}

OwningOpRef<ModuleOp> ReduceModule(OwningOpRef<ModuleOp> module,
                                   BisectState& state, const Options& options) {
  auto strategies = llvm::to_vector(mlir::bisect::detail::GetStrategies());

  auto apply_step = [&]() -> std::optional<OwningOpRef<ModuleOp>> {
    for (auto it = strategies.begin(); it != strategies.end(); ++it) {
      for (auto& candidate :
           detail::GetCandidates(it->second, state, *module)) {
        if (!mlir::verify(*candidate).succeeded()) {
          continue;
        }
        if (options.canonicalize && !Canonicalize(*candidate).succeeded()) {
          continue;
        }

        interpreter::ExecutionTrace trace;
        // Verify that the candidate is still buggy.
        if (!Run(*candidate, &trace, options).succeeded()) {
          continue;
        }

        // Print the new buggy module.
        llvm::outs() << "module after " << it->first << ":\n"
                     << *candidate << "\n\n";

        // Update the trace.
        state.SetTrace(trace);

        // Move failed strategies to the end.
        decltype(strategies) new_strategies;
        std::copy(it, strategies.end(), std::back_inserter(new_strategies));
        std::copy(strategies.begin(), it, std::back_inserter(new_strategies));
        strategies = new_strategies;
        return {std::move(candidate)};
      }
    }
    return std::nullopt;
  };

  while (auto new_module = apply_step()) {
    module = std::move(*new_module);
  }
  return module;
}

void ReplaceArgsWithConstants(ModuleOp module,
                              const xla::HloSnapshot& snapshot) {
  auto main = llvm::cast<func::FuncOp>(module.lookupSymbol("main"));
  OpBuilder b(main.getBody());
  for (auto [arg, bbarg] :
       llvm::zip(snapshot.arguments(), main.getBody().getArguments())) {
    auto attr = interpreter::ValueToAttribute(
        *interpreter::LiteralToValue(*xla::Literal::CreateFromProto(arg)),
        bbarg.getType());
    CHECK_EQ(attr.size(), 1) << "unsupported argument";

    bbarg.replaceAllUsesWith(arith::ConstantOp::materialize(
        b, attr.front(), bbarg.getType(), main.getLoc()));
  }
  while (main.getBody().getNumArguments() > 0) {
    main.getBody().eraseArgument(0);
  }
  main.setFunctionType(FunctionType::get(main.getContext(), /*inputs=*/{},
                                         main.getFunctionType().getResults()));
  main.setArgAttrsAttr(b.getArrayAttr({}));
}

}  // namespace
}  // namespace bisect
}  // namespace mlir

int main(int argc, char* argv[]) {
  llvm::errs().tie(&llvm::outs());
  llvm::outs().tie(&llvm::errs());
  int dummy_argc = 1;
  tsl::port::InitMain("", &dummy_argc, &argv);

  Options options;
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR bisect tool\n");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  mlir::bisect::test::RegisterTestPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  mlir::thlo::registerAllThloPasses();
  mlir::gml_st::registerGmlStPasses();
  mlir::gml_st::registerGmlStTestPasses();
  mlir::mhlo::registerAllMhloDialects(registry);

  registry.insert<mlir::lmhlo::LmhloDialect, mlir::gml_st::GmlStDialect,
                  mlir::deallocation::DeallocationDialect,
                  mlir::thlo::THLODialect, xla::runtime::RuntimeDialect>();

  mlir::MLIRContext context(registry);
  auto module = mlir::bisect::ParseMlirInput(options.input_filename, &context);

  if (!options.hlo_snapshot.empty()) {
    xla::HloSnapshot snapshot;
    TF_CHECK_OK(tsl::ReadBinaryProto(tsl::Env::Default(), options.hlo_snapshot,
                                     &snapshot));
    mlir::bisect::ReplaceArgsWithConstants(*module, snapshot);
  }

  if (options.debug_strategy.empty()) {
    llvm::outs() << "initial module:\n" << *module << "\n";
  }

  mlir::interpreter::ExecutionTrace trace;
  if (!mlir::bisect::Run(*module, &trace, options).succeeded()) {
    llvm::outs() << "Did not find bug in initial module\n";
    if (options.pass_pipeline.hasAnyOccurrences() &&
        mlir::succeeded(mlir::bisect::RunPipeline(*module, options))) {
      llvm::outs() << "Module after running pipeline:\n" << *module << "\n";
    }
    return 1;
  }

  mlir::bisect::BisectState state;
  state.SetTrace(trace);
  if (!options.debug_strategy.empty()) {
    bool some_failed = false;
    for (auto& candidate : mlir::bisect::detail::GetCandidates(
             mlir::bisect::detail::GetStrategies()[options.debug_strategy],
             state, *module)) {
      llvm::outs() << *candidate << "\n\n";
      if (!mlir::verify(*candidate).succeeded()) {
        some_failed = true;
        llvm::errs() << "verification failed\n";
      }
    }
    return some_failed ? 1 : 0;
  }

  module = mlir::bisect::ReduceModule(std::move(module), state, options);

  llvm::outs() << "Final module:\n" << *module << "\n";
  if (options.pass_pipeline.hasAnyOccurrences() &&
      mlir::succeeded(mlir::bisect::RunPipeline(*module, options))) {
    llvm::outs() << "Final module after running pipeline:\n" << *module << "\n";
  }
  return 0;
}
