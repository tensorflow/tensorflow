/* Copyright 2023 The OpenXLA Authors. All Rights Reserved.

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
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "xla/literal.h"
#include "xla/mlir/tools/mlir_bisect/bisect_lib.h"
#include "xla/mlir/tools/mlir_bisect/test_passes.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_replay/public/execution_trace.pb.h"
#include "xla/mlir/tools/mlir_replay/public/execution_trace_utils.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/hlo.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/init_main.h"

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

OwningOpRef<ModuleOp> ParseMlirInput(llvm::StringRef input_filename,
                                     MLIRContext* context) {
  std::string error_message;
  auto file = mlir::openInputFile(input_filename, &error_message);
  if (!file) {
    llvm::errs() << error_message << "\n";
    return {};
  }

  auto source_mgr = std::make_shared<llvm::SourceMgr>();
  source_mgr->AddNewSourceBuffer(std::move(file), SMLoc());
  return parseSourceFile<ModuleOp>(source_mgr, context);
}

LogicalResult RunPipeline(ModuleOp module, const Options& options) {
  if (!options.pass_pipeline.hasAnyOccurrences()) {
    return mlir::success();
  }

  auto error_handler = [&](const llvm::Twine& msg) {
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

LogicalResult Run(mlir::Operation* module, interpreter::ExecutionTrace* trace,
                  const Options& options) {
  SymbolTable symbol_table{module};
  interpreter::ExecutionTraceListener tracer(trace);
  interpreter::InterpreterOptions interpreter_options;
  interpreter_options.listener = &tracer;
  interpreter_options.max_steps = options.max_steps_per_run;
  auto results_before_pass = interpreter::RunInterpreter(
      symbol_table, llvm::cast<func::FuncOp>(symbol_table.lookup("main")), {},
      interpreter_options);

  if (!results_before_pass.ok()) {
    llvm::errs() << "Interpreter failed\n";
    return failure();
  }

  if (!options.debug_strategy.empty()) {
    return success();
  }

  OwningOpRef<ModuleOp> clone(llvm::cast<ModuleOp>(module).clone());
  if (!succeeded(RunPipeline(*clone, options))) {
    return failure();
  }

  SymbolTable symbol_table_after{*clone};
  interpreter_options.listener = nullptr;
  bool found_expected_error = false;
  if (!options.expected_error.empty()) {
    auto original_handler = std::move(interpreter_options.error_handler);
    interpreter_options.error_handler = [&](llvm::StringRef failure) {
      found_expected_error |=
          failure.find(options.expected_error) != std::string::npos;
      original_handler(failure);
    };
  }

  auto results_after_pass = interpreter::RunInterpreter(
      symbol_table_after,
      llvm::cast<func::FuncOp>(symbol_table_after.lookup("main")), {},
      std::move(interpreter_options));

  if (!results_after_pass.ok()) {
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
    llvm::errs() << "  " << result.ToString() << "\n";
  }
  llvm::errs() << "\nresults after:\n";
  for (auto& result : *results_after_pass) {
    llvm::errs() << "  " << result.ToString() << "\n";
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
      for (auto& candidate_fn :
           detail::GetCandidates(it->second, state, *module)) {
        auto candidate = candidate_fn();
        if (!candidate || !mlir::verify(*candidate).succeeded()) {
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
        state.SetTrace(std::move(trace));

        // Move strategies to the end.
        decltype(strategies) new_strategies;
        std::copy(it + 1, strategies.end(), std::back_inserter(new_strategies));
        std::copy(strategies.begin(), it + 1,
                  std::back_inserter(new_strategies));
        strategies = std::move(new_strategies);
        return {candidate.release()};
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

    auto constant = b.create<arith::ConstantOp>(
        main.getLoc(), bbarg.getType(), llvm::cast<TypedAttr>(attr.front()));
    bbarg.replaceAllUsesWith(constant);
  }

  // The remaining ops are output args, so we replace them with allocs.
  for (auto arg :
       main.getBody().getArguments().drop_front(snapshot.arguments().size())) {
    CHECK(llvm::isa<MemRefType>(arg.getType())) << "unsupported argument";
    arg.replaceAllUsesWith(b.create<memref::AllocOp>(
        module.getLoc(), llvm::cast<MemRefType>(arg.getType())));
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
  mlir::mhlo::registerAllMhloDialects(registry);

  registry.insert<mlir::arith::ArithDialect>();

  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  auto module = mlir::bisect::ParseMlirInput(options.input_filename, &context);

  if (!options.hlo_snapshot.empty()) {
    xla::HloSnapshot snapshot;
    CHECK_OK(tsl::ReadBinaryProto(tsl::Env::Default(), options.hlo_snapshot,
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
  state.SetTrace(std::move(trace));
  if (!options.debug_strategy.empty()) {
    bool some_failed = false;
    for (auto& candidate : mlir::bisect::detail::GetCandidates(
             mlir::bisect::detail::GetStrategies()[options.debug_strategy],
             state, *module)) {
      auto new_module = candidate();
      if (!new_module) {
        continue;
      }
      llvm::outs() << *new_module << "\n\n";
      if (!mlir::verify(*new_module).succeeded()) {
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
