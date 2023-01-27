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

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_dialect.h"
#include "tensorflow/compiler/xla/mlir/tools/mlir_replay/mlir_replay_lib.h"
#include "tensorflow/compiler/xla/mlir/tools/mlir_replay/public/compiler_trace.pb.h"
#include "tensorflow/compiler/xla/mlir/tools/mlir_replay/public/execution_trace.pb.h"
#include "tensorflow/compiler/xla/mlir_hlo/gml_st/IR/gml_st_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/register.h"
#include "tensorflow/compiler/xla/mlir_hlo/thlo/IR/thlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/tools/mlir_interpreter/framework/interpreter_value.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/init_main.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/util/command_line_flags.h"

struct ReplayOptions {
  std::string hlo_snapshot;
  std::string mlir_compilation_trace;
  std::string execution_trace_dir = "";
  std::string entry_point = "main";
  bool print_changes_only = false;
  bool stop_after_first_failure = false;
};

int main(int argc, char* argv[]) {
  // Flush llvm::outs before writing errors.
  llvm::errs().tie(&llvm::outs());

  ReplayOptions opts;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("hlo_snapshot", &opts.hlo_snapshot,
                "Filename of an HloSnapshot proto. Only used to read inputs."),
      tsl::Flag("mlir_compilation_trace", &opts.mlir_compilation_trace,
                "Filename of an MlirCompilerTrace proto."),
      tsl::Flag("execution_trace_dir", &opts.execution_trace_dir,
                "Directory where to store the execution traces (optional)."),
      tsl::Flag("entry_point", &opts.entry_point,
                "Program entry function (optional, defaults to 'main')."),
      tsl::Flag("print_changes_only", &opts.print_changes_only,
                "If set, only print changed values"),
      tsl::Flag("stop_after_first_failure", &opts.stop_after_first_failure,
                "If set, stop after the first failed invocation."),
  };
  xla::AppendDebugOptionsFlags(&flag_list);

  // The usage string includes the message at the top of the file, the
  // DebugOptions flags and the flags defined above.
  std::string usage_string = tsl::Flags::Usage(argv[0], flag_list);
  if (!tsl::Flags::Parse(&argc, argv, flag_list)) {
    return 1;
  }
  tsl::port::InitMain(usage_string.c_str(), &argc, &argv);

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  registry.insert<mlir::lmhlo::LmhloDialect, mlir::lmhlo_gpu::LmhloGpuDialect,
                  mlir::gml_st::GmlStDialect, mlir::thlo::THLODialect,
                  xla::runtime::RuntimeDialect>();

  mlir::MLIRContext context(registry);

  xla::HloSnapshot snapshot;
  if (!opts.hlo_snapshot.empty()) {
    TF_CHECK_OK(tsl::ReadBinaryProto(tsl::Env::Default(), opts.hlo_snapshot,
                                     &snapshot));
  }
  mlir::interpreter::MlirCompilationTrace trace;
  TF_CHECK_OK(tsl::ReadBinaryProto(tsl::Env::Default(),
                                   opts.mlir_compilation_trace, &trace));

  llvm::SmallVector<mlir::interpreter::InterpreterValue> previous_results;
  int pass_id = 0;
  for (auto& state : trace.passes()) {
    llvm::outs() << "Running IR after " << state.after_pass() << ".\n";
    mlir::interpreter::ExecutionTrace execution_trace;
    auto results = mlir::interpreter::Run(
        context, state.mlir_module(), snapshot,
        opts.execution_trace_dir.empty() ? nullptr : &execution_trace,
        opts.entry_point);
    if (results.status().ok()) {
      if (!opts.print_changes_only || (*results != previous_results)) {
        llvm::outs() << "Results:\n";
        for (const auto& result : *results) {
          llvm::outs() << result.toString() << "\n";
        }
        previous_results = *results;
        llvm::outs() << "\n";
      }
    } else {
      llvm::errs() << results.status().ToString() << "\n";
      if (opts.stop_after_first_failure) {
        return 1;
      }
    }

    if (!opts.execution_trace_dir.empty()) {
      TF_CHECK_OK(
          tsl::Env::Default()->RecursivelyCreateDir(opts.execution_trace_dir));
      std::string filename = tsl::io::JoinPath(
          opts.execution_trace_dir,
          absl::StrFormat("%.4d.%s.mlir", pass_id, state.after_pass()));
      TF_CHECK_OK(tsl::WriteStringToFile(tsl::Env::Default(), filename,
                                         execution_trace.ir()));

      filename = tsl::io::JoinPath(
          opts.execution_trace_dir,
          absl::StrFormat("%.4d.%s.trace.pb", pass_id, state.after_pass()));
      TF_CHECK_OK(tsl::WriteBinaryProto(tsl::Env::Default(), filename,
                                        execution_trace));
    }
    ++pass_id;
  }

  return 0;
}
