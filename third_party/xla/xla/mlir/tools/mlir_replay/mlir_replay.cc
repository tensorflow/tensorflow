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

#include <cstring>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "xla/debug_options_flags.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_replay/mlir_replay_lib.h"
#include "xla/mlir/tools/mlir_replay/public/compiler_trace.pb.h"
#include "xla/mlir/tools/mlir_replay/public/execution_trace.pb.h"
#include "xla/mlir/tools/mlir_replay/public/execution_trace_utils.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/env.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status.h"

struct ReplayOptions {
  std::string hlo_snapshot;
  std::string mlir_compilation_trace;
  std::string mlir_compilation_trace_dir = "";
  std::string execution_trace_dir = "";
  std::vector<std::string> entry_points = {"main", "main_xla_framework"};
  bool print_changes_only = false;
  bool stop_after_first_failure = false;
  bool print_values = true;
};

bool ResultsMatch(const xla::HloSnapshot& snapshot,
                  const llvm::SmallVector<mlir::interpreter::InterpreterValue>&
                      first_pass_results,
                  std::vector<std::string>& failures,
                  const ReplayOptions& opts) {
  auto actual = mlir::interpreter::LiteralToValue(snapshot.result());
  TF_CHECK_OK(actual.status());

  // We assume this is MHLO, so multiple results will be in a tuple.
  if (first_pass_results.size() != 1) {
    failures.push_back("expected one result");
    return false;
  }

  if (!(*actual == first_pass_results[0])) {
    if (opts.print_values) {
      failures.push_back("result mismatch: " + actual->ToString() +
                         " != " + first_pass_results[0].ToString());
    } else {
      failures.push_back("result mismatch");
    }
    return false;
  }
  return true;
}

void TestAll(mlir::MLIRContext& context, const ReplayOptions& opts) {
  std::vector<std::string> traces;
  TF_CHECK_OK(tsl::Env::Default()->GetMatchingPaths(
      opts.mlir_compilation_trace_dir + "/*.mlir-trace.pb", &traces));

  for (const auto& trace_path : traces) {
    mlir::interpreter::MlirCompilationTrace trace;
    TF_CHECK_OK(tsl::ReadBinaryProto(tsl::Env::Default(), trace_path, &trace))
        << "Failed to load " << trace_path;

    std::vector<std::string> snapshots;
    std::string prefix =
        trace_path.substr(0, trace_path.length() - strlen(".mlir-trace.pb"));
    TF_CHECK_OK(tsl::Env::Default()->GetMatchingPaths(prefix + "*.snapshot.*",
                                                      &snapshots));
    CHECK_NE(snapshots.size(), 0)
        << "No snapshots found for module " << trace_path << ".";

    std::vector<std::string> failures;
    for (const auto& snapshot_path : snapshots) {
      xla::HloSnapshot snapshot;
      TF_CHECK_OK(
          tsl::ReadBinaryProto(tsl::Env::Default(), snapshot_path, &snapshot));

      auto results =
          mlir::interpreter::Run(context, trace.passes(0).mlir_module(),
                                 snapshot, nullptr, opts.entry_points);
      if (!results.status().ok()) {
        failures.push_back("Failed to execute " + snapshot_path + ": " +
                           results.status().ToString());
      } else {
        if (!ResultsMatch(snapshot, *results, failures, opts)) {
          failures.push_back(
              std::string("run :mlir_replay -- --mlir-compilation-trace=") +
              trace_path + " --hlo-snapshot=" + snapshot_path +
              " --print-changes-only --stop-after-first-failure");
        }
      }
    }

    if (!failures.empty()) {
      llvm::errs() << "Failures for " << trace_path << ":\n  "
                   << absl::StrJoin(failures, "\n  ") << "\n";
    }
  }
}

int main(int argc, char* argv[]) {
  // Flush llvm::outs before writing errors.
  llvm::errs().tie(&llvm::outs());

  std::string entry_points;
  ReplayOptions opts;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("hlo-snapshot", &opts.hlo_snapshot,
                "Filename of an HloSnapshot proto. Only used to read inputs."),
      tsl::Flag("mlir-compilation-trace", &opts.mlir_compilation_trace,
                "Filename of an MlirCompilerTrace proto."),
      tsl::Flag("mlir-compilation-trace-dir", &opts.mlir_compilation_trace_dir,
                "Directory from which to load MlirCompilerTrace and "
                "HloSnapshot protos. The tool will run all snapshots and "
                "report the ones with bugs."),
      tsl::Flag("execution-trace-dir", &opts.execution_trace_dir,
                "Directory where to store the execution traces (optional)."),
      tsl::Flag("entry-point", &entry_points,
                "Program entry function (optional, defaults to 'main')."),
      tsl::Flag("print-changes-only", &opts.print_changes_only,
                "If set, only print changed values"),
      tsl::Flag("stop-after-first-failure", &opts.stop_after_first_failure,
                "If set, stop after the first failed invocation."),
      tsl::Flag("print-values", &opts.print_values, "If set, print values."),
  };
  xla::AppendDebugOptionsFlags(&flag_list);

  // The usage string includes the message at the top of the file, the
  // DebugOptions flags and the flags defined above.
  std::string usage_string = tsl::Flags::Usage(argv[0], flag_list);
  if (!tsl::Flags::Parse(&argc, argv, flag_list)) {
    return 1;
  }

  if (!entry_points.empty()) {
    opts.entry_points = absl::StrSplit(entry_points, ',');
  }

  tsl::port::InitMain(usage_string.c_str(), &argc, &argv);

  CHECK(opts.mlir_compilation_trace.empty() !=
        opts.mlir_compilation_trace_dir.empty())
      << "Exactly one of --mlir-compilation-trace and "
         "--mlir-compilation-trace-dir must be specified.";

  CHECK(opts.mlir_compilation_trace_dir.empty() || opts.hlo_snapshot.empty())
      << "If --mlir-compilation-trace-dir is set, --hlo-snapshot must not be.";

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);

  mlir::MLIRContext context(registry);

  if (!opts.mlir_compilation_trace_dir.empty()) {
    TestAll(context, opts);
    return 0;
  }

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
        opts.entry_points);
    if (results.status().ok()) {
      if (opts.print_values &&
          (!opts.print_changes_only || (*results != previous_results))) {
        llvm::outs() << "Results:\n";
        for (const auto& result : *results) {
          llvm::outs() << result.ToString() << "\n";
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
