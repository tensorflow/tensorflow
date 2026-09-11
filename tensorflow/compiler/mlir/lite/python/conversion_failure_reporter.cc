/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/python/conversion_failure_reporter.h"

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/tsl/platform/env.h"

namespace mlir::TFL {
namespace {

bool IsCaretLine(absl::string_view line) {
  absl::string_view trimmed = absl::StripAsciiWhitespace(line);
  return trimmed == "^";
}

}  // namespace

FailureReport ConversionFailureReporter::ParseDiagnostic(
    absl::string_view raw_error, absl::string_view stage,
    absl::string_view status_code, absl::string_view failing_pass,
    absl::string_view failing_pass_arg, absl::string_view failing_function) {
  FailureReport report;
  report.stage = std::string(stage);
  report.status_code = std::string(status_code);
  report.failing_pass = std::string(failing_pass);
  report.failing_pass_arg = std::string(failing_pass_arg);
  report.failing_function = std::string(failing_function);
  report.raw_error = std::string(raw_error);

  std::vector<std::string> lines = absl::StrSplit(raw_error, '\n');

  enum State { kStart, kPrimarySnippet, kCallStackSnippet, kOpContextSnippet };
  State state = kStart;

  for (size_t i = 0; i < lines.size(); ++i) {
    const std::string& line = lines[i];

    // Check for primary error line: "... error: ..."
    auto err_pos = line.find(": error: ");
    if (err_pos != std::string::npos) {
      if (report.primary_error.message.empty()) {
        std::string loc = line.substr(0, err_pos);
        auto file_pos = loc.find("third_party/");
        if (file_pos != std::string::npos) {
          loc = loc.substr(file_pos);
        }
        report.primary_error.location = loc;
        report.primary_error.severity = "error";
        report.primary_error.message = line.substr(err_pos + 9);
        report.primary_error.code_snippet.clear();
        state = kPrimarySnippet;
      } else {
        // Subsequent errors should not mix into the first primary error's
        // snippet.
        state = kStart;
      }
      continue;
    }

    // Check for call stack note: "... note: called from"
    auto call_pos = line.find(": note: called from");
    if (call_pos != std::string::npos) {
      FailureReport::CallStackFrame frame;
      std::string loc = line.substr(0, call_pos);
      auto file_pos = loc.find("third_party/");
      if (file_pos != std::string::npos) {
        loc = loc.substr(file_pos);
      }
      frame.location = loc;
      report.call_stack.push_back(frame);
      state = kCallStackSnippet;
      continue;
    }

    // Check for operation context note: "... note: see current operation: "
    auto op_pos = line.find(": note: see current operation: ");
    if (op_pos != std::string::npos) {
      std::string loc = line.substr(0, op_pos);
      auto file_pos = loc.find("third_party/");
      if (file_pos != std::string::npos) {
        loc = loc.substr(file_pos);
      }
      report.operation_context.location = loc;
      report.operation_context.operation = line.substr(op_pos + 31);
      state = kOpContextSnippet;
      continue;
    }

    // Otherwise, code snippet or continuation line
    absl::string_view trimmed_line = absl::StripAsciiWhitespace(line);
    if (!trimmed_line.empty() && !IsCaretLine(trimmed_line)) {
      if (state == kPrimarySnippet) {
        if (!report.primary_error.code_snippet.empty()) {
          absl::StrAppend(&report.primary_error.code_snippet, "\n");
        }
        absl::StrAppend(&report.primary_error.code_snippet, trimmed_line);
      } else if (state == kCallStackSnippet && !report.call_stack.empty()) {
        if (!report.call_stack.back().code_snippet.empty()) {
          absl::StrAppend(&report.call_stack.back().code_snippet, "\n");
        }
        absl::StrAppend(&report.call_stack.back().code_snippet, trimmed_line);
      } else if (state == kOpContextSnippet) {
        if (!report.operation_context.operation.empty()) {
          absl::StrAppend(&report.operation_context.operation, "\n");
        }
        absl::StrAppend(&report.operation_context.operation, trimmed_line);
      }
    }
  }

  return report;
}

std::string ConversionFailureReporter::GetOrCreateDebugDir(
    absl::string_view working_dir) {
  if (!working_dir.empty()) {
    return std::string(working_dir);
  }
#if defined(_WIN32)
  int pid = _getpid();
#else
  int pid = getpid();
#endif
  return absl::StrFormat(
      "/tmp/litert_conv_%s_%d",
      absl::FormatTime("%E4Y%m%d_%H%M%E6S", absl::Now(), absl::LocalTimeZone()),
      pid);
}

void ConversionFailureReporter::WriteFailureJson(
    absl::string_view working_dir, mlir::ModuleOp module,
    absl::string_view raw_error, absl::string_view stage,
    absl::string_view status_code, absl::string_view failing_pass,
    absl::string_view failing_pass_arg, bool write_module_artifacts,
    absl::string_view failing_function, int64_t elide_elements_larger_than,
    int64_t elide_resource_strings_larger_than) {
  std::string dir = GetOrCreateDebugDir(working_dir);
  if (!tsl::Env::Default()->RecursivelyCreateDir(dir).ok()) return;

  FailureReport report =
      ParseDiagnostic(raw_error, stage, status_code, failing_pass,
                      failing_pass_arg, failing_function);

  // Dump elided MLIR text module if module is valid and artifacts requested
  if (module && write_module_artifacts) {
    std::string elided_path = absl::StrCat(dir, "/before_failure_elided.mlir");
    std::string elided_str;
    llvm::raw_string_ostream elided_os(elided_str);
    mlir::OpPrintingFlags flags;
    flags.elideLargeElementsAttrs(elide_elements_larger_than);
    flags.elideLargeResourceString(elide_resource_strings_larger_than);
    module.print(elided_os, flags);
    if (tsl::WriteStringToFile(tsl::Env::Default(), elided_path, elided_str)
            .ok()) {
      report.elided_mlir_file = elided_path;
    }

    // Dump non-elided binary MLIR bytecode module
    std::string bc_path = absl::StrCat(dir, "/module_bytecode.mlirbc");
    std::string bc_str;
    llvm::raw_string_ostream bc_os(bc_str);
    if (mlir::succeeded(mlir::writeBytecodeToFile(module, bc_os))) {
      if (tsl::WriteStringToFile(tsl::Env::Default(), bc_path, bc_str).ok()) {
        report.bytecode_file = bc_path;
      }
    }
  }

  std::string file_path = absl::StrCat(dir, "/failure.json");

  llvm::json::Object root;
  root["stage"] = report.stage;
  root["status_code"] = report.status_code;

  if (!report.failing_pass.empty()) {
    root["failing_pass"] = report.failing_pass;
  }
  if (!report.failing_pass_arg.empty()) {
    root["failing_pass_arg"] = report.failing_pass_arg;
  }
  if (!report.failing_function.empty()) {
    root["failing_function"] = report.failing_function;
  }

  if (!report.elided_mlir_file.empty()) {
    root["elided_mlir_file"] = report.elided_mlir_file;
  }
  if (!report.bytecode_file.empty()) {
    root["bytecode_file"] = report.bytecode_file;
  }

  if (!report.primary_error.message.empty()) {
    llvm::json::Object prim_err;
    prim_err["location"] = report.primary_error.location;
    prim_err["severity"] = report.primary_error.severity;
    prim_err["message"] = report.primary_error.message;
    if (!report.primary_error.code_snippet.empty()) {
      prim_err["code_snippet"] = report.primary_error.code_snippet;
    }
    root["primary_error"] = std::move(prim_err);
  }

  if (!report.call_stack.empty()) {
    llvm::json::Array stack_arr;
    for (const auto& frame : report.call_stack) {
      llvm::json::Object frame_obj;
      frame_obj["location"] = frame.location;
      if (!frame.code_snippet.empty()) {
        frame_obj["code_snippet"] = frame.code_snippet;
      }
      stack_arr.push_back(std::move(frame_obj));
    }
    root["call_stack"] = std::move(stack_arr);
  }

  if (!report.operation_context.operation.empty()) {
    llvm::json::Object op_ctx;
    op_ctx["location"] = report.operation_context.location;
    op_ctx["operation"] = report.operation_context.operation;
    root["operation_context"] = std::move(op_ctx);
  }

  root["raw_error"] = report.raw_error;

  std::string json_str;
  llvm::raw_string_ostream os(json_str);
  os << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(root)));
  (void)tsl::WriteStringToFile(tsl::Env::Default(), file_path, json_str);
}

}  // namespace mlir::TFL
