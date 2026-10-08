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

#include "tensorflow/compiler/mlir/lite/python/pass_debug_instrumentation.h"

#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassInstrumentation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/python/conversion_failure_reporter.h"
#include "xla/tsl/platform/env.h"

namespace mlir::TFL {

PassDebugInstrumentation::PassDebugInstrumentation(
    std::string* name, std::string* arg, std::string* failing_func,
    mlir::OwningOpRef<mlir::ModuleOp>* clean_module, bool debug,
    absl::string_view dir, absl::string_view print_before,
    absl::string_view print_after, int64_t elide_elements_larger_than,
    int64_t elide_resource_strings_larger_than)
    : name_(name),
      arg_(arg),
      failing_func_(failing_func),
      clean_module_(clean_module),
      enable_debug_(debug),
      debug_dir_(dir),
      elide_elements_larger_than_(elide_elements_larger_than),
      elide_resource_strings_larger_than_(elide_resource_strings_larger_than) {
  if (!print_before.empty()) {
    print_before_regex_ = std::make_unique<llvm::Regex>(print_before);
    std::string error;
    if (!print_before_regex_->isValid(error)) {
      print_before_regex_.reset();
    }
  }
  if (!print_after.empty()) {
    print_after_regex_ = std::make_unique<llvm::Regex>(print_after);
    std::string error;
    if (!print_after_regex_->isValid(error)) {
      print_after_regex_.reset();
    }
  }
}

void PassDebugInstrumentation::DumpIrToFile(mlir::Pass* pass,
                                            mlir::Operation* op,
                                            llvm::StringRef suffix) {
  if (!op) return;
  std::string dumps_dir = absl::StrCat(debug_dir_, "/ir_dumps");
  if (!tsl::Env::Default()->RecursivelyCreateDir(dumps_dir).ok()) return;
  std::string func_name;
  if (auto sym_attr = op->getAttrOfType<mlir::StringAttr>("sym_name")) {
    func_name =
        absl::StrCat("_", absl::string_view(sym_attr.getValue().data(),
                                            sym_attr.getValue().size()));
  }
  std::string filename = absl::StrFormat(
      "%s/%04d_%s%s_%s.mlir", dumps_dir, step_counter_,
      pass ? pass->getName().str() : "unknown", func_name, suffix.str());

  std::string ir_str;
  llvm::raw_string_ostream os(ir_str);
  mlir::OpPrintingFlags flags;
  if (elide_elements_larger_than_ >= 0) {
    flags.elideLargeElementsAttrs(elide_elements_larger_than_);
  }
  if (elide_resource_strings_larger_than_ >= 0) {
    flags.elideLargeResourceString(elide_resource_strings_larger_than_);
  }
  op->print(os, flags);
  (void)tsl::WriteStringToFile(tsl::Env::Default(), filename, ir_str);

  mlir::ModuleOp module_op = mlir::dyn_cast<mlir::ModuleOp>(op);
  if (!module_op) {
    module_op = op->getParentOfType<mlir::ModuleOp>();
  }
  if (module_op) {
    std::string bc_filename = absl::StrFormat(
        "%s/%04d_%s%s_%s.mlirbc", dumps_dir, step_counter_,
        pass ? pass->getName().str() : "unknown", func_name, suffix.str());
    std::string bc_str;
    llvm::raw_string_ostream bc_os(bc_str);
    if (mlir::succeeded(mlir::writeBytecodeToFile(module_op, bc_os))) {
      (void)tsl::WriteStringToFile(tsl::Env::Default(), bc_filename, bc_str);
    }
  }
}

bool PassDebugInstrumentation::MatchPassRegex(llvm::StringRef pass_name,
                                              const llvm::Regex* regex) {
  if (!regex) return false;
  return regex->match(pass_name);
}

void PassDebugInstrumentation::runBeforePass(mlir::Pass* pass,
                                             mlir::Operation* op) {
  if (pass && pass->getName() == "mlir::detail::OpToOpPassAdaptor") {
    return;
  }
  step_counter_++;
  std::string func_name;
  if (op) {
    if (auto sym_attr = op->getAttrOfType<mlir::StringAttr>("sym_name")) {
      func_name = sym_attr.getValue().str();
    } else {
      func_name = op->getName().getStringRef().str();
    }
  }
  if (enable_debug_) {
    std::string trace_file =
        absl::StrCat(debug_dir_, "/pass_execution_trace.log");
    std::ofstream trace_os(trace_file, std::ios::app);
    if (trace_os.is_open()) {
      trace_os << "[" << step_counter_
               << "] Pass: " << (pass ? pass->getName().str() : "unknown")
               << " | Op: " << func_name << "\n";
    }

    if (clean_module_ && op) {
      if (auto mod = mlir::dyn_cast<mlir::ModuleOp>(op)) {
        *clean_module_ = mod.clone();
      } else if (auto mod = op->getParentOfType<mlir::ModuleOp>()) {
        *clean_module_ = mod.clone();
      }
    }
    if (pass && MatchPassRegex(pass->getName(), print_before_regex_.get())) {
      DumpIrToFile(pass, op, "before");
    }
  }
}

void PassDebugInstrumentation::runAfterPass(mlir::Pass* pass,
                                            mlir::Operation* op) {
  if (pass && pass->getName() == "mlir::detail::OpToOpPassAdaptor") {
    return;
  }
  if (enable_debug_ && pass &&
      MatchPassRegex(pass->getName(), print_after_regex_.get())) {
    DumpIrToFile(pass, op, "after");
  }
}

void PassDebugInstrumentation::runAfterPassFailed(mlir::Pass* pass,
                                                  mlir::Operation* op) {
  if (pass && pass->getName() == "mlir::detail::OpToOpPassAdaptor") {
    return;
  }
  if (pass && name_ && name_->empty()) {
    *name_ = std::string(pass->getName());
    if (arg_) *arg_ = std::string(pass->getArgument());
  }
  std::string func_name;
  if (op) {
    if (auto sym_attr = op->getAttrOfType<mlir::StringAttr>("sym_name")) {
      func_name = sym_attr.getValue().str();
    } else {
      func_name = op->getName().getStringRef().str();
    }
  }
  if (!func_name.empty() && failing_func_ && failing_func_->empty()) {
    *failing_func_ = func_name;
  }
}

std::unique_ptr<mlir::PassInstrumentation>
PipelineFailureCoordinator::CreateInstrumentation(
    absl::string_view print_before_pattern,
    absl::string_view print_after_pattern) {
  return std::make_unique<PassDebugInstrumentation>(
      &failing_pass_name_, &failing_pass_arg_, &failing_function_name_,
      &pre_pass_clean_module_, enable_debug_, debug_dir_, print_before_pattern,
      print_after_pattern, elide_elements_larger_than_,
      elide_resource_strings_larger_than_);
}

absl::Status PipelineFailureCoordinator::ReportFailure(
    mlir::ModuleOp fallback_module, const absl::Status& pass_status) const {
  std::string err_msg =
      pass_status.ok() ? "StableHLO to TFLite pipeline failed."
                       : absl::StrCat("StableHLO to TFLite pipeline failed: ",
                                      pass_status.message());
  std::string pass_arg_flag =
      failing_pass_arg_.empty() ? "" : absl::StrCat("--", failing_pass_arg_);
  mlir::ModuleOp dump_module =
      pre_pass_clean_module_ ? pre_pass_clean_module_.get() : fallback_module;
  ConversionFailureReporter::WriteFailureJson(
      debug_dir_, dump_module, err_msg, "StableHLO_to_TFLite_Pass_Pipeline",
      absl::StatusCodeToString(pass_status.code()), failing_pass_name_,
      pass_arg_flag, /*write_module_artifacts=*/enable_debug_,
      failing_function_name_, elide_elements_larger_than_,
      elide_resource_strings_larger_than_);
  return absl::InvalidArgumentError(err_msg);
}

absl::Status PipelineFailureCoordinator::ReportSerializationFailure(
    mlir::ModuleOp module, const absl::Status& status,
    absl::string_view diag_errors) const {
  std::string detail(diag_errors.empty() ? status.message() : diag_errors);
  std::string err_msg =
      absl::StrCat("Failed to serialize to FlatBuffer: ", detail);
  ConversionFailureReporter::WriteFailureJson(
      debug_dir_, module, err_msg, "FlatBuffer_Serialization",
      absl::StatusCodeToString(status.code()),
      /*failing_pass=*/"", /*failing_pass_arg=*/"",
      /*write_module_artifacts=*/enable_debug_,
      /*failing_function=*/"", elide_elements_larger_than_,
      elide_resource_strings_larger_than_);
  return absl::InvalidArgumentError(err_msg);
}

}  // namespace mlir::TFL
