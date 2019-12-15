/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/mlir_gpu/emission_context.h"

#include "absl/strings/substitute.h"
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace mlir_gpu {

EmissionContext::EmissionContext(std::unique_ptr<HloModule> module)
    : module_(std::move(module)), context_() {
  error_handler_ = [](const ErrorMap& instructions_with_error,
                      HloModule* module) {
    std::set<const HloComputation*> computations_with_error;
    for (auto err : instructions_with_error) {
      computations_with_error.insert(err.first->parent());
    }

    LOG(ERROR) << module->ToString(
        HloPrintOptions()
            .set_print_instruction(
                [&instructions_with_error](const HloInstruction* instr) {
                  return instructions_with_error.count(instr);
                })
            .set_format_instruction(
                // Returns the string representation of `instr` in the following
                // format.
                //
                // ROOT? instr_name
                //   FAILED: err_0
                //   FAILED: err_1
                //   ...
                [&instructions_with_error](const HloInstruction* instr,
                                           const string& instr_name, int indent,
                                           bool is_root) {
                  const string tab(2 * indent, ' ');
                  if (!instructions_with_error.count(instr)) {
                    return absl::StrCat(tab, is_root ? "ROOT " : "",
                                        instr_name);
                  }
                  static constexpr char kStartBold[] = "\033[1m";
                  static constexpr char kStartRed[] = "\033[31m";
                  static constexpr char kBackToNormal[] = "\033[0m";

                  string result =
                      absl::StrCat(tab, kStartBold, is_root ? "ROOT " : "",
                                   instr_name, kBackToNormal);

                  for (const string& err : instructions_with_error.at(instr)) {
                    absl::SubstituteAndAppend(
                        &result, "\n$0  $1$2FAILED:$3 $4$5$6", tab, kStartBold,
                        kStartRed, kBackToNormal, kStartBold, err,
                        kBackToNormal);
                  }
                  return result;
                })
            .set_print_computation(
                [&computations_with_error](const HloComputation* comp) {
                  return computations_with_error.find(comp) !=
                         computations_with_error.end();
                }));
  };
  registerDiagnosticHandler();
}

EmissionContext::EmissionContext(
    std::unique_ptr<HloModule> module,
    std::function<void(const ErrorMap&, HloModule*)> callback)
    : module_(std::move(module)), context_(), error_handler_(callback) {
  registerDiagnosticHandler();
}

EmissionContext::~EmissionContext() { callErrorHandlerCallback(); }

mlir::Location EmissionContext::getLocation(const HloInstruction* instr) {
  return mlir::OpaqueLoc::get<const HloInstruction*>(instr, &context_);
}

void EmissionContext::addError(const HloInstruction* hlo_instruction,
                               const string& str) {
  instructions_with_error_[hlo_instruction].push_back(str);
}

void EmissionContext::setErrorHandler(
    std::function<void(const ErrorMap&, HloModule*)> callback) {
  error_handler_ = callback;
}

std::unique_ptr<HloModule> EmissionContext::releaseHloModule() {
  callErrorHandlerCallback();
  return std::move(module_);
}

HloModule* EmissionContext::getHloModule() const { return module_.get(); }

mlir::MLIRContext* EmissionContext::getContext() { return &context_; }

void EmissionContext::registerDiagnosticHandler() {
  context_.getDiagEngine().registerHandler([&](mlir::Diagnostic& diag) {
    const HloInstruction* hloInstruction =
        mlir::OpaqueLoc::getUnderlyingLocationOrNull<const HloInstruction*>(
            diag.getLocation());
    assert(hloInstruction);
    addError(hloInstruction, diag.str());
    return mlir::success();
  });
}

void EmissionContext::callErrorHandlerCallback() {
  if (module_.get() && !instructions_with_error_.empty()) {
    error_handler_(instructions_with_error_, module_.get());
  }
}

}  // namespace mlir_gpu
}  // namespace xla
