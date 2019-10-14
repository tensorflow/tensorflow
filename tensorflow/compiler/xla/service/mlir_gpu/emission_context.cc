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

#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace mlir_gpu {

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

void EmissionContext::addError(const HloInstruction* hloInstruction,
                               const string& str) {
  instructions_with_error_[hloInstruction].push_back(str);
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
