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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_EMISSION_CONTEXT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_EMISSION_CONTEXT_H_

#include <memory>

#include "mlir/IR/Diagnostics.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace mlir_gpu {

// Registers a diagnostic handler and collects all the errors as a map from
// HloInstruction* to a vector of string representations of all the errors that
// occurred at that hlo instruction. Also, it takes a function that handles
// those errors at the point when the instance gets destroyed or
// `releaseHloModule()` is called.
//
// EmissionContext uses an RAII pattern, it owns its hlo module and mlir
// context.
class EmissionContext {
 public:
  using ErrorMap =
      std::unordered_map<const HloInstruction*, std::vector<std::string>>;

  // Gets an hlo module and sets the default error handler which writes to the
  // ERROR log and is executed when the instance gets destroyed or
  // `releaseHloModule()` is called.
  explicit EmissionContext(std::unique_ptr<HloModule> module);

  // Gets an hlo module and an error handler function which is executed when the
  // instance gets destroyed or `releaseHloModule()` is called.
  EmissionContext(std::unique_ptr<HloModule> module,
                  std::function<void(const ErrorMap&, HloModule*)> callback);

  // Handles all the errors according to the error handler function before
  // getting destroyed.
  ~EmissionContext();

  // Returns a location constructed from `instr` that then is used by
  // the diagnostic handler to collect the errors.
  mlir::Location getLocation(const HloInstruction* instr);

  // Adds an error message associated with provided hlo instruction.
  void addError(const HloInstruction* hlo_instruction, const string& str);

  // Sets a function that handles the errors at the point when the instance
  // gets destroyed or `releaseHloModule()` is called.
  void setErrorHandler(
      std::function<void(const ErrorMap&, HloModule*)> callback);

  // Releases hlo module and handles all the errors according to the error
  // handler function.
  std::unique_ptr<HloModule> releaseHloModule();

  HloModule* getHloModule() const;

  mlir::MLIRContext* getContext();

 private:
  void registerDiagnosticHandler();
  void callErrorHandlerCallback();

  std::unique_ptr<HloModule> module_;
  ErrorMap instructions_with_error_;
  mlir::MLIRContext context_;
  std::function<void(const ErrorMap&, HloModule*)> error_handler_;
};

}  // namespace mlir_gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_EMISSION_CONTEXT_H_
