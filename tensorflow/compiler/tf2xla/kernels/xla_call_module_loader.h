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

#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_XLA_CALL_MODULE_LOADER_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_XLA_CALL_MODULE_LOADER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "xla/hlo/builder/xla_computation.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/shape.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

bool IsTokenType(mlir::Type type);

class XlaCallModuleLoader {
 public:
  static absl::StatusOr<std::unique_ptr<XlaCallModuleLoader>> Create(
      mlir::MLIRContext* context, int version, std::string module_str,
      std::vector<std::string> disabled_checks,
      std::vector<std::string> platforms, int num_invocation_args,
      bool main_has_token_input_output);

  int NrInputs() { return main_.getNumArguments(); }
  mlir::TypeRange InputTypes() { return main_.getArgumentTypes(); }

  int NrOutputs() { return main_.getNumResults(); }
  mlir::TypeRange OutputTypes() { return main_.getResultTypes(); }

  // Sets the platform index argument, if the module is compiled for multiple
  // platforms, and then erases the argument.
  absl::Status SetPlatformIndex(absl::string_view compilation_platform);

  // Refines the dynamic module arguments based on the static argument shapes.
  // This assumes that the module has a "main" function without dimension args,
  // but possibly with dynamic shapes. We read the static shapes of the inputs,
  // then set them as the types of the function parameters, and run StableHLO
  // shape refinement to specialize all dynamic shapes in the StableHLO program
  // to static shapes.
  // Starting with version 9, the "main" function may accept token arguments.
  //
  // If the module uses multi-platform lowering, and you called SetPlatformIndex
  // then the refinement will also remove the dead platform code.
  //
  // This method accepts a list of `llvm::ArrayRef` instead of `mlir::Type`.
  // This is to prevent callers from accidentally passing `mlir::Type` owned by
  // a context that's different from the one passed to `Create`, which could
  // cause lifetime issues.
  // The input_shapes includes only the non-token and the non-platform-index
  // arguments.
  absl::Status RefineDynamicShapes(llvm::ArrayRef<xla::Shape> input_shapes);

  // Validates that the module only contains ops from valid dialects.
  absl::Status ValidateDialect();

  // Validates that the module represents a statically-shaped StableHLO program,
  // otherwise all sorts of weirdness might happen in the HLO exporter which is
  // much easier to detect here.
  absl::Status ValidateStaticShapes();

  // Lowers the StableHLO module to MHLO in place.
  absl::Status LowerModuleToMhlo();

  // Lowers the MHLO module to XlaComputation and returns it.
  //
  // REQUIRES: `LowerModuleToMhlo()` is called beforehand.
  absl::StatusOr<xla::XlaComputation> ToXlaComputation();

  // Returns the deserialized stablehlo module.
  mlir::ModuleOp module() & { return *module_; }
  mlir::OwningOpRef<mlir::ModuleOp> module() && { return std::move(module_); }

 private:
  XlaCallModuleLoader() = default;

  // Initializes the loader with the given serialized module string.
  absl::Status LoadModule(mlir::MLIRContext* context, int version,
                          std::string module_str,
                          std::vector<std::string> disabled_checks,
                          std::vector<std::string> platforms,
                          int num_invocation_args,
                          bool main_has_token_input_output);

  // Adds a wrapper for the "main" function to compute the platform index and
  // the dimension arguments.
  absl::Status AddMainWrapper();

  mlir::MLIRContext* context_;
  int version_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::vector<std::string> platforms_;
  bool platform_index_arg_set_ = false;
  // The disabled checks at loading time, including those from the
  // disabled_checks attribute and the TF_XLA_FLAGS environment variable.
  std::vector<std::string> loading_disabled_checks_;
  mlir::func::FuncOp main_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_XLA_CALL_MODULE_LOADER_H_
