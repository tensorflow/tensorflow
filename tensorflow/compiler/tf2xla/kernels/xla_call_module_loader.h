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
#include <vector>

#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {

class XlaCallModuleLoader {
 public:
  static tsl::StatusOr<std::unique_ptr<XlaCallModuleLoader>> Create(
      int version, std::string module_str,
      std::vector<std::string> dim_args_spec, int platform_index);

  int nr_outputs() const { return nr_outputs_; }

  // Refines the dynamic module arguments based on the static argument shapes.
  // This assumes that the module has a "main" function without dimension args,
  // but possibly with dynamic shapes. We read the static shapes of the inputs,
  // then set them as the types of the function parameters, and run StableHLO
  // shape refinement to specialize all dynamic shapes in the StableHLO program
  // to static shapes.
  tsl::Status RefineDynamicShapes(absl::Span<const xla::Shape> input_shapes);

  // Validate that the module represents a statically-shaped StableHLO program,
  // otherwise all sorts of weirdness might happen in the HLO exporter which is
  // much easier to detect here.
  tsl::Status ValidateModule();

  tsl::StatusOr<xla::XlaComputation> ToXlaComputation();

 private:
  XlaCallModuleLoader() = default;

  // Initializes the loader with the given serialized module string.
  tsl::Status LoadAndPreprocessModule(int version, std::string module_str,
                                      std::vector<std::string> dim_args_spec,
                                      int platform_index);

  // Adds a wrapper for the "main" function to compute the platform index and
  // the dimension arguments.
  tsl::Status AddMainWrapper();

  mlir::MLIRContext context_{mlir::MLIRContext::Threading::DISABLED};

  int version_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  int platform_index_;
  std::vector<std::string> dim_args_spec_;
  int nr_outputs_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_XLA_CALL_MODULE_LOADER_H_
