/* Copyright 2023 The JAX Authors.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_REFINE_POLYMORPHIC_SHAPES_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_REFINE_POLYMORPHIC_SHAPES_H_

#include "absl/status/status.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace xla {

// Refines the dynamic shapes for a module whose "main" has static shapes
// and all the intermediate dynamic shapes depend only on the input static
// shapes. Upon refinement, validates that the module does not contain remaining
// dynamic shapes.
// If `enable_shape_assertions` is false, then the shape assertions
// are removed from the module, otherwise they are removed only if the
// assertions hold, and result in an error otherwise.
absl::Status RefinePolymorphicShapes(mlir::ModuleOp module,
                                     bool enable_shape_assertions);

// Like the above but with serialized input and output modules.
// If `validate_static_shapes` is true, then checks that only static shapes
// are left after refinement.
absl::Status RefinePolymorphicShapes(llvm::StringRef module_str,
                                     llvm::raw_ostream &os,
                                     bool enable_shape_assertions,
                                     bool validate_static_shapes);

// Validates that the module has only static shapes.
absl::Status ValidateStaticShapes(mlir::ModuleOp module);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_REFINE_POLYMORPHIC_SHAPES_H_
