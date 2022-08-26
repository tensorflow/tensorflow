/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_MLIR_RUNTIME_SPECIALIZATION_H_
#define XLA_MLIR_RUNTIME_SPECIALIZATION_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/constraints.h"
#include "tensorflow/compiler/xla/runtime/symbolic_shape.h"

namespace xla {
namespace runtime {

// TODO(ezhulenev): A lot of specialization code is written with an assumption
// that we can only specialize Tensor arguments. Make this extendable
// to support user-defined types and user-defined specializations.

// Symbolic shape attached to the argument as a dense i64 array attribute.
// TODO(ezhulenev): Change symbolic shape attribute type to match the comment.
constexpr const char* kSymbolicShapeAttrName = "rt.symbolic_shape";

// Listener class to control notifications during specialization.
struct SpecializationListener {
  virtual ~SpecializationListener() {}

  // Called at the end of module specialization.
  // - 'operands' is a reference to the specialized operands' types.
  // - `attrs` is a list of attributes attached to operands.
  virtual void notifyModuleSpecialized(
      llvm::ArrayRef<mlir::Type> operands,
      llvm::ArrayRef<mlir::DictionaryAttr> attrs) const {}

  // Called once for every value-specialized argument.
  virtual void notifyValueSpecialized(unsigned index, mlir::Type type,
                                      mlir::Attribute value) const {}
};

// Specializes function to the runtime arguments:
//
// - updates all unknown dimensions according to the resolved symbolic shapes
// - attaches symbolic shape attribute to the operands
// - for value-specialized operands sinks small constants into the function body
//
// Returns error if arguments are not compatible with the function signature.
absl::Status SpecializeFunction(
    mlir::func::FuncOp func, ArgumentsRef arguments,
    llvm::ArrayRef<SymbolicShapesResolver::SymbolicShape> symbolic_shapes,
    llvm::ArrayRef<ArgumentConstraint> constraints,
    const SpecializationListener* listener = nullptr);

}  // namespace runtime
}  // namespace xla

#endif  // XLA_MLIR_RUNTIME_SPECIALIZATION_H_
