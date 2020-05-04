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

// This file defines the operations used in the standard MLIR TensorFlow dialect
// after control dependences are raise to the standard form.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_H_

#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffects.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_verifiers.h"

namespace mlir {
namespace TF {

class TensorFlowDialect : public Dialect {
 public:
  TensorFlowDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "tf"; }

  // Gradient attribute ("tf.gradient") in the list of NamedAttributes in a
  // function references to its gradient function. This attribute in TensorFlow
  // Dialect is used to model TF GradientDef. GetGradientAttrName() returns the
  // string description of gradient attribute.
  static StringRef GetGradientAttrName() { return "tf.gradient"; }

  // This attribute marks if a function is stateful.
  // Returns the string description of stateful attribute.
  static StringRef GetStatefulAttrName() { return "tf.signature.is_stateful"; }

  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  void printAttribute(Attribute attr, DialectAsmPrinter &os) const override;

  // Parse a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type ty, DialectAsmPrinter &os) const override;

  // Parses resource type with potential subtypes.
  Type ParseResourceType(DialectAsmParser &parser, Location loc) const;

  // Prints resource type with potential subtypes.
  void PrintResourceType(ResourceType ty, DialectAsmPrinter &os) const;

  // Parse and print variant type. It may have subtypes inferred using shape
  // inference.
  Type ParseVariantType(DialectAsmParser &parser, Location loc) const;
  void PrintVariantType(VariantType ty, DialectAsmPrinter &os) const;

  // Registered hook to materialize a constant operation from a given attribute
  // value with the desired resultant type.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;
};

// TODO(b/131258166): TensorFlow's mutex.h defines a `mutex_lock` macro, whose
// purpose is to catch bug on `tensorflow::mutex_lock`. We don't use
// `tensorflow::mutex_lock` here but we have ops (`tf.MutexLock` and
// `tf.ConsumeMutexLock`) with getter methods named as `mutex_lock()`. Need to
// undefine here to avoid expanding the getter symbol as macro when including
// both mutex.h and this header file.
#undef mutex_lock

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h.inc"
#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h.inc"

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_H_
