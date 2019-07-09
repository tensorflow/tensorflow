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

#include "mlir/Dialect/Traits.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Dialect.h"  // TF:local_config_mlir
#include "mlir/IR/Matchers.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/OpDefinition.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Support/TypeUtilities.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {

class TensorFlowDialect : public Dialect {
 public:
  TensorFlowDialect(MLIRContext *context);

  // Gradient attribute ("tf.gradient") in the list of NamedAttibutes in a
  // function references to its gradient function. This attribute in TensorFlow
  // Dialect is used to model TF GradientDef. GetGradientAttrName() returns the
  // string description of gradient attribute.
  static StringRef GetGradientAttrName() { return "tf.gradient"; }

  // Parses a type registered to this dialect.
  Type parseType(StringRef data, Location loc) const override;

  // Prints a type registered to this dialect.
  void printType(Type ty, raw_ostream &os) const override;

  // Parse and print variant type. It may have subtypes inferred using shape
  // inference.
  Type ParseVariantType(StringRef spec, Location loc) const;
  void PrintVariantType(VariantType ty, raw_ostream &os) const;

  // Registered hook to materialize a constant operation from a given attribute
  // value with the desired resultant type.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;
};

// This verifies that the Op is a well-formed TensorFlow op, checking
// that all inputs and results are Tensor or other TensorFlow types, etc.
static LogicalResult verifyTensorFlowOp(Operation *op);

// This Trait should be used by all TensorFlow Ops.
//
template <typename ConcreteType>
class TensorFlowOp : public OpTrait::TraitBase<ConcreteType, TensorFlowOp> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyTensorFlowOp(op);
  }
};

// TODO(b/131258166): TensorFlow's mutex.h defines a `mutex_lock` macro, whose
// purpose is to catch bug on `tensorflow::mutex_lock`. We don't use
// `tensorflow::mutex_lock` here but we have ops (`tf.MutexLock` and
// `tf.ConsumeMutexLock`) with getter methods named as `mutex_lock()`. Need to
// undefine here to avoid expanding the getter symbol as macro when including
// both mutex.h and this header file.
#undef mutex_lock

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h.inc"

// The "tf.If" operation takes a condition operand, a list of inputs, and a
// function attribute for the then/else branches.  The condition operand
// doesn't have to be a boolean tensor.  It is handled according to these
// rules, quoting the TensorFlow op definition:
//
//   If the tensor is a scalar of non-boolean type, the scalar is converted to
//   a boolean according to the following rule: if the scalar is a numerical
//   value, non-zero means True and zero means False; if the scalar is a
//   string, non-empty means True and empty means False. If the tensor is not a
//   scalar, being empty means False and being non-empty means True.
//
// This is defined in TensorFlow as:
//
// REGISTER_OP("If")
//     .Input("cond: Tcond")
//     .Input("input: Tin")
//     .Output("output: Tout")
//     .Attr("Tcond: type")
//     .Attr("Tin: list(type) >= 0")
//     .Attr("Tout: list(type) >= 0")
//     .Attr("then_branch: func")
//     .Attr("else_branch: func")
//
// Note: Additional result corresponds to the control output.
class IfOp : public Op<IfOp, TensorFlowOp, OpTrait::AtLeastNOperands<1>::Impl,
                       OpTrait::VariadicResults> {
 public:
  using Op::Op;
  static StringRef getOperationName() { return "tf.If"; }

  Value *getCondition() { return getOperand(0); }

  // TODO(b/132271680): This is not following Google naming style
  StringRef getThen() {
    return getAttrOfType<FunctionAttr>("then_branch").getValue();
  }

  StringRef getElse() {
    return getAttrOfType<FunctionAttr>("else_branch").getValue();
  }

  LogicalResult verify();
};

// The "tf.While" operation takes a list of inputs and function attributes for
// the loop condition and body.  Inputs are updated repeatedly by the body
// function while the loop condition with the tensors evaluates to true.  The
// condition result doesn't have to be a boolean tensor.  It is handled
// according to these rules, quoting the TensorFlow op definition:
//
//   If the tensor is a scalar of non-boolean type, the scalar is converted to
//   a boolean according to the following rule: if the scalar is a numerical
//   value, non-zero means True and zero means False; if the scalar is a
//   string, non-empty means True and empty means False. If the tensor is not a
//   scalar, being empty means False and being non-empty means True.
//
// This is defined in TensorFlow as:
//
// REGISTER_OP("While")
//      .Input("input: T")
//      .Output("output: T")
//      .Attr("T: list(type) >= 0")
//      .Attr("cond: func")
//      .Attr("body: func")
//      .Attr("output_shapes: list(shape) = []")
//
class WhileOp : public Op<WhileOp, TensorFlowOp, OpTrait::VariadicOperands,
                          OpTrait::VariadicResults> {
 public:
  using Op::Op;
  static StringRef getOperationName() { return "tf.While"; }

  StringRef getCond() { return getAttrOfType<FunctionAttr>("cond").getValue(); }
  StringRef getBody() { return getAttrOfType<FunctionAttr>("body").getValue(); }

  LogicalResult verify();
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OPS_H_
