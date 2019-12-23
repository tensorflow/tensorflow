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

// This file defines the operations for the "Control Flow" dialect of TensorFlow
// graphs.  The TensorFlow control flow dialect represents control flow with
// Switch/Merge and a few related control flow nodes, along with control
// dependencies.  This dialect can be raised to the standard TensorFlow dialect
// by transforming Switch/Merge and other control flow ops into functional
// control flow ops and removing control dependencies.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_CONTROL_FLOW_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_CONTROL_FLOW_OPS_H_

#include "mlir/IR/Dialect.h"  // TF:local_config_mlir
#include "mlir/IR/OpDefinition.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir

namespace mlir {
namespace TFControlFlow {

class TFControlFlowDialect : public Dialect {
 public:
  explicit TFControlFlowDialect(MLIRContext *context);

  // Parses a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;
};

namespace TensorFlowControlTypes {
enum Kind {
  Control = Type::FIRST_TENSORFLOW_CONTROL_TYPE,
};
}

class TFControlType : public Type::TypeBase<TFControlType, Type> {
 public:
  using Base::Base;

  static TFControlType get(MLIRContext *context) {
    return Base::get(context, TensorFlowControlTypes::Control);
  }

  // Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) {
    return kind == TensorFlowControlTypes::Control;
  }
};

// The "_tf.Enter" operation forwards its input to Tensorflow while loop. Each
// tensor needs its own _tf.Enter to be made available inside the while loop.
//
// More details can be found in Tensorflow Controlflow white paper:
// https://storage.googleapis.com/download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf
//
// This is defined in Tensorflow as:
//
// REGISTER_OP("Enter")
//    .Input("data: T")
//    .Output("output: T")
//    .Attr("T: type")
//    .Attr("frame_name: string")
//    .Attr("is_constant: bool = false")
//    .Attr("parallel_iterations: int = 10")
//
// For example:
//   %1 = "_tf.Enter"(%0#0) {T: "tfdtype$DT_INT32", frame_name:
//        "while/while_context",} : (tensor<i32>) -> (tensor<*xi32>)
//
// Note: Additional result corresponds to the control output.
class EnterOp
    : public Op<EnterOp, OpTrait::AtLeastNOperands<1>::Impl,
                OpTrait::NResults<2>::Impl, OpTrait::HasNoSideEffect> {
 public:
  using Op::Op;

  static StringRef getOperationName() { return "_tf.Enter"; }

  Value getData() { return getOperand(0); }
  void setData(Value value) { setOperand(0, value); }

  LogicalResult verify();
};

// The "_tf.Merge" operation takes a list of input operands and returns a value
// of the operand type along with the index of the first match encountered.
//
// More details can be found in Tensorflow Controlflow white paper:
// https://storage.googleapis.com/download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf
//
// This is defined in TensorFlow as:
//
// REGISTER_OP("Merge")
//    .Input("inputs: N * T")
//    .Output("output: T")
//    .Output("value_index: int32")
//
// For example:
//   %2 = _tf.Merge %0, %1, %2, %3 : tensor<??xf32>
//
// Note: Additional result corresponds to the control output.
class MergeOp : public Op<MergeOp, OpTrait::VariadicOperands,
                          OpTrait::NResults<3>::Impl> {
 public:
  using Op::Op;

  static StringRef getOperationName() { return "_tf.Merge"; }

  LogicalResult verify();
};

// The "_tf.NextIteration.source" and "_tf.NextIteration.sink" operations form
// a logical pair. Together, they represent NextIteration op in Tensorflow.
//
// Tensorflow NextIteration operation forwards its input to the next iteration
// of a while loop. Each loop variable needs its own NextIteration op.
//
// More details can be found in Tensorflow Controlflow white paper:
// https://storage.googleapis.com/download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf
//
// NextIteration op is broken into _tf.NextIteration.sink and
// _tf.NextIteration.source because NextIteration is a back-edge in Tensorflow
// graph, which would form a data flow cycle if expressed naively in a basic
// block. _tf.NextIteration.source takes no input but returns results while
// _tf.NextIteration.sink takes input but doesn't return anything. When
// optimizing these ops, they are paired by op names and considered as a
// single op.
//
// This is defined in Tensorflow as:
//
// REGISTER_OP("NextIteration")
//    .Input("data: T")
//    .Output("output: T")
//    .Attr("T: type")
//
// For example:
//   %11 = "_tf.NextIteration.source"() {name: "while/NextIteration", T:
//         "tfdtype$DT_INT32", id: 0} : () -> (tensor<*xi32>, _tf.control)
//   "_tf.NextIteration.sink"(%10#0) {name: "while/NextIteration", T:
//         "tfdtype$DT_INT32", id: 0} : (tensor<*xi32>) -> ()
//
// Note: Additional result corresponds to the control output.
class NextIterationSourceOp
    : public Op<NextIterationSourceOp, OpTrait::NResults<2>::Impl> {
 public:
  using Op::Op;

  static StringRef getOperationName() { return "_tf.NextIteration.source"; }

  LogicalResult verify();
};

class NextIterationSinkOp
    : public Op<NextIterationSinkOp, OpTrait::AtLeastNOperands<1>::Impl,
                OpTrait::OneResult> {
 public:
  using Op::Op;

  static StringRef getOperationName() { return "_tf.NextIteration.sink"; }

  Value getData() { return getOperand(0); }
  void setData(Value value) { setOperand(0, value); }

  LogicalResult verify();
};

// The "_tf.LoopCond" operation forwards a boolean value as loop condition of
// Tensorflow while loops.
//
// More details can be found in Tensorflow Controlflow white paper:
// https://storage.googleapis.com/download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf
//
// This is defined in Tensorflow as:
//
// REGISTER_OP("LoopCond")
//    .Input("input: bool")
//    .Output("output: bool")
//
// For example:
//   %5 = "_tf.LoopCond"(%4#0) {device: "", name: "while/LoopCond"} :
//        (tensor<*xi1>) -> (i1, !_tf.control)
//
// Note: Additional result corresponds to the control output.
class LoopCondOp
    : public Op<LoopCondOp, OpTrait::AtLeastNOperands<1>::Impl,
                OpTrait::NResults<2>::Impl, OpTrait::HasNoSideEffect> {
 public:
  using Op::Op;
  static StringRef getOperationName() { return "_tf.LoopCond"; }

  Value getData() { return getOperand(0); }
  void setData(Value value) { setOperand(0, value); }

  LogicalResult verify();
};

// The "_tf.Switch" operation takes a data operand and a boolean predicate
// condition, and returns two values matching the type of the data predicate.
//
// More details can be found in Tensorflow Controlflow white paper:
// https://storage.googleapis.com/download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf
//
// This is defined in TensorFlow as:
//
// REGISTER_OP("Switch")
//    .Input("data: T")
//    .Input("pred: bool")
//    .Output("output_false: T")
//    .Output("output_true: T")
//
// For example:
//   %2 = _tf.Switch %0, %1 : tensor<??xf32>
//
// Note: Additional result corresponds to the control output.
class SwitchOp : public Op<SwitchOp, OpTrait::AtLeastNOperands<2>::Impl,
                           OpTrait::NResults<3>::Impl> {
 public:
  using Op::Op;

  static StringRef getOperationName() { return "_tf.Switch"; }

  Value getData() { return getOperand(0); }
  void setData(Value value) { setOperand(0, value); }

  Value getPredicate() { return getOperand(1); }
  void setPredicate(Value value) { setOperand(1, value); }

  LogicalResult verify();
};

// The "_tf.Exit" operation forwards a value from an while loop to its consumer
// outside of loop. Each returned tensor needs its own _tf.Exit.
//
// More details can be found in Tensorflow Controlflow white paper:
// https://storage.googleapis.com/download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf
//
// This is defined in Tensorflow as:
//
// REGISTER_OP("Exit")
//    .Input("data: T")
//    .Output("output: T")
//    .Attr("T: type")
//
// For example:
//  %1 = "_tf.Exit"(%0#0) {T: "tfdtype$DT_INT32",} : (tensor<*xi32>) ->
//       (tensor<*xi32>, !_tf.control)
//
// Note: Additional result corresponds to the control output.
class ExitOp : public Op<ExitOp, OpTrait::AtLeastNOperands<1>::Impl,
                         OpTrait::NResults<2>::Impl, OpTrait::HasNoSideEffect> {
 public:
  using Op::Op;
  static StringRef getOperationName() { return "_tf.Exit"; }

  Value getData() { return getOperand(0); }
  void setData(Value value) { setOperand(0, value); }

  LogicalResult verify();
};

}  // namespace TFControlFlow
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_CONTROL_FLOW_OPS_H_
