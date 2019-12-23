//===- Ops.td - Loop operation definitions ---------------*- tablegen -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Defines MLIR loop operations.
//
//===----------------------------------------------------------------------===//

#ifndef LOOP_OPS
#define LOOP_OPS

include "mlir/IR/OpBase.td"
include "mlir/Transforms/LoopLikeInterface.td"

def Loop_Dialect : Dialect {
  let name = "loop";
  let cppNamespace = "";
}

// Base class for Loop dialect ops.
class Loop_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Loop_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def ForOp : Loop_Op<"for",
      [DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let summary = "for operation";
  let description = [{
    The "loop.for" operation represents a loop nest taking 3 SSA value as
    operands that represent the lower bound, upper bound and step respectively.
    The operation defines an SSA value for its induction variable. It has one
    region capturing the loop body. The induction variable is represented as an
    argument of this region. This SSA value always has type index, which is the
    size of the machine word. The step is a value of type index, required to be
    positive.
    The lower and upper bounds specify a half-open range: the range includes the
    lower bound but does not include the upper bound.

    The body region must contain exactly one block that terminates with
    "loop.terminator".  Calling ForOp::build will create such region and insert
    the terminator, so will the parsing even in cases when it is absent from the
    custom format. For example:

       loop.for %iv = %lb to %ub step %step {
         ... // body
       }
  }];
  let arguments = (ins Index:$lowerBound, Index:$upperBound, Index:$step);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "Value lowerBound, Value upperBound, Value step">
  ];

  let extraClassDeclaration = [{
    Block *getBody() { return &region().front(); }
    Value getInductionVar() { return getBody()->getArgument(0); }
    OpBuilder getBodyBuilder() {
      return OpBuilder(getBody(), std::prev(getBody()->end()));
    }
    void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
    void setStep(Value step) { getOperation()->setOperand(2, step); }
  }];
}

def IfOp : Loop_Op<"if",
      [SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let summary = "if-then-else operation";
  let description = [{
    The "loop.if" operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. The operation produces no results. For example:

       loop.if %b  {
         ...
       } else {
         ...
       }

    The 'else' block is optional, and may be omitted. For
    example:

       loop.if %b  {
         ...
       }
  }];
  let arguments = (ins I1:$condition);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "Value cond, bool withElseRegion">
  ];

  let extraClassDeclaration = [{
    OpBuilder getThenBodyBuilder() {
      assert(!thenRegion().empty() && "Unexpected empty 'then' region.");
      Block &body = thenRegion().front();
      return OpBuilder(&body, std::prev(body.end()));
    }
    OpBuilder getElseBodyBuilder() {
      assert(!elseRegion().empty() && "Unexpected empty 'else' region.");
      Block &body = elseRegion().front();
      return OpBuilder(&body, std::prev(body.end()));
    }
  }];
}

def TerminatorOp :
    Loop_Op<"terminator", [NativeOpTrait<"IsTerminator">]> {
  let summary = "cf terminator operation";
  let description = [{
    "loop.terminator" is a special terminator operation for blocks inside
    loops. It terminates the region. This operation does _not_ have a custom
    syntax. However, `std` control operations omit the terminator in their
    custom syntax for brevity.

       loop.terminator
  }];

  // No custom parsing/printing form.
  let parser = ?;
  let printer = ?;

  // Fully specified by traits.
  let verifier = ?;
}

#endif // LOOP_OPS
