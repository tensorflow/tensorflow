//===- standard_ops.td - Standard operation definitions ----*- tablegen -*-===//
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
// Defines some MLIR standard operations.
//
//===----------------------------------------------------------------------===//

#ifdef STANDARD_OPS
#else
#define STANDARD_OPS

#ifdef OP_BASE
#else
include "mlir/IR/op_base.td"
#endif // OP_BASE

def AnyType : Type;

// Base class for standard arithmetic operations.  Requires operands and
// results to be of the same type, but does not constrain them to specific
// types.  Individual classes will have `lhs` and `rhs` accessor to operands.
class ArithmeticOp<string mnemonic, list<OpProperty> props = [],
                   list<string> traits = []> :
    Op<mnemonic, !listconcat(props, [NoSideEffect])>,
    Traits<!listconcat(traits, ["SameOperandsAndResultType"])>,
    Results<[AnyType]> {

  let opName = mnemonic;

  let builder = [{
    static void build(Builder *builder, OperationState *result, Value *lhs,
                      Value *rhs) {
      impl::buildBinaryOp(builder, result, lhs, rhs);
    }
  }];

  let parser = [{
    return impl::parseBinaryOp(parser, result);
  }];

  let printer = [{
    return impl::printBinaryOp(this->getInstruction(), p);
  }];
}

// Base class for standard arithmetic operations on integers, vectors and
// tensors thereof.  This operation takes two operands and returns one result,
// each of these is required to be of the same type.  This type may be an
// integer scalar type, a vector whose element type is an integer type, or an
// integer tensor.  The short-hand syntax of the operaton is as follows
//
//     <op>i %0, %1 : i32
class IntArithmeticOp<string mnemonic, list<OpProperty> props = [],
                      list<string> traits = []> :
    ArithmeticOp<mnemonic, props, traits>,
    Arguments<(ins IntegerLike:$lhs, IntegerLike:$rhs)>;

// Base class for standard arithmetic binary operations on floats, vectors and
// tensors thereof.  This operation has two operands and returns one result,
// each of these is required to be of the same type.  This type may be a
// floating point scalar type, a vector whose element type is a floating point
// type, or a floating point tensor.  The short-hand syntax of the operation is
// as follows
//
//     <op>f %0, %1 : f32
class FloatArithmeticOp<string mnemonic, list<OpProperty> props = [],
                        list<string> traits = []> :
    ArithmeticOp<mnemonic, props, traits>,
    Arguments<(ins FloatLike:$lhs, FloatLike:$rhs)>;

def AddFOp : FloatArithmeticOp<"addf"> {
  let summary = "floating point addition operation";
  let hasConstantFolder = 0b1;
}

def AddIOp : IntArithmeticOp<"addi", [Commutative]> {
  let summary = "integer addition operation";
  let hasCanonicalizationPatterns = 0b1;
  let hasConstantFolder = 0b1;
}

def DivISOp : IntArithmeticOp<"divis"> {
  let summary = "signed integer division operation";
  let hasConstantFolder = 0b1;
}

def DivIUOp : IntArithmeticOp<"diviu"> {
  let summary = "unsigned integer division operation";
  let hasConstantFolder = 0b1;
}

def MulFOp : FloatArithmeticOp<"mulf"> {
  let summary = "foating point multiplication operation";
  let hasConstantFolder = 0b1;
}

def MulIOp : IntArithmeticOp<"muli", [Commutative]> {
  let summary = "integer multiplication operation";
  let hasCanonicalizationPatterns = 0b1;
  let hasConstantFolder = 0b1;
}

def RemISOp : IntArithmeticOp<"remis"> {
  let summary = "signed integer division remainder operation";
  let hasConstantFolder = 0b1;
}

def RemIUOp : IntArithmeticOp<"remiu"> {
  let summary = "unsigned integer division remainder operation";
  let hasConstantFolder = 0b1;
}

def SubFOp : FloatArithmeticOp<"subf"> {
  let summary = "floating point subtraction operation";
  let hasConstantFolder = 0b1;
}

def SubIOp : IntArithmeticOp<"subi"> {
  let summary = "integer subtraction operation";
  let hasConstantFolder = 0b1;
  let hasCanonicalizationPatterns = 0b1;
}

#endif // STANDARD_OPS
