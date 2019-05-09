//===- Ops.td - Standard operation definitions -------------*- tablegen -*-===//
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
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def Standard_Dialect : Dialect {
  let name = "std";
}

// Base class for standard arithmetic operations.  Requires operands and
// results to be of the same type, but does not constrain them to specific
// types.  Individual classes will have `lhs` and `rhs` accessor to operands.
class ArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Standard_Dialect, mnemonic,
       !listconcat(traits, [NoSideEffect, SameValueType])> {

  let results = (outs AnyType);

  let parser = [{
    return impl::parseBinaryOp(parser, result);
  }];

  let printer = [{
    return detail::printStandardBinaryOp(this->getOperation(), p);
  }];
}

// Base class for standard arithmetic operations on integers, vectors and
// tensors thereof.  This operation takes two operands and returns one result,
// each of these is required to be of the same type.  This type may be an
// integer scalar type, a vector whose element type is an integer type, or an
// integer tensor.  The custom assembly form of the operaton is as follows
//
//     <op>i %0, %1 : i32
class IntArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    ArithmeticOp<mnemonic, traits>,
    Arguments<(ins IntegerLike:$lhs, IntegerLike:$rhs)>;

// Base class for standard arithmetic binary operations on floats, vectors and
// tensors thereof.  This operation has two operands and returns one result,
// each of these is required to be of the same type.  This type may be a
// floating point scalar type, a vector whose element type is a floating point
// type, or a floating point tensor.  The custom assembly form of the operation
// is as follows
//
//     <op>f %0, %1 : f32
class FloatArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    ArithmeticOp<mnemonic, traits>,
    Arguments<(ins FloatLike:$lhs, FloatLike:$rhs)>;

def AddFOp : FloatArithmeticOp<"addf"> {
  let summary = "floating point addition operation";
  let hasConstantFolder = 0b1;
}

def AddIOp : IntArithmeticOp<"addi", [Commutative]> {
  let summary = "integer addition operation";
  let hasFolder = 1;
  let hasConstantFolder = 0b1;
}

def AllocOp : Op<Standard_Dialect, "alloc"> {
  let summary = "memory allocation operation";
  let description = [{
    The "alloc" operation allocates a region of memory, as specified by its
    memref type. For example:

      %0 = alloc() : memref<8x64xf32, (d0, d1) -> (d0, d1), 1>

    The optional list of dimension operands are bound to the dynamic dimensions
    specified in its memref type. In the example below, the ssa value '%d' is
    bound to the second dimension of the memref (which is dynamic).

      %0 = alloc(%d) : memref<8x?xf32, (d0, d1) -> (d0, d1), 1>

    The optional list of symbol operands are bound to the symbols of the
    memrefs affine map. In the example below, the ssa value '%s' is bound to
    the symbol 's0' in the affine map specified in the allocs memref type.

      %0 = alloc()[%s] : memref<8x64xf32, (d0, d1)[s0] -> ((d0 + s0), d1), 1>

    This operation returns a single ssa value of memref type, which can be used
    by subsequent load and store operations.
  }];

  let arguments = (ins Variadic<Index>:$value);
  let results = (outs MemRef<AnyType>);

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, MemRefType memrefType", [{
       result->types.push_back(memrefType);
     }]
  >];

  let parser = [{ return parseAllocOp(parser, result); }];
  let printer = [{ return printAllocOp(p, *this); }];
  let verifier = [{ return ::verify(*this); }];

  let extraClassDeclaration = [{
    MemRefType getType() { return getResult()->getType().cast<MemRefType>(); }
  }];

  let hasCanonicalizer = 0b1;
}

def AndOp : IntArithmeticOp<"and", [Commutative]> {
  let summary = "integer binary and";
  let hasConstantFolder = 0b1;
  let hasFolder = 1;
}

def ConstantOp : Op<Standard_Dialect, "constant", [NoSideEffect]> {
  let summary = "constant";

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Attribute value",
    [{ build(builder, result, value.getType(), value); }]>];

  let parser = [{ return parseConstantOp(parser, result); }];
  let printer = [{ return printConstantOp(p, *this); }];
  let verifier = [{ return ::verify(*this); }];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }
  }];

  let hasConstantFolder = 1;
}

def DeallocOp : Op<Standard_Dialect, "dealloc"> {
  let summary = "memory deallocation operation";
  let description = [{
    The "dealloc" operation frees the region of memory referenced by a memref
    which was originally created by the "alloc" operation.
    The "dealloc" operation should not be called on memrefs which alias an
    alloc'd memref (i.e. memrefs returned by the "view" and "reshape"
    operations).

      %0 = alloc() : memref<8x64xf32, (d0, d1) -> (d0, d1), 1>
      dealloc %0 : memref<8x64xf32, (d0, d1) -> (d0, d1), 1>
  }];

  let arguments = (ins MemRef<AnyType>:$memref);

  let parser = [{ return parseDeallocOp(parser, result); }];
  let printer = [{ return printDeallocOp(p, *this); }];
  let verifier = [{ return ::verify(*this); }];

  let hasCanonicalizer = 0b1;
}

def DivFOp : FloatArithmeticOp<"divf"> {
  let summary = "floating point division operation";
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
  let hasConstantFolder = 0b1;
  let hasFolder = 1;
}

def OrOp : IntArithmeticOp<"or", [Commutative]> {
  let summary = "integer binary or";
  let hasConstantFolder = 0b1;
  let hasFolder = 1;
}

def RemFOp : FloatArithmeticOp<"remf"> {
  let summary = "floating point division remainder operation";
}

def RemISOp : IntArithmeticOp<"remis"> {
  let summary = "signed integer division remainder operation";
  let hasConstantFolder = 0b1;
}

def RemIUOp : IntArithmeticOp<"remiu"> {
  let summary = "unsigned integer division remainder operation";
  let hasConstantFolder = 0b1;
}

def ShlISOp : IntArithmeticOp<"shlis"> {
  let summary = "signed integer shift left";
}

def SubFOp : FloatArithmeticOp<"subf"> {
  let summary = "floating point subtraction operation";
  let hasConstantFolder = 0b1;
}

def SubIOp : IntArithmeticOp<"subi"> {
  let summary = "integer subtraction operation";
  let hasConstantFolder = 0b1;
  let hasCanonicalizer = 0b1;
}

def XOrOp : IntArithmeticOp<"xor", [Commutative]> {
  let summary = "integer binary xor";
  let hasConstantFolder = 0b1;
  let hasCanonicalizer = 0b1;
  let hasFolder = 1;
}

#endif // STANDARD_OPS
