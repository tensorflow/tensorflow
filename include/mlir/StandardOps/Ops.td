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

def Std_Dialect : Dialect {
  let name = "std";
}

// Base class for Standard dialect ops.
class Std_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Std_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter *p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser *parser,
  //                                         OperationState *result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

// Base class for standard cast operations. Requires single operand and result,
// but does not constrain them to specific types.
class CastOp<string mnemonic, list<OpTrait> traits = []> :
    Std_Op<mnemonic, !listconcat(traits, [NoSideEffect])> {

  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Value *source, Type destType", [{
       impl::buildCastOp(builder, result, source, destType);
  }]>];

  let parser = [{
    return impl::parseCastOp(parser, result);
  }];
  let printer = [{
    return printStandardCastOp(this->getOperation(), p);
  }];
  let verifier = [{ return ::verifyCastOp(*this); }];

  let hasFolder = 1;
}

// Base class for standard arithmetic operations.  Requires operands and
// results to be of the same type, but does not constrain them to specific
// types.  Individual classes will have `lhs` and `rhs` accessor to operands.
class ArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Std_Dialect, mnemonic,
       !listconcat(traits, [NoSideEffect, SameValueType])> {

  let results = (outs AnyType);

  let parser = [{
    return impl::parseBinaryOp(parser, result);
  }];

  let printer = [{
    return printStandardBinaryOp(this->getOperation(), p);
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
  let hasConstantFolder = 1;
}

def AddIOp : IntArithmeticOp<"addi", [Commutative]> {
  let summary = "integer addition operation";
  let hasFolder = 1;
  let hasConstantFolder = 1;
}

def AllocOp : Std_Op<"alloc"> {
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

  let extraClassDeclaration = [{
    MemRefType getType() { return getResult()->getType().cast<MemRefType>(); }
  }];

  let hasCanonicalizer = 1;
}

def AndOp : IntArithmeticOp<"and", [Commutative]> {
  let summary = "integer binary and";
  let hasConstantFolder = 1;
  let hasFolder = 1;
}

def BranchOp : Std_Op<"br", [Terminator]> {
  let summary = "branch operation";
  let description = [{
    The "br" operation represents a branch operation in a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types for each successor must match the arguments of
    the block successor. For example:

      ^bb2:
        %2 = call @someFn()
        br ^bb3(%2 : tensor<*xf32>)
      ^bb3(%3: tensor<*xf32>):
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<
    "Builder *, OperationState *result, Block *dest,"
    "ArrayRef<Value *> operands = {}", [{
      result->addSuccessor(dest, operands);
  }]>];

  // BranchOp is fully verified by traits.
  let verifier = ?;

  let extraClassDeclaration = [{
    Block *getDest();
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];
}

def CallOp : Std_Op<"call"> {
  let summary = "call operation";
  let description = [{
    The "call" operation represents a direct call to a function.  The operands
    and result types of the call must match the specified function type.  The
    callee is encoded as a function attribute named "callee".

      %2 = call @my_add(%0, %1) : (f32, f32) -> f32
  }];

  let arguments = (ins FunctionAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Function *callee,"
    "ArrayRef<Value *> operands = {}", [{
      result->addOperands(operands);
      result->addAttribute("callee", builder->getFunctionAttr(callee));
      result->addTypes(callee->getType().getResults());
  }]>];

  let extraClassDeclaration = [{
    Function *getCallee() {
      return getAttrOfType<FunctionAttr>("callee").getValue();
    }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];
}

def CallIndirectOp : Std_Op<"call_indirect"> {
  let summary = "indirect call operation";
  let description = [{
    The "call_indirect" operation represents an indirect call to a value of
    function type.  Functions are first class types in MLIR, and may be passed
    as arguments and merged together with block arguments.  The operands
    and result types of the call must match the specified function type.

      %3 = call_indirect %2(%0, %1) : (f32, f32) -> f32
  }];

  let arguments = (ins FunctionType:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [OpBuilder<
    "Builder *, OperationState *result, Value *callee,"
    "ArrayRef<Value *> operands = {}", [{
      result->operands.push_back(callee);
      result->addOperands(operands);
      result->addTypes(callee->getType().cast<FunctionType>().getResults());
  }]>];

  let extraClassDeclaration = [{
    Value *getCallee() { return getOperand(0); }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];

  let hasCanonicalizer = 1;
}

def ConstantOp : Std_Op<"constant", [NoSideEffect]> {
  let summary = "constant";

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Attribute value",
    [{ build(builder, result, value.getType(), value); }]>];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }
  }];

  let hasConstantFolder = 1;
}

def DeallocOp : Std_Op<"dealloc"> {
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

  let hasCanonicalizer = 1;
}

def DimOp : Std_Op<"dim", [NoSideEffect]> {
  let summary = "dimension index operation";
  let description = [{
    The "dim" operation takes a memref or tensor operand and returns an "index".
    It requires a single integer attribute named "index". It returns the size
    of the specified dimension. For example:

      %1 = dim %0, 2 : tensor<?x?x?xf32>
  }];

  let arguments = (ins AnyTypeOf<[MemRef<AnyType>, Tensor],
                                 "any tensor or memref type">:$memrefOrTensor,
                       APIntAttr:$index);
  let results = (outs Index);

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Value *memrefOrTensor,"
    "unsigned index", [{
      auto indexType = builder->getIndexType();
      auto indexAttr = builder->getIntegerAttr(indexType, index);
      build(builder, result, indexType, memrefOrTensor, indexAttr);
    }]>];

  let extraClassDeclaration = [{
    unsigned getIndex() {
      return getAttrOfType<IntegerAttr>("index").getValue().getZExtValue();
    }
  }];

  let hasConstantFolder = 1;
}

def DivFOp : FloatArithmeticOp<"divf"> {
  let summary = "floating point division operation";
}

def DivISOp : IntArithmeticOp<"divis"> {
  let summary = "signed integer division operation";
  let hasConstantFolder = 1;
}

def DivIUOp : IntArithmeticOp<"diviu"> {
  let summary = "unsigned integer division operation";
  let hasConstantFolder = 1;
}

def ExtractElementOp : Std_Op<"extract_element", [NoSideEffect]> {
  let summary = "element extract operation";
  let description = [{
    The "extract_element" op reads a tensor or vector and returns one element
    from it specified by an index list. The output of extract is a new value
    with the same type as the elements of the tensor or vector. The arity of
    indices matches the rank of the accessed value (i.e., if a tensor is of rank
    3, then 3 indices are required for the extract).  The indices should all be
    of affine_int type. For example:

      %0 = extract_element %0[%1, %2] : vector<4x4xi32>
  }];

  let arguments = (ins VectorOrTensor:$aggregate,
                       Variadic<Index>:$indices);
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Value *aggregate,"
    "ArrayRef<Value *> indices = {}", [{
      auto resType = aggregate->getType().cast<VectorOrTensorType>()
                                         .getElementType();
      build(builder, result, resType, aggregate, indices);
    }]>];

  let extraClassDeclaration = [{
    Value *getAggregate() { return getOperand(0); }

    operand_range getIndices() {
      return {getOperation()->operand_begin() + 1,
              getOperation()->operand_end()};
    }
  }];

  let hasConstantFolder = 1;
}

def MemRefCastOp : CastOp<"memref_cast"> {
  let summary = "memref cast operation";
  let description = [{
    The "memref_cast" operation converts a memref from one type to an equivalent
    type with a compatible shape. The source and destination types are
    when both are memref types with the same element type, affine mappings,
    address space, and rank but where the individual dimensions may add or
    remove constant dimensions from the memref type.

    If the cast converts any dimensions from an unknown to a known size, then it
    acts as an assertion that fails at runtime of the dynamic dimensions
    disagree with resultant destination size.

    Assert that the input dynamic shape matches the destination static shape.
       %2 = memref_cast %1 : memref<?x?xf32> to memref<4x4xf32>
    Erase static shape information, replacing it with dynamic information.
       %3 = memref_cast %1 : memref<4xf32> to memref<?xf32>
  }];

  let arguments = (ins MemRef<AnyType>);
  let results = (outs MemRef<AnyType>);

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);

    /// The result of a memref_cast is always a memref.
    MemRefType getType() { return getResult()->getType().cast<MemRefType>(); }
  }];
}

def MulFOp : FloatArithmeticOp<"mulf"> {
  let summary = "foating point multiplication operation";
  let hasConstantFolder = 1;
}

def MulIOp : IntArithmeticOp<"muli", [Commutative]> {
  let summary = "integer multiplication operation";
  let hasConstantFolder = 1;
  let hasFolder = 1;
}

def OrOp : IntArithmeticOp<"or", [Commutative]> {
  let summary = "integer binary or";
  let hasConstantFolder = 1;
  let hasFolder = 1;
}

def RemFOp : FloatArithmeticOp<"remf"> {
  let summary = "floating point division remainder operation";
}

def RemISOp : IntArithmeticOp<"remis"> {
  let summary = "signed integer division remainder operation";
  let hasConstantFolder = 1;
}

def RemIUOp : IntArithmeticOp<"remiu"> {
  let summary = "unsigned integer division remainder operation";
  let hasConstantFolder = 1;
}

def ReturnOp : Std_Op<"return", [Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation. For example:

      func @foo() : (i32, f8) {
      ...
      return %0, %1 : i32, f8
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<
    "Builder *b, OperationState *result", [{ build(b, result, llvm::None); }]
  >];
}

def ShlISOp : IntArithmeticOp<"shlis"> {
  let summary = "signed integer shift left";
}

def SubFOp : FloatArithmeticOp<"subf"> {
  let summary = "floating point subtraction operation";
  let hasConstantFolder = 1;
}

def SubIOp : IntArithmeticOp<"subi"> {
  let summary = "integer subtraction operation";
  let hasConstantFolder = 1;
  let hasCanonicalizer = 1;
}

def TensorCastOp : CastOp<"tensor_cast"> {
  let summary = "tensor cast operation";
  let description = [{
    The "tensor_cast" operation converts a tensor from one type to an equivalent
    type without changing any data elements.  The source and destination types
    must both be tensor types with the same element type.  If both are ranked
    then the rank should be the same and static dimensions should match.  The
    operation is invalid if converting to a mismatching constant dimension.

    Convert from unknown rank to rank 2 with unknown dimension sizes.
       %2 = tensor_cast %1 : tensor<??f32> to tensor<?x?xf32>
  }];

  let arguments = (ins Tensor);
  let results = (outs Tensor);

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);

    /// The result of a tensor_cast is always a tensor.
    TensorType getType() { return getResult()->getType().cast<TensorType>(); }
  }];
}

def XOrOp : IntArithmeticOp<"xor", [Commutative]> {
  let summary = "integer binary xor";
  let hasConstantFolder = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

#endif // STANDARD_OPS
