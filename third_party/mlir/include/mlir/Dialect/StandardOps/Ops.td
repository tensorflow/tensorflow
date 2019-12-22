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

#ifndef STANDARD_OPS
#define STANDARD_OPS

include "mlir/Analysis/CallInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

def Std_Dialect : Dialect {
  let name = "std";
  let cppNamespace = "";
}

// Base class for Standard dialect ops.
class Std_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Std_Dialect, mnemonic, traits> {
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

// Base class for standard cast operations. Requires single operand and result,
// but does not constrain them to specific types.
class CastOp<string mnemonic, list<OpTrait> traits = []> :
    Std_Op<mnemonic, !listconcat(traits, [NoSideEffect])> {

  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *source, Type destType", [{
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

// Base class for unary ops. Requires single operand and result. Individual
// classes will have `operand` accessor.
class UnaryOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Std_Dialect, mnemonic, !listconcat(traits, [NoSideEffect])> {
  let results = (outs AnyType);
  let printer = [{
    return printStandardUnaryOp(this->getOperation(), p);
  }];
}

class UnaryOpSameOperandAndResultType<string mnemonic,
                                      list<OpTrait> traits = []> :
    UnaryOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])> {
  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];
}

class FloatUnaryOp<string mnemonic, list<OpTrait> traits = []> :
    UnaryOpSameOperandAndResultType<mnemonic, traits>,
    Arguments<(ins FloatLike:$operand)>;

// Base class for standard arithmetic operations.  Requires operands and
// results to be of the same type, but does not constrain them to specific
// types.  Individual classes will have `lhs` and `rhs` accessor to operands.
class ArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Std_Dialect, mnemonic,
       !listconcat(traits, [NoSideEffect, SameOperandsAndResultType])> {

  let results = (outs AnyType);

  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];

  let printer = [{
    return printStandardBinaryOp(this->getOperation(), p);
  }];
}

// Base class for standard arithmetic operations on integers, vectors and
// tensors thereof.  This operation takes two operands and returns one result,
// each of these is required to be of the same type.  This type may be an
// integer scalar type, a vector whose element type is an integer type, or an
// integer tensor.  The custom assembly form of the operation is as follows
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

def AbsFOp : FloatUnaryOp<"absf"> {
  let summary = "floating point absolute-value operation";
  let description = [{
    The `absf` operation computes the absolute value. It takes one operand and
    returns one result of the same type. This type may be a float scalar type,
    a vector whose element type is float, or a tensor of floats. It has no
    standard attributes.
  }];
}

def AddFOp : FloatArithmeticOp<"addf"> {
  let summary = "floating point addition operation";
  let hasFolder = 1;
}

def AddIOp : IntArithmeticOp<"addi", [Commutative]> {
  let summary = "integer addition operation";
  let hasFolder = 1;
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

    The optional `alignment` attribute may be specified to ensure that the
    region of memory that will be indexed is aligned at the specified byte
    boundary. TODO(b/144281289) optional alignment attribute to MemRefType.

      %0 = alloc()[%s] {alignment = 8} :
        memref<8x64xf32, (d0, d1)[s0] -> ((d0 + s0), d1), 1>
  }];

  let arguments = (ins Variadic<Index>:$value,
                   Confined<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$alignment);
  let results = (outs AnyMemRef);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, MemRefType memrefType", [{
       result.types.push_back(memrefType);
     }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, MemRefType memrefType, " #
    "ArrayRef<Value*> operands, IntegerAttr alignment = IntegerAttr()", [{
       result.addOperands(operands);
       result.types.push_back(memrefType);
       if (alignment)
         result.addAttribute(getAlignmentAttrName(), alignment);
     }]>];

  let extraClassDeclaration = [{
    static StringRef getAlignmentAttrName() { return "alignment"; }

    MemRefType getType() { return getResult()->getType().cast<MemRefType>(); }

    /// Returns the number of symbolic operands (the ones in square brackets),
    /// which bind to the symbols of the memref's layout map.
    unsigned getNumSymbolicOperands() {
      return getNumOperands() - getType().getNumDynamicDims();
    }

    /// Returns the symbolic operands (the ones in square brackets), which bind
    /// to the symbols of the memref's layout map.
    operand_range getSymbolicOperands() {
      return {operand_begin() + getType().getNumDynamicDims(), operand_end()};
    }

    /// Returns the dynamic sizes for this alloc operation if specified.
    operand_range getDynamicSizes() { return getOperands(); }
  }];

  let hasCanonicalizer = 1;
}

def AndOp : IntArithmeticOp<"and", [Commutative]> {
  let summary = "integer binary and";
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
    "Builder *, OperationState &result, Block *dest,"
    "ValueRange operands = {}", [{
      result.addSuccessor(dest, operands);
  }]>];

  // BranchOp is fully verified by traits.
  let verifier = ?;

  let extraClassDeclaration = [{
    Block *getDest();
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];

  let hasCanonicalizer = 1;
}

def CallOp : Std_Op<"call", [CallOpInterface]> {
  let summary = "call operation";
  let description = [{
    The "call" operation represents a direct call to a function that is within
    the same symbol scope as the call.  The operands and result types of the
    call must match the specified function type. The callee is encoded as a
    function attribute named "callee".

      %2 = call @my_add(%0, %1) : (f32, f32) -> f32
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, FuncOp callee,"
    "ValueRange operands = {}", [{
      result.addOperands(operands);
      result.addAttribute("callee", builder->getSymbolRefAttr(callee));
      result.addTypes(callee.getType().getResults());
  }]>, OpBuilder<
    "Builder *builder, OperationState &result, SymbolRefAttr callee,"
    "ArrayRef<Type> results, ValueRange operands = {}", [{
      result.addOperands(operands);
      result.addAttribute("callee", callee);
      result.addTypes(results);
  }]>, OpBuilder<
    "Builder *builder, OperationState &result, StringRef callee,"
    "ArrayRef<Type> results, ValueRange operands = {}", [{
      build(builder, result, builder->getSymbolRefAttr(callee), results,
            operands);
  }]>];

  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getAttrOfType<SymbolRefAttr>("callee");
    }
  }];
}

def CallIndirectOp : Std_Op<"call_indirect", [CallOpInterface]> {
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
    "Builder *, OperationState &result, Value *callee,"
    "ValueRange operands = {}", [{
      result.operands.push_back(callee);
      result.addOperands(operands);
      result.addTypes(callee->getType().cast<FunctionType>().getResults());
  }]>];

  let extraClassDeclaration = [{
    Value *getCallee() { return getOperand(0); }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }
  }];

  let hasCanonicalizer = 1;
}

def CeilFOp : FloatUnaryOp<"ceilf"> {
  let summary = "ceiling of the specified value";
  let description = [{
    The `ceilf` operation computes the ceiling of a given value. It takes one
    operand and returns one result of the same type. This type may be a float
    scalar type, a vector whose element type is float, or a tensor of floats.
    It has no standard attributes.
  }];
}

def CmpFOp : Std_Op<"cmpf",
    [NoSideEffect, SameTypeOperands, SameOperandsAndResultShape]> {
  let summary = "floating-point comparison operation";
  let description = [{
    The "cmpf" operation compares its two operands according to the float
    comparison rules and the predicate specified by the respective attribute.
    The predicate defines the type of comparison: (un)orderedness, (in)equality
    and signed less/greater than (or equal to) as well as predicates that are
    always true or false.  The operands must have the same type, and this type
    must be a float type, or a vector or tensor thereof.  The result is an i1,
    or a vector/tensor thereof having the same shape as the inputs. Unlike cmpi,
    the operands are always treated as signed. The u prefix indicates
    *unordered* comparison, not unsigned comparison, so "une" means unordered or
    not equal. For the sake of readability by humans, custom assembly form for
    the operation uses a string-typed attribute for the predicate.  The value of
    this attribute corresponds to lower-cased name of the predicate constant,
    e.g., "one" means "ordered not equal".  The string representation of the
    attribute is merely a syntactic sugar and is converted to an integer
    attribute by the parser.

      %r1 = cmpf "oeq" %0, %1 : f32
      %r2 = cmpf "ult" %0, %1 : tensor<42x42xf64>
      %r3 = "std.cmpf"(%0, %1) {predicate: 0} : (f8, f8) -> i1
  }];

  let arguments = (ins FloatLike:$lhs, FloatLike:$rhs);
  let results = (outs BoolLike);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, CmpFPredicate predicate,"
    "Value *lhs, Value *rhs", [{
      ::buildCmpFOp(builder, result, predicate, lhs, rhs);
  }]>];

  let extraClassDeclaration = [{
    static StringRef getPredicateAttrName() { return "predicate"; }
    static CmpFPredicate getPredicateByName(StringRef name);

    CmpFPredicate getPredicate() {
      return (CmpFPredicate)getAttrOfType<IntegerAttr>(getPredicateAttrName())
          .getInt();
    }
  }];

  let hasFolder = 1;
}

def CMPI_P_EQ  : I64EnumAttrCase<"eq", 0>;
def CMPI_P_NE  : I64EnumAttrCase<"ne", 1>;
def CMPI_P_SLT : I64EnumAttrCase<"slt", 2>;
def CMPI_P_SLE : I64EnumAttrCase<"sle", 3>;
def CMPI_P_SGT : I64EnumAttrCase<"sgt", 4>;
def CMPI_P_SGE : I64EnumAttrCase<"sge", 5>;
def CMPI_P_ULT : I64EnumAttrCase<"ult", 6>;
def CMPI_P_ULE : I64EnumAttrCase<"ule", 7>;
def CMPI_P_UGT : I64EnumAttrCase<"ugt", 8>;
def CMPI_P_UGE : I64EnumAttrCase<"uge", 9>;

def CmpIPredicateAttr : I64EnumAttr<
    "CmpIPredicate", "",
    [CMPI_P_EQ, CMPI_P_NE, CMPI_P_SLT, CMPI_P_SLE, CMPI_P_SGT,
     CMPI_P_SGE, CMPI_P_ULT, CMPI_P_ULE, CMPI_P_UGT, CMPI_P_UGE]> {
  let cppNamespace = "::mlir";
}

def CmpIOp : Std_Op<"cmpi",
    [NoSideEffect, SameTypeOperands, SameOperandsAndResultShape]> {
  let summary = "integer comparison operation";
  let description = [{
    The "cmpi" operation compares its two operands according to the integer
    comparison rules and the predicate specified by the respective attribute.
    The predicate defines the type of comparison: (in)equality, (un)signed
    less/greater than (or equal to).  The operands must have the same type, and
    this type must be an integer type, a vector or a tensor thereof.  The result
    is an i1, or a vector/tensor thereof having the same shape as the inputs.
    Since integers are signless, the predicate also explicitly indicates
    whether to interpret the operands as signed or unsigned integers for
    less/greater than comparisons.  For the sake of readability by humans,
    custom assembly form for the operation uses a string-typed attribute for
    the predicate.  The value of this attribute corresponds to lower-cased name
    of the predicate constant, e.g., "slt" means "signed less than".  The string
    representation of the attribute is merely a syntactic sugar and is converted
    to an integer attribute by the parser.

      %r1 = cmpi "eq" %0, %1 : i32
      %r2 = cmpi "slt" %0, %1 : tensor<42x42xi64>
      %r3 = "std.cmpi"(%0, %1){predicate: 0} : (i8, i8) -> i1
  }];

  let arguments = (ins
      CmpIPredicateAttr:$predicate,
      IntegerLike:$lhs,
      IntegerLike:$rhs
  );
  let results = (outs BoolLike);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, CmpIPredicate predicate,"
    "Value *lhs, Value *rhs", [{
      ::buildCmpIOp(builder, result, predicate, lhs, rhs);
  }]>];

  let extraClassDeclaration = [{
    static StringRef getPredicateAttrName() { return "predicate"; }
    static CmpIPredicate getPredicateByName(StringRef name);

    CmpIPredicate getPredicate() {
      return (CmpIPredicate)getAttrOfType<IntegerAttr>(getPredicateAttrName())
          .getInt();
    }
  }];

  let verifier = [{ return success(); }];

  let hasFolder = 1;
}

def CondBranchOp : Std_Op<"cond_br", [Terminator]> {
  let summary = "conditional branch operation";
  let description = [{
    The "cond_br" operation represents a conditional branch operation in a
    function. The operation takes variable number of operands and produces
    no results. The operand number and types for each successor must match the
    arguments of the block successor. For example:

      ^bb0:
         %0 = extract_element %arg0[] : tensor<i1>
         cond_br %0, ^bb1, ^bb2
      ^bb1:
         ...
      ^bb2:
         ...
  }];

  let arguments = (ins I1:$condition, Variadic<AnyType>:$branchOperands);

  let builders = [OpBuilder<
    "Builder *, OperationState &result, Value *condition,"
    "Block *trueDest, ValueRange trueOperands,"
    "Block *falseDest, ValueRange falseOperands", [{
      result.addOperands(condition);
      result.addSuccessor(trueDest, trueOperands);
      result.addSuccessor(falseDest, falseOperands);
  }]>];

  // CondBranchOp is fully verified by traits.
  let verifier = ?;

  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };

    // The condition operand is the first operand in the list.
    Value *getCondition() { return getOperand(0); }

    /// Return the destination if the condition is true.
    Block *getTrueDest() {
      return getSuccessor(trueIndex);
    }

    /// Return the destination if the condition is false.
    Block *getFalseDest() {
      return getSuccessor(falseIndex);
    }

    // Accessors for operands to the 'true' destination.
    Value *getTrueOperand(unsigned idx) {
      assert(idx < getNumTrueOperands());
      return getOperand(getTrueDestOperandIndex() + idx);
    }

    void setTrueOperand(unsigned idx, Value *value) {
      assert(idx < getNumTrueOperands());
      setOperand(getTrueDestOperandIndex() + idx, value);
    }

    operand_iterator true_operand_begin() {
      return operand_begin() + getTrueDestOperandIndex();
    }
    operand_iterator true_operand_end() {
      return true_operand_begin() + getNumTrueOperands();
    }
    operand_range getTrueOperands() {
      return {true_operand_begin(), true_operand_end()};
    }

    unsigned getNumTrueOperands()  {
      return getNumSuccessorOperands(trueIndex);
    }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index)  {
      getOperation()->eraseSuccessorOperand(trueIndex, index);
    }

    // Accessors for operands to the 'false' destination.
    Value *getFalseOperand(unsigned idx) {
      assert(idx < getNumFalseOperands());
      return getOperand(getFalseDestOperandIndex() + idx);
    }
    void setFalseOperand(unsigned idx, Value *value) {
      assert(idx < getNumFalseOperands());
      setOperand(getFalseDestOperandIndex() + idx, value);
    }

    operand_iterator false_operand_begin() { return true_operand_end(); }
    operand_iterator false_operand_end() {
      return false_operand_begin() + getNumFalseOperands();
    }
    operand_range getFalseOperands() {
      return {false_operand_begin(), false_operand_end()};
    }

    unsigned getNumFalseOperands() {
      return getNumSuccessorOperands(falseIndex);
    }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      getOperation()->eraseSuccessorOperand(falseIndex, index);
    }

  private:
    /// Get the index of the first true destination operand.
    unsigned getTrueDestOperandIndex() { return 1; }

    /// Get the index of the first false destination operand.
    unsigned getFalseDestOperandIndex() {
      return getTrueDestOperandIndex() + getNumTrueOperands();
    }
  }];

  let hasCanonicalizer = 1;
}

def ConstantOp : Std_Op<"constant",
    [NoSideEffect, DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
  let summary = "constant";

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Attribute value",
    [{ build(builder, result, value.getType(), value); }]>];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }

    /// Returns true if a constant operation can be built with the given value
    /// and result type.
    static bool isBuildableWith(Attribute value, Type type);
  }];

  let hasFolder = 1;
}

def CopySignOp : FloatArithmeticOp<"copysign"> {
  let summary = "A copysign operation";
  let description = [{
    The `copysign` returns a value with the magnitude of the first operand and
    the sign of the second operand. It takes two operands and returns one
    result of the same type. This type may be a float scalar type, a vector
    whose element type is float, or a tensor of floats. It has no standard
    attributes.
  }];
}

def CosOp : FloatUnaryOp<"cos"> {
  let summary = "cosine of the specified value";
  let description = [{
    The `cos` operation computes the cosine of a given value. It takes one
    operand and returns one result of the same type. This type may be a float
    scalar type, a vector whose element type is float, or a tensor of floats.
    It has no standard attributes.
  }];
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

  let arguments = (ins AnyMemRef:$memref);

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def DimOp : Std_Op<"dim", [NoSideEffect]> {
  let summary = "dimension index operation";
  let description = [{
    The "dim" operation takes a memref or tensor operand and returns an "index".
    It requires a single integer attribute named "index". It returns the size
    of the specified dimension. For example:

      %1 = dim %0, 2 : tensor<?x?x?xf32>
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyTensor],
                                 "any tensor or memref type">:$memrefOrTensor,
                       APIntAttr:$index);
  let results = (outs Index);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *memrefOrTensor,"
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

  let hasFolder = 1;
}

def DivFOp : FloatArithmeticOp<"divf"> {
  let summary = "floating point division operation";
}

def SignedDivIOp : IntArithmeticOp<"divi_signed"> {
  let summary = "signed integer division operation";
  let hasFolder = 1;
}

def UnsignedDivIOp : IntArithmeticOp<"divi_unsigned"> {
  let summary = "unsigned integer division operation";
  let hasFolder = 1;
}

def ExpOp : FloatUnaryOp<"exp"> {
  let summary = "base-e exponential of the specified value";
}

def ExtractElementOp : Std_Op<"extract_element", [NoSideEffect]> {
  let summary = "element extract operation";
  let description = [{
    The "extract_element" op reads a tensor or vector and returns one element
    from it specified by an index list. The output of extract is a new value
    with the same type as the elements of the tensor or vector. The arity of
    indices matches the rank of the accessed value (i.e., if a tensor is of rank
    3, then 3 indices are required for the extract).  The indices should all be
    of index type. For example:

      %3 = extract_element %0[%1, %2] : vector<4x4xi32>
  }];

  let arguments = (ins AnyTypeOf<[AnyVector, AnyTensor]>:$aggregate,
                       Variadic<Index>:$indices);
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *aggregate,"
    "ValueRange indices = {}", [{
      auto resType = aggregate->getType().cast<ShapedType>()
                                         .getElementType();
      build(builder, result, resType, aggregate, indices);
    }]>];

  let extraClassDeclaration = [{
    Value *getAggregate() { return getOperand(0); }

    operand_range getIndices() {
      return {operand_begin() + 1, operand_end()};
    }
  }];

  let hasFolder = 1;
}

def IndexCastOp : CastOp<"index_cast">, Arguments<(ins AnyType:$in)> {
  let summary = "cast between index and integer types";
  let description = [{
    Casts between integer scalars and 'index' scalars.  Index is an integer of
    platform-specific bit width.  If casting to a wider integer, the value is
    sign-extended.  If casting to a narrower integer, the value is truncated.
  }];

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);
  }];

  let hasFolder = 0;
}

def FPExtOp : CastOp<"fpext">, Arguments<(ins AnyType:$in)> {
  let summary = "cast from floating-point to wider floating-point";
  let description = [{
    Cast a floating-point value to a larger floating-point-typed value.
    The destination type must to be strictly wider than the source type.
    Only scalars are currently supported.
  }];

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);
  }];

  let hasFolder = 0;
}

def FPTruncOp : CastOp<"fptrunc">, Arguments<(ins AnyType:$in)> {
  let summary = "cast from floating-point to narrower floating-point";
  let description = [{
    Truncate a floating-point value to a smaller floating-point-typed value.
    The destination type must be strictly narrower than the source type.
    If the value cannot be exactly represented, it is rounded using the default
    rounding mode. Only scalars are currently supported.
  }];

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);
  }];

  let hasFolder = 0;
}

def LoadOp : Std_Op<"load"> {
  let summary = "load operation";
  let description = [{
    The "load" op reads an element from a memref specified by an index list. The
    output of load is a new value with the same type as the elements of the
    memref. The arity of indices is the rank of the memref (i.e., if the memref
    loaded from is of rank 3, then 3 indices are required for the load following
    the memref identifier). For example:

      %3 = load %0[%1, %1] : memref<4x4xi32>
  }];

  let arguments = (ins AnyMemRef:$memref, Variadic<Index>:$indices);
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *, OperationState &result, Value *memref,"
    "ValueRange indices = {}", [{
      auto memrefType = memref->getType().cast<MemRefType>();
      result.addOperands(memref);
      result.addOperands(indices);
      result.types.push_back(memrefType.getElementType());
  }]>];

  let extraClassDeclaration = [{
    Value *getMemRef() { return getOperand(0); }
    void setMemRef(Value *value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return getMemRef()->getType().cast<MemRefType>();
    }

    operand_range getIndices() { return {operand_begin() + 1, operand_end()}; }
  }];

  let hasFolder = 1;
}

def LogOp : FloatUnaryOp<"log"> {
  let summary = "base-e logarithm of the specified value";
}

def Log10Op : FloatUnaryOp<"log10"> {
  let summary = "base-10 logarithm of the specified value";
}

def Log2Op : FloatUnaryOp<"log2"> {
  let summary = "base-2 logarithm of the specified value";
}

def MemRefCastOp : CastOp<"memref_cast"> {
  let summary = "memref cast operation";
  let description = [{
    The "memref_cast" operation converts a memref from one type to an equivalent
    type with a compatible shape. The source and destination types are
    compatible if:
    a. both are ranked memref types with the same element type, affine mappings,
    address space, and rank but where the individual dimensions may add or
    remove constant dimensions from the memref type.

    If the cast converts any dimensions from an unknown to a known size, then it
    acts as an assertion that fails at runtime of the dynamic dimensions
    disagree with resultant destination size.

    Example:
    Assert that the input dynamic shape matches the destination static shape.
       %2 = memref_cast %1 : memref<?x?xf32> to memref<4x4xf32>
    Erase static shape information, replacing it with dynamic information.
       %3 = memref_cast %1 : memref<4xf32> to memref<?xf32>

    The same holds true for offsets and strides.

    Assert that the input dynamic shape matches the destination static stride.
       %4 = memref_cast %1 : memref<12x4xf32, offset:?, strides: [?, ?]> to
                             memref<12x4xf32, offset:5, strides: [4, 1]>
    Erase static offset and stride information, replacing it with
    dynamic information.
       %5 = memref_cast %1 : memref<12x4xf32, offset:5, strides: [4, 1]> to
                             memref<12x4xf32, offset:?, strides: [?, ?]>

    b. either or both memref types are unranked with the same element type, and
    address space.

    Example:
    Cast to concrete shape.
        %4 = memref_cast %1 : memref<*xf32> to memref<4x?xf32>

    Erase rank information.
        %5 = memref_cast %1 : memref<4x?xf32> to memref<*xf32>
  }];

  let arguments = (ins AnyRankedOrUnrankedMemRef:$source);
  let results = (outs AnyRankedOrUnrankedMemRef);

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);

    /// The result of a memref_cast is always a memref.
    Type getType() { return getResult()->getType(); }
  }];
}

def MulFOp : FloatArithmeticOp<"mulf"> {
  let summary = "floating point multiplication operation";
  let hasFolder = 1;
}

def MulIOp : IntArithmeticOp<"muli", [Commutative]> {
  let summary = "integer multiplication operation";
  let hasFolder = 1;
}

def NegFOp : FloatUnaryOp<"negf"> {
  let summary = "floating point negation";
  let description = [{
    The `negf` operation computes the negation of a given value. It takes one
    operand and returns one result of the same type. This type may be a float
    scalar type, a vector whose element type is float, or a tensor of floats.
    It has no standard attributes.
  }];
}

def OrOp : IntArithmeticOp<"or", [Commutative]> {
  let summary = "integer binary or";
  let hasFolder = 1;
}

def PrefetchOp : Std_Op<"prefetch"> {
  let summary = "prefetch operation";
  let description = [{
    The "prefetch" op prefetches data from a memref location described with
    subscript indices similar to std.load, and with three attributes: a
    read/write specifier, a locality hint, and a cache type specifier as shown
    below:

      prefetch %0[%i, %j], read, locality<3>, data : memref<400x400xi32>

    The read/write specifier is either 'read' or 'write', the locality hint
    ranges from locality<0> (no locality) to locality<3> (extremely local keep
    in cache). The cache type specifier is either 'data' or 'instr'
    and specifies whether the prefetch is performed on data cache or on
    instruction cache.
  }];

  let arguments = (ins AnyMemRef:$memref, Variadic<Index>:$indices,
                   BoolAttr:$isWrite,
                   Confined<I32Attr, [IntMinValue<0>,
                     IntMaxValue<3>]>:$localityHint,
                   BoolAttr:$isDataCache);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *memref,"
    "ArrayRef<Value *> indices, bool isWrite, unsigned hint, bool isData",
    [{
      auto hintAttr = builder->getI32IntegerAttr(hint);
      auto isWriteAttr = builder->getBoolAttr(isWrite);
      auto isDataCacheAttr = builder->getBoolAttr(isData);
      result.addOperands(memref);
      result.addOperands(indices);
      result.addAttribute("localityHint", hintAttr);
      result.addAttribute("isWrite", isWriteAttr);
      result.addAttribute("isDataCache", isDataCacheAttr);
    }]>];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return memref()->getType().cast<MemRefType>();
    }
    static StringRef getLocalityHintAttrName() { return "localityHint"; }
    static StringRef getIsWriteAttrName() { return "isWrite"; }
    static StringRef getIsDataCacheAttrName() { return "isDataCache"; }
  }];

  let hasFolder = 1;
}

def RankOp : Std_Op<"rank", [NoSideEffect]> {
  let summary = "rank operation";
  let description = [{
    The "rank" operation takes a tensor operand and returns its rank.

      %1 = rank %0 : index
  }];

  let arguments = (ins AnyTensor);
  let results = (outs Index);
  let verifier = ?;

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *tensor", [{
      auto indexType = builder->getIndexType();
      build(builder, result, indexType, tensor);
    }]>];

  let hasFolder = 1;
}

def RemFOp : FloatArithmeticOp<"remf"> {
  let summary = "floating point division remainder operation";
}

def SignedRemIOp : IntArithmeticOp<"remi_signed"> {
  let summary = "signed integer division remainder operation";
  let hasFolder = 1;
}

def UnsignedRemIOp : IntArithmeticOp<"remi_unsigned"> {
  let summary = "unsigned integer division remainder operation";
  let hasFolder = 1;
}

def ReturnOp : Std_Op<"return", [Terminator, HasParent<"FuncOp">]> {
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
    "Builder *b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];
}

def SelectOp : Std_Op<"select", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "select operation";
  let description = [{
    The "select" operation chooses one value based on a binary condition
    supplied as its first operand. If the value of the first operand is 1, the
    second operand is chosen, otherwise the third operand is chosen. The second
    and the third operand must have the same type. The operation applies
    elementwise to vectors and tensors.  The shape of all arguments must be
    identical. For example, the maximum operation is obtained by combining
    "select" with "cmpi" as follows.

      %2 = cmpi "gt" %0, %1 : i32         // %2 is i1
      %3 = select %2, %0, %1 : i32
  }];

  let arguments = (ins BoolLike:$condition, AnyType:$true_value,
                       AnyType:$false_value);
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *condition,"
    "Value *trueValue, Value *falseValue", [{
      result.addOperands({condition, trueValue, falseValue});
      result.addTypes(trueValue->getType());
  }]>];

  let extraClassDeclaration = [{
      Value *getCondition() { return condition(); }
      Value *getTrueValue() { return true_value(); }
      Value *getFalseValue() { return false_value(); }
  }];

  let hasFolder = 1;
}

def SignExtendIOp : Std_Op<"sexti",
    [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "integer sign extension operation";
  let description = [{
    The integer sign extension operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be larger than the input bit-width (N > M).
    The top-most (N - M) bits of the output are filled with copies
    of the most-significant bit of the input.

      %1 = constant 5 : i3            // %1 is 0b101
      %2 = sexti %1 : i3 to i6        // %2 is 0b111101
      %3 = constant 2 : i3            // %3 is 0b010
      %4 = sexti %3 : i3 to i6        // %4 is 0b000010

      %5 = sexti %0 : vector<2 x i32> to vector<2 x i64>
  }];

  let arguments = (ins IntegerLike:$value);
  let results = (outs IntegerLike);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *value, Type destType", [{
      result.addOperands(value);
      result.addTypes(destType);
  }]>];

  let parser = [{
    return impl::parseCastOp(parser, result);
  }];
  let printer = [{
    return printStandardCastOp(this->getOperation(), p);
  }];
}

def ShiftLeftOp : IntArithmeticOp<"shift_left"> {
  let summary = "integer left-shift";
  let description = [{
    The shift_left operation shifts an integer value to the left by a variable
    amount. The low order bits are filled with zeros.

      %1 = constant 5 : i8                       // %1 is 0b00000101
      %2 = constant 3 : i8
      %3 = shift_left %1, %2 : (i8, i8) -> i8    // %3 is 0b00101000
  }];
}

def SignedShiftRightOp : IntArithmeticOp<"shift_right_signed"> {
  let summary = "signed integer right-shift";
  let description = [{
    The shift_right_signed operation shifts an integer value to the right by
    a variable amount. The integer is interpreted as signed. The high order
    bits in the output are filled with copies of the most-significant bit
    of the shifted value (which means that the sign of the value is preserved).

      %1 = constant 160 : i8                             // %1 is 0b10100000
      %2 = constant 3 : i8
      %3 = shift_right_signed %1, %2 : (i8, i8) -> i8    // %3 is 0b11110100
      %4 = constant 96 : i8                              // %4 is 0b01100000
      %5 = shift_right_signed %4, %2 : (i8, i8) -> i8    // %5 is 0b00001100
  }];
}

def UnsignedShiftRightOp : IntArithmeticOp<"shift_right_unsigned"> {
  let summary = "unsigned integer right-shift";
  let description = [{
    The shift_right_unsigned operation shifts an integer value to the right by
    a variable amount. The integer is interpreted as unsigned. The high order
    bits are always filled with zeros.

      %1 = constant 160 : i8                               // %1 is 0b10100000
      %2 = constant 3 : i8
      %3 = shift_right_unsigned %1, %2 : (i8, i8) -> i8    // %3 is 0b00010100
  }];
}

def SIToFPOp : CastOp<"sitofp">, Arguments<(ins AnyType:$in)> {
  let summary = "cast from integer type to floating-point";
  let description = [{
    Cast from a value interpreted as signed integer to the corresponding
    floating-point value. If the value cannot be exactly represented, it is
    rounded using the default rounding mode. Only scalars are currently
    supported.
  }];

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);
  }];

  let hasFolder = 0;
}

def SplatOp : Std_Op<"splat", [NoSideEffect]> {
  let summary = "splat or broadcast operation";
  let description = [{
    The "splat" op reads a value of integer or float type and broadcasts it into
    a vector or a tensor. The output of splat is thus a new value of either
    vector or tensor type with elemental type being its operand's type.
    When the result is a tensor, it has to be statically shaped.

      %1 = splat %0 : vector<8xi32>
      %2 = splat %0 : tensor<4x8xi32>

    TODO: Extend this operation to broadcast to dynamically shaped tensors in
    the same way dynamically shaped memrefs are handled.

    // Broadcasts %s to a 2-d dynamically shaped tensor, with %m, %n binding
    // to the sizes of the two dynamic dimensions.

      %m = "foo"() : () -> (index)
      %n = "bar"() : () -> (index)
      %t = splat %s [%m, %n] : tensor<?x?xi32>

  }];

  let arguments = (ins AnyTypeOf<[AnyInteger, AnyFloat],
                                 "integer or float type">:$input);
  let results = (outs AnyTypeOf<[AnyVector, AnyStaticShapeTensor]>:$aggregate);

  let builders =
      [OpBuilder<"Builder *builder, OperationState &result, Value *element, "
                  "Type aggregateType",
                  [{ build(builder, result, aggregateType, element); }]>];

  let hasFolder = 1;
}

def StoreOp : Std_Op<"store"> {
  let summary = "store operation";
  let description = [{
    The "store" op writes an element to a memref specified by an index list.
    The arity of indices is the rank of the memref (i.e. if the memref being
    stored to is of rank 3, then 3 indices are required for the store following
    the memref identifier). The store operation does not produce a result.

    In the following example, the ssa value '%v' is stored in memref '%A' at
    indices [%i, %j]:
      store %v, %A[%i, %j] : memref<4x128xf32, (d0, d1) -> (d0, d1), 0>
  }];

  let arguments = (ins AnyType:$value, AnyMemRef:$memref,
                   Variadic<Index>:$indices);

  let builders = [OpBuilder<
    "Builder *, OperationState &result, Value *valueToStore, Value *memref", [{
      result.addOperands(valueToStore);
      result.addOperands(memref);
  }]>];

  let extraClassDeclaration = [{
      Value *getValueToStore() { return getOperand(0); }

      Value *getMemRef() { return getOperand(1); }
      void setMemRef(Value *value) { setOperand(1, value); }
      MemRefType getMemRefType() {
        return getMemRef()->getType().cast<MemRefType>();
      }

      operand_range getIndices() {
        return {operand_begin() + 2, operand_end()};
      }
  }];

  let hasFolder = 1;
}

def SubFOp : FloatArithmeticOp<"subf"> {
  let summary = "floating point subtraction operation";
  let hasFolder = 1;
}

def SubIOp : IntArithmeticOp<"subi"> {
  let summary = "integer subtraction operation";
  let hasFolder = 1;
}

def SubViewOp : Std_Op<"subview", [AttrSizedOperandSegments, NoSideEffect]> {
  let summary = "memref subview operation";
  let description = [{
    The "subview" operation converts a memref type to another memref type
    which represents a reduced-size view of the original memref as specified by
    the operation's offsets, sizes and strides arguments.

    The SubView operation supports the following arguments:
    *) Memref: the "base" memref on which to create a "view" memref.
    *) Offsets: zero or memref-rank number of dynamic offsets into the "base"
                memref at which to create the "view" memref.
    *) Sizes: zero or memref-rank dynamic size operands which specify the
              dynamic sizes of the result "view" memref type.
    *) Strides: zero or memref-rank number of dynamic strides which are applied
                multiplicatively to the base memref strides in each dimension.

    Note on the number of operands for offsets, sizes and strides: For
    each of these, the number of operands must either be same as the
    memref-rank number or empty. For the latter, those values will be
    treated as constants.

    Example 1:

      %0 = alloc() : memref<64x4xf32, (d0, d1) -> (d0 * 4 + d1)>

      // Create a sub-view of "base" memref '%0' with offset arguments '%c0',
      // dynamic sizes for each dimension, and stride arguments '%c1'.
      %1 = subview %0[%c0, %c0][%size0, %size1][%c1, %c1]
        : memref<64x4xf32, (d0, d1) -> (d0 * 4 + d1) > to
          memref<?x?xf32, (d0, d1)[s0, s1] -> (d0 * s1 + d1 + s0)>

    Example 2:

      %0 = alloc() : memref<8x16x4xf32, (d0, d1, d1) -> (d0 * 64 + d1 * 4 + d2)>

      // Create a sub-view of "base" memref '%0' with dynamic offsets, sizes,
      // and strides.
      // Note that dynamic offsets are represented by the linearized dynamic
      // offset symbol 's0' in the subview memref layout map, and that the
      // dynamic strides operands, after being applied to the base memref
      // strides in each dimension, are represented in the view memref layout
      // map as symbols 's1', 's2' and 's3'.
      %1 = subview %0[%i, %j, %k][%size0, %size1, %size2][%x, %y, %z]
        : memref<8x16x4xf32, (d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)> to
          memref<?x?x?xf32,
            (d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + d1 * s2 + d2 * s3 + s0)>

    Example 3:

      %0 = alloc() : memref<8x16x4xf32, (d0, d1, d1) -> (d0 * 64 + d1 * 4 + d2)>

      // Subview with constant offsets, sizes and strides.
      %1 = subview %0[][][]
        : memref<8x16x4xf32, (d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)> to
          memref<4x4x4xf32, (d0, d1, d2) -> (d0 * 16 + d1 * 4 + d2 + 8)>

    Example 4:

      %0 = alloc(%arg0, %arg1) : memref<?x?xf32>

      // Subview with constant size, but dynamic offsets and
      // strides. The resulting memref has a static shape, but if the
      // base memref has an affine map to describe the layout, the result
      // memref also uses an affine map to describe the layout. The
      // strides of the result memref is computed as follows:
      //
      // Let #map1 represents the layout of the base memref, and #map2
      // represents the layout of the result memref. A #mapsubview can be
      // constructed to map an index from the result memref to the base
      // memref (note that the description below uses more convenient
      // naming for symbols, while in affine maps, symbols are
      // represented as unsigned numbers that identify that symbol in the
      // given affine map.
      //
      // #mapsubview = (d0, d1)[o0, o1, t0, t1] -> (d0 * t0 + o0, d1 * t1 + o1)
      //
      // where, o0, o1, ... are offsets, and t0, t1, ... are strides. Then,
      //
      // #map2 = #map1.compose(#mapsubview)
      //
      // If the layout map is represented as
      //
      // #map1 = (d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)
      //
      // then,
      //
      // #map2 = (d0, d1)[s0, s1, s2, o0, o1, t0, t1] ->
      //              (d0 * s1 * t0 + d1 * s2 * t1 + o0 * s1 + o1 * s2 + s0)
      //
      // Representing this canonically
      //
      // #map2 = (d0, d1)[r0, r1, r2] -> (d0 * r1 + d1 * r2 + r0)
      //
      // where, r0 = o0 * s1 + o1 * s2 + s0, r1 = s1 * t0, r2 = s2 * t1.
      %1 = subview %0[%i, %j][][%x, %y] :
        : memref<?x?xf32, (d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)> to
          memref<4x4xf32, (d0, d1)[r0, r1, r2] -> (d0 * r1 + d1 * r2 + r0)>

      // Note that the subview op does not guarantee that the result
      // memref is "inbounds" w.r.t to base memref. It is upto the client
      // to ensure that the subview is accessed in a manner that is
      // in-bounds.

    }
  }];

  // TODO(b/144779634, ravishankarm) : Use different arguments for
  // offsets, sizes and strides.
  let arguments = (ins
    AnyMemRef:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    I32ElementsAttr:$operand_segment_sizes
  );
  let results = (outs AnyMemRef);

  let builders = [
    OpBuilder<
      "Builder *b, OperationState &result, Value *source, "
      "ValueRange offsets, ValueRange sizes, "
      "ValueRange strides, Type resultType = Type(), "
      "ArrayRef<NamedAttribute> attrs = {}">,
    OpBuilder<
      "Builder *builder, OperationState &result, "
      "Type resultType, Value *source">
  ];

  let extraClassDeclaration = [{
    /// Returns the type of the base memref operand.
    MemRefType getBaseMemRefType() {
      return source()->getType().cast<MemRefType>();
    }

    /// The result of a subview is always a memref.
    MemRefType getType() { return getResult()->getType().cast<MemRefType>(); }

    /// Returns as integer value the number of offset operands.
    int64_t getNumOffsets() { return llvm::size(offsets()); }

    /// Returns as integer value the number of size operands.
    int64_t getNumSizes() { return llvm::size(sizes()); }

    /// Returns as integer value the number of stride operands.
    int64_t getNumStrides() { return llvm::size(strides()); }

    /// Returns the dynamic sizes for this subview operation if specified.
    operand_range getDynamicSizes() { return sizes(); }

    /// Returns in `staticStrides` the static value of the stride
    /// operands. Returns failure() if the static value of the stride
    /// operands could not be retrieved.
    LogicalResult getStaticStrides(SmallVectorImpl<int64_t> &staticStrides);

    // Auxiliary range data structure and helper function that unpacks the
    // offset, size and stride operands of the SubViewOp into a list of triples.
    // Such a list of triple is sometimes more convenient to manipulate.
    struct Range {
      Value *offset, *size, *stride;
    };
    SmallVector<Range, 8> getRanges();
  }];

  let hasCanonicalizer = 1;
}

def TanhOp : FloatUnaryOp<"tanh"> {
  let summary = "hyperbolic tangent of the specified value";
  let description = [{
    The `tanh` operation computes the hyperbolic tangent. It takes one operand
    and returns one result of the same type. This type may be a float scalar
    type, a vector whose element type is float, or a tensor of floats. It has
    no standard attributes.
  }];
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
       %2 = tensor_cast %1 : tensor<*xf32> to tensor<?x?xf32>
  }];

  let arguments = (ins AnyTensor);
  let results = (outs AnyTensor);

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);

    /// The result of a tensor_cast is always a tensor.
    TensorType getType() { return getResult()->getType().cast<TensorType>(); }
  }];
}

def TensorLoadOp : Std_Op<"tensor_load",
    [SameOperandsAndResultShape, SameOperandsAndResultElementType]> {
  let summary = "tensor load operation";
  let description = [{
    The "tensor_load" operation creates a tensor from a memref, making an
    independent copy of the element data. The result value is a tensor whose
    shape and element type match the memref operand.

    Produce a value of tensor<4x?xf32> type.
       %12 = tensor_load %10 : memref<4x?xf32, #layout, memspace0>
  }];

  let arguments = (ins AnyMemRef);
  let results = (outs AnyTensor);
  // TensorLoadOp is fully verified by traits.
  let verifier = ?;

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *memref", [{
      auto memrefType = memref->getType().cast<MemRefType>();
      auto resultType = RankedTensorType::get(memrefType.getShape(),
                                              memrefType.getElementType());
      result.addOperands(memref);
      result.addTypes(resultType);
  }]>];


  let extraClassDeclaration = [{
    /// The result of a tensor_load is always a tensor.
    TensorType getType() { return getResult()->getType().cast<TensorType>(); }
  }];
}

def TensorStoreOp : Std_Op<"tensor_store",
    [SameOperandsShape, SameOperandsElementType]> {
  let summary = "tensor store operation";
  let description = [{
    The "tensor_store" operation stores the contents of a tensor into a memref.
    The first operand is a value of tensor type, the second operand is a value
    of memref type. The shapes and element types of these must match, and are
    specified by the memref type.

    Example:
       %9 = dim %8, 1 : tensor<4x?xf32>
       %10 = alloc(%9) : memref<4x?xf32, #layout, memspace0>
       tensor_store %8, %10 : memref<4x?xf32, #layout, memspace0>
  }];

  let arguments = (ins AnyTensor:$tensor, AnyMemRef:$memref);
  // TensorStoreOp is fully verified by traits.
  let verifier = ?;
}

def TruncateIOp : Std_Op<"trunci", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "integer truncation operation";
  let description = [{
    The integer truncation operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be smaller than the input bit-width (N < M).
    The top-most (N - M) bits of the input are discarded.

      %1 = constant 21 : i5           // %1 is 0b10101
      %2 = trunci %1 : i5 to i4       // %2 is 0b0101
      %3 = trunci %1 : i5 to i3       // %3 is 0b101

      %5 = trunci %0 : vector<2 x i32> to vector<2 x i16>
  }];

  let arguments = (ins IntegerLike:$value);
  let results = (outs IntegerLike);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *value, Type destType", [{
      result.addOperands(value);
      result.addTypes(destType);
  }]>];

  let parser = [{
    return impl::parseCastOp(parser, result);
  }];
  let printer = [{
    return printStandardCastOp(this->getOperation(), p);
  }];
}

def ViewOp : Std_Op<"view", [NoSideEffect]> {
  let summary = "memref view operation";
  let description = [{
    The "view" operation converts a 1-D memref with i8 element type,
    to an N-D memref with arbitrary element type. In addition, the ViewOp
    supports the following arguments:
    *) A single dynamic offset operand can be specified which represents a
       a dynamic offset within the base 1-D memref at which to create the
       resulting memref view.
    *) A dynamic size operand must be specified for each dynamic dimension
       in the resulting view memref type.

    // Allocate a flat 1D/i8 memref.
    %0 = alloc() : memref<2048xi8>

    // ViewOp with static offset and sizes.
    %1 = view %0[][] : memref<2048xi8> to memref<64x4xf32>

    // ViewOp with dynamic offset and one dynamic size.
    %2 = view %0[%offset_1024][%size0]
      : memref<2048xi8> to memref<?x4xf32, (d0, d1)[s0] -> (d0 * 4 + d1 + s0)>

    // ViewOp creating 3D shape where two of the dim sizes are dynamic.
    // *) The dynamic offset specified in the ViewOp is applied to the
    //    base 1-D memref, and is represented by the symbol 's0' in the
    //    layout map of the ViewOp result memref type.
    // *) The dynamic size for the second dimension induces a dynamic
    //    stride for the first dimension, which is represented by the
    //    symbol 's1' in the layout map of the ViewOp result memref type.
    //    Note that this dynamic stride will be computed from the view
    //    shape and dynamic sizes.
    %3 = view %0[%offset_1024][%size0, %size1]
      : memref<2048xi8> to memref<?x?x4xf32,
        (d0, d1, d2)[s0, s1] -> (d0 * s1 + d1 * 4 + d2 + s0)>
  }];

  let arguments = (ins MemRefRankOf<[I8], [1]>:$source,
                       Variadic<Index>:$operands);
  let results = (outs AnyMemRef);

  let extraClassDeclaration = [{
    /// The result of a view is always a memref.
    MemRefType getType() { return getResult()->getType().cast<MemRefType>(); }

    /// Returns the dynamic offset for this view operation if specified.
    /// Returns nullptr if no dynamic offset was specified.
    Value *getDynamicOffset();

    /// Returns the starting operand list position of the dynamic size operands.
    unsigned getDynamicSizesOperandStart() {
      return getDynamicOffset() == nullptr ? 1 : 2;
    }

    /// Returns the dynamic sizes for this view operation.
    operand_range getDynamicSizes() {
      return {operand_begin() + getDynamicSizesOperandStart(), operand_end()};
    }
  }];

  let hasCanonicalizer = 1;
}

def XOrOp : IntArithmeticOp<"xor", [Commutative]> {
  let summary = "integer binary xor";
  let hasFolder = 1;
}

def ZeroExtendIOp : Std_Op<"zexti", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "integer zero extension operation";
  let description = [{
    The integer zero extension operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be larger than the input bit-width (N > M).
    The top-most (N - M) bits of the output are filled with zeros.

      %1 = constant 5 : i3            // %1 is 0b101
      %2 = zexti %1 : i3 to i6        // %2 is 0b000101
      %3 = constant 2 : i3            // %3 is 0b010
      %4 = zexti %3 : i3 to i6        // %4 is 0b000010

      %5 = zexti %0 : vector<2 x i32> to vector<2 x i64>
  }];

  let arguments = (ins IntegerLike:$value);
  let results = (outs IntegerLike);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *value, Type destType", [{
      result.addOperands(value);
      result.addTypes(destType);
  }]>];

  let parser = [{
    return impl::parseCastOp(parser, result);
  }];
  let printer = [{
    return printStandardCastOp(this->getOperation(), p);
  }];
}

#endif // STANDARD_OPS
