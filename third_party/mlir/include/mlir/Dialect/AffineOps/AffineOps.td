//===- AffineOps.td - Affine operation definitions ---------*- tablegen -*-===//
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
// Defines MLIR affine operations.
//
//===----------------------------------------------------------------------===//

#ifndef AFFINE_OPS
#define AFFINE_OPS

include "mlir/Dialect/AffineOps/AffineOpsBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Transforms/LoopLikeInterface.td"

def Affine_Dialect : Dialect {
  let name = "affine";
  let cppNamespace = "";
}

// Base class for Affine dialect ops.
class Affine_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Affine_Dialect, mnemonic, traits> {
  // For every affine op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

// Require regions to have affine terminator.
def ImplicitAffineTerminator
    : SingleBlockImplicitTerminator<"AffineTerminatorOp">;

def AffineForOp : Affine_Op<"for",
    [ImplicitAffineTerminator,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
  let summary = "for operation";
  let description = [{
    The "affine.for" operation represents an affine loop nest, defining an SSA
    value for its induction variable. It has one region capturing the loop body.
    The induction variable is represented as a argument of this region. This SSA
    value always has type index, which is the size of the machine word. The
    stride, represented by step, is a positive constant integer which defaults
    to "1" if not present. The lower and upper bounds specify a half-open range:
    the range includes the lower bound but does not include the upper bound.

    The body region must contain exactly one block that terminates with
    "affine.terminator".  Calling AffineForOp::build will create such region
    and insert the terminator, so will the parsing even in cases if it is absent
    from the custom format.

    The lower and upper bounds of a for operation are represented as an
    application of an affine mapping to a list of SSA values passed to the map.
    The same restrictions hold for these SSA values as for all bindings of SSA
    values to dimensions and symbols. The affine mappings for the bounds may
    return multiple results, in which case the max/min keywords are required
    (for the lower/upper bound respectively), and the bound is the
    maximum/minimum of the returned values.

    Example:

      affine.for %i = 1 to 10 {
        ...
      }

  }];
  let arguments = (ins Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "int64_t lowerBound, int64_t upperBound, int64_t step = 1">,
    OpBuilder<"Builder *builder, OperationState &result, "
              "ValueRange lbOperands, AffineMap lbMap, "
              "ValueRange ubOperands, AffineMap ubMap, "
              "int64_t step = 1">
  ];

  let extraClassDeclaration = [{
    static StringRef getStepAttrName() { return "step"; }
    static StringRef getLowerBoundAttrName() { return "lower_bound"; }
    static StringRef getUpperBoundAttrName() { return "upper_bound"; }

    Block *getBody() { return &region().front(); }
    Value *getInductionVar() { return getBody()->getArgument(0); }
    OpBuilder getBodyBuilder() {
      return OpBuilder(getBody(), std::prev(getBody()->end()));
    }

    // TODO: provide iterators for the lower and upper bound operands
    // if the current access via getLowerBound(), getUpperBound() is too slow.

    /// Returns operands for the lower bound map.
    operand_range getLowerBoundOperands();

    /// Returns operands for the upper bound map.
    operand_range getUpperBoundOperands();

    /// Returns information about the lower bound as a single object.
    AffineBound getLowerBound();

    /// Returns information about the upper bound as a single object.
    AffineBound getUpperBound();

    /// Returns loop step.
    int64_t getStep() {
      return getAttr(getStepAttrName()).cast<IntegerAttr>().getInt();
    }

    /// Returns affine map for the lower bound.
    AffineMap getLowerBoundMap() { return getLowerBoundMapAttr().getValue(); }
    AffineMapAttr getLowerBoundMapAttr() {
      return getAttr(getLowerBoundAttrName()).cast<AffineMapAttr>();
    }
    /// Returns affine map for the upper bound. The upper bound is exclusive.
    AffineMap getUpperBoundMap() { return getUpperBoundMapAttr().getValue(); }
    AffineMapAttr getUpperBoundMapAttr() {
      return getAttr(getUpperBoundAttrName()).cast<AffineMapAttr>();
    }

    /// Set lower bound. The new bound must have the same number of operands as
    /// the current bound map. Otherwise, 'replaceForLowerBound' should be used.
    void setLowerBound(ValueRange operands, AffineMap map);
    /// Set upper bound. The new bound must not have more operands than the
    /// current bound map. Otherwise, 'replaceForUpperBound' should be used.
    void setUpperBound(ValueRange operands, AffineMap map);

    /// Set the lower bound map without changing operands.
    void setLowerBoundMap(AffineMap map);

    /// Set the upper bound map without changing operands.
    void setUpperBoundMap(AffineMap map);

    /// Set loop step.
    void setStep(int64_t step) {
      assert(step > 0 && "step has to be a positive integer constant");
      auto *context = getLowerBoundMap().getContext();
      setAttr(Identifier::get(getStepAttrName(), context),
              IntegerAttr::get(IndexType::get(context), step));
    }

    /// Returns true if the lower bound is constant.
    bool hasConstantLowerBound();
    /// Returns true if the upper bound is constant.
    bool hasConstantUpperBound();
    /// Returns true if both bounds are constant.
    bool hasConstantBounds() {
      return hasConstantLowerBound() && hasConstantUpperBound();
    }
    /// Returns the value of the constant lower bound.
    /// Fails assertion if the bound is non-constant.
    int64_t getConstantLowerBound();
    /// Returns the value of the constant upper bound. The upper bound is
    /// exclusive. Fails assertion if the bound is non-constant.
    int64_t getConstantUpperBound();
    /// Sets the lower bound to the given constant value.
    void setConstantLowerBound(int64_t value);
    /// Sets the upper bound to the given constant value.
    void setConstantUpperBound(int64_t value);

    /// Returns true if both the lower and upper bound have the same operand
    /// lists (same operands in the same order).
    bool matchingBoundOperandList();
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def AffineIfOp : Affine_Op<"if", [ImplicitAffineTerminator]> {
  let summary = "if-then-else operation";
  let description = [{
    The "if" operation represents an if-then-else construct for conditionally
    executing two regions of code. The operands to an if operation are an
    IntegerSet condition and a set of symbol/dimension operands to the
    condition set. The operation produces no results. For example:

       affine.if #set(%i)  {
         ...
       } else {
         ...
       }

    The 'else' blocks to the if operation are optional, and may be omitted. For
    example:

       affine.if #set(%i)  {
         ...
       }
  }];
  let arguments = (ins Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "IntegerSet set, ValueRange args, bool withElseRegion">
  ];

  let extraClassDeclaration = [{
    static StringRef getConditionAttrName() { return "condition"; }

    IntegerSet getIntegerSet();
    void setIntegerSet(IntegerSet newSet);

    /// Sets the integer set with its operands. The size of 'operands' must not
    /// exceed the current number of operands for this instance, as the operands
    /// list of AffineIf is not resizable.
    void setConditional(IntegerSet set, ValueRange operands);

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

  let hasFolder = 1;
}

def AffineMinOp : Affine_Op<"min"> {
  let summary = "min operation";
  let description = [{
    The "min" operation computes the minimum value result from a multi-result
    affine map.

    Example:

       %0 = affine.min (d0) -> (1000, d0 + 512) (%i0) : index
  }];
  let arguments = (ins AffineMapAttr:$map, Variadic<Index>:$operands);
  let results = (outs Index);
  let extraClassDeclaration = [{
    static StringRef getMapAttrName() { return "map"; }
  }];
  let hasFolder = 1;
}

def AffinePrefetchOp : Affine_Op<"prefetch"> {
  let summary = "affine prefetch operation";
  let description = [{
    The "affine.prefetch" op prefetches data from a memref location described
    with an affine subscript similar to affine.load, and has three attributes:
    a read/write specifier, a locality hint, and a cache type specifier as shown
    below:

      affine.prefetch %0[%i, %j + 5], read, locality<3>, data
                          : memref<400x400xi32>

    The read/write specifier is either 'read' or 'write', the locality hint
    specifier ranges from locality<0> (no locality) to locality<3> (extremely
    local keep in cache). The cache type specifier is either 'data' or 'instr'
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
    "AffineMap map, ArrayRef<Value *> mapOperands, bool isWrite,"
    "unsigned localityHint, bool isDataCache",
    [{
      assert(map.getNumInputs() == mapOperands.size()
             && "inconsistent index info");
      auto localityHintAttr = builder->getI32IntegerAttr(localityHint);
      auto isWriteAttr = builder->getBoolAttr(isWrite);
      auto isDataCacheAttr = builder->getBoolAttr(isDataCache);
      result.addOperands(memref);
      result.addAttribute(getMapAttrName(), AffineMapAttr::get(map));
      result.addOperands(mapOperands);
      result.addAttribute(getLocalityHintAttrName(), localityHintAttr);
      result.addAttribute(getIsWriteAttrName(), isWriteAttr);
      result.addAttribute(getIsDataCacheAttrName(), isDataCacheAttr);
    }]>];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return memref()->getType().cast<MemRefType>();
    }

    /// Returns the affine map used to index the memref for this operation.
    AffineMap getAffineMap() { return getAffineMapAttr().getValue(); }
    AffineMapAttr getAffineMapAttr() {
      return getAttr(getMapAttrName()).cast<AffineMapAttr>();
    }

    /// Returns the AffineMapAttr associated with 'memref'.
    NamedAttribute getAffineMapAttrForMemRef(Value *mref) {
      assert(mref == memref());
      return {Identifier::get(getMapAttrName(), getContext()),
        getAffineMapAttr()};
    }

    /// Get affine map operands.
    operand_range getMapOperands() {
      return {operand_begin() + 1, operand_end()};
    }

    static StringRef getMapAttrName() { return "map"; }
    static StringRef getLocalityHintAttrName() { return "localityHint"; }
    static StringRef getIsWriteAttrName() { return "isWrite"; }
    static StringRef getIsDataCacheAttrName() { return "isDataCache"; }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def AffineTerminatorOp :
    Affine_Op<"terminator", [Terminator]> {
  let summary = "affine terminator operation";
  let description = [{
    Affine terminator is a special terminator operation for blocks inside affine
    loops and branches. It unconditionally transmits the control flow to the
    successor of the operation enclosing the region.

    This operation does _not_ have a custom syntax. However, affine control
    operations omit the terminator in their custom syntax for brevity.
  }];

  // No custom parsing/printing form.
  let parser = ?;
  let printer = ?;

  // Fully specified by traits.
  let verifier = ?;
}

#endif // AFFINE_OPS
