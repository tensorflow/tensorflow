//===- LinalgOps.td - Linalg dialect ops -------------------*- tablegen -*-===//
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
// This is the operation definition file for linear algebra operations.
//
//===----------------------------------------------------------------------===//

include "mlir/Linalg/IR/LinalgBase.td"

#ifdef LINALG_OPS
#else
#define LINALG_OPS

// Base class for Linalg dialect ops that do not correspond to library calls.
class Linalg_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Linalg_Dialect, mnemonic, traits> {
  // For every linalg op, there needs to be a:
  //   * void print(OpAsmPrinter *p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser *parser,
  //                                         OperationState *result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def BufferSizeOp :
    Linalg_Op<"buffer_size", [NoSideEffect]>,
    Arguments<(ins Buffer)>,
    Results<(outs Index)> {
  let summary = "buffer size operation";
  let description = [{
    The "linalg.buffer_size" operation takes a linalg.buffer and returns an
    "index". For example:

       %0 = linalg.buffer_size %arg0 : !linalg.buffer<f32>
  }];
  // Fully specified by traits.
  let verifier = ?;
}

def DimOp : Linalg_Op<"dim", [NoSideEffect]>,
    Arguments<(ins View:$view, APIntAttr:$index)>,
    Results<(outs Index)> {
  let summary = "dimension index operation";
  let description = [{
    The "linalg.dim" operation takes a linalg.view and returns an
    "index". It requires a single integer attribute named "index". It
     returns the size of the specified dimension. For example:

      %1 = linalg.dim %0, 2 : view<?x?x?xf32>
  }];

  let verifier = [{
    if (getIndex() >= getViewType().getRank())
      return emitOpError("index is out of range");
    return success();
  }];

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Value *view, unsigned index",
    [{
      result->addOperands(view);
      result->addAttribute(
        "index", builder->getIntegerAttr(builder->getIndexType(), index));
      result->types.push_back(builder->getIndexType());
    }]>];

  let extraClassDeclaration = [{
    unsigned getIndex() {
      return getAttrOfType<IntegerAttr>("index").getValue().getZExtValue();
    }
    ViewType getViewType() { return getOperand()->getType().cast<ViewType>(); }
  }];
}

def RangeIntersectOp : Linalg_Op<"range_intersect", [NoSideEffect]>,
    Arguments<(ins Range, Range)>,
    Results<(outs Range)> {
  let summary = "range intersection operation";
  let description = [{
    The "linalg.range_intersect" operation takes two linalg.range and returns a
    linalg.range that represents their intersection. This assumes both steps
    are one for now. For example:

      %2 = linalg.range_intersect %0, %1 : !linalg.range
  }];

  // Fully verified by traits.
  let verifier = ?;

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Value *range1, Value *range2",
    [{
      result->addOperands({range1, range2});
      result->types.push_back(builder->getType<RangeType>());
    }]>];
}

def TerminatorOp :
    Linalg_Op<"terminator", [NativeOpTrait<"IsTerminator">]> {
  let summary = "linalg terminator operation";
  let description = [{
    "linalg.terminator" is a special terminator operation for blocks inside
    linalg loops and branches. It unconditionally transmits the control flow to
    the successor of the operation enclosing the region.

    This operation does _not_ have a custom syntax. However, linalg control
    operations omit the terminator in their custom syntax for brevity.

       linalg.terminator
  }];

  // No custom parsing/printing form.
  let parser = ?;
  let printer = ?;

  // Fully specified by traits.
  let verifier = ?;
}

def SubViewOp : Linalg_Op<"subview", [NoSideEffect]>,
    Arguments<(ins View:$view, Variadic<Index>:$ranges)>,
    Results<(outs View)> {
  let summary = "subview operation";
  let description = [{
    The "linalg.subview" operation takes a linalg.view, a list of indices and
    returns a new linalg.view of the same type that is contained within the
    operand view.
    This operation is equivalent to a non-rank-reducing slice operation. The
    main difference is the operands are all of type `index` and no intermediate
    linalg.range operations are required. A "linalg.subview" is thus a
    specialized linalg.slice with a higher level of abstraction.

      %1 = linalg.subview %0[%1, %2, %3, %4, %5, %6] : view<?x?xf32>

  }];
  // TODO(ntv) evolve syntax towards:
  //   linalg.subview %0[%1:%2:%3][%4:%5:%6] : view<?x?xf32>

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Value *view, "
    "ArrayRef<Value *> ranges",
    [{
      result->addOperands(view);
      result->addOperands(ranges);
      result->types.push_back(view->getType());
    }]>];

  let verifier = [{
    auto numRanges = (getNumOperands() - 1) / 3;
    if (getNumOperands() != 3 * numRanges + 1 ||
        numRanges != getViewType().getRank())
      return emitOpError("expected a view followed by 3 indices specifying ") <<
        "a range for each dimension";
    return success();
  }];

  let extraClassDeclaration = [{
    Value *getView() { return getOperand(0); }
    ViewType getViewType() { return getView()->getType().cast<ViewType>(); }
    struct Range { Value *min; Value *max; Value *step; };
    Range getRange(unsigned i) {
      return Range{
        getOperand(1 + 3*i), getOperand(1 + 3*i + 1), getOperand(1 + 3*i + 2)};
    }
    SmallVector<Range, 8> getRanges() {
      SmallVector<Range, 8> res;
      unsigned rank = getViewType().getRank();
      res.reserve(rank);
      for (unsigned i = 0; i < rank; ++i)
        res.push_back(getRange(i));
      return res;
    }
    // This requires `SubViewOp` to be declared, in the future it should be
    // folded into the builders.
    static void build(Builder *builder, OperationState *result, Value *view,
        ArrayRef<SubViewOp::Range> ranges) {
      result->addOperands(view);
      for (auto r : ranges)
        result->addOperands({r.min, r.max, r.step});
      result->types.push_back(view->getType());
    }
  }];
}

#endif // LINALG_OPS
