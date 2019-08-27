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

#ifdef LINALG_OPS
#else
#define LINALG_OPS

include "mlir/Dialect/AffineOps/AffineOpsBase.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"

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

def BufferAllocOp :
    Linalg_Op<"buffer_alloc">,
    Arguments<(ins Variadic<Index>:$size, OptionalAttr<I64Attr>:$alignment)>,
    Results<(outs Buffer)> {
  let summary = "buffer allocation operation";
  let description = [{
    The "buffer_alloc" op creates a 1-D linalg.buffer of the specified type,
    upon which a base view can be laid out to give it indexing semantics.
    "buffer_alloc" takes a single argument, the size of the buffer to allocate
    (in number of elements).
    An optional alignment attribute may be specified in which case the actual
    underlying allocation size may be increased. The base pointer is guaranteed
    to be a multiple of `alignment`. Such an alignment must be a positive power
    of 2.

    Examples:

        %0 = linalg.buffer_alloc(%arg0) : !linalg.buffer<?xf32>

        %1 = linalg.buffer_alloc(%arg0) { alignment = 16 } :
          !linalg.buffer<?xf32>

    The size argument may be omitted if it is statically known, in which case it
    must be reflected in the type.

    Example:

        %0 = linalg.buffer_alloc() : !linalg.buffer<4xf32>
  }];
  let builders = [
    OpBuilder<
      "Builder *b, OperationState *result, BufferType bufferType", [{
          result->addTypes(bufferType);
       }]>,
    OpBuilder<
      "Builder *b, OperationState *result, BufferType bufferType, "
      "unsigned alignment", [{
        build(b, result, bufferType);
        if (alignment != 0)
          result->addAttribute(BufferAllocOp::getAlignmentAttrName(),
                               b->getI64IntegerAttr(alignment));
      }]>,
    OpBuilder<
      "Builder *b, OperationState *result, BufferType bufferType, "
      "Value *size, unsigned alignment", [{
        if (alignment == 0)
          return build(b, result, bufferType, size);
        build(b, result, bufferType, size, b->getI64IntegerAttr(alignment));
      }]>,
    OpBuilder<
      "Builder *b, OperationState *result, BufferType bufferType, Value *size",
      [{
        result->addOperands(size);
        result->addTypes(bufferType);
      }]>
  ];
  let extraClassDeclaration = [{
    static StringRef getAlignmentAttrName() { return "alignment"; }
    BufferType getBufferType() { return getType().cast<BufferType>(); }
    Type getElementType() { return getBufferType().getElementType(); }
  }];
}

def BufferDeallocOp :
    Linalg_Op<"buffer_dealloc">,
    Arguments<(ins Buffer:$buffer)>,
    Results<(outs)> {
  let summary = "buffer allocation operation";
  let description = [{
    The "buffer_dealloc" op frees a 1-D linalg.buffer of the specified type.

    Example:

        linalg.buffer_dealloc %0 : !linalg.buffer<f32>
  }];
  let extraClassDeclaration = [{
    BufferType getBufferType() {
      return buffer()->getType().cast<BufferType>();
    }
  }];

  // Fully specified by traits.
  let verifier = ?;
}

def BufferSizeOp :
    Linalg_Op<"buffer_size", [NoSideEffect]>,
    Arguments<(ins Buffer:$buffer)>,
    Results<(outs Index)> {
  let summary = "buffer size operation";
  let description = [{
    The "linalg.buffer_size" operation takes a linalg.buffer and returns an
    "index".

    Example:

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
     returns the size of the specified dimension.

     Example:

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

  let hasCanonicalizer = 1;
}

def LoadOp :
    Linalg_Op<"load"
      // TODO(ntv): activate once ViewType can be made a ShapeType (i.e.
      // shape type is extensible or standard adopts a reasonable view type).
      // , [ PredOpTrait<"operand and result have same element type",
      //             TCresVTEtIsSameAsOpBase<0, 0>>]
      >,
    Arguments<(ins View:$view, Variadic<Index>:$indices)>,
    Results<(outs AnyType:$value)> {
  let summary = "Read an elemental value from a view at a certain index";
  let description = [{
    The `linalg.load` op reads an elemental value from a view at a certain
    index. This is the counterpart of other load ops but operating on ViewType.

    Example:

       %0 = linalg.load %V[%c0] : !linalg.view<?xf32>
  }];
  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Value *view, "
    "ArrayRef<Value*> indices",
    [{
      auto viewType = view->getType().cast<ViewType>();
      build(builder, result, viewType.getElementType(), view, indices);
    }]>];
  let extraClassDeclaration = [{
    unsigned getRank() { return getViewType().getRank(); }
    ViewType getViewType() { return view()->getType().cast<ViewType>(); }
  }];
}

def RangeOp :
    Linalg_Op<"range", [NoSideEffect]>,
    Arguments<(ins Index:$min, Index:$max, Index:$step)>,
    Results<(outs Range)> {
  let summary = "Create a range type value, used to create views";
  let description = [{
    The `linalg.range` op creates a linalg.range from 3 values of type `index`
    that represent the min, max and step values of the range.

    Example:

      %3 = linalg.range %0:%1:%2 : !linalg.range
  }];
  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Value *min, Value *max, "
    "Value *step",
    [{
      auto rangeType = RangeType::get(builder->getContext());
      build(builder, result, rangeType, min, max, step);
    }]>];

  // Fully specified by traits.
  let verifier = ?;
}

def SliceOp : Linalg_Op<"slice", [NoSideEffect]>,
    Arguments<(ins View:$view, Variadic<AnyTypeOf<[Range, Index]>>:$indexings)>,
    Results<(outs View)> {
  let summary = "Produce a linalg.view which is a subview of a base view.";
  let description = [{
    The "linalg.slice" op produces a linalg.view which is a subview of a given
    base view. This allows defining a subregion within the underlying buffer to
    operate on only a subset of the buffer.

    A "linalg.slice" op takes a view and a variadic number of indexings and
    produces a linalg.view of the same elemental type. An indexing is either:
      1. a linalg.range, in which case it does not reduce the rank of the parent
         view.
      2. an index, in which case it reduces the rank of the parent view by one.

    If an indexing extends past the size of the view, the slice operation
    automatically truncates it to be within the bounds.

    Examples:

      1. rank-preserving slice:

        %4 = linalg.slice %0[%1, %2] : !linalg.view<?x?xf32>, !linalg.range,
               !linalg.range, !linalg.view<?x?xf32>

      2. rank-reducing slice (from 2-D to 1-D):

        %4 = linalg.slice %0[%1, %2] : !linalg.view<?x?xf32>, index,
               !linalg.range, !linalg.view<?xf32>

      3. rank-reducing slice (from 2-D to 0-D):

        %4 = linalg.slice %0[%1, %2] : !linalg.view<?x?xf32>, index, index,
               !linalg.view<f32>
  }];

  let builders = [OpBuilder<
    "Builder *b, OperationState *result, Value *base, "
    "ArrayRef<Value *> indexings">];

  let extraClassDeclaration = [{
    enum { FirstIndexingOperand = 1 };
    unsigned getRank() { return getViewType().getRank(); }
    Type getElementType() { return getViewType().getElementType(); }
    ViewType getViewType() { return getType().cast<ViewType>(); }
    unsigned getBaseViewRank() { return getBaseViewType().getRank(); }
    ViewType getBaseViewType() { return view()->getType().cast<ViewType>(); }

    // Get the underlying indexing at a given rank.
    Value *indexing(unsigned rank) { return *(indexings().begin() + rank); }

    // Get the subset of indexings that are of RangeType.
    SmallVector<Value *, 8> getRanges() {
      llvm::SmallVector<Value *, 8> res;
      for (auto *operand : indexings())
        if (!operand->getType().isa<IndexType>())
          res.push_back(operand);
      return res;
    }
  }];
}

def StoreOp :
    Linalg_Op<"store"
      // TODO(ntv): activate once ViewType can be made a ShapeType (i.e.
      // shape type is extensible or standard adopts a reasonable view type).
      // , [ PredOpTrait<"value to store and view have the same element type",
      //             TCopVTEtIsSameAs<0, 1>>]
      >,
    Arguments<(ins AnyType:$value, View:$view, Variadic<Index>:$indices)>,
    Results<(outs)> {
  let summary = "Write an elemental value in a view at a certain index";
  let description = [{
    The `linalg.store` op writes an elemental value in a view at a certain
    index. This is the counterpart of other store ops but operating on ViewType.

    Example:

      linalg.store %f, %V[%c0] : !linalg.view<?xf32>
  }];
  let extraClassDeclaration = [{
    unsigned getRank() { return getViewType().getRank(); }
    ViewType getViewType() { return view()->getType().cast<ViewType>(); }
  }];
}

def SubViewOp : Linalg_Op<"subview", [NoSideEffect]>,
    Arguments<(ins View:$view, Variadic<Index>:$ranges)>,
    Results<(outs View)> {
  let summary = "subview operation";
  let description = [{
    The "linalg.subview" op produces a linalg.view which is a subview of a given
    base view. This allows defining a subregion within the underlying buffer.

    The "linalg.subview" operation takes a base view, a list of indices and
    returns a new linalg.view of the same type that is contained within the
    view. This operation is equivalent to a non-rank-reducing slice operation.
    The main difference is the operands are all of type `index` and no
    intermediate linalg.range operations are required. A "linalg.subview" is
    thus a specialized linalg.slice with a higher level of abstraction.

    Similary to linalg.slice, if a range extends past the size of the base view,
    the slice operation automatically truncates it to be within the bounds of
    the view.

    Example:

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
    auto rank = getViewType().getRank();
    if (getNumOperands() != 3 * rank + 1)
      return emitOpError("expected a view followed by ") << (3 * rank) <<
        " indices specifying a range for each dimension";
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

def TransposeOp : Linalg_Op<"transpose", [NoSideEffect]>,
    Arguments<(ins View:$view, AffineMapAttr:$permutation)>,
    Results<(outs View)> {
  let summary = "transpose operation produces a new view (metadata-only)";
  let description = [{
    The "linalg.transpose" op produces a linalg.view whose sizes and strides are
    a permutation of the original. This is a pure metadata transformation.

    Example:

       %1 = linalg.transpose %0 (i, j) -> (j, i) : !linalg.view<?x?xf32>
  }];

  let builders = [OpBuilder<
    "Builder *b, OperationState *result, Value *view, "
    "AffineMapAttr permutation, ArrayRef<NamedAttribute> attrs = {}">];

  let verifier = [{
    if (!permutation().isPermutation())
      return emitOpError("expected a permutation map");
    if (permutation().getNumDims() != getViewType().getRank())
      return emitOpError("expected a permutation map of same rank as the view");
    return success();
  }];

  let extraClassDeclaration = [{
    static StringRef getPermutationAttrName() { return "permutation"; }
    ViewType getViewType() { return view()->getType().cast<ViewType>(); }
  }];
}

def ViewOp : Linalg_Op<"view", [NoSideEffect]>,
    Arguments<(ins Buffer:$buffer, Variadic<Range>:$ranges)>,
    Results<(outs View)> {
  let summary = "view operation";
  let description = [{
    The "linalg.view" op produces a linalg.view which is a multi-dimensional
    range abstraction on top of an underlying linalg.buffer. This gives an
    indexing structure to an otherwise non-indexable linalg.buffer.

    A "linalg.view" takes a buffer and a variadic number of ranges and produces
    a `view` of rank the number of ranges. The elemental type may not match the
    buffer element type:

    Example:

       %1 = linalg.buffer_alloc %0 : !linalg.buffer<f32>
       %2 = linalg.range %arg2:%arg3:%arg4 : !linalg.range
       %3 = linalg.view %1[%2, %2] : !linalg.view<?x?xvector<4xf32>>
  }];

  let builders = [OpBuilder<
    "Builder *b, OperationState *result, Value *buffer, "
    "ArrayRef<Value *> ranges, Type resultType = Type(), "
    "ArrayRef<NamedAttribute> attrs = {}">];

  let verifier = [{
    if (getViewType().getRank() != llvm::size(ranges()))
      return emitOpError("the view rank must be the number of its ranges");
    return success();
  }];

  let extraClassDeclaration = [{
    enum { FirstIndexingOperand = 1 };
    unsigned getRank() { return getViewType().getRank(); }
    Type getElementType() { return getViewType().getElementType(); }
    ViewType getViewType() { return getType().cast<ViewType>(); }
    /// Get the underlying indexing at a given rank.
    Value *getRange(unsigned rank) {
      assert(rank < getRank() && "rank overflow");
      return *(ranges().begin() + rank);
    }
  }];
}

def YieldOp : Linalg_Op<"yield", [NativeOpTrait<"IsTerminator">]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Linalg yield operation";
  let description = [{
    "linalg.yield" is a special terminator operation for blocks inside regions
    in linalg ops. It returns values to the immediately enclosing linalg op.

    Example:

       linalg.yield %f0, %f1 : f32, f32
  }];
}

#endif // LINALG_OPS
