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

#endif // LINALG_OPS