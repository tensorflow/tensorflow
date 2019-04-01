//===- SliceOp.cpp - Implementation of the linalg SliceOp operation -------===//
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
// This file implements an IR operation to extract a "sub-View" from a ViewType
// in the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "linalg/SliceOp.h"
#include "linalg/Ops.h"
#include "linalg/RangeOp.h"
#include "linalg/RangeType.h"
#include "linalg/ViewOp.h"
#include "linalg/ViewType.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

using mlir::Builder;
using mlir::IndexType;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::OperationState;
using mlir::Type;
using mlir::Value;

using namespace linalg;

ViewOp linalg::ViewOrSliceOp::view() {
  return v->getDefiningOp()->dyn_cast<ViewOp>();
}
SliceOp linalg::ViewOrSliceOp::slice() {
  return v->getDefiningOp()->dyn_cast<SliceOp>();
}
linalg::ViewOrSliceOp::operator bool() {
  return static_cast<bool>(view()) || static_cast<bool>(slice());
}
unsigned linalg::ViewOrSliceOp::getRank() {
  assert(*this && "Not a ViewOp or a SliceOp!");
  return view() ? view().getRank() : slice().getRank();
}
ViewType linalg::ViewOrSliceOp::getViewType() {
  assert(*this && "Not a ViewOp or a SliceOp!");
  return view() ? view().getViewType() : slice().getViewType();
}
std::pair<Value *, unsigned>
linalg::ViewOrSliceOp::getRootIndexing(unsigned dim) {
  assert(*this && "Not a ViewOp or a SliceOp!");
  return view() ? view().getRootIndexing(dim) : slice().getRootIndexing(dim);
}
llvm::iterator_range<mlir::Operation::operand_iterator>
linalg::ViewOrSliceOp::getIndexings() {
  assert(*this && "Not a ViewOp or a SliceOp!");
  return view() ? view().getIndexings() : slice().getIndexings();
}
Value *linalg::ViewOrSliceOp::getSupportingMemRef() {
  assert(*this && "Not a ViewOp or a SliceOp!");
  return view() ? view().getSupportingMemRef() : slice().getSupportingMemRef();
}

// A view may itself be coming either from a ViewOp or from a SliceOp.
// TODO assert statically or dynamically that indexing is within the bounds of
// view.
void linalg::SliceOp::build(Builder *b, OperationState *result, Value *view,
                            Value *indexing, unsigned dim) {
  // Early sanity checks + extract rank.
  ViewOrSliceOp op(view);
  unsigned rank = op.getRank();
  ViewType viewType = op.getViewType();
  Type elementType = viewType.getElementType();

  result->addOperands({view, indexing});
  result->addAttribute(getSlicingDimAttrName(),
                       b->getIntegerAttr(b->getIndexType(), dim));
  if (indexing->getType().isa<RangeType>()) {
    // Taking a range slice does not decrease the rank, the view has the same
    // type.
    result->addTypes({viewType});
  } else {
    assert(indexing->getType().cast<IndexType>());
    result->addTypes(
        {linalg::ViewType::get(b->getContext(), elementType, rank - 1)});
  }
}

bool linalg::SliceOp::verify() {
  unsigned dim = getSlicingDim();
  if (dim >= getParentRank())
    return emitOpError("slicing dim must be in the [0 .. parent_rank) range");
  ViewOrSliceOp op(getOperand(0));
  if (!op)
    return emitOpError(
        "first operand must be of ViewType (i.e. a ViewOp or a SliceOp)");
  auto type = getOperand(1)->getType().dyn_cast<IndexType>();
  auto *inst = getOperand(1)->getDefiningOp();
  auto range = inst ? inst->dyn_cast<RangeOp>() : RangeOp();
  if (!range && !type)
    return emitOpError(
        "second operand must be of RangeType (i.e. a RangeOp) or IndexType");
  return false;
}

// Parsing of the linalg dialect is not supported in this tutorial.
bool linalg::SliceOp::parse(OpAsmParser *parser, OperationState *result) {
  assert(false && "NYI");
  return false;
}

// A SliceOp prints as:
//
// ```{.mlir}
//   linalg.slice %0[*, %i0] { dim : 1 } : !linalg<"view<f32>">
// ```
//
// Where %0 is an ssa-value holding a `view<f32xf32>`, %i0 is an ssa-value
// holding an index.
void linalg::SliceOp::print(OpAsmPrinter *p) {
  unsigned dim = getSlicingDim();
  *p << getOperationName() << " " << *getParentView() << "[";
  for (unsigned idx = 0, rank = getParentRank(); idx < rank; ++idx) {
    if (idx != dim) {
      *p << "*";
    } else {
      auto *v = getIndexing();
      if (v->getDefiningOp() && v->getDefiningOp()->isa<RangeOp>()) {
        *p << *v << "..";
      } else {
        *p << *v;
      }
    }
    *p << ((idx == rank - 1) ? "" : ", ");
  }
  *p << "] { " << getSlicingDimAttrName() << " : " << dim << " }"
     << " : " << getViewType();
}

ViewType linalg::SliceOp::getViewType() { return getType().cast<ViewType>(); }

unsigned linalg::SliceOp::getRank() { return getViewType().getRank(); }

mlir::Type linalg::SliceOp::getElementType() {
  return getViewType().getElementType();
}

ViewType linalg::SliceOp::getParentViewType() {
  ViewOrSliceOp op(getParentView());
  return op.getViewType();
}

unsigned linalg::SliceOp::getParentRank() {
  return getParentViewType().getRank();
}

mlir::Type linalg::SliceOp::getParentElementType() {
  return getParentViewType().getElementType();
}

Value *linalg::SliceOp::getBaseView() {
  Value *parent = getParentView();
  while (!parent->getDefiningOp()->isa<ViewOp>()) {
    parent = parent->getDefiningOp()->cast<SliceOp>().getParentView();
  }
  assert(parent && "null parent");
  return parent;
}

// We want to extract the range from the original ViewOp that this slice
// captures along `dim`. To achieve this, we want to walk back the chain of
// SliceOp and determine the first slice that constrains `dim`.
std::pair<Value *, unsigned> linalg::SliceOp::getRootIndexing(unsigned dim) {
  assert(dim < getRank());
  auto *view = getParentView();
  unsigned sliceDim = getSlicingDim();
  auto *indexing = getIndexing();
  if (indexing->getDefiningOp()) {
    if (auto rangeOp = indexing->getDefiningOp()->cast<RangeOp>()) {
      // If I sliced with a range and I sliced at this dim, then I'm it.
      if (dim == sliceDim) {
        return make_pair(rangeOp.getResult(), dim);
      }
      // Otherwise, I did not change the rank, just go look for `dim` into my
      // parent.
      ViewOrSliceOp op(view);
      return op.getRootIndexing(dim);
    }
  }
  assert(indexing->getType().isa<IndexType>());
  // If I get here, I indexed and reduced along the dim `sliceDim` from my
  // parent. I need to query my parent for `dim` or `dim + 1` depending on
  // whether dim > sliceDim or not.
  unsigned parentDim = dim > sliceDim ? dim + 1 : dim;
  ViewOrSliceOp op(view);
  return op.getRootIndexing(parentDim);
}

Value *linalg::SliceOp::getSupportingMemRef() {
  auto view = getBaseView()->getDefiningOp()->cast<ViewOp>();
  return view.getSupportingMemRef();
}

mlir::Operation::operand_range linalg::SliceOp::getIndexings() {
  return {this->getOperation()->operand_begin() + SliceOp::FirstIndexingOperand,
          this->getOperation()->operand_end()};
}
