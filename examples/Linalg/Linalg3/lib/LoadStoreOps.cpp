//===- LoadStoreOps.cpp - Implementation of linalg Load/Store operations --===//
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
// This file implements linalg.load and linalg.store operations which allow
// accessing memory through ViewType values.
//
//===----------------------------------------------------------------------===//

#include "linalg3/LoadStoreOps.h"
#include "linalg3/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

using llvm::ArrayRef;
using namespace mlir;
using namespace linalg;

////////////////////////////////////////////////////////////////////////////////
// LoadOp.
////////////////////////////////////////////////////////////////////////////////
void linalg::LoadOp::build(Builder *b, OperationState *result, Value *view,
                           ArrayRef<Value *> indices) {
  auto viewType = view->getType().cast<ViewType>();
  result->addOperands(view);
  result->addOperands(indices);
  result->addTypes(viewType.getElementType());
}

void linalg::LoadOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " " << *getView() << '[';
  p->printOperands(getIndices());
  *p << ']';
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << getViewType();
}

ParseResult linalg::LoadOp::parse(OpAsmParser *parser, OperationState *result) {
  llvm_unreachable("Parsing linalg dialect is not supported in this tutorial");
  return success();
}

LogicalResult linalg::LoadOp::verify() {
  if (getNumOperands() == 0)
    return emitOpError("expected a view to load from");

  auto viewType = getView()->getType().dyn_cast<ViewType>();
  if (!viewType)
    return emitOpError("first operand must be a view");

  if (getType() != viewType.getElementType())
    return emitOpError("result type must match element type of the view");

  if (getRank() != getNumOperands() - 1)
    return emitOpError("incorrect number of indices for load");

  for (auto *idx : getIndices())
    if (!idx->getType().isIndex())
      return emitOpError("index to load must have 'index' type");

  return success();
}

ViewType linalg::LoadOp::getViewType() {
  return getView()->getType().cast<ViewType>();
}

unsigned linalg::LoadOp::getRank() { return getViewType().getRank(); }

////////////////////////////////////////////////////////////////////////////////
// StoreOp.
////////////////////////////////////////////////////////////////////////////////
void linalg::StoreOp::build(Builder *b, OperationState *result,
                            Value *valueToStore, Value *view,
                            ArrayRef<Value *> indices) {
  result->addOperands(valueToStore);
  result->addOperands(view);
  result->addOperands(indices);
}

void linalg::StoreOp::print(OpAsmPrinter *p) {
  *p << getOperationName() << " " << *getValueToStore();
  *p << ", " << *getView() << '[';
  p->printOperands(getIndices());
  *p << ']';
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << getViewType();
}

ParseResult linalg::StoreOp::parse(OpAsmParser *parser,
                                   OperationState *result) {
  assert(false && "NYI");
  return success();
}

LogicalResult linalg::StoreOp::verify() {
  if (getNumOperands() < 2)
    return emitOpError("expected a value to store and a view");

  // Second operand is a memref type.
  auto viewType = getView()->getType().dyn_cast<ViewType>();
  if (!viewType)
    return emitOpError("second operand must be a view");

  // First operand must have same type as memref element type.
  if (getValueToStore()->getType() != viewType.getElementType())
    return emitOpError("first operand must have same element type as the view");

  if (getNumOperands() != 2 + viewType.getRank())
    return emitOpError("store index operand count not equal to view rank");

  for (auto *idx : getIndices())
    if (!idx->getType().isIndex())
      return emitOpError("index to store must have 'index' type");

  return success();
}

unsigned linalg::StoreOp::getRank() { return getViewType().getRank(); }

ViewType linalg::StoreOp::getViewType() {
  return getView()->getType().cast<ViewType>();
}
