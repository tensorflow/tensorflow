//===- GPUDialect.cpp - MLIR Dialect for GPU Kernels implementation -------===//
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
// This file implements the GPU kernel-related dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/GPU/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;

StringRef GPUDialect::getDialectName() { return "gpu"; }

GPUDialect::GPUDialect(MLIRContext *context)
    : Dialect(getDialectName(), context) {
  addOperations<LaunchOp>();
}

//===----------------------------------------------------------------------===//
// LaunchOp
//===----------------------------------------------------------------------===//

static SmallVector<Type, 4> getValueTypes(ArrayRef<Value *> values) {
  SmallVector<Type, 4> types;
  types.reserve(values.size());
  for (Value *v : values)
    types.push_back(v->getType());
  return types;
}

void LaunchOp::build(Builder *builder, OperationState *result, Value *gridSizeX,
                     Value *gridSizeY, Value *gridSizeZ, Value *blockSizeX,
                     Value *blockSizeY, Value *blockSizeZ,
                     ArrayRef<Value *> operands) {
  // Add grid and block sizes as op operands, followed by the data operands.
  result->addOperands(
      {gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY, blockSizeZ});
  result->addOperands(operands);

  // Create a kernel body region with 12 + N arguments, where the first 12
  // arguments have `index` type and the rest have the same types as the data
  // operands.
  Region *kernelRegion = result->addRegion();
  Block *body = new Block();
  body->addArguments(std::vector<Type>(12, builder->getIndexType()));
  body->addArguments(getValueTypes(operands));
  kernelRegion->push_back(body);
}

Region &LaunchOp::getBody() { return getOperation()->getRegion(0); }

KernelDim3 LaunchOp::getBlockIds() {
  auto args = getBody().getBlocks().front().getArguments();
  return KernelDim3{args[0], args[1], args[2]};
}

KernelDim3 LaunchOp::getThreadIds() {
  auto args = getBody().getBlocks().front().getArguments();
  return KernelDim3{args[3], args[4], args[5]};
}

KernelDim3 LaunchOp::getGridSize() {
  auto args = getBody().getBlocks().front().getArguments();
  return KernelDim3{args[6], args[7], args[8]};
}

KernelDim3 LaunchOp::getBlockSize() {
  auto args = getBody().getBlocks().front().getArguments();
  return KernelDim3{args[9], args[10], args[11]};
}

LogicalResult LaunchOp::verify() {
  // Kernel launch takes 6 leading operands for grid/block sizes and transforms
  // them into 12 region arguments for block/thread identifiers and grid/block
  // sizes.
  if (!getBody().empty()) {
    Block &entryBlock = getBody().front();
    if (entryBlock.getNumArguments() != 6 + getNumOperands())
      return emitError("unexpected number of region arguments");
  }

  return success();
}
