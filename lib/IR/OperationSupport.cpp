//===- OperationSupport.cpp -----------------------------------------------===//
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
// This file contains out-of-line implementations of the support types that
// Instruction and related classes build on top of.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Block.h"

namespace mlir {

OperationState::OperationState(MLIRContext *context, Location location,
                               StringRef name)
    : context(context), location(location), name(name, context) {}

OperationState::OperationState(MLIRContext *context, Location location,
                               OperationName name)
    : context(context), location(location), name(name) {}

OperationState::OperationState(MLIRContext *context, Location location,
                               StringRef name, ArrayRef<Value *> operands,
                               ArrayRef<Type> types,
                               ArrayRef<NamedAttribute> attributes,
                               ArrayRef<Block *> successors,
                               MutableArrayRef<std::unique_ptr<Region>> regions,
                               bool resizableOperandList)
    : context(context), location(location), name(name, context),
      operands(operands.begin(), operands.end()),
      types(types.begin(), types.end()),
      attributes(attributes.begin(), attributes.end()),
      successors(successors.begin(), successors.end()) {
  for (std::unique_ptr<Region> &r : regions) {
    this->regions.push_back(std::move(r));
  }
}

Region *OperationState::addRegion() {
  regions.emplace_back(new Region);
  return regions.back().get();
}

void OperationState::addRegion(std::unique_ptr<Region> &&region) {
  regions.push_back(std::move(region));
}

} // end namespace mlir
