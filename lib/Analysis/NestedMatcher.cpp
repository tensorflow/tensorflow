//===- NestedMatcher.cpp - NestedMatcher Impl  ------------------*- C++ -*-===//
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

#include "mlir/Analysis/NestedMatcher.h"
#include "mlir/AffineOps/AffineOps.h"
#include "mlir/StandardOps/StandardOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

llvm::BumpPtrAllocator *&NestedMatch::allocator() {
  thread_local llvm::BumpPtrAllocator *allocator = nullptr;
  return allocator;
}

NestedMatch NestedMatch::build(Instruction *instruction,
                               ArrayRef<NestedMatch> nestedMatches) {
  auto *result = allocator()->Allocate<NestedMatch>();
  auto *children = allocator()->Allocate<NestedMatch>(nestedMatches.size());
  std::uninitialized_copy(nestedMatches.begin(), nestedMatches.end(), children);
  new (result) NestedMatch();
  result->matchedInstruction = instruction;
  result->matchedChildren =
      ArrayRef<NestedMatch>(children, nestedMatches.size());
  return *result;
}

llvm::BumpPtrAllocator *&NestedPattern::allocator() {
  thread_local llvm::BumpPtrAllocator *allocator = nullptr;
  return allocator;
}

NestedPattern::NestedPattern(Instruction::Kind k,
                             ArrayRef<NestedPattern> nested,
                             FilterFunctionType filter)
    : kind(k), nestedPatterns(), filter(filter), skip(nullptr) {
  if (!nested.empty()) {
    auto *newNested = allocator()->Allocate<NestedPattern>(nested.size());
    std::uninitialized_copy(nested.begin(), nested.end(), newNested);
    nestedPatterns = ArrayRef<NestedPattern>(newNested, nested.size());
  }
}

unsigned NestedPattern::getDepth() const {
  if (nestedPatterns.empty()) {
    return 1;
  }
  unsigned depth = 0;
  for (auto &c : nestedPatterns) {
    depth = std::max(depth, c.getDepth());
  }
  return depth + 1;
}

/// Matches a single instruction in the following way:
///   1. checks the kind of instruction against the matcher, if different then
///      there is no match;
///   2. calls the customizable filter function to refine the single instruction
///      match with extra semantic constraints;
///   3. if all is good, recursivey matches the nested patterns;
///   4. if all nested match then the single instruction matches too and is
///      appended to the list of matches;
///   5. TODO(ntv) Optionally applies actions (lambda), in which case we will
///      want to traverse in post-order DFS to avoid invalidating iterators.
void NestedPattern::matchOne(Instruction *inst,
                             SmallVectorImpl<NestedMatch> *matches) {
  if (skip == inst) {
    return;
  }
  // Structural filter
  if (inst->getKind() != kind) {
    return;
  }
  // Local custom filter function
  if (!filter(*inst)) {
    return;
  }

  if (nestedPatterns.empty()) {
    SmallVector<NestedMatch, 8> nestedMatches;
    matches->push_back(NestedMatch::build(inst, nestedMatches));
    return;
  }
  // Take a copy of each nested pattern so we can match it.
  for (auto nestedPattern : nestedPatterns) {
    SmallVector<NestedMatch, 8> nestedMatches;
    // Skip elem in the walk immediately following. Without this we would
    // essentially need to reimplement walkPostOrder here.
    nestedPattern.skip = inst;
    nestedPattern.match(inst, &nestedMatches);
    // If we could not match even one of the specified nestedPattern, early exit
    // as this whole branch is not a match.
    if (nestedMatches.empty()) {
      return;
    }
    matches->push_back(NestedMatch::build(inst, nestedMatches));
  }
}

static bool isAffineForOp(const Instruction &inst) {
  return cast<OperationInst>(inst).isa<AffineForOp>();
}

static bool isAffineIfOp(const Instruction &inst) {
  return isa<OperationInst>(inst) &&
         cast<OperationInst>(inst).isa<AffineIfOp>();
}

namespace mlir {
namespace matcher {

NestedPattern Op(FilterFunctionType filter) {
  return NestedPattern(Instruction::Kind::OperationInst, {}, filter);
}

NestedPattern If(NestedPattern child) {
  return NestedPattern(Instruction::Kind::OperationInst, child, isAffineIfOp);
}
NestedPattern If(FilterFunctionType filter, NestedPattern child) {
  return NestedPattern(Instruction::Kind::OperationInst, child,
                       [filter](const Instruction &inst) {
                         return isAffineIfOp(inst) && filter(inst);
                       });
}
NestedPattern If(ArrayRef<NestedPattern> nested) {
  return NestedPattern(Instruction::Kind::OperationInst, nested, isAffineIfOp);
}
NestedPattern If(FilterFunctionType filter, ArrayRef<NestedPattern> nested) {
  return NestedPattern(Instruction::Kind::OperationInst, nested,
                       [filter](const Instruction &inst) {
                         return isAffineIfOp(inst) && filter(inst);
                       });
}

NestedPattern For(NestedPattern child) {
  return NestedPattern(Instruction::Kind::OperationInst, child, isAffineForOp);
}
NestedPattern For(FilterFunctionType filter, NestedPattern child) {
  return NestedPattern(Instruction::Kind::OperationInst, child,
                       [=](const Instruction &inst) {
                         return isAffineForOp(inst) && filter(inst);
                       });
}
NestedPattern For(ArrayRef<NestedPattern> nested) {
  return NestedPattern(Instruction::Kind::OperationInst, nested, isAffineForOp);
}
NestedPattern For(FilterFunctionType filter, ArrayRef<NestedPattern> nested) {
  return NestedPattern(Instruction::Kind::OperationInst, nested,
                       [=](const Instruction &inst) {
                         return isAffineForOp(inst) && filter(inst);
                       });
}

// TODO(ntv): parallel annotation on loops.
bool isParallelLoop(const Instruction &inst) {
  auto loop = cast<OperationInst>(inst).cast<AffineForOp>();
  return loop || true; // loop->isParallel();
};

// TODO(ntv): reduction annotation on loops.
bool isReductionLoop(const Instruction &inst) {
  auto loop = cast<OperationInst>(inst).cast<AffineForOp>();
  return loop || true; // loop->isReduction();
};

bool isLoadOrStore(const Instruction &inst) {
  const auto *opInst = dyn_cast<OperationInst>(&inst);
  return opInst && (opInst->isa<LoadOp>() || opInst->isa<StoreOp>());
};

} // end namespace matcher
} // end namespace mlir
