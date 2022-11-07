/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_TOPOLOGICAL_SORT_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_TOPOLOGICAL_SORT_H_

#include <vector>

#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace mlir {
namespace TF {

// A function that determines which op to emit next in the case of ties.
// The predecessor (which can be null) is the last op we emitted,
// and op is the candidate we're considering. A larger returned integer
// means the op has a higher chance of being emitted first.
typedef int (*PriorityFunction)(Operation *predecessor, Operation *op);

// Sort a block topologically, so that for all ops, all operands are
// available at the time of execution.  This is similar to MLIR's topological
// sort (lib/Transforms/TopologicalSort.cpp) but also takes a priority
// function to determine the next op to emit in the case of ambiguity. This
// makes it possible to group operations by certain attributes. For example,
// the order_by_dialect pass uses this function to group by dialect.
std::vector<Operation *> SortBlockTopologically(
    Block &block, PriorityFunction priorityFunction);

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_TOPOLOGICAL_SORT_H_
