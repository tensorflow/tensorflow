/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_common.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hlo {

// Verifies the source target pairs attached to collective permute.
LogicalResult VerifyCollectivePermuteSourceTargetPairs(
    Operation *op, DenseIntElementsAttr attr) {
  auto type = attr.getType().dyn_cast<RankedTensorType>();
  if (type.getRank() != 2)
    return op->emitError() << "expect source_target_pairs attribute to be of "
                              "rank 2, but got rank "
                           << type.getRank();
  if (type.getShape()[1] != 2)
    return op->emitError()
           << "expect source_target_pairs attribute of shape (N, 2), but got ("
           << type.getShape() << ")";
  // Check source target pairs for duplicate sources or targets.
  llvm::DenseSet<int64_t> sources;
  llvm::DenseSet<int64_t> targets;
  for (auto i = attr.begin(), e = attr.end(); i != e; ++i) {
    auto val = (*i).getSExtValue();
    if (i.getIndex() % 2 == 0) {
      bool is_unique = sources.insert(val).second;
      if (!is_unique)
        return op->emitError() << "duplicate sources not allowed.";
    } else {
      bool is_unique = targets.insert(val).second;
      if (!is_unique)
        return op->emitError() << "duplicate targets not allowed.";
    }
  }
  return success();
}

}  // namespace hlo
}  // namespace mlir
