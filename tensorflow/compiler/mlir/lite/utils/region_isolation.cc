/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/utils/region_isolation.h"

#define DEBUG_TYPE "tfl_isolate_regions"

#include <optional>

#include "absl/strings/str_format.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project

namespace mlir {
namespace TFL {

std::optional<llvm::SetVector<Value>> IsolateRegions(Operation* op_with_regions,
                                                     OpBuilder& b) {
  LLVM_DEBUG(
      llvm::dbgs() << absl::StrFormat("Isolating Op with %u regions...\n",
                                      op_with_regions->getNumRegions()));
  LLVM_DEBUG(op_with_regions->print(llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  if (op_with_regions->getNumRegions() == 0) {
    return {};
  }

  llvm::SetVector<Value> shared_signature;
  getUsedValuesDefinedAbove(op_with_regions->getRegions(), shared_signature);

  for (auto& reg : op_with_regions->getRegions()) {
    if (!reg.hasOneBlock()) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Region isolation only supports regions with a single block\n");
      return {};
    }
    auto& block = reg.getBlocks().front();
    if (block.getNumArguments() != 0) {
      LLVM_DEBUG(llvm::dbgs() << "Region isolation reguires empty blargs\n");
    }
    for (auto val : shared_signature) {
      auto blarg = block.addArgument(val.getType(), b.getUnknownLoc());
      replaceAllUsesInRegionWith(val, blarg, reg);
    }
  }

  return shared_signature;
}

}  // namespace TFL
}  // namespace mlir
