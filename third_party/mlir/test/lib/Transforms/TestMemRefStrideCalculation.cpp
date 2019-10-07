//===- TestMemRefStrideCalculation.cpp - Pass to test strides computation--===//
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

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
/// Simple constant folding pass.
struct TestMemRefStrideCalculation
    : public FunctionPass<struct TestMemRefStrideCalculation> {
  void runOnFunction() override;
};
} // end anonymous namespace

// Traverse AllocOp and compute strides of each MemRefType independently.
void TestMemRefStrideCalculation::runOnFunction() {
  llvm::outs() << "Testing: " << getFunction().getName() << "\n";
  getFunction().walk([&](AllocOp allocOp) {
    auto memrefType = allocOp.getResult()->getType().cast<MemRefType>();
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    if (failed(getStridesAndOffset(memrefType, strides, offset))) {
      llvm::outs() << "MemRefType " << memrefType << " cannot be converted to "
                   << "strided form\n";
      return;
    }
    llvm::outs() << "MemRefType offset: ";
    if (offset == MemRefType::getDynamicStrideOrOffset())
      llvm::outs() << "?";
    else
      llvm::outs() << offset;
    llvm::outs() << " strides: ";
    interleaveComma(strides, llvm::outs(), [&](int64_t v) {
      if (v == MemRefType::getDynamicStrideOrOffset())
        llvm::outs() << "?";
      else
        llvm::outs() << v;
    });
    llvm::outs() << "\n";
  });
  llvm::outs().flush();
}

static PassRegistration<TestMemRefStrideCalculation>
    pass("test-memref-stride-calculation", "Test operation constant folding");
