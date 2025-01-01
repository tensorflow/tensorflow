/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace tf_test {
namespace {

// A pass that annotates each operation with a resource type result with the
// aliasing values for each such result. Each value is assigned a unique ID, and
// that ID is used to annotate the operations.
struct TestResourceAliasAnalysis
    : public TF::PerFunctionAggregateAnalysisConsumerPass<
          TestResourceAliasAnalysis, TF::ResourceAliasAnalysis> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestResourceAliasAnalysis)

  StringRef getArgument() const final {
    return "tf-test-resource-alias-analysis";
  }

  StringRef getDescription() const final {
    return "Add remarks based on resource alias analysis result, for testing "
           "purpose.";
  }

  void runOnFunction(func::FuncOp func,
                     const TF::ResourceAliasAnalysis::Info& analysis) {
    int64_t next_id = 0;
    llvm::SmallDenseMap<Value, int64_t, 8> ids;

    auto assign_id = [&](Value value) {
      if (ids.find(value) == ids.end()) ids.insert({value, next_id++});
    };

    auto get_id = [&](Value value) -> int64_t {
      auto it = ids.find(value);
      assert(it != ids.end());
      return it->second;
    };

    auto print_aliases = [&](InFlightDiagnostic& diag, Value value) {
      diag << ", ID " << get_id(value) << " : ";
      if (analysis.IsUnknownResource(value)) {
        diag << "Unknown";
      } else {
        auto aliases = llvm::to_vector<4>(analysis.GetResourceAliases(value));
        llvm::sort(aliases,
                   [&](Value v1, Value v2) { return get_id(v1) < get_id(v2); });
        llvm::interleaveComma(aliases, diag,
                              [&](Value v) { diag << get_id(v); });
      }
    };

    // Assign a unique ID to each value seen in this function.
    func.walk([&](Operation* op) {
      // For all attached regions, assign ID to the region arguments.
      for (Region& region : op->getRegions()) {
        for (auto region_arg : TF::filter_resources(region.getArguments()))
          assign_id(region_arg);
      }

      // Assign ID for all results.
      for (auto result : TF::filter_resources(op->getResults()))
        assign_id(result);
    });

    // Now walk each operation, and annotate it wil remarks for aliases for
    // each resource type result
    func.walk([&](Operation* op) {
      // For all attached regions, assign ID to the region arguments.
      for (Region& region : op->getRegions()) {
        for (auto region_arg : TF::filter_resources(region.getArguments())) {
          InFlightDiagnostic diag = op->emitRemark("Region #")
                                    << region.getRegionNumber() << ", Arg #"
                                    << region_arg.getArgNumber();
          print_aliases(diag, region_arg);
        }
      }

      for (auto result : TF::filter_resources(op->getResults())) {
        InFlightDiagnostic diag = op->emitRemark("Result #")
                                  << result.getResultNumber();
        print_aliases(diag, result);
      }
    });
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTestResourceAliasAnalysisPass() {
  return std::make_unique<TestResourceAliasAnalysis>();
}

}  // namespace tf_test
}  // namespace mlir
