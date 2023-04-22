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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_PER_FUNCTION_AGGREGATE_ANALYSIS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_PER_FUNCTION_AGGREGATE_ANALYSIS_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace TF {
namespace detail {

// This template defines an aggregate analysis base class, which analyzes a
// module but the analysis info is stored per function.
template <typename InfoT>
class PerFunctionAggregateAnalysis {
 public:
  using Info = InfoT;

  // Returns the analysis info for the given function.
  const Info& GetAnalysisForFunc(FuncOp func) const {
    auto it = info_map_.find(func);
    assert(it != info_map_.end());
    return it->second;
  }

 protected:
  // Since `InfoT` might be large, DenseMap is used instead of SmallDenseMap to
  // avoid stack overflow.
  llvm::DenseMap<FuncOp, InfoT> info_map_;
};

}  // namespace detail

// Base CRTP class to help write passes that are consumes a per-function
// aggregate analysis and operate on all non-extern functions (similar to a
// FunctionPass, but with no concurrency between functions). The derived classes
// need to provide a runOnFunction() method that accepts the function and the
// analysis information for that function.
template <typename DerivedT, typename AnalysisT>
class PerFunctionAggregateAnalysisConsumerPass
    : public PassWrapper<
          PerFunctionAggregateAnalysisConsumerPass<DerivedT, AnalysisT>,
          OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp op = this->getOperation();
    DerivedT& derived = *static_cast<DerivedT*>(this);
    auto& analysis = this->template getAnalysis<AnalysisT>();

    for (auto func : op.getOps<FuncOp>())
      if (!func.isExternal())
        derived.runOnFunction(func, analysis.GetAnalysisForFunc(func));
  }
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_PER_FUNCTION_AGGREGATE_ANALYSIS_H_
