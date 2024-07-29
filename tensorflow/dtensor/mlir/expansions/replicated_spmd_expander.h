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

#ifndef TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_REPLICATED_SPMD_EXPANDER_H_
#define TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_REPLICATED_SPMD_EXPANDER_H_

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/spmd_expander.h"

namespace tensorflow {
namespace dtensor {

// General SPMD Expander that enforces input(s) and output(s) are replicated.
class ReplicatedOpSPMDExpander : public SPMDExpanderBase {
 protected:
  // If true, then during ExpandOp, relayouts all operands and outputs
  // to be have replicated layout. If false, then will emit
  // an error if not all operand and output layouts are replicated after layout
  // propagation.
  //
  // This is needed because some ops like RngReadAndSkip need to enforce input
  // and output are replicated, while some ops don't need to enforce it, so
  // we can just relayout to replicated on those during ExpandOp.
  bool relayout_when_sharded_;

  // Expand the op to the local op after checking all layouts are replicated.
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  // Compute the layouts as always replicated.
  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& input_layouts) override;

  // Compute the layouts as always replicated.
  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& output_layouts) override;

 public:
  explicit ReplicatedOpSPMDExpander(bool relayout_when_sharded = false) {
    relayout_when_sharded_ = relayout_when_sharded;
  }

 private:
  // Relayouts all operands and outputs to a replicated layout.
  StatusOr<mlir::Operation*> ReplicatedRelayoutOperandsAndOutputs(
      mlir::Operation* op, const std::vector<Layout>& operand_layouts,
      const std::vector<Layout>& output_layouts);
};
}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_REPLICATED_SPMD_EXPANDER_H_
