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

#ifndef TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_MATMUL_SPMD_EXPANDER_H_
#define TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_MATMUL_SPMD_EXPANDER_H_

#include <optional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/core/platform/status.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/spmd_expander.h"

namespace tensorflow {
namespace dtensor {

class MatMulSPMDExpander : public SPMDExpanderBase {
 public:
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& input_layouts) override;

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& output_layouts) override;

 private:
  StatusOr<Layout> OutputLayoutAndReducedDims(
      bool allow_unknown_layouts, mlir::Operation* op,
      absl::flat_hash_set<std::string>* reduced_dims,
      std::optional<Layout>* left, std::optional<Layout>* right);

  // This function prepares the inputs (x, y or a, b) to (Batch)MatMul by
  // possibly computing a new layout for each input that allows us to simply
  // emit a local (Batch)MatMul op. Once the layouts are computed, the function
  // calls EmitRelayout to transform from left_layout, right_layout to the
  // newly computed layouts.
  // The left and right arguments are set to the mlir::Values representing the
  // tensors with the possibly new layout.
  // reduced_dim will contain the dim that must be reduced on after the local
  // MatMul. It may be set to Layout::kUnsharded if no reduction is needed.
  // matmul_layout will be set to the layout of the output of the local matmul
  // (after the above reduction). This may be different from the desired output
  // layout.
  absl::Status MaybeRelayoutInputs(
      mlir::Operation* op, const Layout& left_layout, bool left_transposed,
      const Layout& right_layout, bool right_transposed,
      const Layout& output_layout, std::string& reduced_dim,
      Layout& matmul_layout, mlir::Value& left, mlir::Value& right);
};

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_MATMUL_SPMD_EXPANDER_H_
