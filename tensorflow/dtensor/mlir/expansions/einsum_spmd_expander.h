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

#ifndef TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_EINSUM_SPMD_EXPANDER_H_
#define TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_EINSUM_SPMD_EXPANDER_H_

#include <string>
#include <vector>

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

class EinsumSPMDExpander : public SPMDExpanderBase {
 public:
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& input_layouts) override;

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& output_layouts) override;

 private:
  // This function will take the inputs, possibly slice or add AllConcat along
  // various mesh dimensions before the einsum operation takes place. This also
  // returns:
  // * The mesh dimensions, if any, that the output of the einsum should be
  //   summed along.
  // * The resulting output layout of the einsum operation, so we can insert an
  //   AllConcat/split to make the output have the desired layout.
  // * The new inputs to fed into the einsum.
  absl::Status MaybeRelayoutInputs(
      const std::vector<Layout>& input_layouts, mlir::Operation* op,
      const Layout& output_layout,
      absl::flat_hash_set<std::string>& reduce_dims, Layout& einsum_layout,
      std::vector<mlir::Value>& new_inputs);
};

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_EINSUM_SPMD_EXPANDER_H_
