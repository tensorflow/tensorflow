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

#include "tensorflow/dtensor/mlir/expansions/top_k_spmd_expander.h"

#include <string>
#include <vector>

#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {

// layout -> layout[:-1] + unsharded
StatusOr<Layout> GetSuggestedLayout(const Layout& input_layout) {
  std::vector<std::string> layout_specs(input_layout.rank());

  for (int i = 0; i < input_layout.rank() - 1; ++i) {
    layout_specs[i] = input_layout.sharding_spec(i);
  }
  layout_specs[input_layout.rank() - 1] = Layout::kUnshardedDim;
  return Layout::GetLayout(layout_specs, input_layout.mesh());
}

StatusOr<mlir::Operation*> TopKSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto top_k_op = mlir::cast<mlir::TF::TopKV2Op>(op);
  mlir::Value input = top_k_op.getInput();
  TF_ASSIGN_OR_RETURN(auto input_layout, ExtractLayoutFromOperand(input));

  if (!input_layout)
    return errors::InvalidArgument(
        "layout of TopKV2Op must be known before SPMD expansion.");

  TF_ASSIGN_OR_RETURN(auto layouts, ExtractLayoutFromOp(op));
  for (const auto& layout : layouts) {
    if (layout.has_value() && !layout->IsLastDimReplicated()) {
      return errors::InvalidArgument(
          "The last dimensions of TopKV2Op outputs should be UNSHARDED.");
    }
  }
  mlir::OpBuilder builder(op);
  if (!input_layout->IsLastDimReplicated()) {
    TF_ASSIGN_OR_RETURN(Layout new_layout, GetSuggestedLayout(*input_layout));
    TF_ASSIGN_OR_RETURN(
        input, EmitAllGather(builder, input, *input_layout, new_layout));
    mlir::IRMapping mapping;
    mapping.map(op->getOperand(0), input);
    mlir::Operation* new_op = builder.clone(*op, mapping);
    new_op = InferSPMDExpandedLocalShape(new_op);

    op->getResult(0).replaceAllUsesWith(new_op->getResult(0));
    op->getResult(1).replaceAllUsesWith(new_op->getResult(1));
    op->erase();
    return new_op;
  }

  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>> TopKSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // If the input layout is missing, don't return an output layout.
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  TF_ASSIGN_OR_RETURN(Layout output_layout,
                      GetSuggestedLayout(input_layouts.lookup(0)));

  return llvm::DenseMap<int, Layout>({
      {0, output_layout},  // values
      {1, output_layout},  // indices
  });
}

StatusOr<llvm::DenseMap<int, Layout>> TopKSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // If the output values layout is missing, don't return an input layout.
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  TF_ASSIGN_OR_RETURN(Layout input_layout,
                      GetSuggestedLayout(output_layouts.lookup(0)));
  const Mesh& mesh = input_layout.mesh();

  return llvm::DenseMap<int, Layout>({
      {0, input_layout},                                // input
      {1, Layout::ReplicatedOnMesh(mesh, /*rank=*/0)},  // k
  });
}

}  // namespace dtensor
}  // namespace tensorflow
