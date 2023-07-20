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

#include "tensorflow/dtensor/mlir/expansions/nullary_spmd_expander.h"

#include "absl/strings/str_join.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/utils/array_container_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> NullarySPMDExpander::ExpandOp(mlir::Operation* op) {
  if (op->getNumResults() == 0) return op;

  bool all_operands_fully_replicated = true;
  TF_ASSIGN_OR_RETURN(auto op_layouts, ExtractLayoutFromOp(op));
  for (const auto& op_layout : op_layouts) {
    if (!op_layout)
      return errors::InvalidArgument(
          "Nullary op layouts must be known before SPMD expansion.");
    all_operands_fully_replicated =
        all_operands_fully_replicated && op_layout->IsFullyReplicated();
  }

  if (all_operands_fully_replicated) return op;

  if (auto const_op = mlir::dyn_cast<mlir::TF::ConstOp>(op)) {
    if (auto dense = const_op.getValue().dyn_cast<mlir::DenseElementsAttr>()) {
      if (dense.isSplat()) {
        // A 'splat' value for a DenseElementsAttr, has a single value for
        // all its elements. For these inputs, we don't need to slice. We just
        // need to update the shape of the attribute given the requested
        // sharding.
        assert(dense.getType().getRank() == op_layouts[0]->rank());
        auto shape = dense.getType().getShape();
        std::vector<int64_t> new_shape(dense.getType().getRank());
        for (int i = 0; i < op_layouts[0]->rank(); ++i) {
          const int num_shards = op_layouts[0]->num_shards_for_dim(i);
          if (shape[i] % num_shards != 0)
            return errors::InvalidArgument(
                "has output dimension size ", shape[i],
                " which is not evenly divisible by the number of shards ",
                num_shards, " in the layout for that dimension.");
          new_shape[i] = shape[i] / num_shards;
        }
        const_op.setValueAttr(mlir::DenseElementsAttr::get(
            mlir::RankedTensorType::get(new_shape,
                                        dense.getType().getElementType()),
            dense.getSplatValue<mlir::Attribute>()));
        return InferSPMDExpandedLocalShape(op);
      }
    }
  }

  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  llvm::SmallVector<mlir::Value, 4> generated_outputs;
  llvm::SmallVector<mlir::Type, 4> generated_types;

  mlir::OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  for (int i = 0; i < op_layouts.size(); ++i) {
    // Split each output to the correct layout by assuming the input is
    // replicated.
    TF_ASSIGN_OR_RETURN(
        const mlir::Value output,
        EmitAllScatter(builder, op->getOpResult(i),
                       Layout::ReplicatedOnMesh(op_layouts[i]->mesh(),
                                                op_layouts[i]->rank()),
                       *op_layouts[i], &newly_created_ops));
    generated_outputs.emplace_back(output);
    generated_types.emplace_back(output.getType());
  }

  auto identity_op = builder.create<mlir::TF::IdentityNOp>(
      op->getLoc(), generated_types, generated_outputs);

  newly_created_ops.insert(identity_op);
  for (int i = 0; i < op_layouts.size(); ++i)
    op->getOpResult(i).replaceAllUsesExcept(identity_op.getResult(i),
                                            newly_created_ops);

  return identity_op.getOperation();
}

StatusOr<llvm::DenseMap<int, Layout>> NullarySPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  auto enclosing_mesh = op->getParentOfType<mlir::tf_device::ClusterOp>();
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshFromOp(enclosing_mesh));
  if (!mesh.has_value())
    return errors::Internal("Failure in extracting mesh from Nullary Op.");
  llvm::DenseMap<int, Layout> output_layouts;
  // Nullary ops always output replicated layout for output values.
  for (auto i = 0; i < op->getNumResults(); ++i) {
    auto output_ranked_type =
        op->getResult(i).getType().dyn_cast<mlir::RankedTensorType>();
    if (!output_ranked_type) {
      return errors::InvalidArgument(
          llvm::formatv("requires output type to have statically known rank, "
                        "but got : {0}",
                        output_ranked_type)
              .str());
    }
    output_layouts[i] =
        Layout::ReplicatedOnMesh(*mesh, output_ranked_type.getRank());
  }
  return output_layouts;
}

StatusOr<llvm::DenseMap<int, Layout>>
NullarySPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // No operand inputs.
  return llvm::DenseMap<int, Layout>();
}

}  // namespace dtensor
}  // namespace tensorflow
