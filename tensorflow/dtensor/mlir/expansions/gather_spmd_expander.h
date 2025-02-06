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

#ifndef TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_GATHER_SPMD_EXPANDER_H_
#define TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_GATHER_SPMD_EXPANDER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

class GatherCommonSPMDExpander : public SPMDExpanderBase {
 public:
  template <typename OpType>
  StatusOr<mlir::Operation*> ExpandOpHelper(mlir::Operation* op) {
    TF_ASSIGN_OR_RETURN(int64_t axis, GetAxis(op));
    TF_ASSIGN_OR_RETURN(uint64_t batch_dims, GetBatchDim(op));

    mlir::Value params = op->getOperand(0);
    mlir::Value indices = op->getOperand(1);

    TF_ASSIGN_OR_RETURN(std::vector<Layout> operand_layouts,
                        ExtractRequiredLayoutFromOperands(op));
    TF_ASSIGN_OR_RETURN(const Layout& output_layout,
                        ExtractRequiredSingleLayoutFromOp(op));

    const Layout& params_layout = operand_layouts[0];
    const Layout& indices_layout = operand_layouts[1];

    const int params_rank = ValueRank(params);
    const int indices_rank = ValueRank(indices);

    if (params_rank == -1)
      return errors::InvalidArgument("Missing rank for params input.");
    if (indices_rank == -1)
      return errors::InvalidArgument("Missing rank for indices input.");

    // Handle the case of negative axis.
    if (axis < 0) axis += params_rank;
    if (batch_dims < 0) batch_dims += indices_rank;
    mlir::OpBuilder builder(op);

    // Step 1: If the params are sharded on dim axis, an unconditional
    // all-concat is generated. Alternatively, we could do: all-concating
    // indices, followed by tf.Gather + slicing with correct masks.
    //
    // Currently we only support the case that the output layout matching the
    // params layout for all non-axis dim. Other cases needs either a slicing or
    // all-concat, which can be added later.
    {
      Mesh mesh = params_layout.mesh();
      std::vector<std::string> tgt_params_sharding_specs;
      tgt_params_sharding_specs.reserve(params_rank);
      // check the first half
      for (int i = 0; i < axis; ++i) {
        const std::string& dim_name = params_layout.sharding_spec(i);
        if (dim_name != output_layout.sharding_spec(i)) {
          return errors::InvalidArgument(
              llvm::formatv(
                  "input and output layout do not agree on non-axis dim {0}. "
                  "\n  params: {1}\n  output: {2}, axis: {3}",
                  i, params_layout.ToString(), output_layout.ToString(), axis)
                  .str());
        }
        tgt_params_sharding_specs.push_back(dim_name);
      }
      // Set replicated for `axis` dim.
      tgt_params_sharding_specs.push_back(Layout::kUnshardedDim);
      // Check the second half
      for (int i = axis + 1; i < params_rank; ++i) {
        const std::string& dim_name = params_layout.sharding_spec(i);
        // To align the param dim with output, we can think we insert
        // indices_rank
        // - batch_dims dims from indices and remove one from param (axis), so
        // the shifting is indices_rank - batch_dims - 1.
        if (dim_name !=
            output_layout.sharding_spec(i + indices_rank - batch_dims - 1)) {
          return errors::InvalidArgument(
              llvm::formatv(
                  "input and output layout do not agree on non-axis dim {0}. "
                  "\n  params: {1}\n  output: {2}, axis: {3}",
                  i, params_layout.ToString(), output_layout.ToString(), axis)
                  .str());
        }
        tgt_params_sharding_specs.push_back(dim_name);
      }

      if (!Layout::IsUnshardedDimension(params_layout.sharding_spec(axis))) {
        if (llvm::isa<mlir::TF::ResourceGatherOp>(op)) {
          return errors::InvalidArgument(
              "DTensor does not support sharded 0th dimension for the resource "
              "tensor for ResourceGatherOp. Please unshard dimension ",
              axis);
        }
        TF_ASSIGN_OR_RETURN(Layout tgt_params_layout,
                            Layout::GetLayout(params_layout.type(),
                                              tgt_params_sharding_specs, mesh));
        TF_ASSIGN_OR_RETURN(
            params,
            EmitAllGather(builder, params, params_layout, tgt_params_layout));
      }
    }

    // Step 2: Check the output layout. If it requires all-relayouting indices.
    // Do it.
    //
    // Indices shape is not big typically. Relayouting is expected to be cheap.
    {
      bool indices_relayout_needed = false;
      Mesh mesh = output_layout.mesh();
      std::vector<std::string> tgt_indices_sharding_specs;
      tgt_indices_sharding_specs.reserve(indices_rank);
      for (int i = 0; i < indices_rank; ++i) {
        int index_in_output;
        int index_in_indices;
        if (i < batch_dims) {
          // For dim within batch_dims, indices dim is aligning at the same
          // index as output.
          index_in_output = i;
          index_in_indices = i;
        } else {
          // For dim after batch_dims, we can remove batch_dims from outputs and
          // indices first, i.e., (i - batch_dims), add axis back, i.e., axis -
          // batch_dims, and then put batch_dims back, so the target position in
          // output is
          //
          //   i - batch_dims + axis - batch_dims + batch_dims
          //
          // which is as follows:
          index_in_output = i + axis - batch_dims;
          index_in_indices = i;
        }
        tgt_indices_sharding_specs.push_back(
            output_layout.sharding_spec(index_in_output));

        if (output_layout.sharding_spec(index_in_output) !=
            indices_layout.sharding_spec(index_in_indices)) {
          indices_relayout_needed = true;
        }
      }

      if (indices_relayout_needed) {
        TF_ASSIGN_OR_RETURN(
            Layout tgt_indices_layout,
            Layout::GetLayout(indices_layout.type(), tgt_indices_sharding_specs,
                              mesh));
        TF_ASSIGN_OR_RETURN(
            indices, EmitRelayout(indices, indices_layout, tgt_indices_layout));
      }
    }

    auto new_operands = llvm::to_vector<4>(op->getOperands());
    new_operands[0] = params;
    new_operands[1] = indices;

    mlir::Operation* new_gather =
        builder
            .create<OpType>(op->getLoc(), op->getResultTypes(),
                            mlir::ValueRange(new_operands), op->getAttrs())
            .getOperation();

    op->getResult(0).replaceAllUsesWith(new_gather->getResult(0));
    op->erase();

    return InferSPMDExpandedLocalShape(new_gather);
  }

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& input_layouts) override;

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& output_layouts) override;

  virtual StatusOr<int64_t> GetAxis(mlir::Operation* op) = 0;
  virtual StatusOr<uint64_t> GetBatchDim(mlir::Operation* op) = 0;
};

class GatherV2SPMDExpander : public GatherCommonSPMDExpander {
 public:
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  StatusOr<int64_t> GetAxis(mlir::Operation* op) override;

  StatusOr<uint64_t> GetBatchDim(mlir::Operation* op) override;
};

class ResourceGatherSPMDExpander : public GatherCommonSPMDExpander {
 public:
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  StatusOr<int64_t> GetAxis(mlir::Operation* op) override;

  StatusOr<uint64_t> GetBatchDim(mlir::Operation* op) override;
};

class GatherNdSPMDExpander : public SPMDExpanderBase {
 public:
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& input_layouts) override;

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& output_layouts) override;
};

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_GATHER_SPMD_EXPANDER_H_
