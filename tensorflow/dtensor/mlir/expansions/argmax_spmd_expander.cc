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

#include "tensorflow/dtensor/mlir/expansions/argmax_spmd_expander.h"

#include <cstdint>
#include <string>
#include <utility>

#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {
namespace {

StatusOr<Layout> ComputeResultLayout(mlir::Operation* op,
                                     const Layout& input_layout) {
  if (!mlir::isa<mlir::TF::ArgMaxOp>(op))
    return errors::Unimplemented("SPMD expansion for op type: ", OpName(op),
                                 " not yet implemented.");

  auto argmax_op = llvm::cast<mlir::TF::ArgMaxOp>(op);
  const auto input_rank = ValueRank(argmax_op.getInput());
  TF_ASSIGN_OR_RETURN(int64_t axis,
                      ExtractConstIntFromValue(argmax_op.getDimension()));

  if (axis < 0) axis += input_rank;

  LayoutProto output_layout_proto;
  TF_ASSIGN_OR_RETURN(*output_layout_proto.mutable_mesh_config(),
                      input_layout.mesh().ToProto());

  for (int i = 0; i < input_rank; ++i) {
    if (i != axis)
      output_layout_proto.add_sharding_specs()->set_sharding_spec(
          input_layout.sharding_spec(i));
  }
  return Layout::FromProto(output_layout_proto).value();
}
}  // namespace

StatusOr<mlir::Operation*> ArgMaxSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto argmax_op = llvm::cast<mlir::TF::ArgMaxOp>(op);
  TF_ASSIGN_OR_RETURN(int64_t axis,
                      ExtractConstIntFromValue(argmax_op.getDimension()));
  TF_ASSIGN_OR_RETURN(auto input_layout,
                      ExtractLayoutFromOperand(argmax_op.getInput()));
  TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(argmax_op));
  if (!input_layout || !output_layout)
    return errors::InvalidArgument(
        OpName(op), " is missing layouts during SPMD Expansion.");

  mlir::Value input = argmax_op.getInput();
  const auto input_rank = ValueRank(input);

  TF_ASSIGN_OR_RETURN(auto input_shape, GetShapeOfValue(input));

  if (input_rank == -1) return errors::Unimplemented("missing rank for input.");
  if (axis < 0) axis += input_rank;

  mlir::OpBuilder builder(op);
  {
    LayoutProto tgt_input_layout_proto;
    TF_ASSIGN_OR_RETURN(*tgt_input_layout_proto.mutable_mesh_config(),
                        input_layout->mesh().ToProto());

    for (int i = 0; i < input_shape.size(); ++i) {
      // const auto dim_name
      if (i == axis) {
        // Set replicated for `axis` dim.
        tgt_input_layout_proto.add_sharding_specs()->set_sharding_spec(
            Layout::kUnshardedDim);
      } else {
        // Keep the rest dimension.
        tgt_input_layout_proto.add_sharding_specs()->set_sharding_spec(
            input_layout->sharding_spec(i));
      }
    }

    if (!Layout::IsUnshardedDimension(input_layout->sharding_spec(axis))) {
      TF_ASSIGN_OR_RETURN(
          input,
          EmitAllGather(builder, input, *input_layout,
                        Layout::FromProto(tgt_input_layout_proto).value()));
    }
  }

  auto new_argmax = builder.create<mlir::TF::ArgMaxOp>(
      argmax_op.getLoc(), argmax_op.getResult().getType(), input,
      argmax_op.getDimension());
  op->getResult(0).replaceAllUsesWith(new_argmax.getOutput());
  op->erase();

  return InferSPMDExpandedLocalShape(new_argmax);
}

StatusOr<llvm::DenseMap<int, Layout>> ArgMaxSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // If the input layout is missing, don't return an output layout.
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  const Layout& input_layout = input_layouts.lookup(0);
  TF_ASSIGN_OR_RETURN(auto result_layout,
                      ComputeResultLayout(op, input_layout));
  if (result_layout.rank() != input_layout.rank() - 1)
    return errors::FailedPrecondition(
        OpName(op), " derived output layout rank is ", result_layout.rank(),
        " not ", input_layout.rank() - 1, " as expected.");

  return llvm::DenseMap<int, Layout>({{0, result_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>> ArgMaxSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // If no output layout, then do not infer any operand layouts.
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  auto argmax_op = llvm::cast<mlir::TF::ArgMaxOp>(op);
  TF_ASSIGN_OR_RETURN(int64_t axis,
                      ExtractConstIntFromValue(argmax_op.getDimension()));
  auto input = argmax_op.getInput();
  const auto input_rank = ValueRank(input);

  // Handle the case of negative axis.
  if (axis < 0) axis += input_rank;

  const Layout& output_layout = output_layouts.lookup(0);

  TF_ASSIGN_OR_RETURN(auto input_shape, GetShapeOfValue(input));

  std::vector<std::string> layout_sharding;

  int output_dim = 0;
  for (int i = 0; i < input_shape.size(); ++i) {
    if (i == axis) {
      layout_sharding.emplace_back(Layout::kUnshardedDim);
    } else {
      layout_sharding.emplace_back(output_layout.sharding_spec(output_dim));
      output_dim += 1;
    }
  }

  // Add Layout for first input attribute, while the second one is axis as a
  // scalar, we don't need to set its layout.
  TF_ASSIGN_OR_RETURN(const Layout result_layout,
                      Layout::GetLayout(layout_sharding, output_layout.mesh()));
  return llvm::DenseMap<int, Layout>({{0, result_layout}});
}

}  // namespace dtensor
}  // namespace tensorflow
