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

#include "tensorflow/dtensor/mlir/expansions/meta_spmd_expander.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"
#include "tensorflow/compiler/mlir/utils/array_container_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/dtensor_location.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/spmd_expander.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Validates `axis` for pack/unpack and resolves negative values.
//
// Returns a valid positive axis or an error.
StatusOr<int> CanonicalizeAxis(int axis, int packed_rank) {
  // Axis can be in range [-packed_rank, packed_rank), so we add packed_rank
  // to wrap it around.
  if (axis >= -packed_rank && axis < 0) {
    axis += packed_rank;
  } else if (axis < -packed_rank || axis >= packed_rank) {
    return errors::InvalidArgument(
        "Invalid axis; expected a value in [-packed_rank, packed_rank)");
  }
  return axis;
}

// Implements, for pack or unpack, layout propagation from a suggested layout
// for the packed tensor to suggested layouts for the unpacked tensors.
StatusOr<llvm::DenseMap<int, Layout>> LayoutsFromPackedTensor(
    int axis, const Layout& packed_layout, size_t num_unpacked_tensors) {
  TF_ASSIGN_OR_RETURN(axis,
                      CanonicalizeAxis(axis,
                                       /*packed_rank=*/packed_layout.rank()));
  const Layout unpacked_layout =
      packed_layout.GetLayoutWithReducedDims({axis}, false);
  llvm::DenseMap<int, Layout> layouts(num_unpacked_tensors);
  for (int i = 0; i < num_unpacked_tensors; ++i) {
    layouts[i] = unpacked_layout;
  }
  return layouts;
}

// Implements, for pack or unpack, layout propagation from suggested layouts for
// the unpacked tensors to a suggested layout for the packed tensor.
StatusOr<llvm::DenseMap<int, Layout>> LayoutFromUnpackedTensors(
    int axis, const llvm::DenseMap<int, Layout>& unpacked_layouts) {
  if (unpacked_layouts.empty()) return llvm::DenseMap<int, Layout>();

  auto it = unpacked_layouts.begin();
  const Layout& first_layout = it->getSecond();
  const Mesh& mesh = first_layout.mesh();

  // Record the mesh and rank of the first input layout that exists.
  // The rank + mesh for others will be the same.
  const int unpacked_rank = first_layout.rank();
  TF_ASSIGN_OR_RETURN(axis,
                      CanonicalizeAxis(axis,
                                       /*packed_rank=*/unpacked_rank + 1));

  std::vector<std::string> inferred_packed_layout_specs;
  for (int rank_index = 0; rank_index <= unpacked_rank; ++rank_index) {
    if (rank_index == axis)
      inferred_packed_layout_specs.push_back(Layout::kUnshardedDim);
    if (rank_index == unpacked_rank) {
      break;
    }
    // When we have multiple input with conflicting shardings, set that
    // dimension to replicated (aka unsharded).
    std::string dimension = Layout::kUnshardedDim;
    bool found_sharded_dim = false;
    for (; it != unpacked_layouts.end(); ++it) {
      const std::string& sharding_spec =
          it->getSecond().sharding_spec(rank_index);
      if (!Layout::IsUnshardedDimension(sharding_spec)) {
        if (!found_sharded_dim) {
          found_sharded_dim = true;
          dimension = sharding_spec;
        } else if (sharding_spec != dimension) {
          dimension = Layout::kUnshardedDim;
        }
      }
    }
    inferred_packed_layout_specs.push_back(dimension);
  }
  TF_ASSIGN_OR_RETURN(auto inferred_packed_layout,
                      Layout::GetLayout(inferred_packed_layout_specs, mesh));
  return llvm::DenseMap<int, Layout>({{0, inferred_packed_layout}});
}

}  // namespace

StatusOr<mlir::Operation*> PackSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto pack = llvm::cast<mlir::TF::PackOp>(op);
  TF_ASSIGN_OR_RETURN(const absl::optional<Layout> output_layout,
                      ExtractSingleLayoutFromOp(op));

  const int output_rank = ValueRank(pack.getOutput());
  if (output_rank == -1)
    return errors::Unimplemented("output must have a rank");

  TF_ASSIGN_OR_RETURN(
      int axis, CanonicalizeAxis(pack.getAxis(), /*packed_rank=*/output_rank));

  // TODO(bfontain): This may not be the best, but for now relayout all inputs
  // to match the output layout. E.g. if the output layout is not but the input
  // is, this would force a AllConcat on all inputs, rather than first packing
  // and emitting one AllConcat.
  const Layout new_input_layout =
      output_layout->GetLayoutWithReducedDims({axis}, /*keep_dims=*/false);

  for (int i = 0; i < op->getNumOperands(); ++i) {
    TF_ASSIGN_OR_RETURN(const absl::optional<Layout> layout,
                        ExtractLayoutFromOperand(pack.getOperand(i)));
    if (!layout) return errors::InvalidArgument("missing layout for input ", i);

    TF_ASSIGN_OR_RETURN(
        mlir::Value new_input,
        EmitRelayout(pack.getOperand(i), *layout, new_input_layout));

    pack.setOperand(i, new_input);
  }

  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>> PackSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  auto pack = llvm::cast<mlir::TF::PackOp>(op);
  const int axis = pack.getAxis();
  return LayoutFromUnpackedTensors(axis, input_layouts);
}

StatusOr<llvm::DenseMap<int, Layout>> PackSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  auto pack = llvm::cast<mlir::TF::PackOp>(op);
  const int axis = pack.getAxis();
  return LayoutsFromPackedTensor(axis, output_layouts.lookup(0),
                                 pack->getNumOperands());
}

StatusOr<mlir::Operation*> UnpackSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto unpack = llvm::cast<mlir::TF::UnpackOp>(op);
  TF_ASSIGN_OR_RETURN(const absl::optional<Layout> input_layout,
                      ExtractLayoutFromOperand(unpack.getOperand()));
  if (!input_layout) {
    return errors::Unimplemented("input must have a layout");
  }

  const int input_rank = ValueRank(unpack.getOperand());
  if (input_rank == -1) {
    return errors::Unimplemented("input must have a rank");
  }

  TF_ASSIGN_OR_RETURN(
      int axis, CanonicalizeAxis(unpack.getAxis(), /*packed_rank=*/input_rank));

  if (input_layout->num_shards_for_dim(input_layout->dim(axis)) != 1) {
    // If the axis being unpacked is sharded, relayout to replicated along that
    // axis since each device needs to split across it.
    std::vector<ShardingSpec> new_layout_specs(input_rank);
    for (int input_index = 0; input_index < input_rank; ++input_index) {
      if (input_index == axis) {
        new_layout_specs[input_index].set_sharding_spec(Layout::kUnshardedDim);
      } else {
        new_layout_specs[input_index] = input_layout->dim(input_index);
      }
    }
    TF_ASSIGN_OR_RETURN(
        Layout new_input_layout,
        Layout::GetLayout(std::move(new_layout_specs), input_layout->mesh()));
    TF_ASSIGN_OR_RETURN(
        mlir::Value new_input,
        EmitRelayout(unpack.getOperand(), *input_layout, new_input_layout));
    unpack.setOperand(new_input);
  }
  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>> UnpackSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  auto unpack = llvm::cast<mlir::TF::UnpackOp>(op);
  const int axis = unpack.getAxis();
  return LayoutsFromPackedTensor(axis, input_layouts.lookup(0),
                                 unpack->getNumResults());
}

StatusOr<llvm::DenseMap<int, Layout>> UnpackSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  auto unpack = llvm::cast<mlir::TF::UnpackOp>(op);
  const int axis = unpack.getAxis();
  return LayoutFromUnpackedTensors(axis, output_layouts);
}

namespace {

Status VerifyPaddedDimensionNotSharded(const Layout& layout,
                                       mlir::Value pad_input,
                                       mlir::Value pad_output) {
  auto input_type = pad_input.getType().dyn_cast<mlir::RankedTensorType>();
  auto output_type = pad_output.getType().dyn_cast<mlir::RankedTensorType>();
  if (!input_type || !output_type)
    return errors::InvalidArgument(
        "pad op input/output should have statically known shape for SPMD.");

  const auto input_shape = input_type.getShape();
  const auto output_shape = input_type.getShape();
  for (const auto& dim_shard_and_index :
       llvm::enumerate(layout.sharding_specs())) {
    const int index = dim_shard_and_index.index();
    const auto& tensor_dimension = dim_shard_and_index.value();
    const int input_shape_for_dim = input_shape[index];
    const int output_shape_for_dim = output_shape[index];
    if ((input_shape_for_dim == -1 || output_shape_for_dim == -1 ||
         output_shape_for_dim != input_shape_for_dim) &&
        layout.num_shards_for_dim(tensor_dimension) > 1) {
      return errors::InvalidArgument(
          "Padding over sharded dimension is not allowed.");
    }
  }
  return OkStatus();
}

}  // namespace

StatusOr<mlir::Operation*> PadSPMDExpander::ExpandOp(mlir::Operation* op) {
  // TODO(b/170666884): Implement sharded SPMD logic for tf.Pad op.
  TF_ASSIGN_OR_RETURN(auto op_layout, ExtractSingleLayoutFromOp(op));
  auto pad_input = op->getOperand(0);
  auto pad_output = op->getResult(0);

  TF_ASSIGN_OR_RETURN(auto input_layout, ExtractLayoutFromOperand(pad_input));
  assert(input_layout && op_layout);

  if (op_layout != input_layout)
    return errors::Unimplemented(
        "pad op with input layout different from op output layout is not yet "
        "supported.");

  TF_RETURN_IF_ERROR(
      VerifyPaddedDimensionNotSharded(*op_layout, pad_input, pad_output));
  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>> PadSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  const Layout input_layout = input_layouts.lookup(0);
  mlir::Value pad_input;
  mlir::Value pad_output;

  if (auto pad_v2 = llvm::dyn_cast<mlir::TF::PadV2Op>(op)) {
    pad_output = pad_v2.getOutput();
    pad_input = pad_v2.getInput();
  } else {
    auto pad_op = llvm::cast<mlir::TF::PadOp>(op);
    pad_output = pad_op.getOutput();
    pad_input = pad_op.getInput();
  }

  TF_RETURN_IF_ERROR(
      VerifyPaddedDimensionNotSharded(input_layout, pad_input, pad_output));
  return llvm::DenseMap<int, Layout>({{0, input_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>> PadSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  mlir::Value pad_input;
  mlir::Value pad_output;

  llvm::DenseMap<int, Layout> input_layouts(op->getNumOperands());
  // Pad op `padding` operand always has rank 2 tensor.
  input_layouts[1] = Layout::ReplicatedOnMesh(mesh, /*rank=*/2);

  if (auto pad_v2 = llvm::dyn_cast<mlir::TF::PadV2Op>(op)) {
    pad_output = pad_v2.getOutput();
    pad_input = pad_v2.getInput();
    // `constant_values` operand
    input_layouts[2] = Layout::ReplicatedOnMesh(mesh, /*rank=*/0);
  } else {
    auto pad_op = llvm::cast<mlir::TF::PadOp>(op);
    pad_output = pad_op.getOutput();
    pad_input = pad_op.getInput();
  }

  if (output_layouts.find(0) != output_layouts.end()) {
    const Layout output_layout = output_layouts.lookup(0);
    TF_RETURN_IF_ERROR(
        VerifyPaddedDimensionNotSharded(output_layout, pad_input, pad_output));
    // `input` operand
    input_layouts[0] = output_layout;
  }
  return input_layouts;
}

namespace {

Status VerifyTileOperandLayout(const Layout& operand_layout,
                               llvm::ArrayRef<int64_t> static_multiples) {
  for (const auto& tensor_dim_and_multiple :
       llvm::zip(operand_layout.sharding_specs(), static_multiples)) {
    const auto& tensor_dimension = std::get<0>(tensor_dim_and_multiple);
    const int64_t multiple_factor = std::get<1>(tensor_dim_and_multiple);
    if (multiple_factor > 1 &&
        operand_layout.num_shards_for_dim(tensor_dimension) > 1)
      return errors::InvalidArgument(
          "tile op with input sharded at dimension where `multiple` > 1 is not "
          "supported.");
  }
  return OkStatus();
}

}  // namespace

StatusOr<mlir::Operation*> TileSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto tile_op = llvm::cast<mlir::TF::TileOp>(op);
  // After layout propagation, tile op should already have the proper output
  // layout tagged on itself.
  TF_ASSIGN_OR_RETURN(absl::optional<Layout> output_layout,
                      ExtractSingleLayoutFromOp(op));
  if (!output_layout)
    return errors::InvalidArgument(
        "TileOP doesn't have a layout after layout propagation");

  TF_ASSIGN_OR_RETURN(absl::optional<Layout> operand_layout,
                      ExtractLayoutFromOperand(tile_op.getInput()));
  if (!operand_layout)
    return errors::InvalidArgument(
        "Input operand to TileOp doesn't have a layout after layout "
        "propagation.");

  if (operand_layout->IsFullyReplicated() &&
      output_layout->IsFullyReplicated()) {
    // There's nothing to do; we can avoid some unimplemented cases.
    return InferSPMDExpandedLocalShape(op);
  }

  llvm::SmallVector<int64_t, 4> static_multiples;
  auto status =
      ExtractConstVectorFromValue(tile_op.getMultiples(), &static_multiples);
  if (!status.ok())
    return errors::Unimplemented(
        "Tile with a sharded output is not implemented for dynamic "
        "`multiples`.");

  // If `multiples` values can be statically known, verify that all dimensions
  // with `multiples` > 1 is replicated.
  TF_RETURN_IF_ERROR(
      VerifyTileOperandLayout(*operand_layout, static_multiples));

  llvm::SmallVector<int, 4> local_tile_multiples;
  std::vector<int32> operand_shards = operand_layout->num_shards();
  std::vector<int32> output_shards = output_layout->num_shards();
  if (operand_shards.size() != output_shards.size()) {
    return errors::InvalidArgument(
        "Expected inputs and outputs to have the same rank.");
  }

  for (int dim_index = 0; dim_index < operand_shards.size(); ++dim_index) {
    if (static_multiples[dim_index] == 1) {
      local_tile_multiples.push_back(static_multiples[dim_index]);
      continue;
    }
    if (output_shards[dim_index] > static_multiples[dim_index])
      // TODO(b/161012891): Split the input to support sharding the output
      // more than `multiples` ways.
      return errors::Unimplemented(
          "Sharding the output of Tile into more than `multiples` shards is "
          "not currently supported.");
    if (static_multiples[dim_index] % output_shards[dim_index] != 0)
      return errors::Unimplemented(
          "The output sharding of Tile must evenly divide `multiples`.");
    if (!Layout::IsUnshardedDimension(
            operand_layout->sharding_spec(dim_index)) &&
        (Layout::IsUnshardedDimension(
             output_layout->sharding_spec(dim_index)) ||
         (operand_layout->sharding_spec(dim_index) !=
          output_layout->sharding_spec(dim_index))))
      return errors::Unimplemented(
          "Input is replicated on tensor dimension ", dim_index,
          " but the "
          "output is not replicated or is replicated on a different mesh "
          "dimension.");
    local_tile_multiples.push_back(static_multiples[dim_index] /
                                   output_shards[dim_index]);
  }
  mlir::OpBuilder builder(op);
  auto location = DT_LOC(tile_op.getLoc());
  auto multiples_const = IntConst(builder, location, local_tile_multiples);

  auto global_output_type =
      tile_op.getResult().getType().cast<mlir::TensorType>();
  TF_ASSIGN_OR_RETURN(
      auto local_type,
      LocalTypeFromGlobalType(output_layout.value(), global_output_type));

  auto new_tile =
      builder.create<mlir::TF::TileOp>(location, /*output=*/local_type,
                                       /*input=*/tile_op.getInput(),
                                       /*multiples=*/multiples_const);
  tile_op.getResult().replaceAllUsesWith(new_tile.getOutput());
  tile_op.erase();
  return new_tile.getOperation();
}

StatusOr<llvm::DenseMap<int, Layout>> TileSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto tile_op = llvm::cast<mlir::TF::TileOp>(op);

  auto output_ranked_type =
      tile_op.getOutput().getType().dyn_cast<mlir::RankedTensorType>();
  if (!output_ranked_type || !output_ranked_type.hasStaticShape()) {
    return errors::InvalidArgument(
        llvm::formatv(
            "requires output type to have statically known rank, but got : {0}",
            output_ranked_type)
            .str());
  }
  auto tile_output_shape = output_ranked_type.getShape();

  llvm::SmallVector<int64_t, 4> static_multiple;
  auto status =
      ExtractConstVectorFromValue(tile_op.getMultiples(), &static_multiple);

  // If multiple operands cannot be statically known, output is set to
  // replicated.
  if (!status.ok()) {
    return llvm::DenseMap<int, Layout>(
        {{0, Layout::ReplicatedOnMesh(mesh, tile_output_shape.size())}});
  }

  // When suggested input layout exists then forward the input sharding for all
  // dimensions where `multiple` == 1.
  const Layout input_layout = input_layouts.lookup(0);
  std::vector<std::string> output_layout_specs;
  for (const auto& multiple_and_dim_sharding :
       llvm::zip(static_multiple, input_layout.sharding_specs())) {
    const int multiple = std::get<0>(multiple_and_dim_sharding);
    const auto& tensor_dimension = std::get<1>(multiple_and_dim_sharding);
    output_layout_specs.push_back(multiple == 1
                                      ? tensor_dimension.sharding_spec()
                                      : Layout::kUnshardedDim);
  }

  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      Layout::GetLayout(output_layout_specs, mesh));
  return llvm::DenseMap<int, Layout>({{0, output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>> TileSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto tile_op = llvm::cast<mlir::TF::TileOp>(op);

  // Retrieve operand/output shapes of tile op.
  auto input_ranked_type =
      tile_op.getInput().getType().dyn_cast<mlir::RankedTensorType>();
  if (!input_ranked_type || !input_ranked_type.hasStaticShape()) {
    return errors::InvalidArgument(
        llvm::formatv(
            "requires input type to have statically known rank, but got : {0}",
            input_ranked_type)
            .str());
  }
  auto tile_input_shape = input_ranked_type.getShape();

  llvm::DenseMap<int, Layout> input_layouts(op->getNumOperands());

  // `multiples` operand is always set to have replicated layout.
  input_layouts[1] =
      Layout::ReplicatedOnMesh(mesh, tile_op.getMultiples()
                                         .getType()
                                         .cast<mlir::RankedTensorType>()
                                         .getRank());

  llvm::SmallVector<int64_t, 4> static_multiple;
  auto status =
      ExtractConstVectorFromValue(tile_op.getMultiples(), &static_multiple);

  // If multiple operands cannot be statically known they are set to replicated.
  if (!status.ok()) {
    input_layouts[0] = Layout::ReplicatedOnMesh(mesh, tile_input_shape.size());
    return input_layouts;
  }

  // When suggested output layout exists, then override operand layout with
  // consumer suggested output layout if `multiple` of dimension == 1 and
  // dimension size can be evenly divisible by the sharding.
  if (output_layouts.find(0) != output_layouts.end()) {
    const Layout output_layout = output_layouts.lookup(0);
    std::vector<std::string> input_layout_specs;
    for (const auto& multiple_and_dim_sharding :
         llvm::zip(static_multiple, output_layout.sharding_specs())) {
      const int multiple = std::get<0>(multiple_and_dim_sharding);
      const auto& tensor_dimension = std::get<1>(multiple_and_dim_sharding);
      input_layout_specs.push_back(multiple == 1
                                       ? tensor_dimension.sharding_spec()
                                       : Layout::kUnshardedDim);
    }
    TF_ASSIGN_OR_RETURN(const Layout input_layout,
                        Layout::GetLayout(input_layout_specs, mesh));
    input_layouts[0] = input_layout;
  }
  return input_layouts;
}

namespace {

// From input shape and output shape, extract a maximal list segments where
// the product of the input shape from input_segment_start to input_segment_end
// is equal to the product of the output shape from output_segment_start
// to output_segment_end and is not equal for the product of any subsequence.
// Note that dimensions of shape are skipped over if they would be at the start
// of a segment.
// Note that shapes with unknown dimension size (represented by -1) are
// unsupported.
void ComputeReshapeSegments(
    llvm::ArrayRef<int64_t> input_shape, llvm::ArrayRef<int64_t> output_shape,
    llvm::SmallVectorImpl<int64_t>& input_segment_start,
    llvm::SmallVectorImpl<int64_t>& input_segment_end,
    llvm::SmallVectorImpl<int64_t>& output_segment_start,
    llvm::SmallVectorImpl<int64_t>& output_segment_end) {
  int input_offset = 0;
  int output_offset = 0;

  while (input_offset < input_shape.size() &&
         output_offset < output_shape.size()) {
    while (input_offset < input_shape.size() && input_shape[input_offset] == 1)
      input_offset++;
    while (output_offset < output_shape.size() &&
           output_shape[output_offset] == 1)
      output_offset++;
    if (input_offset >= input_shape.size() ||
        output_offset >= output_shape.size()) {
      // Since the input and output tensors the same number of entries, we are
      // guaranteed to reach the end of both shapes at the same time.
      assert(input_offset >= input_shape.size() &&
             output_offset >= output_shape.size());
      return;
    }

    input_segment_start.emplace_back(input_offset);
    output_segment_start.emplace_back(output_offset);

    int64 input_prod = input_shape[input_offset++];
    int64 output_prod = output_shape[output_offset++];
    while (input_prod != output_prod) {
      if (input_prod < output_prod)
        input_prod *= input_shape[input_offset++];
      else
        output_prod *= output_shape[output_offset++];
    }
    input_segment_end.emplace_back(input_offset);
    output_segment_end.emplace_back(output_offset);
  }
}

// For reshape we want to reduce the number of all-to-alls and slices needed.
// Note that the forward layout propagation for reshape will be the same
// algorithm as backwards propagation.
//
// Suppose we have input shape a_0,...,a_n and output shape b_0,...,b_k  such
// that a_0*...*a_i != b_0*...*b_j except with (i,j)=(n,k).
// The forward propagation of an input layout depends only on size of axis 0
// of the output shape and the mesh dimension axis 0 of input is sharded on:
//
// 1. In any case we must all to all on any input axis from 1 to n and the
//    output layout from output axis 1 to k will always be replicated.
//
// 2. If input axis 0 is replicated, we do a local reshape and set the layout
//    of the output axis 0 to replicated.
//
// 3. If input axis 0 is sharded and the number of shards *does not divide*
//    b_0, then we must all-to-all on input axis 0 (as well as the axis
//    mentioned in 1) do a local reshape and set the layout of output axis 0
//    to replicated.
//
// 4. If input axis 0 is sharded and the number of shards does divide b_0, we
//    can do a local reshape and set the layout of output axis 0 to the same
//    mesh dimension as the input layout axis 0.
//
// Finally if for a general input and output shape, if we partition the input
// and output shape into such segments, we can apply the above rule on each
// segment. The ComputeReshapeSegments function above computes the starting
// and ending index of each segment.
StatusOr<Layout> MakeLayoutForReshape(
    const Layout& input_layout, const llvm::ArrayRef<int64_t> output_shape,
    llvm::SmallVectorImpl<int64_t>& input_segment_start,
    llvm::SmallVectorImpl<int64_t>& output_segment_start) {
  std::vector<std::string> layout_specs;
  layout_specs.reserve(output_shape.size());
  // Initialy set the layout to be all replicated.
  for (int i = 0; i < output_shape.size(); ++i)
    layout_specs.push_back(Layout::kUnshardedDim);
  // Now process each segment, for each segment if the number of shards on the
  // first entry of the input segment divides the output shape on the first
  // entry of the output segment, we request a sharded layout on that axis.
  for (int i = 0; i < input_segment_start.size(); ++i) {
    const int num_shards = input_layout.num_shards_for_dim(
        input_layout.dim(input_segment_start[i]));
    if (output_shape[output_segment_start[i]] % num_shards == 0)
      layout_specs[output_segment_start[i]] =
          input_layout.sharding_spec(input_segment_start[i]);
  }
  return Layout::GetLayout(layout_specs, input_layout.mesh());
}

}  // namespace

// TODO(b/171335075): Implement the SPMD for generic Reshape.
StatusOr<mlir::Operation*> ReshapeSPMDExpander::ExpandOp(mlir::Operation* op) {
  // Update input/output shape based on the sharding information.
  TF_ASSIGN_OR_RETURN(auto input_layout,
                      ExtractLayoutFromOperand(op->getOperand(0)));
  TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(op));

  if (!input_layout || !output_layout)
    return errors::InvalidArgument(
        "Input and output layouts of Reshape op must be known before SPMD "
        "expansion.");

  if (input_layout->IsFullyReplicated() && output_layout->IsFullyReplicated())
    return InferSPMDExpandedLocalShape(op);

  TF_ASSIGN_OR_RETURN(auto global_input_shape,
                      ExtractGlobalInputShape(op->getOpOperand(0)));
  TF_ASSIGN_OR_RETURN(auto global_output_shape,
                      ExtractGlobalOutputShape(op->getOpResult(0)));

  llvm::SmallVector<int64_t, 4> input_segment_start;
  llvm::SmallVector<int64_t, 4> input_segment_end;
  llvm::SmallVector<int64_t, 4> output_segment_start;
  llvm::SmallVector<int64_t, 4> output_segment_end;

  llvm::SmallVector<int64_t, 4> local_reshape_const;

  // Break up input and output shapes into segments which multiply to the same
  // number. We will treat each segment seaparately when constructing the input
  // shape from the output shape and vica versa.
  ComputeReshapeSegments(global_input_shape, global_output_shape,
                         input_segment_start, input_segment_end,
                         output_segment_start, output_segment_end);

  // Compute the shape for the local reshape op. For each input segment,
  // 1) Check the sharding status of all dimensions in that segment.
  // 2) Create entries in the output shape and layout for the segment.
  //
  // Also insert the necessary 1 dimensions between input and output segments.
  //
  // Currently the algorithm supports Reshape with limited cases.
  // - For example, reshape a [2, 16] shape tensor with layout ['not_sharded',
  //   'x'],
  //      - to a [2, 4, 4] shape tensor with layout ['not_sharded', 'x',
  //        'not_sharded'] does not need cross device data shuffling.
  //      - to a [2, 4, 4] shape tensor with layout ['not_sharded',
  //        'not_sharded', 'x'] needs cross device AllToAll on the input and
  //        a slice on the output.
  // - For trivial cases, which AllToAll can support, an AllToAll will be
  //   inserted. For example, reshape a [2, 4, 3] shape tensor with layout
  //   ['not_sharded', 'x', 'not_sharded'] to [2, 12] shape tensor fully
  //   replicated can be supported.
  std::vector<ShardingSpec> tgt_input_layout(input_layout->rank());
  std::vector<ShardingSpec> tgt_output_layout(output_layout->rank());

  for (int i = 0; i < input_segment_start.size(); ++i) {
    const int input_start = input_segment_start[i];
    const int output_start = output_segment_start[i];
    const int prev_input_segment_end = (i == 0 ? 0 : input_segment_end[i - 1]);
    const int prev_output_segment_end =
        (i == 0 ? 0 : output_segment_end[i - 1]);

    // Between this segment and the last segment, if there is a gap, insert
    // dimensions of size 1 and kUnshardedDim as output layout dim.
    for (int j = prev_input_segment_end; j < input_start; ++j)
      tgt_input_layout[j].set_sharding_spec(Layout::kUnshardedDim);
    for (int j = prev_output_segment_end; j < output_start; ++j) {
      local_reshape_const.emplace_back(1);
      tgt_output_layout[j].set_sharding_spec(Layout::kUnshardedDim);
    }

    const int num_input_shards =
        input_layout->num_shards_for_dim(input_layout->dim(input_start));

    // Decide on the sharding of the input for this segment.
    // If the input is already sharded, we try to keep this sharding (unless
    // the output size of first output dimension is incompatible).
    // NOTE: If the input is unsharded in a dimension, and the output is sharded
    // we could 'preshard' the input on this dimension before the reshape.
    // This is unlikely to have any major gains in performance.
    if (global_output_shape[output_start] % num_input_shards != 0) {
      tgt_input_layout[input_start].set_sharding_spec(Layout::kUnshardedDim);
      tgt_output_layout[output_start].set_sharding_spec(Layout::kUnshardedDim);
      local_reshape_const.emplace_back(global_output_shape[output_start]);
    } else {
      tgt_input_layout[input_start] = input_layout->dim(input_start);
      tgt_output_layout[output_start] = input_layout->dim(input_start);
      local_reshape_const.emplace_back(global_output_shape[output_start] /
                                       num_input_shards);
    }

    for (int j = input_start + 1; j < input_segment_end[i]; ++j)
      tgt_input_layout[j].set_sharding_spec(Layout::kUnshardedDim);
    for (int j = output_start + 1; j < output_segment_end[i]; ++j) {
      local_reshape_const.emplace_back(global_output_shape[j]);
      tgt_output_layout[j].set_sharding_spec(Layout::kUnshardedDim);
    }
  }

  // Fill any remaining dimensions of size 1 and sharding dim on the end of the
  // layout.
  for (int j = input_segment_end.back(); j < tgt_input_layout.size(); ++j)
    tgt_input_layout[j].set_sharding_spec(Layout::kUnshardedDim);
  for (int j = output_segment_end.back(); j < tgt_output_layout.size(); ++j) {
    local_reshape_const.emplace_back(1);
    tgt_output_layout[j].set_sharding_spec(Layout::kUnshardedDim);
  }

  TF_ASSIGN_OR_RETURN(
      auto desired_input_layout,
      Layout::GetLayout(tgt_input_layout, input_layout->mesh()));
  TF_ASSIGN_OR_RETURN(
      auto desired_output_layout,
      Layout::GetLayout(tgt_output_layout, input_layout->mesh()));

  auto reshape_op = mlir::cast<mlir::TF::ReshapeOp>(op);
  TF_ASSIGN_OR_RETURN(mlir::Value new_input,
                      EmitRelayout(reshape_op.getTensor(), *input_layout,
                                   desired_input_layout));

  mlir::OpBuilder builder(op);

  // Update shape op to use the local shape as input. Importantly, this updates
  // the shape attr in the Op, which `InferSPMDExpandedLocalShape` does not
  // help.
  auto new_shape = mlir::RankedTensorType::get(
      {static_cast<int64_t>(local_reshape_const.size())}, builder.getI64Type());
  auto const_attr =
      mlir::DenseIntElementsAttr::get(new_shape, local_reshape_const);
  auto new_reshape_const_op =
      builder.create<mlir::TF::ConstOp>(DT_LOC(op), const_attr);
  mlir::TF::ReshapeOp new_reshape_op = builder.create<mlir::TF::ReshapeOp>(
      op->getLoc(), new_input, new_reshape_const_op);

  TF_ASSIGN_OR_RETURN(auto final_output,
                      EmitRelayout(new_reshape_op.getOutput(),
                                   desired_output_layout, *output_layout));

  op->getResult(0).replaceAllUsesWith(final_output);
  op->erase();
  return final_output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>> ReshapeSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  auto reshape_op = mlir::cast<mlir::TF::ReshapeOp>(op);
  TF_ASSIGN_OR_RETURN(
      auto input_shape,
      GetShapeOfValue(reshape_op.getTensor(), /*fail_on_dynamic=*/true));
  TF_ASSIGN_OR_RETURN(
      auto output_shape,
      GetShapeOfValue(reshape_op.getOutput(), /*fail_on_dynamic=*/true));

  llvm::SmallVector<int64_t, 4> input_segment_start;
  llvm::SmallVector<int64_t, 4> input_segment_end;
  llvm::SmallVector<int64_t, 4> output_segment_start;
  llvm::SmallVector<int64_t, 4> output_segment_end;

  // Break up input and output shapes into segments which multiply to the same
  // number. We will treat each segment seaparately when constructing the input
  // shape from the output shape and vica versa.
  ComputeReshapeSegments(input_shape, output_shape, input_segment_start,
                         input_segment_end, output_segment_start,
                         output_segment_end);

  TF_ASSIGN_OR_RETURN(
      const Layout output_layout,
      MakeLayoutForReshape(input_layouts.lookup(0), output_shape,
                           input_segment_start, output_segment_start));
  return llvm::DenseMap<int, Layout>({{0, output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>>
ReshapeSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  auto reshape_op = mlir::cast<mlir::TF::ReshapeOp>(op);
  TF_ASSIGN_OR_RETURN(
      auto input_shape,
      GetShapeOfValue(reshape_op.getTensor(), /*fail_on_dynamic=*/true));
  TF_ASSIGN_OR_RETURN(
      auto output_shape,
      GetShapeOfValue(reshape_op.getOutput(), /*fail_on_dynamic=*/true));

  llvm::SmallVector<int64_t, 4> input_segment_start;
  llvm::SmallVector<int64_t, 4> input_segment_end;
  llvm::SmallVector<int64_t, 4> output_segment_start;
  llvm::SmallVector<int64_t, 4> output_segment_end;

  // Break up input and output shapes into segments which multiply to the same
  // number. We will treat each segment seaparately when constructing the input
  // shape from the output shape and vica versa.
  ComputeReshapeSegments(input_shape, output_shape, input_segment_start,
                         input_segment_end, output_segment_start,
                         output_segment_end);

  TF_ASSIGN_OR_RETURN(
      const Layout input_layout,
      MakeLayoutForReshape(output_layouts.lookup(0), input_shape,
                           output_segment_start, input_segment_start));
  return llvm::DenseMap<int, Layout>({{0, input_layout}});
}

StatusOr<mlir::Operation*> TransposeSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  // Currently we only support transpose without shuffling data. When use cases
  // come, we can add support as we need to figure the best strategy to keep the
  // cost as low as possible. Before that, add a check with good error message.
  {
    TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(op));
    TF_ASSIGN_OR_RETURN(auto operand_layout,
                        ExtractLayoutFromOperand(op->getOperand(0)));

    if (!output_layout)
      return errors::InvalidArgument(
          "output layout of TransposeOp must be known before SPMD expansion.");
    if (!operand_layout)
      return errors::InvalidArgument(
          "operand layout of TransposeOp must be known before SPMD expansion.");

    auto transpose = mlir::cast<mlir::TF::TransposeOp>(op);
    llvm::SmallVector<int64, 4> perm;
    TF_RETURN_IF_ERROR(ExtractConstVectorFromValue(transpose.getPerm(), &perm));

    for (const auto& p : llvm::enumerate(perm)) {
      if (operand_layout->dim(p.value()).sharding_spec() !=
          output_layout->dim(p.index()).sharding_spec()) {
        return errors::InvalidArgument(
            "TransposeOp SPMD needs communication is not supported yet. \n "
            "operand layout: ",
            operand_layout->ToString(),
            "\n output layout: ", output_layout->ToString());
      }
    }
  }

  // Do nothing but infer local shape for now.
  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>>
TransposeSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  auto transpose = mlir::cast<mlir::TF::TransposeOp>(op);
  llvm::SmallVector<int64, 4> perm;
  TF_RETURN_IF_ERROR(ExtractConstVectorFromValue(transpose.getPerm(), &perm));

  const Layout input_layout = input_layouts.lookup(0);
  std::vector<std::string> output_layout_specs;
  for (int64 p : perm)
    output_layout_specs.push_back(input_layout.sharding_spec(p));

  TF_ASSIGN_OR_RETURN(
      const Layout output_layout,
      Layout::GetLayout(output_layout_specs, input_layout.mesh()));
  return llvm::DenseMap<int, Layout>({{0, output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>>
TransposeSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  auto transpose = mlir::cast<mlir::TF::TransposeOp>(op);
  llvm::SmallVector<int64, 4> perm;
  TF_RETURN_IF_ERROR(ExtractConstVectorFromValue(transpose.getPerm(), &perm));
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> input_layouts(transpose->getNumOperands());
  input_layouts[1] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);

  if (output_layouts.find(0) != output_layouts.end()) {
    const Layout output_layout = output_layouts.lookup(0);

    llvm::SmallVector<int64, 4> inverse_perm(perm.size());
    for (const auto& p : llvm::enumerate(perm)) {
      inverse_perm[p.value()] = p.index();
    }

    std::vector<std::string> input_layout_specs;
    // For example, if perm [2, 0, 1], then inverse perm is [1, 2, 0].
    // So for input_dim[i] it is output[reverse_perm[i]]
    for (auto dim_in_output : inverse_perm)
      input_layout_specs.push_back(output_layout.sharding_spec(dim_in_output));

    TF_ASSIGN_OR_RETURN(const Layout input_layout,
                        Layout::GetLayout(input_layout_specs, mesh));
    input_layouts[0] = input_layout;
  }

  return input_layouts;
}

namespace {

Status RelayoutOneHotInput(const absl::optional<Layout>& input_layout,
                           const absl::optional<Layout>& output_layout,
                           const int axis, mlir::TF::OneHotOp& one_hot) {
  if (!input_layout || !output_layout)
    return errors::InvalidArgument(
        "layout for tf.OneHot operation inputs and outputs must be known before"
        " SPMD expansion. Consider adding Relayout() op to specify the "
        "layout.");

  std::vector<ShardingSpec> sharding_specs(input_layout->rank());
  for (int i = 0; i < input_layout->rank(); ++i) {
    if (i < axis)
      sharding_specs[i] = output_layout->dim(i);
    else
      sharding_specs[i] = output_layout->dim(i + 1);
  }
  TF_ASSIGN_OR_RETURN(const Layout new_input_layout,
                      Layout::GetLayout(sharding_specs, input_layout->mesh()));

  TF_ASSIGN_OR_RETURN(
      mlir::Value new_input,
      EmitRelayout(one_hot.getIndices(), *input_layout, new_input_layout));

  one_hot->setOperand(0, new_input);

  return OkStatus();
}

}  // namespace

StatusOr<mlir::Operation*> OneHotSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto one_hot_op = llvm::cast<mlir::TF::OneHotOp>(op);

  mlir::OpBuilder builder(op);
  TF_ASSIGN_OR_RETURN(const auto input_layout,
                      ExtractLayoutFromOperand(one_hot_op->getOperand(0)));

  TF_ASSIGN_OR_RETURN(const auto output_layout,
                      ExtractSingleLayoutFromOp(one_hot_op));
  int axis = one_hot_op.getAxisAttr().getInt();
  if (axis == -1) axis = output_layout->rank() - 1;

  // For tf.OneHot, relayout input so that it matches the output layout (outside
  // of the one hot dimension).
  TF_RETURN_IF_ERROR(
      RelayoutOneHotInput(input_layout, output_layout, axis, one_hot_op));

  const int num_shards = output_layout->num_shards()[axis];
  const auto depth = ExtractConstIntFromValue(one_hot_op.getDepth());
  const bool depth_statically_divisible_by_sharding =
      (depth.ok() && (*depth) % num_shards == 0);

  // If axis dimension of tf.OneHot is sharded and number of shards evenly
  // divides the `depth` input of the one hot operations, we can mutate the
  // `depth` and parameter and `indices` parameter to calculate local tensor
  // directly.
  const std::string& mesh_dim_name = output_layout->sharding_spec(axis);

  if (mesh_dim_name != Layout::kUnshardedDim) {
    if (!depth_statically_divisible_by_sharding)
      return errors::InvalidArgument(
          "OneHot axis dimension is sharded with incorrect layout. OneHot op "
          "depth should be evenly divisible by number of shards.");

    // Recalculate new local depth. Namely: new_depth = depth / num_shards
    mlir::Value new_depth = CreateIntScalarConst((*depth) / num_shards, builder,
                                                 one_hot_op->getLoc(), false);

    // Calculate shard id at mesh dimension for the sharded axis.
    TF_ASSIGN_OR_RETURN(const Mesh mesh,
                        ExtractDeviceMeshEnclosingCluster(one_hot_op));
    mlir::tf_device::ClusterOp cluster =
        one_hot_op->getParentOfType<mlir::tf_device::ClusterOp>();

    // `mesh_coordinates` is tensor of size [1, num mesh dimensions] where each
    // element in the tensor refers to shard id for the specified mesh
    // dimension.
    TF_ASSIGN_OR_RETURN(mlir::Value mesh_coordinates,
                        GetMeshCoordinatesFromCluster(cluster));
    const int num_mesh_dimensions = output_layout->mesh().dims().size();
    llvm::SmallVector<int32_t, 4> multiplier(num_mesh_dimensions);
    const int mesh_dim_index =
        output_layout->mesh().GetMeshDimIndexWithName(mesh_dim_name);

    mlir::TF::SliceOp selected_sharding_at_dimension = builder.create<
        mlir::TF::SliceOp>(
        one_hot_op.getLoc(),
        mlir::RankedTensorType::get({1, 1}, mesh_coordinates.getType()
                                                .cast<mlir::TensorType>()
                                                .getElementType()),
        /*input=*/mesh_coordinates,
        /*begin=*/IntConst(builder, one_hot_op.getLoc(), {0, mesh_dim_index}),
        /*size=*/IntConst(builder, one_hot_op.getLoc(), {1, 1}));

    // Reshape the sliced shape (1,1) tensor to shape 0 scalar.
    auto scalar_size_type =
        mlir::RankedTensorType::get({}, builder.getIntegerType(32));
    mlir::Value scalar_shape = mlir::TF::collection_ops_util::GetR1Const(
        scalar_size_type.getShape(), builder, one_hot_op->getLoc());
    mlir::Value selected_sharding_scalar_value =
        builder.create<mlir::TF::ReshapeOp>(
            one_hot_op.getLoc(), mlir::ArrayRef<mlir::Type>{scalar_size_type},
            mlir::ArrayRef<mlir::Value>{
                selected_sharding_at_dimension.getOutput(), scalar_shape},
            mlir::ArrayRef<mlir::NamedAttribute>{});

    // `new_indices` =  `original_indices` - `selected_sharding_scalar_value` *
    // (depth/num_shards)
    mlir::Value id_offset = builder.create<mlir::TF::MulOp>(
        one_hot_op->getLoc(), new_depth, selected_sharding_scalar_value);
    mlir::Value original_indices = one_hot_op.getIndices();
    mlir::Value new_indices = builder.create<mlir::TF::SubOp>(
        one_hot_op->getLoc(), original_indices, id_offset);

    // Replace onehot operation inputs with mutated `new_depth` and `new_input`
    // tensors so that local tensors can be calculated directly without
    // calculating intermediate global tensors.
    one_hot_op->getOpOperand(0).set(new_indices);
    one_hot_op->getOpOperand(1).set(new_depth);
  }
  return InferSPMDExpandedLocalShape(one_hot_op);
}

StatusOr<llvm::DenseMap<int, Layout>> OneHotSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  auto one_hot = mlir::dyn_cast<mlir::TF::OneHotOp>(op);
  int axis = one_hot.getAxis();
  if (axis == -1) axis = ValueRank(one_hot.getIndices());
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  const Layout indices_layout = input_layouts.lookup(0);
  std::vector<std::string> output_layout_specs;
  for (int i = 0; i < indices_layout.rank(); ++i) {
    // Insert an onehot dimension for expanded axis.
    if (i == axis) {
      output_layout_specs.push_back(Layout::kUnshardedDim);
    }
    output_layout_specs.push_back(indices_layout.sharding_spec(i));
  }
  if (axis == indices_layout.rank() || axis == -1) {
    output_layout_specs.push_back(Layout::kUnshardedDim);
  }

  TF_ASSIGN_OR_RETURN(auto output_layout,
                      Layout::GetLayout(output_layout_specs, mesh));
  return llvm::DenseMap<int, Layout>({{0, output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>> OneHotSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  auto one_hot = mlir::dyn_cast<mlir::TF::OneHotOp>(op);
  int axis = one_hot.getAxis();
  if (axis == -1) axis = ValueRank(one_hot.getIndices());
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> input_layouts(one_hot->getNumOperands());
  const auto scalar_replicated_layout =
      Layout::ReplicatedOnMesh(mesh, /*rank=*/0);
  input_layouts[1] = scalar_replicated_layout;  // depth
  input_layouts[2] = scalar_replicated_layout;  // on_value
  input_layouts[3] = scalar_replicated_layout;  // off_value

  // If output layout is specified, then propagate all dimensions (except axis
  // dimension) as operand layout.
  if (output_layouts.find(0) != output_layouts.end()) {
    const Layout output_layout = output_layouts.lookup(0);

    std::vector<std::string> indices_layout_specs;
    for (int i = 0; i < output_layout.rank(); ++i) {
      if (i == axis) continue;
      indices_layout_specs.push_back(output_layout.sharding_spec(i));
    }

    TF_ASSIGN_OR_RETURN(auto input_layout,
                        Layout::GetLayout(indices_layout_specs, mesh));
    input_layouts[0] = input_layout;
  }

  return input_layouts;
}

StatusOr<mlir::Operation*> ShapeSPMDExpander::ExpandOp(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto result_layouts, ExtractLayoutFromOp(op));
  for (const auto& layout : result_layouts) {
    if (!layout.has_value())
      return errors::Internal(
          "All op result layouts must be specified for SPMD expansion.");

    if (!layout->IsFullyReplicated()) {
      return errors::Internal(
          "Shape/Rank ops must output value with replicated layout.");
    }
  }
  InferSPMDExpandedLocalShape(op);

  // DTensors shards are always full rank -- local rank == global rank
  if (mlir::isa<mlir::TF::RankOp>(op)) return op;

  // We have Shape/ShapeN op.

  // Find enclosing device_cluster op and update attributes for it if the
  // shape op result is returned to the cluster.
  auto enclosing_cluster = op->getParentOfType<mlir::tf_device::ClusterOp>();
  if (!enclosing_cluster)
    return errors::InvalidArgument(
        "Error during SPMD expansion of Shape op. Op must be enclosed in a "
        "device_cluster.");

  // Record output result index -> input_layout mapping.
  llvm::SmallVector<std::string, 4> input_layouts;
  std::vector<int> return_indices;

  // For each operand, extract global shape if necessary. If global shape
  // transformation is needed, and the transformed shape is returned to
  // outside of the device cluster, also attach input layout as additional
  // information so that future stack could infer local shape from the result.
  llvm::SmallVector<mlir::TF::MulOp, 4> output_ops;
  for (int i = 0; i < op->getNumOperands(); ++i) {
    // Fetch layout from _input_, not current op.
    TF_ASSIGN_OR_RETURN(auto input_layout,
                        ExtractLayoutFromOperand(op->getOperand(i)));
    if (!input_layout)
      return errors::InvalidArgument(
          "Input layout to shape op must be known before SPMD expansion.");

    // Fully replicated tensors: local shape = global shape.
    if (input_layout->IsFullyReplicated()) {
      continue;
    }

    // If a DTensor is sharded over a dimension, shards have equal size.
    // GlobalShape[Dim] = LocalShape[Dim] * NumShards[Dim]
    mlir::OpBuilder builder(op->getBlock(), ++mlir::Block::iterator(op));
    auto num_shards =
        IntConst(builder, op->getLoc(), input_layout->num_shards());
    auto global_shape = builder.create<mlir::TF::MulOp>(
        op->getLoc(), op->getResult(i), num_shards);

    op->getResult(i).replaceAllUsesExcept(
        global_shape.getResult(),
        llvm::SmallPtrSet<mlir::Operation*, 1>{global_shape});

    // Find the returned global shape and attach Metadata information.
    for (auto& use : global_shape.getOperation()->getUses()) {
      if (use.getOwner() == enclosing_cluster.GetBody().getTerminator()) {
        input_layouts.emplace_back(input_layout->ToString());
        return_indices.emplace_back(use.getOperandNumber());
        break;
      }
    }
    output_ops.emplace_back(global_shape);
  }

  // Attach original input for the enclosing device_cluster op.
  if (!input_layouts.empty()) {
    mlir::OpBuilder builder(op);
    enclosing_cluster->setAttr(kShapeOpInputLayoutIndices,
                               builder.getI32VectorAttr(return_indices));
    enclosing_cluster->setAttr(
        kShapeOpInputLayout,
        builder.getStrArrayAttr(llvm::SmallVector<llvm::StringRef, 4>(
            input_layouts.begin(), input_layouts.end())));
  }

  // TODO(hthu): Support multiple returns in ShapeN op.
  return output_ops.empty() ? op : output_ops[0].getOperation();
}

StatusOr<llvm::DenseMap<int, Layout>> ShapeSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  assert(op->getNumResults() == 1);
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  const int output_rank = ValueRank(op->getResult(0));
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, output_rank)}});
}

StatusOr<llvm::DenseMap<int, Layout>> ShapeSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  return llvm::DenseMap<int, Layout>();
}

}  // namespace dtensor
}  // namespace tensorflow
