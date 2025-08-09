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

#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/device_utils.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

// Checks that all layouts are fully replicated
bool AllReplicated(const std::vector<Layout>& layouts) {
  for (const Layout& layout : layouts) {
    if (!layout.IsFullyReplicated()) return false;
  }
  return true;
}

StatusOr<mlir::TensorType> LocalTypeFromGlobalType(
    const Layout& layout, const mlir::TensorType& original_type) {
  if (!original_type.hasRank()) {
    return original_type;
  }
  auto shape = llvm::to_vector<4>(original_type.getShape());
  auto shard_values = layout.num_shards();
  for (int output_axis = 0; output_axis < shape.size(); ++output_axis) {
    if (shape[output_axis] != mlir::ShapedType::kDynamic) {
      if (shape[output_axis] % shard_values[output_axis] != 0) {
        return errors::InvalidArgument(
            "The sharding spec for axis ", output_axis, " splits among ",
            shard_values[output_axis],
            " values, which does not evenly divide the length of that axis "
            "(",
            shape[output_axis], "). The full requested layout is ",
            layout.ToString(), ".");
      }
      shape[output_axis] /= shard_values[output_axis];
    }
  }
  mlir::RankedTensorType new_output_type =
      mlir::RankedTensorType::get(shape, original_type.getElementType());
  return new_output_type;
}

StatusOr<mlir::TensorType> GlobalTypeFromLocalType(
    const Layout& layout, const mlir::TensorType& original_type) {
  if (!original_type.hasRank()) {
    return original_type;
  }
  auto shape = llvm::to_vector<4>(original_type.getShape());
  auto shard_values = layout.num_shards();
  for (int output_axis = 0; output_axis < shape.size(); ++output_axis)
    if (shape[output_axis] != mlir::ShapedType::kDynamic)
      shape[output_axis] *= shard_values[output_axis];
  mlir::RankedTensorType new_output_type =
      mlir::RankedTensorType::get(shape, original_type.getElementType());
  return new_output_type;
}

absl::Status CreateSplitOp(const int num_split, const int split_dimension,
                           const mlir::Location location, mlir::Value src_input,
                           mlir::OpBuilder* builder,
                           mlir::TF::SplitOp* split_op) {
  // Creates a const op to hold split dimension value.
  auto split_dim_type =
      mlir::RankedTensorType::get({}, builder->getIntegerType(32));
  auto split_dimension_attr =
      mlir::DenseElementsAttr::get(split_dim_type, split_dimension);
  auto split_dimension_op = builder->create<mlir::TF::ConstOp>(
      location, split_dim_type, split_dimension_attr);

  // Correctly set output shapes of split op output if input shape is statically
  // known.
  mlir::Type output_type;
  auto input_type = mlir::cast<mlir::TensorType>(src_input.getType());

  if (input_type.hasRank()) {
    if (input_type.getShape()[split_dimension] == mlir::ShapedType::kDynamic) {
      output_type = input_type;
    } else {
      auto shape = llvm::to_vector<4>(input_type.getShape());
      if (shape[split_dimension] % num_split != 0) {
        return errors::InvalidArgument(
            llvm::formatv(
                "incorrect input sharding configuration received. "
                "{0}-th dimension of the input must be evenly divisible by {1}",
                split_dimension, num_split)
                .str());
      }

      shape[split_dimension] = shape[split_dimension] / num_split;
      output_type =
          mlir::RankedTensorType::get(shape, input_type.getElementType());
    }
  } else {
    output_type = input_type;
  }

  // Creates a split op that splits |src_input| along |split_dimension|.
  llvm::SmallVector<mlir::Type, 4> output_types(num_split, output_type);
  *split_op = builder->create<mlir::TF::SplitOp>(
      location, output_types, split_dimension_op.getOutput(), src_input);
  return absl::OkStatus();
}

// Given layouts + shapes, determines if the two are broadcasting compatible.
// When broadcasting we effectively line up the shapes and layouts by the end.
// The input with lower rank can be thought of as having abs(rank_a-rank_b)
// replicated dims of size 1 prepended to it.
//
// Returns the broadcast layout and the splits in the two inputs needed to run
// an elementwise op efficiently.
//
// Checks that a given mesh dimension is not used in different tensor dimensions
// in the two input layouts.
// E.g. a layout like (unsharded,x,unsharded) is not compatible with
// (unsharded,x) or (x,unsharded,unsharded) but is compatible with
// (x,unsharded), (unsharded,unsharded) or (unsharded,x,unsharded).
// (Note that due to broadcasting, we compare the dimensions from the end).
//
// If dims_to_ignore is > 0, then we ignore when a mesh dimension is used in
// different tensor dimensions when those dimensions are both in the last
// dims_to_ignore tensor dimensions of each input.
// E.g. If dims_to_ignore = 2, then (unsharded,x,unsharded) is now compatible
// with (unsharded,x) and it not compatible with (x,unsharded,unsharded).
//
// The output layout will be of rank max(layout_a.rank(), layout_b.rank()) -
// dims_to_ignore and will be replicated on a dimension if either one of the
// input layouts is replicated on that dimension. Once again recall due to
// broadcasting, layouts are aligned by their ends and not their beginnings.
// E.g. if dims_to_ignore is zero, the output layout for the inputs
// (unsharded,x,unsharded) and (unsharded,y) is (unsharded,x,y).
// If dims_to_ignore is two, the output for (y,x,unsharded) and
// (unsharded,x) is just (y).
//
// In the case that one tensor is sharded and the other is not on a given
// dimension, element wise operations *may* need to split the unsharded tensor
// along the same mesh dimension that the other input is split on. Note that
// the split is *not* needed if the unsharded tensor has dimension of size 1,
// due to broadcasting.
//
// To help with the needed splittings, the vectors to_split_* are resized to the
// rank of each input and if that dimension of the tensor needs to be split for
// and elementwise op, we record the mesh dimension it should be split along in
// the vector.
// E.g. in the case of input layouts (unsharded,x,unsharded) and
// (unsharded,unsharded) with dimensions (10,10,10) and (10,10),
// to_split_a = {"unsharded", "unsharded", "unsharded"} and to_split_b =
// {"x", "unsharded"}.
// If the shapes were (10,10,10) and (1,10), then to_split_a = {"unsharded",
// "unsharded", "unsharded"} and to_split_b = {"unsharded", "unsharded"}.
//
// Note that "unsharded" == Layout::kUnshardedDim.
// NOTE: shape_a and shape_b are *global* shapes.
StatusOr<Layout> GetBroadcastLayoutForElementWise(
    const Layout& layout_a, const Layout& layout_b,
    mlir::ArrayRef<int64_t> shape_a, mlir::ArrayRef<int64_t> shape_b,
    int64_t dims_to_ignore, std::vector<std::string>& to_split_a,
    std::vector<std::string>& to_split_b) {
  if (layout_a.mesh() != layout_b.mesh())
    return errors::InvalidArgument(
        "layout_a and layout_b cannot be broadcast as they are on different "
        "meshes.");

  const int rank_a = layout_a.rank();
  const int rank_b = layout_b.rank();
  const int rank_offset_a = std::max(0, rank_b - rank_a);
  const int rank_offset_b = std::max(0, rank_a - rank_b);
  absl::flat_hash_map<std::string, int> mesh_dim_map_a;
  absl::flat_hash_map<std::string, int> mesh_dim_map_b;
  std::vector<string> output_layout_specs;

  auto unsharded_specs = [](const int new_size) -> std::vector<std::string> {
    std::vector<std::string> spec_strs(new_size, Layout::kUnshardedDim);
    return spec_strs;
  };

  to_split_a = unsharded_specs(rank_a - dims_to_ignore);
  to_split_b = unsharded_specs(rank_b - dims_to_ignore);

  // Note that we record ranks over all dimensions even ones we ignore.
  // We will check that a non-ignored dimension of a tensor does not use a
  // mesh dimension that is used by an ignored dimension in the other tensor.
  for (int i = 0; i < rank_a; ++i)
    if (!Layout::IsUnshardedDimension(layout_a.sharding_spec(i)))
      mesh_dim_map_a[layout_a.sharding_spec(i)] = i;
  for (int i = 0; i < rank_b; ++i)
    if (!Layout::IsUnshardedDimension(layout_b.sharding_spec(i)))
      mesh_dim_map_b[layout_b.sharding_spec(i)] = i;

  for (int i = 0; i < std::max(rank_a, rank_b) - dims_to_ignore; ++i) {
    const int dim_a = i - rank_offset_a;
    const int dim_b = i - rank_offset_b;
    // When ranks are not equal we treat the first rank_offset_* dims of the
    // shorter layout as not sharded.
    const std::string mesh_dim_a =
        dim_a >= 0 ? layout_a.sharding_spec(dim_a) : Layout::kUnshardedDim;
    const std::string mesh_dim_b =
        dim_b >= 0 ? layout_b.sharding_spec(dim_b) : Layout::kUnshardedDim;
    // When ranks are not equal, we treat the first rank_offset_* dims of the
    // shorter shape as if they were 1.
    const int64_t tensor_dim_a = dim_a >= 0 ? shape_a[dim_a] : 1;
    const int64_t tensor_dim_b = dim_b >= 0 ? shape_b[dim_b] : 1;

    // Check for conflicted dimensions. If occurred, chose unsharded as merged
    // result, if generate_unsharded_dim_for_conflicts is set by call site.
    bool have_conflicted_dim = false;
    if (!Layout::IsUnshardedDimension(mesh_dim_a) &&
        mesh_dim_map_b.contains(mesh_dim_a) &&
        mesh_dim_map_b[mesh_dim_a] != dim_b)
      have_conflicted_dim = true;

    if (!Layout::IsUnshardedDimension(mesh_dim_b) &&
        mesh_dim_map_a.contains(mesh_dim_b) &&
        mesh_dim_map_a[mesh_dim_b] != dim_a)
      have_conflicted_dim = true;

    // If both dimensions are sharded, we have already verified that they are
    // sharded on the same mesh dim.
    if (have_conflicted_dim) {
      output_layout_specs.emplace_back(Layout::kUnshardedDim);
    } else {
      output_layout_specs.emplace_back(
          Layout::IsUnshardedDimension(mesh_dim_a) ? mesh_dim_b : mesh_dim_a);
    }
    if (dim_a >= 0 && tensor_dim_a > 1 &&
        Layout::IsUnshardedDimension(mesh_dim_a) &&
        !Layout::IsUnshardedDimension(mesh_dim_b)) {
      to_split_a[dim_a] = mesh_dim_b;
    }
    if (dim_b >= 0 && tensor_dim_b > 1 &&
        Layout::IsUnshardedDimension(mesh_dim_b) &&
        !Layout::IsUnshardedDimension(mesh_dim_a)) {
      to_split_b[dim_b] = mesh_dim_a;
    }
  }
  return Layout::GetLayout(output_layout_specs, layout_a.mesh());
}

StatusOr<std::optional<Layout>> GetMergedOperandLayout(
    const llvm::DenseMap<int, Layout>& operand_layouts, mlir::Operation* op) {
  // Represents list of Layouts and it's operand index where layout value is
  // defined (i.e. layout is not absl::nullopt).
  llvm::SmallVector<std::pair<const Layout&, llvm::ArrayRef<int64_t>>, 4>
      filtered_preferred_operand_layouts;
  filtered_preferred_operand_layouts.reserve(op->getNumOperands());

  for (const auto& index_and_layout : operand_layouts) {
    TF_ASSIGN_OR_RETURN(
        llvm::ArrayRef<int64_t> shape_to_merge,
        GetShapeOfValue(op->getOperand(index_and_layout.first)));
    filtered_preferred_operand_layouts.emplace_back(index_and_layout.second,
                                                    shape_to_merge);
  }

  if (filtered_preferred_operand_layouts.empty())
    return std::optional<Layout>();

  // Merged all operands and it's layouts to a single broadcasted layout.
  Layout merged_operand_layout = filtered_preferred_operand_layouts[0].first;
  llvm::ArrayRef<int64_t> merged_shape =
      filtered_preferred_operand_layouts[0].second;

  // Statically analyze merged input operands layouts. Broadcasting is allowed
  // but no cross device communication should be incurred.
  for (int i = 1; i < filtered_preferred_operand_layouts.size(); ++i) {
    const auto& operand_index_and_layout_to_merge =
        filtered_preferred_operand_layouts[i];
    const Layout& layout_to_merge = operand_index_and_layout_to_merge.first;
    llvm::ArrayRef<int64_t> shape_to_merge =
        operand_index_and_layout_to_merge.second;

    std::vector<std::string> left_splits;
    std::vector<std::string> right_splits;
    TF_ASSIGN_OR_RETURN(merged_operand_layout,
                        GetBroadcastLayoutForElementWise(
                            merged_operand_layout, layout_to_merge,
                            merged_shape, shape_to_merge,
                            /*dims_to_ignore=*/0, left_splits, right_splits));
  }
  return std::optional<Layout>(merged_operand_layout);
}

mlir::Value GetForwardedDTensorLayoutInput(mlir::Value value) {
  auto layout_op =
      llvm::dyn_cast_or_null<mlir::TF::DTensorLayout>(value.getDefiningOp());
  if (!layout_op) return value;

  return layout_op.getInput();
}

// Takes an operand and traces its use across function call and
// tf_device.cluster boundaries. Note that this may turn one operand into many.
// TODO(bfontain): Assumes that a function is only called once. This is checked
// when creating func_to_caller.
llvm::SmallVector<mlir::OpOperand*, 4> TraceUseToNextTFOp(
    mlir::OpOperand* operand,
    const llvm::DenseMap<llvm::StringRef, mlir::Operation*>& func_to_caller,
    llvm::SmallVector<mlir::Value, 4>* skipped_values) {
  mlir::Operation* owner = operand->getOwner();
  llvm::SmallVector<mlir::Value, 4> values;
  llvm::SmallVector<mlir::Value, 4> unused_values;
  if (mlir::isa<mlir::TF::PartitionedCallOp>(owner) ||
      mlir::isa<mlir::TF::StatefulPartitionedCallOp>(owner)) {
    mlir::func::FuncOp func;
    if (mlir::isa<mlir::TF::PartitionedCallOp>(owner))
      func = mlir::cast<mlir::TF::PartitionedCallOp>(owner).func();
    else
      func = mlir::cast<mlir::TF::StatefulPartitionedCallOp>(owner).func();
    values.emplace_back(func.getArgument(operand->getOperandNumber()));
  } else if (mlir::isa<mlir::tf_device::ReturnOp>(owner)) {
    auto device_return = mlir::cast<mlir::tf_device::ReturnOp>(owner);
    auto enclosing_cluster =
        device_return->getParentOfType<mlir::tf_device::ClusterOp>();
    auto value = enclosing_cluster.getResult(operand->getOperandNumber());
    values.emplace_back(value);
  } else if (mlir::isa<mlir::func::ReturnOp>(owner)) {
    auto func = mlir::cast<mlir::func::ReturnOp>(owner)
                    ->getParentOfType<mlir::func::FuncOp>();
    if (func) {
      // The one function we don't have a caller for is the main function.
      // In this case return the empty list as there are no consumers.
      auto caller = func_to_caller.find(func.getName());
      if (caller != func_to_caller.end()) {
        auto value = caller->second->getOpResult(operand->getOperandNumber());
        values.emplace_back(value);
      }
    } else {
      LOG(WARNING) << "func is null. "
                   << "owner is " << owner;
    }
  } else if (auto yield = mlir::dyn_cast<mlir::TF::YieldOp>(owner)) {
    auto op = owner->getParentOp();
    while (op != nullptr) {
      if (mlir::isa<mlir::TF::IfRegionOp>(op)) {
        break;
      }
      if (mlir::isa<mlir::TF::WhileRegionOp>(op)) {
        break;
      }
      op = op->getParentOp();
    }
    if (auto if_op = mlir::dyn_cast<mlir::TF::IfRegionOp>(op)) {
      auto value = if_op.getResult(operand->getOperandNumber());
      values.emplace_back(value);
    } else if (auto while_op = mlir::dyn_cast<mlir::TF::WhileRegionOp>(op)) {
      if (while_op &&
          !while_op.getCond().isAncestor(yield->getParentRegion())) {
        auto value = while_op.getResult(operand->getOperandNumber());
        values.emplace_back(value);
      }
    } else {
      LOG(WARNING)
          << "Found terminator op for unsupported controlflow operations.";
    }
  } else if (mlir::isa<mlir::TF::DTensorLayout>(owner)) {
    auto dtensor_layout = mlir::cast<mlir::TF::DTensorLayout>(owner);
    auto value = dtensor_layout.getOutput();
    values.emplace_back(value);
  } else if (auto while_op = mlir::dyn_cast<mlir::TF::WhileRegionOp>(owner)) {
    // Handle loop variant inputs of while op.
    mlir::Region& cond = while_op.getCond();
    mlir::Region& body = while_op.getBody();
    const int operand_index = operand->getOperandNumber();
    auto value1 = cond.front().getArgument(operand_index);
    values.emplace_back(value1);
    auto value2 = body.front().getArgument(operand_index);
    values.emplace_back(value2);
  } else {
    return {operand};
  }
  llvm::SmallVector<mlir::OpOperand*, 4> ret;
  for (mlir::Value value : values) {
    if (skipped_values != nullptr) skipped_values->emplace_back(value);
    if (value.use_empty()) continue;
    for (mlir::OpOperand& use : value.getUses()) {
      //  TODO(bfontain): Remove recursion here.
      const auto& traced_operands =
          TraceUseToNextTFOp(&use, func_to_caller, skipped_values);
      ret.append(traced_operands.begin(), traced_operands.end());
    }
  }

  return ret;
}

mlir::LogicalResult GetFuncToCaller(
    mlir::ModuleOp module,
    llvm::DenseMap<llvm::StringRef, mlir::Operation*>& func_to_caller) {
  // For now this is a 1:1 mapping and we will error out if a function is called
  // by more than one op. The layout code assumes there is 1:many relationship
  // between producers and consumers. If we allow a function to be called
  // multiple times, then its consumers consume from multiple producers, which
  // breaks this assumption.
  // TODO(bfontain): Fix this, possibly by duplicating all functions in order to
  // make this mapping 1:1 in truth.
  auto result = module->walk([&](mlir::Operation* op) -> mlir::WalkResult {
    mlir::StringRef func;
    if (mlir::TF::PartitionedCallOp call_op =
            mlir::dyn_cast<mlir::TF::PartitionedCallOp>(op))
      func = call_op.func().getName();
    else if (mlir::TF::StatefulPartitionedCallOp call_op =
                 mlir::dyn_cast<mlir::TF::StatefulPartitionedCallOp>(op))
      func = call_op.func().getName();
    else
      return mlir::WalkResult::advance();
    if (func_to_caller.find(func) != func_to_caller.end())
      return op->emitOpError()
             << "multiple calls found to " << func << " found.";
    func_to_caller[func] = op;
    return mlir::WalkResult::advance();
  });
  return mlir::failure(result.wasInterrupted());
}

mlir::LogicalResult PopulateConsumersFromModule(
    mlir::ModuleOp* module, mlir::Dialect* tf_dialect,
    llvm::DenseMap<mlir::Value, std::vector<mlir::OpOperand*>>& consumers) {
  mlir::func::FuncOp main_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  llvm::DenseMap<llvm::StringRef, mlir::Operation*> func_to_caller;

  if (mlir::failed(GetFuncToCaller(*module, func_to_caller)))
    return mlir::failure();

  module->walk([&](mlir::Operation* op) {
    if (op->getDialect() != tf_dialect) return;

    if (mlir::isa<mlir::TF::PartitionedCallOp>(op) ||
        mlir::isa<mlir::TF::StatefulPartitionedCallOp>(op) ||
        mlir::isa<mlir::TF::WhileRegionOp>(op) ||
        mlir::isa<mlir::TF::IfRegionOp>(op) ||
        mlir::isa<mlir::TF::DTensorLayout>(op))
      return;

    for (const auto& value : op->getOpResults()) {
      // Call clear so that value is in consumers (with an empty vector)even if
      // there are no 'uses'. This should only happen for ops whose outputs are
      // directly to main return, e.g. eagerly executed ops.
      consumers[value].clear();
      for (auto& operand : value.getUses())
        for (auto& traced_operand :
             TraceUseToNextTFOp(&operand, func_to_caller))
          consumers[value].emplace_back(traced_operand);
    }
  });

  // Note that we need to add in the inputs from the main function (otherwise
  // we won't have any layouts to propagate!).
  for (auto& value : main_func.getArguments())
    for (auto& operand : value.getUses())
      for (auto* traced_operand : TraceUseToNextTFOp(&operand, func_to_caller))
        consumers[value].emplace_back(traced_operand);
  return mlir::success();
}

// Compute the mesh coordinates from a device id + the current cluster.
//
// If the mesh shape is [a, b, c, d], then the mesh coordinates are
// [device_id/b/c/d, device_id/c/d%b, device_id/d%c, device_id%d]
// for convenience, since device_id < a*b*c*d, we can apply %a on the first
// coordinate as well for simplicity's sake.
// Thus we can decompose this calculation into the following tf ops:
// tf.FloorMod(tf.Div(device_id, [b*c*d, c*d, d, 1]), [a, b, c, d]) where
// [a, b, c, d] and [b*c*d, c*d, d, 1] are simply precomputed constants.
//
// Note that this returns a tensor of shape [1, mesh.rank()], suitable for
// using with MatMul.
StatusOr<mlir::Value> GetMeshCoordinatesFromCluster(
    mlir::tf_device::ClusterOp cluster) {
  // First try to find a FloorMod op with kMeshCoordinatesAttr attribute that
  // has the given mesh in it. If it exists, simply return that op's value.
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshFromOp(cluster));
  if (!mesh) return errors::InvalidArgument("missing mesh on cluster");
  string serialized_mesh = mesh->ToString();
  mlir::Value ret_val;
  auto result = cluster.walk([&](mlir::TF::FloorModOp op) -> mlir::WalkResult {
    if (op->hasAttrOfType<mlir::StringAttr>(kMeshCoordinatesAttr) &&
        op->getAttrOfType<mlir::StringAttr>(kMeshCoordinatesAttr)
                .getValue()
                .str() == serialized_mesh) {
      ret_val = op.getZ();
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) return ret_val;

  // We didn't find a FloorModOp for the given mesh, so we must produce the
  // FloorModOp and add the attr so we can find it on next call.
  std::vector<int32> mesh_shape(mesh->rank());
  for (int i = 0; i < mesh->rank(); ++i) mesh_shape[i] = mesh->dim(i).size;

  // This product represents the [b*c*d, c*d, d, 1] from the function
  // documentation.
  std::vector<int32> running_product(mesh->rank());
  running_product[mesh->rank() - 1] = 1;
  for (int i = mesh->rank() - 1; i > 0; --i)
    running_product[i - 1] = running_product[i] * mesh_shape[i];

  mlir::OpBuilder builder(cluster.getContext());
  builder.setInsertionPointToStart(&cluster.GetBody());

  auto mesh_shape_type = mlir::RankedTensorType::get(
      {1, mesh->rank()}, builder.getIntegerType(32));
  mlir::Attribute mesh_shape_attr =
      mlir::DenseIntElementsAttr::get(mesh_shape_type, mesh_shape);
  auto mesh_shape_value =
      mlir::TF::ConstOp::create(builder, cluster.getLoc(), mesh_shape_attr)
          .getResult();

  auto running_product_value =
      IntConst(builder, cluster.getLoc(), running_product);

  TF_ASSIGN_OR_RETURN(mlir::Value device_id, DeviceId(cluster));

  auto div_op = mlir::TF::DivOp::create(builder, cluster.getLoc(), device_id,
                                        running_product_value);

  auto mod_op = mlir::TF::FloorModOp::create(builder, cluster.getLoc(),
                                             div_op.getZ(), mesh_shape_value);

  mod_op->setAttr(kMeshCoordinatesAttr, builder.getStringAttr(serialized_mesh));
  return mod_op.getZ();
}

StatusOr<Mesh> GetMeshOnParentCluster(mlir::Operation* op) {
  mlir::tf_device::ClusterOp cluster =
      op->getParentOfType<mlir::tf_device::ClusterOp>();

  auto mesh_attr = cluster->getAttrOfType<mlir::StringAttr>(kMeshAttr);
  if (mesh_attr) {
    return Mesh::FromString(mesh_attr.getValue().str());
  }
  return errors::InvalidArgument("missing mesh attribute on cluster.");
}

mlir::LogicalResult ValidateMetadataAttributes(mlir::Operation* op) {
  // If cluster function has attributes containing inferred layout of resource
  // handle arguments, then add the attributes to the newly created
  // StatefulPartitonedCallOp.
  auto inferred_resource_handle_indices =
      op->getAttrOfType<mlir::DenseIntElementsAttr>(kNewResourceLayoutIndices);
  auto inferred_resource_handle_layouts =
      op->getAttrOfType<mlir::ArrayAttr>(kNewResourceArgLayouts);
  if (inferred_resource_handle_indices || inferred_resource_handle_layouts) {
    if (!inferred_resource_handle_indices ||
        !inferred_resource_handle_layouts ||
        inferred_resource_handle_indices.getNumElements() !=
            inferred_resource_handle_layouts.size())
      return op->emitOpError(
                 "inferred layout args doesn't match. indices size: ")
             << (inferred_resource_handle_indices
                     ? inferred_resource_handle_indices.getNumElements()
                     : 0)
             << ", layouts size : "
             << (inferred_resource_handle_layouts
                     ? inferred_resource_handle_layouts.size()
                     : 0);
  }

  auto shape_layouts = op->getAttrOfType<mlir::ArrayAttr>(kShapeOpInputLayout);
  auto shape_op_indices =
      op->getAttrOfType<mlir::DenseIntElementsAttr>(kShapeOpInputLayoutIndices);
  if (shape_op_indices || shape_layouts) {
    if (!shape_op_indices || !shape_layouts ||
        shape_op_indices.getNumElements() != shape_layouts.size())
      return op->emitOpError("shape layout args doesn't match. indices size: ")
             << (shape_op_indices ? shape_op_indices.getNumElements() : 0)
             << ", layouts size : "
             << (shape_layouts ? shape_layouts.size() : 0);
  }
  return mlir::success();
}

void RemoveUnusedClusterResults(mlir::tf_device::ClusterOp cluster) {
  llvm::SmallVector<mlir::OpResult, 4> new_result_values;
  llvm::SmallVector<mlir::Value, 4> result_producing_values;
  new_result_values.reserve(cluster->getNumResults());
  result_producing_values.reserve(cluster->getNumResults());
  for (mlir::OpResult result : cluster.getResults()) {
    if (!result.use_empty()) {
      new_result_values.emplace_back(result);
      result_producing_values.emplace_back(
          cluster.GetBody().getTerminator()->getOperand(
              result.getResultNumber()));
    }
  }

  if (new_result_values.size() == cluster.getNumResults()) return;

  llvm::SmallVector<mlir::Type, 4> new_result_types;
  llvm::transform(new_result_values, std::back_inserter(new_result_types),
                  [](mlir::Value v) { return v.getType(); });

  mlir::OpBuilder builder(cluster);
  auto new_cluster = mlir::tf_device::ClusterOp::create(
      builder, cluster.getLoc(), new_result_types);
  new_cluster->setAttr(kMeshAttr,
                       cluster->getAttrOfType<mlir::StringAttr>(kMeshAttr));
  new_cluster.getBody().push_back(new mlir::Block);

  auto& cluster_body = cluster.GetBody().getOperations();
  new_cluster.GetBody().getOperations().splice(
      new_cluster.GetBody().end(), cluster_body, cluster_body.begin(),
      std::prev(cluster_body.end()));

  builder.setInsertionPointToEnd(&new_cluster.GetBody());
  mlir::tf_device::ReturnOp::create(builder, cluster.getLoc(),
                                    result_producing_values);

  assert(new_cluster.getNumResults() == new_result_values.size());
  for (auto it : llvm::zip(new_result_values, new_cluster.getResults())) {
    mlir::Value value_to_replace = std::get<0>(it);
    mlir::Value new_result = std::get<1>(it);
    value_to_replace.replaceAllUsesWith(new_result);
  }
  cluster.erase();
}

namespace {

// Keeps track of number of functions added to the global graph for adding
// control flows. When converting regional control flow to functional control
// flow ops, function names may collide if non-unique branch function names are
// used. In order to ensure that all branch functions of TF control flow ops are
// unique, we keep track of atomic counter for each control flow functions.
// See b/174253694 for more details.
std::atomic<int32> dtensor_controlflow_function_counter{0};

}  // namespace

mlir::StringAttr GetUniqueControlflowFnName(const std::string& prefix,
                                            mlir::OpBuilder& builder) {
  int32 unique_id = dtensor_controlflow_function_counter++;
  return builder.getStringAttr(
      absl::StrCat(prefix, "_dtensor_function_", unique_id));
}

absl::Status SetBuilderInsertionAfterValue(mlir::Value value,
                                           mlir::OpBuilder& builder) {
  if (mlir::isa<mlir::OpResult>(value)) {
    builder.setInsertionPointAfterValue(value);
    return absl::OkStatus();
  }
  mlir::tf_device::ClusterOp cluster;
  for (mlir::Operation* op : value.getUsers()) {
    mlir::tf_device::ClusterOp new_cluster =
        op->getParentOfType<mlir::tf_device::ClusterOp>();
    if (!new_cluster) continue;
    if (!cluster) cluster = new_cluster;
    if (cluster != new_cluster)
      return errors::Internal("value has multiple uses in different clusters");
  }
  if (!cluster) return errors::Internal("value not used in any cluster");

  builder.setInsertionPointToStart(cluster.SingleBlock::getBody());
  return absl::OkStatus();
}

absl::Status PrintTensor(mlir::Value value,
                         const std::string& format_string = "%s") {
  mlir::OpBuilder builder(value.getContext());
  builder.setInsertionPointAfterValue(value);
  TF_ASSIGN_OR_RETURN(mlir::Value device_id, DeviceId(value));
  std::string all_format = absl::StrCat("Core %s: ", format_string);
  // Scalar string type
  mlir::RankedTensorType scalar_string =
      mlir::RankedTensorType::get({}, builder.getType<mlir::TF::StringType>());
  mlir::TF::StringFormatOp format =
      mlir::TF::StringFormatOp::create(builder, value.getLoc(), scalar_string,
                                       mlir::ValueRange({device_id, value}));
  format->setAttr("template", builder.getStringAttr(all_format));
  mlir::TF::PrintV2Op::create(builder, value.getLoc(), format.getOutput(),
                              /*output_stream=*/"log(info)",
                              /*end=*/"\n");
  return absl::OkStatus();
}

absl::Status ExtractConstStringVectorFromValue(
    mlir::Value value, llvm::SmallVectorImpl<std::string>& out_vector) {
  value = GetForwardedDTensorLayoutInput(value);
  if (mlir::isa<mlir::BlockArgument>(value))
    return errors::Internal("Unable get constant value from block argument.");
  mlir::DenseStringElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) {
    return errors::Internal(
        llvm::formatv("failed to extract constant string vector from : {0}",
                      value)
            .str());
  }
  for (const auto& str : attr.getRawStringData()) {
    out_vector.push_back(str.str());
  }
  return absl::OkStatus();
}

StatusOr<std::string> ExtractConstScalarStringFromValue(mlir::Value value) {
  value = GetForwardedDTensorLayoutInput(value);
  if (mlir::isa<mlir::BlockArgument>(value))
    return errors::Internal("Unable get constant value from block argument.");
  mlir::DenseStringElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) {
    return errors::Internal(absl::StrCat("required constant value for ",
                                         OpName(value.getDefiningOp())));
  }
  if (attr.size() != 1) {
    return errors::Internal(absl::StrCat("expected 1 element, got ",
                                         attr.size(), " for ",
                                         OpName(value.getDefiningOp())));
  }
  return std::string(*attr.getRawStringData().begin());
}

}  // namespace dtensor
}  // namespace tensorflow
