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

#include "tensorflow/dtensor/mlir/expansions/random_op_spmd_expander.h"

#include <algorithm>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_location.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

absl::Status CheckLayoutIsSupported(const Layout& layout) {
  // Currently we support small mesh rank for arbitrary layout.
  if (layout.mesh().rank() > 3)
    return errors::InvalidArgument("Large mesh rank size is not supported",
                                   layout.ToString());

  return absl::OkStatus();
}

absl::Status ValidateShapeAndGetNewShape(
    const llvm::SmallVector<int64_t, 4>& op_shape, const Layout& layout,
    llvm::SmallVectorImpl<int64_t>& new_random_shape) {
  TF_RETURN_IF_ERROR(CheckLayoutIsSupported(layout));

  // Validate that sharding of random op is compatible with it's user defined
  // shape and calculate new shape of local random op.
  const auto op_sharding = layout.num_shards();
  new_random_shape.reserve(op_shape.size());

  if (op_sharding.size() != op_shape.size())
    return errors::InvalidArgument(
        "Sharding dimension of random op does not match rank of the "
        "random op. Received sharding: ",
        layout.ToString());

  for (int i = 0; i < op_sharding.size(); ++i) {
    const auto dimension_sharding = op_sharding[i];
    const auto op_dimension_size = op_shape[i];
    if (op_dimension_size % dimension_sharding != 0) {
      return errors::InvalidArgument(
          "Sharding of random op incompatible with shape. Received "
          "sharding: ",
          layout.ToString());
    }
    new_random_shape.emplace_back(op_dimension_size / dimension_sharding);
  }
  return absl::OkStatus();
}

// Get a device seed for this layout and device_id.
//
// The computation will be inserted directly after the mesh coordinate
// computation in the current cluster. First we search for a Squeeze with the
// attribute kDeviceSeedForMeshDims = layout.mesh_dims
// If it exists, we return that, otherwise we insert the ops to compute a device
// seed.
StatusOr<mlir::Value> GetDeviceSeed(const Layout& layout, mlir::Operation* op) {
  // We need both a set, to check for membership and a vector that we sort
  // to use as the attribute attached to the squeeze op.
  llvm::SmallVector<int32_t, 4> layout_dims;
  llvm::SmallSet<int32_t, 4> layout_dims_set;
  for (const auto& spec : layout.sharding_spec_strs()) {
    if (Layout::IsUnshardedDimension(spec)) continue;
    layout_dims.emplace_back(layout.mesh().GetMeshDimIndexWithName(spec));
    layout_dims_set.insert(layout_dims.back());
  }
  llvm::sort(layout_dims);

  mlir::tf_device::ClusterOp cluster =
      op->getParentOfType<mlir::tf_device::ClusterOp>();
  if (!cluster)
    return errors::InvalidArgument(
        "random op not in ClusterOp when it should be");

  for (mlir::TF::SqueezeOp squeeze : cluster.getOps<mlir::TF::SqueezeOp>())
    if (squeeze->hasAttrOfType<mlir::DenseIntElementsAttr>(
            kDeviceSeedForMeshDims) &&
        std::equal(layout_dims.begin(), layout_dims.end(),
                   squeeze
                       ->getAttrOfType<mlir::DenseIntElementsAttr>(
                           kDeviceSeedForMeshDims)
                       .getValues<uint32_t>()
                       .begin()))
      return squeeze.getOutput();

  TF_ASSIGN_OR_RETURN(mlir::Value mesh_coordinates,
                      GetMeshCoordinatesFromCluster(cluster));

  mlir::OpBuilder builder(cluster.getContext());
  builder.setInsertionPointAfterValue(mesh_coordinates);

  // mesh_coordinates is a [1, mesh.rank()] shaped tensor containing the current
  // mesh coordinates of the device.
  // If there are 4 mesh dimensions [w, x, y, z] and only [w, x, z] are used in
  // this layout then one way of getting the device id would be
  // w_coord + x_coord*size_w + z_coord*size_x*size_w
  // Note that only the dims in layout_dims count.
  llvm::SmallVector<uint32_t, 4> multipliers(layout.mesh().rank(), 0);

  // By starting with 65536, we effective perform a left shift of the id by
  // 16 bits.
  int32_t running_product = 65536;
  for (int i = 0; i < layout.mesh().rank(); ++i) {
    if (layout_dims_set.contains(i)) {
      multipliers[i] = running_product;
      running_product = running_product * layout.mesh().dim_sizes()[i];
    }
  }

  mlir::RankedTensorType const_type = mlir::RankedTensorType::get(
      {static_cast<int64>(multipliers.size()), 1}, builder.getIntegerType(32));
  mlir::Attribute const_attr =
      mlir::DenseIntElementsAttr::get(const_type, multipliers);
  mlir::Value multiplier =
      builder.create<mlir::TF::ConstOp>(cluster.getLoc(), const_attr)
          .getOutput();

  const mlir::RankedTensorType one_by_one =
      mlir::RankedTensorType::get({1, 1}, builder.getIntegerType(32));

  mlir::Value seed = builder.create<mlir::TF::MatMulOp>(
      cluster.getLoc(), one_by_one, mesh_coordinates, multiplier);

  // Largest prime in 16 bits.
  mlir::Value prime = CreateIntScalarConst(
      /*value=*/65521, builder, cluster.getLoc(), /*use_int64=*/false);

  mlir::Value seed_plus_prime =
      builder
          .create<mlir::TF::AddV2Op>(cluster.getLoc(), one_by_one, seed, prime)
          .getZ();

  mlir::TF::SqueezeOp squeeze = builder.create<mlir::TF::SqueezeOp>(
      cluster.getLoc(),
      mlir::RankedTensorType::get({}, builder.getIntegerType(32)),
      seed_plus_prime, builder.getI64ArrayAttr({0, 1}));

  squeeze->setAttr(kDeviceSeedForMeshDims,
                   builder.getI32TensorAttr(layout_dims));

  return squeeze.getOutput();
}

// Compute the new local shape for SPMD expansion and ensure it is valid.
template <typename RandomOp>
StatusOr<llvm::SmallVector<int64_t, 4>> GetNewLocalShape(mlir::Operation* op,
                                                         const Layout& layout) {
  auto random_op = llvm::cast<RandomOp>(op);
  llvm::SmallVector<int64_t, 4> op_shape;
  TF_RETURN_IF_ERROR(
      ExtractConstVectorFromValue(random_op.getShape(), &op_shape));

  // Validate that sharding of random op is compatible with it's user defined
  // shape and calculate new shape of local random op.
  llvm::SmallVector<int64_t, 4> new_random_shape;
  TF_RETURN_IF_ERROR(
      ValidateShapeAndGetNewShape(op_shape, layout, new_random_shape));
  return new_random_shape;
}

// Calculate the new local seed
template <typename RandomOp>
StatusOr<mlir::Value> ComputeNewSeed(mlir::OpBuilder& builder,
                                     mlir::Operation* op, const Layout& layout,
                                     mlir::Location& location,
                                     mlir::Value op_seed) {
  TF_ASSIGN_OR_RETURN(auto device_id_seed, GetDeviceSeed(layout, op));
  mlir::Type seed_type =
      mlir::cast<mlir::TensorType>(op_seed.getType()).getElementType();

  device_id_seed = builder.create<mlir::TF::CastOp>(
      location, mlir::RankedTensorType::get({}, seed_type), device_id_seed);

  mlir::Value seed_xor =
      builder.create<mlir::TF::BitwiseXorOp>(location, op_seed, device_id_seed);
  return seed_xor;
}

template <typename RandomOp>
StatusOr<mlir::Operation*> CreatedShardedLocalRandomOpV1(const Layout& layout,
                                                         mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto new_random_shape,
                      GetNewLocalShape<RandomOp>(op, layout));

  // Create new seed using already existing seed and a device id.
  mlir::OpBuilder builder(op);
  auto location = DT_LOC(op);

  auto random_op = llvm::cast<RandomOp>(op);
  // Create device_id_seed for local RNG.
  TF_ASSIGN_OR_RETURN(auto seed_xor,
                      ComputeNewSeed<RandomOp>(builder, op, layout, location,
                                               random_op.getSeed()));

  // Create a new random op with new `local` shape and newly generated seed.
  // StatelessRandom op is used to make random op SPMD expansion
  // deterministic.
  mlir::Type new_random_type = mlir::RankedTensorType::get(
      new_random_shape, mlir::cast<mlir::TensorType>(op->getResult(0).getType())
                            .getElementType());

  auto new_shape_value = Int64Const(builder, location, new_random_shape);
  // TODO(zhonglinhan) : check different input for StatelessRandomUniformInt
  auto local_random = builder.create<RandomOp>(location, new_random_type,
                                               new_shape_value, seed_xor);
  op->getResult(0).replaceAllUsesWith(local_random.getOutput());
  op->erase();
  return local_random.getOperation();
}

template <typename RandomOp>
StatusOr<mlir::Operation*> CreatedShardedLocalRandomOpV2(const Layout& layout,
                                                         mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto new_random_shape,
                      GetNewLocalShape<RandomOp>(op, layout));

  // Create new seed using already existing seed and a device id.
  mlir::OpBuilder builder(op);
  auto location = DT_LOC(op);

  auto random_op = llvm::cast<RandomOp>(op);
  // Create device_id_seed for local RNG.
  TF_ASSIGN_OR_RETURN(auto seed_xor,
                      ComputeNewSeed<RandomOp>(builder, op, layout, location,
                                               random_op.getKey()));

  // Create a new random op with new `local` shape and newly generated seed.
  // StatelessRandom op is used to make random op SPMD expansion
  // deterministic.
  mlir::Type new_random_type = mlir::RankedTensorType::get(
      new_random_shape, mlir::cast<mlir::TensorType>(op->getResult(0).getType())
                            .getElementType());

  auto new_shape_value = Int64Const(builder, location, new_random_shape);

  auto local_random = builder.create<RandomOp>(
      location, new_random_type, new_shape_value, seed_xor,
      random_op.getCounter(), random_op.getAlg());
  op->getResult(0).replaceAllUsesWith(local_random.getOutput());
  op->erase();
  return local_random.getOperation();
}

template <typename RandomOp>
StatusOr<mlir::Operation*> CreatedShardedLocalRandomOpV2Range(
    const Layout& layout, mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto new_random_shape,
                      GetNewLocalShape<RandomOp>(op, layout));

  // Create new seed using already existing seed and a device id.
  mlir::OpBuilder builder(op);
  auto location = DT_LOC(op);

  auto random_op = llvm::cast<RandomOp>(op);
  // Create device_id_seed for local RNG.
  TF_ASSIGN_OR_RETURN(auto seed_xor,
                      ComputeNewSeed<RandomOp>(builder, op, layout, location,
                                               random_op.getKey()));

  // Create a new random op with new `local` shape and newly generated seed.
  // StatelessRandom op is used to make random op SPMD expansion
  // deterministic.
  mlir::Type new_random_type = mlir::RankedTensorType::get(
      new_random_shape, mlir::cast<mlir::TensorType>(op->getResult(0).getType())
                            .getElementType());

  auto new_shape_value = Int64Const(builder, location, new_random_shape);

  auto local_random = builder.create<RandomOp>(
      location, new_random_type, new_shape_value, seed_xor,
      random_op.getCounter(), random_op.getAlg(), random_op.getMinval(),
      random_op.getMaxval());
  op->getResult(0).replaceAllUsesWith(local_random.getOutput());
  op->erase();
  return local_random.getOperation();
}

}  // namespace

StatusOr<mlir::Operation*> RandomOpSPMDExpander::ExpandOp(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto layout, ExtractSingleLayoutFromOp(op));

  if (!layout)
    return errors::InvalidArgument(
        "layout of Random op must be known before SPMD expansion.");

  // For fully replicated random ops, all devices have the same random
  // value. As so, SPMD expansion is a no-op.
  if (layout->IsFullyReplicated()) return op;
  if (llvm::isa<mlir::TF::StatelessRandomUniformOp>(op)) {
    return CreatedShardedLocalRandomOpV1<mlir::TF::StatelessRandomUniformOp>(
        *layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessRandomUniformFullIntOp>(op)) {
    return CreatedShardedLocalRandomOpV1<
        mlir::TF::StatelessRandomUniformFullIntOp>(*layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessRandomNormalOp>(op)) {
    return CreatedShardedLocalRandomOpV1<mlir::TF::StatelessRandomNormalOp>(
        *layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessTruncatedNormalOp>(op)) {
    return CreatedShardedLocalRandomOpV1<mlir::TF::StatelessTruncatedNormalOp>(
        *layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessRandomUniformFullIntV2Op>(op)) {
    return CreatedShardedLocalRandomOpV2<
        mlir::TF::StatelessRandomUniformFullIntV2Op>(*layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessRandomNormalV2Op>(op)) {
    return CreatedShardedLocalRandomOpV2<mlir::TF::StatelessRandomNormalV2Op>(
        *layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessTruncatedNormalV2Op>(op)) {
    return CreatedShardedLocalRandomOpV2<
        mlir::TF::StatelessTruncatedNormalV2Op>(*layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessRandomUniformV2Op>(op)) {
    return CreatedShardedLocalRandomOpV2<mlir::TF::StatelessRandomUniformV2Op>(
        *layout, op);
  }
  if (llvm::isa<mlir::TF::StatelessRandomUniformIntV2Op>(op)) {
    return CreatedShardedLocalRandomOpV2Range<
        mlir::TF::StatelessRandomUniformIntV2Op>(*layout, op);
  }
  return errors::Unimplemented(absl::StrCat(
      "SPMD expansion for op : ", OpName(op), " is not implemented"));
}

StatusOr<llvm::DenseMap<int, Layout>>
RandomOpSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  llvm::DenseMap<int, Layout> output_layouts;

  // For random op, input is always replicated and we always respect layouts
  // from consumers.
  for (int i = 0; i < op->getNumResults(); ++i) {
    output_layouts[i] =
        Layout::ReplicatedOnMesh(mesh, /*rank=*/ValueRank(op->getResult(i)));
  }
  return output_layouts;
}

StatusOr<llvm::DenseMap<int, Layout>>
RandomOpSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  llvm::DenseMap<int, Layout> input_layouts;

  // For random op, default the input layout as replicated layout.
  for (int i = 0; i < op->getNumOperands(); ++i) {
    input_layouts[i] =
        Layout::ReplicatedOnMesh(mesh, /*rank=*/ValueRank(op->getOperand(i)));
  }
  return input_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
