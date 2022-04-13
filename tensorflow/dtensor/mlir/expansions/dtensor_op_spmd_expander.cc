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

#include "tensorflow/dtensor/mlir/expansions/dtensor_op_spmd_expander.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/device_utils.h"
#include "tensorflow/dtensor/mlir/dtensor_send_recv.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Validates send/recv layout and mesh configurations. Among other things, this
// checks for below constraints.
// 1. Src/target layouts have non empty mesh.
// 2. Src/target layouts have the same host.
// 3. Src/target layouts are from different mesh.
// 4. One of scr/target layout is from host mesh cluster.
// 5. CPU host cluster mesh has 1 device.
Status ValidateSendRecvLayoutConfiguration(mlir::TF::DTensorSend dtensor_send,
                                           mlir::TF::DTensorRecv dtensor_recv) {
  // If either one of the send/recv ops has already been lowered, then send/recv
  // configuration has already been verified.
  if (!dtensor_send || !dtensor_recv) return Status::OK();

  TF_ASSIGN_OR_RETURN(const absl::optional<Layout> send_layout_or_null,
                      ExtractLayoutFromOperand(dtensor_send.input()));

  if (!send_layout_or_null.has_value())
    return errors::InvalidArgument(
        "Input to DTensorSend must have specified layout.");

  const Layout& send_layout = send_layout_or_null.value();
  const Layout recv_layout = dtensor_recv.layout();

  const Mesh& send_mesh = send_layout.mesh();
  const Mesh& recv_mesh = recv_layout.mesh();

  // If any one of send/recv mesh are empty, return error.
  if (send_mesh.IsEmpty() || recv_mesh.IsEmpty())
    return errors::InvalidArgument(
        "Found empty mesh when sending/receiving tensor across clusters.");

  // If send host not found in list of receiving hosts, return error.
  std::vector<std::string> send_hosts = send_layout.ReducedMesh().hosts();
  std::vector<std::string> recv_hosts = recv_layout.ReducedMesh().hosts();
  if (send_hosts != recv_hosts)
    return errors::InvalidArgument("Send and receive hosts don't match");

  // Check shards in sending host match those in the receiving host.
  const auto send_host_shard_map = send_layout.HostShardMap();
  const auto recv_host_shard_map = recv_layout.HostShardMap();
  for (const std::string& host : send_hosts) {
    const ShardVector& shards_in_send_host =
        send_host_shard_map.find(host)->second;
    ShardVector shards_in_recv_host = recv_host_shard_map.find(host)->second;
    if (shards_in_send_host != shards_in_recv_host)
      return errors::InvalidArgument(
          "Send and receive host shard vectors don't match. Send shard_vector:",
          shards_in_send_host.ToString(),
          " / Recv host spec : ", shards_in_recv_host.ToString());
  }

  // Send/Recv mesh must be different.
  if (recv_mesh == send_mesh)
    return errors::InvalidArgument(
        "Found CopyToMesh op sending tensor to same mesh. Only use "
        "CopyToMesh to transfer data across different mesh cluster. For "
        "changing layout within the same mesh, use tf.Relayout op.");

  // Either one of send/recv pair must be to/from CPU mesh.
  // For example, TPU mesh -> GPU mesh or TPU mesh -> another TPU mesh
  // is disallowed.
  if (!send_mesh.is_cpu_mesh() && !recv_mesh.is_cpu_mesh())
    return errors::InvalidArgument(
        "tf.CopyToMesh op must be used to send data from/to host mesh.");

  return Status::OK();
}

// Returns whether to lower DTensorSend/DTensorRecv op to xla backend ops.
// Xla backend ops are used when either sending/receiving device uses XLA
// compiler.
bool SendRecvOpUsesXla(const Mesh& send_mesh, const Mesh& recv_mesh) {
  assert(!(send_mesh.is_tpu_mesh() && recv_mesh.is_tpu_mesh()));
  return (send_mesh.is_tpu_mesh() || recv_mesh.is_tpu_mesh());
}

// Takes relayout which may have kMatch dimensions and uses it to mask input.
// Here source_layout
StatusOr<Layout> MergeLayouts(
    const absl::flat_hash_set<std::string>& used_mesh_dimensions,
    const Layout& mask_layout, const Layout& target_layout) {
  std::vector<std::string> sharding_specs(mask_layout.sharding_spec_strs());
  for (int i = 0; i < target_layout.rank(); ++i) {
    if (sharding_specs[i] == Layout::kMatch &&
        !used_mesh_dimensions.contains(target_layout.sharding_spec(i)))
      sharding_specs[i] = target_layout.sharding_spec(i);
  }
  return Layout::GetLayout(sharding_specs, target_layout.mesh());
}

// Given one side of layouts, compute the other side of the layouts.
// Note that this implies that we compute the same layout for the
// operand and output.
StatusOr<llvm::DenseMap<int, Layout>> ComputeRelayoutLayout(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& layouts) {
  mlir::TF::RelayoutOp relayout = llvm::cast<mlir::TF::RelayoutOp>(op);
  mlir::StringRef layout_attr = relayout.layout();
  TF_ASSIGN_OR_RETURN(const Layout mask_layout,
                      Layout::FromString(layout_attr.str()));

  absl::flat_hash_set<std::string> used_dimensions;
  bool match_present = false;
  for (const std::string& sharding_spec : mask_layout.sharding_spec_strs()) {
    if (sharding_spec == Layout::kMatch)
      match_present = true;
    else if (Layout::IsShardedDimension(sharding_spec))
      used_dimensions.insert(sharding_spec);
  }
  if (!match_present) {
    return llvm::DenseMap<int, Layout>({{0, mask_layout}});
  }

  if (layouts.find(0) != layouts.end()) {
    TF_ASSIGN_OR_RETURN(
        Layout new_layout,
        MergeLayouts(used_dimensions, mask_layout, layouts.lookup(0)));
    return llvm::DenseMap<int, Layout>({{0, new_layout}});
  }
  return llvm::DenseMap<int, Layout>();
}
}  // namespace

StatusOr<mlir::Operation*> RelayoutSPMDExpander::ExpandOp(mlir::Operation* op) {
  mlir::TF::RelayoutOp relayout = mlir::cast<mlir::TF::RelayoutOp>(op);
  mlir::StringRef layout_attr = relayout.layout();
  TF_ASSIGN_OR_RETURN(const Layout target_layout,
                      Layout::FromString(layout_attr.str()));
  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(const Layout input_layout,
                      ExtractRequiredLayoutFromOperand(relayout.input()));
  bool match_present = false;
  for (const std::string& sharding_spec : target_layout.sharding_spec_strs())
    if (sharding_spec == Layout::kMatch) match_present = true;

  if (!match_present && output_layout != target_layout)
    return errors::Internal(
        "output layout of Relayout op after layout propagation does not match "
        "layout specified by Relayout op.");

  if (input_layout == output_layout) {
    // Input of RelayoutOp must be output value from DTensorLayout operation
    // as layout propagation adds DTensorLayout op for each tensor values.
    // Replace with identity op.
    mlir::OpBuilder builder(relayout);
    mlir::TF::IdentityOp op = builder.create<mlir::TF::IdentityOp>(
        relayout.getLoc(), relayout.input().getType(), relayout.input());
    relayout.output().replaceAllUsesWith(op.output());
    relayout.erase();
    return op.getOperation();
  }

  auto value_or_status =
      EmitRelayout(relayout.input(), input_layout, output_layout);
  if (!value_or_status.ok())
    return errors::InvalidArgument(
        llvm::formatv("Unsupported layout received for tf.Relayout op. Trying "
                      "to set tensor "
                      "to layout : {0}. Found error {1}",
                      layout_attr.str(),
                      value_or_status.status().error_message())
            .str());
  mlir::Value output = value_or_status.ValueOrDie();
  relayout.output().replaceAllUsesWith(output);
  relayout.erase();
  return output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>>
RelayoutSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  return ComputeRelayoutLayout(op, input_layouts);
}

StatusOr<llvm::DenseMap<int, Layout>>
RelayoutSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  return ComputeRelayoutLayout(op, output_layouts);
}

namespace {

// Returns whether send/recv layout represents send/recv of tensor between
// i-th TPU device and i-th device of the host mesh. Host mesh represents the
// CPU devices that are 1-to-1 mapped with the TPU mesh devices, having the same
// global and local device IDs.
bool IsOneToOneHostMeshTransfer(const Layout& send_layout,
                                const Layout& recv_layout) {
  const Mesh& send_mesh = send_layout.mesh();
  const Mesh& recv_mesh = recv_layout.mesh();

  // Check tensor is being transferred between CPU <-> TPU.
  if (!(send_mesh.is_tpu_mesh() && recv_mesh.is_cpu_mesh()) &&
      !(recv_mesh.is_tpu_mesh() && send_mesh.is_cpu_mesh()))
    return false;

  // Check tensor transfer is happening between TPU and its host mesh.
  if (!((send_mesh.is_tpu_mesh() &&
         send_mesh.tpu_host_mesh() == recv_mesh.ToString()) ||
        (recv_mesh.is_tpu_mesh() &&
         recv_mesh.tpu_host_mesh() == send_mesh.ToString())))
    return false;

  // Check local device IDs are fully matching so that there is no cross-host
  // transfer.
  if (send_mesh.local_device_ids() != recv_mesh.local_device_ids())
    return false;

  return send_layout.GetShardVector() == recv_layout.GetShardVector();
}

}  // namespace

StatusOr<mlir::Operation*> DTensorSendSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
  auto dtensor_send = llvm::cast<mlir::TF::DTensorSend>(op);

  TF_ASSIGN_OR_RETURN(mlir::Operation * recv_op,
                      GetCorrespondingDTensorSendRecvOp<mlir::TF::DTensorSend>(
                          module, dtensor_send));
  auto dtensor_recv = llvm::dyn_cast<mlir::TF::DTensorRecv>(recv_op);

  TF_RETURN_IF_ERROR(
      ValidateSendRecvLayoutConfiguration(dtensor_send, dtensor_recv));

  TF_ASSIGN_OR_RETURN(const Layout input_layout,
                      ExtractRequiredLayoutFromOperand(dtensor_send.input()));

  // Is tensor transfer is from TPU mesh to host mesh and send layout and recv
  // layout is identical, then tensor from each source device is sent to
  // target device asynchronously.
  if (IsOneToOneHostMeshTransfer(input_layout, dtensor_send.target_layout())) {
    return LowerDTensorSendToXlaOp(input_layout, dtensor_send.input(),
                                   dtensor_send,
                                   /*send_from_device_zero=*/false);
  }

  // Calculate input tensor layout of data to send and target fully replicated
  // layout. For now, we ensure that all data transfer happen with fully
  // replicated tensors.
  const int rank = ValueRank(dtensor_send.input());
  const Layout target_layout =
      Layout::ReplicatedOnMesh(input_layout.mesh(), rank);

  // Convert tensor to send to replicated layout.
  mlir::OpBuilder builder(dtensor_send);
  TF_ASSIGN_OR_RETURN(mlir::Value send_input,
                      EmitAllGather(builder, dtensor_send.input(), input_layout,
                                    target_layout));

  // Insert control flow such that only device with device ordinal == 0 sends
  // the tensor data across mesh.
  auto send_cluster =
      dtensor_send->getParentOfType<mlir::tf_device::ClusterOp>();
  TF_ASSIGN_OR_RETURN(absl::optional<Mesh> mesh,
                      ExtractDeviceMeshFromOp(send_cluster));
  if (!mesh.has_value())
    return errors::InvalidArgument(
        "failed to lower DTensor CopyToMesh op as sending side mesh is not "
        "specified.");

  mlir::Location loc = dtensor_send.getLoc();
  TF_ASSIGN_OR_RETURN(
      mlir::Value device_ordinal,
      GetDeviceOrdinal(*mesh, loc,
                       send_cluster->getParentOfType<mlir::func::FuncOp>(),
                       &builder));
  mlir::Value predicate = builder.create<mlir::TF::EqualOp>(
      loc, device_ordinal, CreateIntScalarConst(0, builder, loc),
      /*incompatible_shape_error=*/builder.getBoolAttr(true));

  auto send_if = builder.create<mlir::TF::IfRegionOp>(
      loc, llvm::SmallVector<mlir::Type, 4>{}, predicate,
      /*is_stateless=*/builder.getBoolAttr(true),
      GetUniqueControlflowFnName("copy_to_mesh_send_if_then", builder),
      GetUniqueControlflowFnName("copy_to_mesh_send_if_else", builder));

  // Create empty else branch region.
  auto& else_branch = send_if.else_branch();
  else_branch.push_back(new mlir::Block);
  builder.setInsertionPointToEnd(&else_branch.front());
  builder.create<mlir::TF::YieldOp>(loc,
                                    /*operands=*/llvm::ArrayRef<mlir::Value>{});

  // Create then branch region with DTensorSend op.
  auto& then_branch = send_if.then_branch();
  then_branch.push_back(new mlir::Block);
  builder.setInsertionPointToEnd(&then_branch.front());
  auto yield = builder.create<mlir::TF::YieldOp>(
      loc, /*operands=*/llvm::ArrayRef<mlir::Value>{});
  dtensor_send->moveBefore(yield);

  // Lower DTensorSend op to actual TF op.
  TF_ASSIGN_OR_RETURN(const Mesh recv_mesh,
                      ExtractDeviceMeshEnclosingCluster(recv_op));
  mlir::Operation* lowered_send;
  if (SendRecvOpUsesXla(input_layout.mesh(), recv_mesh)) {
    // Lower DTensorSend op to Xla Send ops.
    TF_ASSIGN_OR_RETURN(
        lowered_send,
        LowerDTensorSendToXlaOp(input_layout, send_input, dtensor_send,
                                /*send_from_device_zero=*/true));
  } else if (input_layout.mesh().is_cpu_mesh() &&
             target_layout.mesh().is_cpu_mesh()) {
    // Lower DTensorSend op to TF Host Send op.
    TF_ASSIGN_OR_RETURN(
        lowered_send,
        LowerDTensorSendFromCPUToTFOp(input_layout, send_input, dtensor_send));
  } else {
    // TODO(hongjunchoi): Implement SPMD transformation lowering that lowers
    // DTensorSend to vanilla TF Send op.
    return errors::Unimplemented(
        "CopyToMesh between CPU/GPU not implemented yet.");
  }

  return lowered_send;
}

// DTensorSend op respects input layout from input operations and does not
// set any preferred inputs layouts. During SPMD expansion, however, tensor
// values are changed to replicated layout before transferring data across mesh
// cluster.
StatusOr<llvm::DenseMap<int, Layout>>
DTensorSendSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  return llvm::DenseMap<int, Layout>();
}

StatusOr<llvm::DenseMap<int, Layout>>
DTensorSendSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  return llvm::DenseMap<int, Layout>();
}

StatusOr<mlir::Operation*> DTensorRecvSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
  auto dtensor_recv = llvm::cast<mlir::TF::DTensorRecv>(op);

  TF_ASSIGN_OR_RETURN(mlir::Operation * send_op,
                      GetCorrespondingDTensorSendRecvOp<mlir::TF::DTensorRecv>(
                          module, dtensor_recv));
  auto dtensor_send = llvm::dyn_cast<mlir::TF::DTensorSend>(send_op);

  TF_RETURN_IF_ERROR(
      ValidateSendRecvLayoutConfiguration(dtensor_send, dtensor_recv));

  TF_ASSIGN_OR_RETURN(const Layout send_layout,
                      ExtractRequiredLayoutFromOperand(send_op->getOperand(0)));

  TF_ASSIGN_OR_RETURN(const Mesh send_mesh,
                      ExtractDeviceMeshEnclosingCluster(send_op));

  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));

  mlir::Operation* lowered_recv;
  const Layout recv_layout = dtensor_recv.layout();
  const Mesh& recv_mesh = recv_layout.mesh();
  mlir::OpBuilder builder(dtensor_recv);

  if (SendRecvOpUsesXla(send_mesh, recv_mesh)) {
    if (recv_mesh.is_cpu_mesh() ||
        IsOneToOneHostMeshTransfer(send_layout, recv_layout)) {
      // Recv can be lowered directly for a 1-to-1 transfer between host and
      // device.
      TF_ASSIGN_OR_RETURN(mlir::TensorType local_output_type,
                          LocalTypeFromGlobalType(
                              dtensor_recv.layout(),
                              dtensor_recv.getType().cast<mlir::TensorType>()));
      TF_ASSIGN_OR_RETURN(lowered_recv, LowerDTensorRecvToXlaOp(
                                            dtensor_recv, local_output_type));
      dtensor_recv->replaceAllUsesWith(lowered_recv);
      dtensor_recv.erase();
    } else {
      // For other send/recv layouts, the tensor needs to be replicated.
      if (!dtensor_recv.layout().IsFullyReplicated()) {
        return errors::InvalidArgument(
            "CopyToMesh where target mesh is TPU requires target layout to be "
            "replicated.");
      }

      // For Receiving at TPU, only receive for device with device ordinal 0.
      auto recv_cluster =
          dtensor_recv->getParentOfType<mlir::tf_device::ClusterOp>();
      mlir::Location loc = dtensor_recv.getLoc();
      TF_ASSIGN_OR_RETURN(
          mlir::Value device_ordinal,
          GetDeviceOrdinal(recv_mesh, loc,
                           recv_cluster->getParentOfType<mlir::func::FuncOp>(),
                           &builder));
      mlir::Value predicate = builder.create<mlir::TF::EqualOp>(
          loc, device_ordinal, CreateIntScalarConst(0, builder, loc),
          /*incompatible_shape_error=*/builder.getBoolAttr(true));

      auto recv_if = builder.create<mlir::TF::IfRegionOp>(
          loc, llvm::SmallVector<mlir::Type, 4>{dtensor_recv.getType()},
          predicate,
          /*is_stateless=*/builder.getBoolAttr(true),
          GetUniqueControlflowFnName("copy_to_mesh_recv_if_then", builder),
          GetUniqueControlflowFnName("copy_to_mesh_recv_if_else", builder));

      // Create empty else branch region that outputs zeros.
      auto& else_branch = recv_if.else_branch();
      else_branch.push_back(new mlir::Block);
      builder.setInsertionPointToEnd(&else_branch.front());

      // Create a zero constant.
      mlir::Attribute const_attr;
      if (dtensor_recv.getType().getElementType().isIntOrIndex()) {
        const_attr = mlir::DenseIntElementsAttr::get(
            dtensor_recv.getType(), llvm::SmallVector<int32_t>{0});
      } else {
        const_attr = mlir::DenseFPElementsAttr::get(
            dtensor_recv.getType(), llvm::SmallVector<float>{0.0});
      }

      mlir::Value zeros = builder.create<mlir::TF::ConstOp>(loc, const_attr);
      builder.create<mlir::TF::YieldOp>(
          loc, /*operands=*/llvm::ArrayRef<mlir::Value>{zeros});

      // Create then branch region with DTensorRecv op.
      auto& then_branch = recv_if.then_branch();
      then_branch.push_back(new mlir::Block);
      builder.setInsertionPointToEnd(&then_branch.front());
      dtensor_recv->moveBefore(&then_branch.front(), then_branch.front().end());

      TF_ASSIGN_OR_RETURN(mlir::Operation * xla_recv,
                          LowerDTensorRecvToXlaOp(dtensor_recv));
      builder.create<mlir::TF::YieldOp>(
          loc,
          /*operands=*/llvm::ArrayRef<mlir::Value>{xla_recv->getResult(0)});

      // Broadcast the received output to all TPU cores.
      mlir::Value if_output = recv_if->getResult(0);
      builder.setInsertionPointAfterValue(if_output);
      absl::flat_hash_set<std::string> reduced_dims;
      for (const auto& mesh_dim : recv_mesh.dims())
        reduced_dims.insert(mesh_dim.name);

      TF_ASSIGN_OR_RETURN(lowered_recv,
                          EmitAllReduce(builder, recv_layout, reduced_dims,
                                        recv_if, kReduceOpAdd));

      // Replaces usages of DTensorRecv op with the broadcasted value.
      dtensor_recv.output().replaceUsesWithIf(
          lowered_recv->getResult(0), [&](mlir::OpOperand& operand) {
            return !recv_if->isProperAncestor(operand.getOwner());
          });
      dtensor_recv.erase();
    }
  } else if (dtensor_recv.layout().mesh().is_cpu_mesh() &&
             send_mesh.is_cpu_mesh()) {
    // Lower DTensorRecv op to TF Host Recv op.
    TF_ASSIGN_OR_RETURN(lowered_recv,
                        LowerDTensorRecvFromCPUToTFOp(send_mesh, dtensor_recv));
  } else {
    // TODO(hongjunchoi): Implement SPMD transformation lowering that lowers
    // DTensorRecv to vanilla TF Recv op.
    return errors::Unimplemented(
        "CopyToMesh between CPU/GPU not implemented yet.");
  }

  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  builder.setInsertionPointAfter(lowered_recv);
  TF_ASSIGN_OR_RETURN(
      mlir::Value recv_output,
      EmitAllScatter(builder, lowered_recv->getResult(0), recv_layout,
                     output_layout, &newly_created_ops));
  lowered_recv->getResult(0).replaceAllUsesExcept(recv_output,
                                                  newly_created_ops);
  return recv_output.getDefiningOp();
}

// DTensorRecv always returns tensors with fully replicated layout.
StatusOr<llvm::DenseMap<int, Layout>>
DTensorRecvSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  mlir::TF::DTensorRecv dtensor_recv =
      mlir::dyn_cast<mlir::TF::DTensorRecv>(op);
  if (!dtensor_recv) {
    return errors::InvalidArgument(
        llvm::formatv("Expecting DTensorRecvOp but got {0}", OpName(op)).str());
  }
  return llvm::DenseMap<int, Layout>({{0, dtensor_recv.layout()}});
}

StatusOr<llvm::DenseMap<int, Layout>>
DTensorRecvSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  return llvm::DenseMap<int, Layout>();
}

}  // namespace dtensor
}  // namespace tensorflow
