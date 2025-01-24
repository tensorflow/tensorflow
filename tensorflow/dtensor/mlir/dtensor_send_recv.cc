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

#include "tensorflow/dtensor/mlir/dtensor_send_recv.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/device_utils.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

bool IsStringType(mlir::Type type) {
  if (mlir::isa<mlir::TF::StringType>(type)) return true;

  auto sub_type = mlir::dyn_cast<mlir::TF::TensorFlowTypeWithSubtype>(type);
  if (!sub_type) return false;

  bool has_string =
      llvm::any_of(sub_type.GetSubtypes(), [](mlir::TensorType type) {
        return mlir::isa<mlir::TF::StringType>(type.getElementType());
      });
  return has_string;
}

// Returns compilation key placeholder. This placeholder will be replaced with
// output of TPUCompile op during TPURewrite pass. Program key (output of
// TPUCompile op) is used to differentiate TPU computation from which to receive
// data.
mlir::Value GetOrCreateCompilationKey(mlir::Operation* op) {
  mlir::Value key;
  auto cluster = op->getParentOfType<mlir::tf_device::ClusterOp>();
  assert(cluster);
  cluster.walk(
      [&](mlir::TF::_XlaCompileMlirPlaceholderProgramKeyOp compilation_key) {
        key = compilation_key.getProgram();
      });
  if (key) return key;

  mlir::OpBuilder builder(&cluster.GetBody().front());
  auto result_type =
      mlir::RankedTensorType::get({3}, builder.getType<mlir::TF::StringType>());
  auto new_compilation_key =
      builder.create<mlir::TF::_XlaCompileMlirPlaceholderProgramKeyOp>(
          cluster.getLoc(), /*program=*/result_type,
          llvm::ArrayRef<mlir::Value>{});
  return new_compilation_key.getProgram();
}

}  // namespace

StatusOr<mlir::Value> GetDeviceOrdinal(const Mesh& mesh,
                                       const mlir::Location& loc,
                                       mlir::func::FuncOp function,
                                       mlir::OpBuilder* builder,
                                       bool return_int64_type) {
  // Create as many entries as the number of devices in the entire mesh.
  llvm::SmallVector<int32, 4> device_id_to_ordinal(mesh.num_devices(), 0);
  // Only fill in entries with indices equal to local device IDs. For TPUs,
  // there are usually 8 local devices.
  for (int i = 0; i < mesh.local_device_ids().size(); ++i) {
    device_id_to_ordinal[mesh.local_device_ids()[i]] = i;
  }
  // Slice out the device ordinal using the device ID as index.
  TF_ASSIGN_OR_RETURN(mlir::Value device_id, DeviceId(function));
  mlir::TF::SliceOp device_ordinal = builder->create<mlir::TF::SliceOp>(
      loc,
      /*output=*/EffectivelyScalarR1Type(builder->getIntegerType(32)),
      /*input=*/IntConst(*builder, loc, device_id_to_ordinal),
      /*begin=*/
      mlir::TF::collection_ops_util::ReshapeScalarToSizeType(*builder,
                                                             device_id, loc),
      /*size=*/IntConst(*builder, loc, {1}));
  mlir::Value device_ordinal_scalar =
      ReshapeSizeTypeToScalar(*builder, loc, device_ordinal);
  if (return_int64_type) {
    device_ordinal_scalar = builder->create<mlir::TF::CastOp>(
        loc, mlir::RankedTensorType::get({}, builder->getI64Type()),
        device_ordinal_scalar);
  }
  return device_ordinal_scalar;
}

StatusOr<mlir::Operation*> LowerDTensorSendToTFOp(
    const Layout& send_input_layout, mlir::Value send_input,
    mlir::TF::DTensorSend dtensor_send) {
  mlir::OpBuilder builder(dtensor_send);
  builder.setInsertionPointAfter(send_input.getDefiningOp());
  std::string tensor_name = dtensor_send.getKey().str();

  Mesh target_mesh = dtensor_send.getTargetMesh();
  absl::Span<const std::string> sending_devices =
      send_input_layout.mesh().local_devices();
  absl::Span<const std::string> receiving_devices = target_mesh.local_devices();

  mlir::Operation* lowered_send_op;
  lowered_send_op = builder.create<mlir::TF::_HostSendOp>(
      send_input.getLoc(), send_input, tensor_name, sending_devices[0],
      /*send_device_incarnation=*/0, receiving_devices[0],
      /*client_terminated=*/false);

  dtensor_send.erase();
  return lowered_send_op;
}

// Lowers DTensorSend Op to either one of XlaSendFromHost op or XlaSendToHost,
// depending on the src mesh cluster.
StatusOr<mlir::Operation*> LowerDTensorSendToXlaOp(
    const Layout& send_input_layout, mlir::Value send_input,
    mlir::TF::DTensorSend dtensor_send, bool send_from_device_zero) {
  const bool send_from_cpu = !send_input_layout.mesh().is_tpu_mesh();
  mlir::OpBuilder builder(dtensor_send);

  mlir::Location loc = dtensor_send.getLoc();
  mlir::Operation* lowered_send_op;
  if (send_from_cpu) {
    llvm::SmallVector<mlir::Value, 4> value_to_send{send_input};
    mlir::OpBuilder::InsertPoint insertion_point = builder.saveInsertionPoint();
    mlir::Value program_key = GetOrCreateCompilationKey(dtensor_send);
    builder.restoreInsertionPoint(insertion_point);

    mlir::Value device_ordinal;
    if (send_from_device_zero) {
      // For CopyToMesh, we currently only support sending from host device 0
      // to target TPUs.
      device_ordinal = CreateIntScalarConst(0, builder, loc);
    } else {
      // For special topologies, always send from CPU device i to TPU device i.
      auto send_cluster =
          dtensor_send->getParentOfType<mlir::tf_device::ClusterOp>();
      if (!send_cluster) {
        return errors::InvalidArgument("DTensorSend is not inside a ClusterOp");
      }
      auto send_func = send_cluster->getParentOfType<mlir::func::FuncOp>();
      if (!send_func) {
        return errors::InvalidArgument("DTensorSend is not inside a FuncOp");
      }
      TF_ASSIGN_OR_RETURN(
          device_ordinal,
          GetDeviceOrdinal(send_input_layout.mesh(), loc, send_func, &builder));
    }
    // Create XlaSendFromHostV2 op
    lowered_send_op = builder.create<mlir::TF::_XlaSendFromHostV2Op>(
        loc, value_to_send, program_key, device_ordinal, dtensor_send.getKey());
  } else {
    // Note that for ops running in XLA/TPU, device ordinal input is not needed.
    lowered_send_op = builder.create<mlir::TF::XlaSendToHostOp>(
        loc, send_input, dtensor_send.getKey());
  }

  dtensor_send.erase();
  return lowered_send_op;
}

// Creates a shape attribute of the local shape version of RecvFromOp's result.
StatusOr<mlir::TF::ShapeAttr> GetDTensorRecvLocalShapeAttr(
    mlir::TF::DTensorRecv dtensor_recv) {
  if (dtensor_recv->getNumResults() != 1) {
    return absl::InvalidArgumentError(
        "XlaRecvFromHostOp must have exactly one result.");
  }
  TF_ASSIGN_OR_RETURN(std::vector<Layout> layouts,
                      ExtractRequiredLayoutFromOp(dtensor_recv));
  if (layouts.empty() || layouts.size() > 1) {
    return absl::InvalidArgumentError(
        "invalid layout for XlaRecvFromHostOp specified");
  }
  auto result_type = dtensor_recv->getResult(0);
  TF_ASSIGN_OR_RETURN(auto result_shape, GetShapeOfValue(result_type));
  auto local_shape = layouts[0].LocalShapeFromGlobalShape(result_shape);
  return mlir::TF::ShapeAttr::get(result_type.getContext(), local_shape);
}

// Lowers DTensorRecv op to either one of XlaRecvAtHost or XlaRecvFromHost,
// depending on src mesh cluster configuration. `output_type` can be set to the
// specific local tensor type needed, if different from the Recv op output type.
StatusOr<mlir::Operation*> LowerDTensorRecvToXlaOp(
    mlir::TF::DTensorRecv dtensor_recv, mlir::Type output_type) {
  const bool recv_at_cpu = dtensor_recv.getMesh().is_cpu_mesh();
  mlir::Operation* recv_xla_op = nullptr;
  mlir::OpBuilder builder(dtensor_recv);

  if (recv_at_cpu) {
    // Create XlaRecvAtHostV2 op.
    llvm::SmallVector<mlir::Type, 4> output_types{output_type};
    auto recv_cluster =
        dtensor_recv->getParentOfType<mlir::tf_device::ClusterOp>();

    TF_ASSIGN_OR_RETURN(std::optional<Mesh> mesh,
                        ExtractDeviceMeshFromOp(recv_cluster));
    if (!mesh.has_value())
      return errors::InvalidArgument(
          "failed to get device ordinal as mesh for operation is not "
          "specified.");

    mlir::OpBuilder builder(&recv_cluster.GetBody().front());
    TF_ASSIGN_OR_RETURN(
        mlir::Value device_ordinal,
        GetDeviceOrdinal(*mesh, recv_cluster.getLoc(),
                         recv_cluster->getParentOfType<mlir::func::FuncOp>(),
                         &builder));

    auto program_key = GetOrCreateCompilationKey(dtensor_recv);
    builder.setInsertionPoint(dtensor_recv);
    recv_xla_op = builder.create<mlir::TF::_XlaRecvAtHostV2Op>(
        dtensor_recv.getLoc(), output_types,
        /*dynamic_key=*/program_key, device_ordinal, dtensor_recv.getKeyAttr());
  } else {
    TF_ASSIGN_OR_RETURN(auto local_shape_attr,
                        GetDTensorRecvLocalShapeAttr(dtensor_recv));

    // Create XlaRecvFromHost op.
    recv_xla_op = builder.create<mlir::TF::XlaRecvFromHostOp>(
        dtensor_recv.getLoc(), output_type, local_shape_attr,
        dtensor_recv.getKeyAttr());
  }

  assert(recv_xla_op);

  // TODO(hongjunchoi): After receiving tensor, convert tensor to requested
  // layout with EmitRelayout.
  return recv_xla_op;
}

// Lowers DTensorRecv op to either one of XlaRecvAtHost or XlaRecvFromHost,
// depending on src mesh cluster configuration.
StatusOr<mlir::Operation*> LowerDTensorRecvToXlaOp(
    mlir::TF::DTensorRecv dtensor_recv) {
  return LowerDTensorRecvToXlaOp(dtensor_recv, dtensor_recv.getType());
}

StatusOr<mlir::Operation*> LowerDTensorRecvToXlaOp(
    const Mesh&, mlir::TF::DTensorRecv dtensor_recv, mlir::Type output_type) {
  return LowerDTensorRecvToXlaOp(dtensor_recv, output_type);
}

// Lowers a DTensorSend Op from a CPU to a TF Send op.
StatusOr<mlir::Operation*> LowerDTensorSendFromCPUToTFOp(
    const Layout& send_input_layout, mlir::Value send_input,
    mlir::TF::DTensorSend dtensor_send) {
  mlir::OpBuilder builder(dtensor_send);
  builder.setInsertionPointAfter(send_input.getDefiningOp());

  llvm::SmallVector<mlir::Value, 4> value_to_send{send_input};

  // Create multiple send from host. There should be #number of local
  // devices(in target mesh) number of sends.
  absl::Span<const std::string> sending_devices =
      send_input_layout.mesh().local_devices();

  Mesh target_mesh = dtensor_send.getTargetMesh();
  absl::Span<const std::string> receiving_devices = target_mesh.local_devices();

  std::string tensor_name = dtensor_send.getKey().str();

  mlir::Operation* lowered_send_op;
  for (size_t i = 0; i < receiving_devices.size(); ++i)
    lowered_send_op = builder.create<mlir::TF::_HostSendOp>(
        send_input.getLoc(), dtensor_send.getInput(), tensor_name,
        sending_devices[0],
        /*send_device_incarnation=*/0, receiving_devices[i]);

  dtensor_send.erase();
  return lowered_send_op;
}

// Lowers DTensorRecv op to TF Recv Op.
StatusOr<mlir::Operation*> LowerDTensorRecvFromCPUToTFOp(
    const Mesh& send_mesh, mlir::TF::DTensorRecv dtensor_recv) {
  const Mesh& recv_mesh = dtensor_recv.getMesh();

  auto recv_cluster =
      dtensor_recv->getParentOfType<mlir::tf_device::ClusterOp>();

  mlir::OpBuilder builder(&recv_cluster.GetBody().front());
  llvm::SmallVector<mlir::Type, 4> output_types{dtensor_recv.getType()};
  builder.setInsertionPoint(dtensor_recv);
  std::string tensor_name = dtensor_recv.getKey().str();
  absl::Span<const std::string> sending_devices = send_mesh.local_devices();
  absl::Span<const std::string> receiving_devices = recv_mesh.local_devices();

  mlir::Operation* lowered_recv_op;
  mlir::Location loc = dtensor_recv.getLoc();
  for (size_t i = 0; i < receiving_devices.size(); ++i)
    lowered_recv_op = builder.create<mlir::TF::_HostRecvOp>(
        loc, dtensor_recv.getType(), tensor_name, sending_devices[0],
        /*send_device_incarnation=*/0, receiving_devices[i]);

  // Replace dtensor_recv with newly created recv op and remove DTensorRecv op.
  assert(lowered_recv_op);
  dtensor_recv.replaceAllUsesWith(lowered_recv_op);
  dtensor_recv.erase();
  return lowered_recv_op;
}

StatusOr<mlir::Operation*> LowerDTensorRecvToTFOp(
    const Mesh& send_mesh, mlir::TF::DTensorRecv dtensor_recv,
    mlir::Type output_type) {
  const Mesh& recv_mesh = dtensor_recv.getMesh();
  auto recv_cluster =
      dtensor_recv->getParentOfType<mlir::tf_device::ClusterOp>();

  mlir::OpBuilder builder(&recv_cluster.GetBody().front());
  builder.setInsertionPoint(dtensor_recv);
  std::string tensor_name = dtensor_recv.getKey().str();
  absl::Span<const std::string> sending_devices = send_mesh.local_devices();
  absl::Span<const std::string> receiving_devices = recv_mesh.local_devices();

  mlir::Location loc = dtensor_recv.getLoc();
  mlir::Operation* lowered_recv_op = builder.create<mlir::TF::_HostRecvOp>(
      loc, output_type, tensor_name, sending_devices[0],
      /*send_device_incarnation=*/0, receiving_devices[0]);

  return lowered_recv_op;
}

namespace {
template <typename It, typename Fn>
llvm::SmallVector<mlir::Attribute, 4> GenerateBranches(
    mlir::Operation* op, mlir::SymbolTable& symbol_table,
    llvm::ArrayRef<mlir::Type> result_types, const char* fmt, const It& values,
    const Fn& fn) {
  llvm::SmallVector<mlir::Attribute, 4> branches;
  for (const auto& it : llvm::enumerate(values)) {
    // Builds the restore op on device_id.
    mlir::OpBuilder builder(op);
    auto func_type = mlir::FunctionType::get(
        builder.getContext(), op->getOperandTypes(), result_types);

    mlir::Location location = op->getLoc();
    mlir::func::FuncOp func_op = mlir::func::FuncOp::create(
        location, llvm::formatv(fmt, OpName(op), OpHash(op), it.index()).str(),
        func_type, llvm::ArrayRef<mlir::NamedAttribute>{});

    func_op.setVisibility(mlir::SymbolTable::Visibility::Private);
    symbol_table.insert(func_op);

    mlir::Block* fn_block = func_op.addEntryBlock();
    mlir::OpBuilder fn_builder = mlir::OpBuilder::atBlockBegin(fn_block);
    mlir::BlockArgument arg = (func_op.getNumArguments() > 0)
                                  ? func_op.getArgument(0)
                                  : mlir::BlockArgument{};
    auto branch_op = fn(fn_builder, location, arg, it.value());
    fn_builder.create<mlir::func::ReturnOp>(location, branch_op->getResults());

    branches.push_back(mlir::SymbolRefAttr::get(func_op));
  }
  return branches;
}
}  // namespace

StatusOr<mlir::Operation*> LowerOneToOneDTensorSendToTFHostSend(
    const Layout& send_layout, const Mesh& recv_mesh,
    mlir::TF::DTensorSend dtensor_send) {
  const auto& send_mesh = send_layout.mesh();
  bool i32_copy =
      dtensor_send.getInput().getType().getElementType().isInteger(32);
  auto module = dtensor_send->getParentOfType<mlir::ModuleOp>();
  mlir::SymbolTable symbol_table(module);
  auto device_pairs =
      llvm::zip(send_mesh.local_devices(), recv_mesh.local_devices());
  mlir::OpBuilder builder(dtensor_send);

  auto send_cluster =
      dtensor_send->getParentOfType<mlir::tf_device::ClusterOp>();
  auto send_fn = send_cluster->getParentOfType<mlir::func::FuncOp>();
  TF_ASSIGN_OR_RETURN(std::optional<Mesh> mesh,
                      ExtractDeviceMeshFromOp(send_cluster));
  TF_ASSIGN_OR_RETURN(
      mlir::Value device_ordinal,
      GetDeviceOrdinal(*mesh, dtensor_send.getLoc(), send_fn, &builder,
                       /*return_int64_type=*/false));

  mlir::StringAttr tensor_name =
      builder.getStringAttr(dtensor_send.getKey().str());
  auto branches = GenerateBranches(
      dtensor_send, symbol_table, llvm::ArrayRef<mlir::Type>{},
      "{0}_send_{1}_{2}", device_pairs,
      [&](mlir::OpBuilder& op_builder, auto& loc, mlir::BlockArgument arg,
          auto device_pair) -> mlir::Operation* {
        auto func_op = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
            arg.getOwner()->getParentOp());
        func_op.setArgAttr(arg.getArgNumber(), kCustomDeviceAttr,
                           op_builder.getStringAttr(send_layout.ToString()));
        mlir::Value val = arg;
        if (i32_copy) {
          auto val_type = mlir::cast<mlir::TensorType>(val.getType());
          val = op_builder
                    .create<mlir::TF::CastOp>(
                        loc,
                        mlir::RankedTensorType::get(
                            val_type.getShape(), op_builder.getIntegerType(64)),
                        val)
                    ->getResult(0);
        }
        return op_builder.create<mlir::TF::_HostSendOp>(
            loc, val, tensor_name, std::get<0>(device_pair),
            /*send_device_incarnation=*/0, std::get<1>(device_pair));
      });
  mlir::Operation* case_op = builder.create<mlir::TF::CaseOp>(
      dtensor_send.getLoc(),
      /*output=*/llvm::ArrayRef<mlir::Type>{},
      /*branch_index=*/device_ordinal,
      /*input=*/dtensor_send->getOperands(),
      /*branches=*/builder.getArrayAttr(branches),
      /*is_stateless=*/builder.getBoolAttr(false));

  // erase the send op here iff targeting a gpu
  // otherwise there will be 'op not within cluster' error(s)
  if (recv_mesh.device_type() == "GPU") {
    dtensor_send.erase();
  }

  return case_op;
}

StatusOr<mlir::Operation*> LowerOneToOneDTensorRecvToTFHostRecv(
    const Mesh& send_mesh, const Layout& recv_layout,
    mlir::TF::DTensorRecv dtensor_recv) {
  auto module = dtensor_recv->getParentOfType<mlir::ModuleOp>();
  const auto& recv_mesh = recv_layout.mesh();
  mlir::SymbolTable symbol_table(module);
  auto device_pairs =
      llvm::zip(send_mesh.local_devices(), recv_mesh.local_devices());
  mlir::OpBuilder builder(dtensor_recv);

  auto recv_cluster =
      dtensor_recv->getParentOfType<mlir::tf_device::ClusterOp>();
  auto recv_fn = recv_cluster->getParentOfType<mlir::func::FuncOp>();
  TF_ASSIGN_OR_RETURN(std::optional<Mesh> mesh,
                      ExtractDeviceMeshFromOp(recv_cluster));
  TF_ASSIGN_OR_RETURN(
      mlir::Value device_ordinal,
      GetDeviceOrdinal(*mesh, recv_cluster.getLoc(), recv_fn, &builder,
                       /*return_int64_type=*/false));

  mlir::TensorType recv_type = dtensor_recv.getType();
  bool i32_copy = recv_type.getElementType().isInteger(32);
  TF_ASSIGN_OR_RETURN(mlir::TensorType local_recv_type,
                      LocalTypeFromGlobalType(recv_layout, recv_type));
  mlir::TensorType local_output_type =
      i32_copy ? mlir::RankedTensorType::get(local_recv_type.getShape(),
                                             builder.getIntegerType(64))
               : local_recv_type;

  mlir::StringAttr tensor_name =
      builder.getStringAttr(dtensor_recv.getKey().str());
  auto branches = GenerateBranches(
      dtensor_recv, symbol_table, llvm::ArrayRef<mlir::Type>{local_output_type},
      "{0}_receive_{1}_{2}", device_pairs,
      [&](mlir::OpBuilder& op_builder, auto& loc, auto _,
          auto device_pair) -> mlir::Operation* {
        auto recv_op = op_builder.create<mlir::TF::_HostRecvOp>(
            loc, local_output_type, tensor_name, std::get<0>(device_pair),
            /*send_device_incarnation=*/0, std::get<1>(device_pair));
        SetSingleLayoutOnOp(recv_op, recv_layout);
        return recv_op;
      });
  mlir::Operation* case_op = builder.create<mlir::TF::CaseOp>(
      dtensor_recv.getLoc(),
      /*output=*/llvm::ArrayRef<mlir::Type>{local_output_type},
      /*branch_index=*/device_ordinal,
      /*input=*/dtensor_recv->getOperands(),
      /*branches=*/builder.getArrayAttr(branches),
      /*is_stateless=*/builder.getBoolAttr(false));

  mlir::Operation* lowered_recv;
  if (i32_copy) {
    lowered_recv = builder.create<mlir::TF::CastOp>(
        dtensor_recv.getLoc(), local_recv_type, case_op->getResult(0));
  } else {
    lowered_recv = case_op;
  }

  dtensor_recv.getOutput().replaceAllUsesWith(lowered_recv->getResult(0));
  dtensor_recv.erase();

  return lowered_recv;
}

namespace {
bool IsTpuToHostMeshTransfer(const Mesh& send_mesh, const Mesh& recv_mesh) {
  // Check tensor is being transferred between CPU <-> TPU.
  if (!(send_mesh.is_tpu_mesh() && recv_mesh.is_cpu_mesh()) &&
      !(recv_mesh.is_tpu_mesh() && send_mesh.is_cpu_mesh()))
    return false;

  // Check tensor transfer is happening between TPU and its host mesh.
  return ((send_mesh.is_tpu_mesh() &&
           send_mesh.tpu_host_mesh() == recv_mesh.ToString()) ||
          (recv_mesh.is_tpu_mesh() &&
           recv_mesh.tpu_host_mesh() == send_mesh.ToString()));
}

bool IsGpuToHostMeshTransfer(const Mesh& send_mesh, const Mesh& recv_mesh) {
  return ((send_mesh.device_type() == "GPU") && recv_mesh.is_cpu_mesh()) ||
         ((recv_mesh.device_type() == "GPU") && send_mesh.is_cpu_mesh());
}

// Returns whether send/recv layout represents send/recv of tensor between
// i-th TPU device and i-th device of the host mesh. Host mesh represents the
// CPU devices that are 1-to-1 mapped with the TPU mesh devices, having the same
// global and local device IDs.
bool IsOneToOneMeshTransfer(const Layout& send_layout,
                            const Layout& recv_layout) {
  const Mesh& send_mesh = send_layout.mesh();
  const Mesh& recv_mesh = recv_layout.mesh();

  // Check local device IDs are fully matching so that there is no cross-host
  // transfer.
  if (send_mesh.local_device_ids() != recv_mesh.local_device_ids())
    return false;

  return send_layout.GetShardVector() == recv_layout.GetShardVector();
}

// Returns whether to lower DTensorSend/DTensorRecv op to xla backend ops.
// Xla backend ops are used when either sending/receiving device uses XLA
// compiler.
bool SendRecvOpUsesXla(const Mesh& send_mesh, const Mesh& recv_mesh) {
  assert(!(send_mesh.is_tpu_mesh() && recv_mesh.is_tpu_mesh()));
  return (send_mesh.is_tpu_mesh() || recv_mesh.is_tpu_mesh());
}
}  // namespace

// FIXME(b/271292250): Remove the recv_op argument.
StatusOr<mlir::Operation*> LowerDTensorSend(mlir::Operation* send_op,
                                            mlir::Operation* recv_op) {
  auto dtensor_send = llvm::cast<mlir::TF::DTensorSend>(send_op);

  TF_ASSIGN_OR_RETURN(
      const Layout input_layout,
      ExtractRequiredLayoutFromOperand(dtensor_send.getInput()));

  const Mesh& input_mesh = input_layout.mesh();
  const Mesh& target_mesh = dtensor_send.getTargetMesh();

  auto layout_attr =
      dtensor_send->getAttrOfType<mlir::dtensor::LayoutAttr>(kTargetLayoutAttr);

  if (!layout_attr) {
    return absl::InvalidArgumentError("target_layout is not found");
  }

  const Layout& recv_layout = layout_attr.getValue();

  bool one_to_one = IsOneToOneMeshTransfer(input_layout, recv_layout);
  // Force string type to not use the allreduce/broadcast optimization as there
  // is no string type allreduce.
  bool is_string_type =
      IsStringType(dtensor_send.getInput().getType().getElementType());
  // Is tensor transfer is from TPU mesh to host mesh and send layout and recv
  // layout is identical, then tensor from each source device is sent to
  // target device asynchronously.
  mlir::Operation* lowered_send;
  if (IsTpuToHostMeshTransfer(input_mesh, target_mesh) && one_to_one) {
    TF_ASSIGN_OR_RETURN(lowered_send,
                        LowerDTensorSendToXlaOp(
                            input_layout, dtensor_send.getInput(), dtensor_send,
                            /*send_from_device_zero=*/false));
  } else if (IsGpuToHostMeshTransfer(input_mesh, target_mesh) &&
             (one_to_one &&
              (!recv_layout.IsFullyReplicated() || is_string_type))) {
    TF_ASSIGN_OR_RETURN(lowered_send,
                        LowerOneToOneDTensorSendToTFHostSend(
                            input_layout, target_mesh, dtensor_send));
  } else {
    // Calculate input tensor layout of data to send and target fully replicated
    // layout. For now, we ensure that all data transfer happen with fully
    // replicated tensors.
    const int rank = ValueRank(dtensor_send.getInput());
    const Layout target_layout = Layout::ReplicatedOnMesh(input_mesh, rank);

    // Convert tensor to send to replicated layout.
    mlir::OpBuilder builder(dtensor_send);
    TF_ASSIGN_OR_RETURN(mlir::Value send_input,
                        EmitAllGather(builder, dtensor_send.getInput(),
                                      input_layout, target_layout));

    // Insert control flow such that only device with device ordinal == 0 sends
    // the tensor data across mesh.
    auto send_cluster =
        dtensor_send->getParentOfType<mlir::tf_device::ClusterOp>();
    TF_ASSIGN_OR_RETURN(std::optional<Mesh> mesh,
                        ExtractDeviceMeshFromOp(send_cluster));
    if (!mesh.has_value()) {
      return absl::InvalidArgumentError(
          "failed to lower DTensor CopyToMesh op as sending side mesh is not "
          "specified.");
    }

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
    auto& else_branch = send_if.getElseBranch();
    else_branch.push_back(new mlir::Block);
    builder.setInsertionPointToEnd(&else_branch.front());
    builder.create<mlir::TF::YieldOp>(
        loc,
        /*operands=*/llvm::ArrayRef<mlir::Value>{});

    // Create then branch region with DTensorSend op.
    auto& then_branch = send_if.getThenBranch();
    then_branch.push_back(new mlir::Block);
    builder.setInsertionPointToEnd(&then_branch.front());
    auto yield = builder.create<mlir::TF::YieldOp>(
        loc, /*operands=*/llvm::ArrayRef<mlir::Value>{});
    dtensor_send->moveBefore(yield);

    // Lower DTensorSend op to actual TF op.
    const Mesh& recv_mesh = recv_layout.mesh();
    if (SendRecvOpUsesXla(input_layout.mesh(), recv_mesh)) {
      // Lower DTensorSend op to Xla Send ops.
      TF_ASSIGN_OR_RETURN(
          lowered_send,
          LowerDTensorSendToXlaOp(input_layout, send_input, dtensor_send,
                                  /*send_from_device_zero=*/true));
    } else if (input_layout.mesh().is_cpu_mesh() && recv_mesh.is_cpu_mesh()) {
      // Lower DTensorSend op to TF Host Send op.
      TF_ASSIGN_OR_RETURN(
          lowered_send, LowerDTensorSendFromCPUToTFOp(input_layout, send_input,
                                                      dtensor_send));
    } else {
      mlir::TensorType send_type =
          mlir::cast<mlir::TensorType>(send_input.getType());
      if (!recv_mesh.is_cpu_mesh() &&
          send_type.getElementType().isInteger(32)) {
        builder.setInsertionPointAfter(send_input.getDefiningOp());
        auto cast_to_int64 = builder.create<mlir::TF::CastOp>(
            send_input.getLoc(),
            mlir::RankedTensorType::get(send_type.getShape(),
                                        builder.getIntegerType(64)),
            send_input);
        send_input = cast_to_int64->getResult(0);
      }
      TF_ASSIGN_OR_RETURN(
          lowered_send,
          LowerDTensorSendToTFOp(input_layout, send_input, dtensor_send));
    }
  }

  return lowered_send;
}

// FIXME(b/271292250): Remove the send_op argument.
StatusOr<mlir::Operation*> LowerDTensorRecv(mlir::Operation* send_op,
                                            mlir::Operation* recv_op) {
  auto dtensor_recv = llvm::cast<mlir::TF::DTensorRecv>(recv_op);

  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(recv_op));

  mlir::Operation* lowered_recv;
  auto layout_attr =
      dtensor_recv->getAttrOfType<mlir::dtensor::LayoutAttr>(kSourceLayoutAttr);
  if (!layout_attr) {
    return absl::InvalidArgumentError("source_layout is not found");
  }
  const Layout send_layout = layout_attr.getValue();

  const Mesh send_mesh = send_layout.mesh();

  const Mesh& recv_mesh = dtensor_recv.getMesh();
  const Layout& recv_layout = output_layout;
  mlir::OpBuilder builder(dtensor_recv);

  bool cpu_to_cpu = recv_mesh.is_cpu_mesh() && send_mesh.is_cpu_mesh();
  bool one_to_one = IsOneToOneMeshTransfer(send_layout, recv_layout);
  bool send_recv_xla = SendRecvOpUsesXla(send_mesh, recv_mesh);
  // Force string type to not use the allreduce/broadcast optimization as there
  // is no string type allreduce.
  bool is_string_type = IsStringType(dtensor_recv.getType().getElementType());

  if (IsGpuToHostMeshTransfer(send_mesh, recv_mesh) &&
      (one_to_one && (!recv_layout.IsFullyReplicated() || is_string_type))) {
    TF_ASSIGN_OR_RETURN(lowered_recv,
                        LowerOneToOneDTensorRecvToTFHostRecv(
                            send_mesh, recv_layout, dtensor_recv));

    // erase the send op here iff not targeting a gpu
    if (recv_mesh.device_type() != "GPU") {
      send_op->erase();
    }

    return lowered_recv;
  } else if (cpu_to_cpu) {
    // Lower DTensorRecv op to TF Host Recv op.
    TF_ASSIGN_OR_RETURN(lowered_recv,
                        LowerDTensorRecvFromCPUToTFOp(send_mesh, dtensor_recv));
  } else if ((IsTpuToHostMeshTransfer(send_mesh, recv_mesh) && one_to_one) ||
             (send_recv_xla && recv_mesh.is_cpu_mesh())) {
    // Recv can be lowered directly for a 1-to-1 transfer between host and
    // device (*for XLA/TPUs).
    TF_ASSIGN_OR_RETURN(
        mlir::TensorType local_output_type,
        LocalTypeFromGlobalType(
            recv_layout, mlir::cast<mlir::TensorType>(dtensor_recv.getType())));
    TF_ASSIGN_OR_RETURN(
        lowered_recv, LowerDTensorRecvToXlaOp(dtensor_recv, local_output_type));
    dtensor_recv->replaceAllUsesWith(lowered_recv);
    dtensor_recv.erase();
  } else {
    // Choose which receive lowering function to use.
    auto lower_fn =
        send_recv_xla
            ? (decltype(&LowerDTensorRecvToTFOp))LowerDTensorRecvToXlaOp
            : LowerDTensorRecvToTFOp;

    // For other send/recv layouts, the tensor needs to be replicated.
    if (!recv_layout.IsFullyReplicated()) {
      return absl::InvalidArgumentError(
          "CopyToMesh where target mesh is GPU/TPU requires a replicated "
          "target layout.");
    }

    // For Receiving at GPU/TPU, only device 0 (ordinal) receives from the
    // host, then it shares the tensor with its peers.
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

    mlir::TensorType recv_type = dtensor_recv.getType();
    bool i32_copy = recv_type.getElementType().isInteger(32);
    bool need_i32_to_i64_upcast =
        i32_copy && !(recv_mesh.is_cpu_mesh() || send_recv_xla);
    mlir::TensorType output_type =
        need_i32_to_i64_upcast
            ? mlir::RankedTensorType::get(recv_type.getShape(),
                                          builder.getIntegerType(64))
            : recv_type;

    auto recv_if = builder.create<mlir::TF::IfRegionOp>(
        loc, llvm::SmallVector<mlir::Type, 4>{output_type}, predicate,
        /*is_stateless=*/builder.getBoolAttr(true),
        GetUniqueControlflowFnName("copy_to_mesh_recv_if_then", builder),
        GetUniqueControlflowFnName("copy_to_mesh_recv_if_else", builder));

    // Create empty else branch region that outputs zeros.
    auto& else_branch = recv_if.getElseBranch();
    else_branch.push_back(new mlir::Block);
    builder.setInsertionPointToEnd(&else_branch.front());

    // Create a zero constant.
    mlir::Attribute const_attr;
    auto output_element_type = output_type.getElementType();
    if (output_element_type.isIntOrIndex()) {
      if (output_element_type.isInteger(64)) {
        const_attr = mlir::DenseIntElementsAttr::get(
            output_type, llvm::SmallVector<int64_t>{0});
      } else {
        const_attr = mlir::DenseIntElementsAttr::get(
            output_type, llvm::SmallVector<int32_t>{0});
      }
    } else if (output_element_type.isBF16()) {
      mlir::FloatAttr zero = mlir::FloatAttr::get(output_element_type, 0.);
      const_attr = mlir::DenseElementsAttr::get(
          output_type, llvm::SmallVector<mlir::Attribute>{zero});
    } else if (output_element_type.isF16() || output_element_type.isF32()) {
      const_attr = mlir::DenseFPElementsAttr::get(
          output_type, llvm::SmallVector<float>{0.0});
    } else if (output_element_type.isF64()) {
      const_attr = mlir::DenseFPElementsAttr::get(
          output_type, llvm::SmallVector<double>{0.0});
    } else {
      return absl::InvalidArgumentError("unsupported output type");
    }

    mlir::Value zeros = builder.create<mlir::TF::ConstOp>(loc, const_attr);
    builder.create<mlir::TF::YieldOp>(
        loc, /*operands=*/llvm::ArrayRef<mlir::Value>{zeros});

    // Create then branch region with DTensorRecv op.
    auto& then_branch = recv_if.getThenBranch();
    then_branch.push_back(new mlir::Block);
    builder.setInsertionPointToEnd(&then_branch.front());
    dtensor_recv->moveBefore(&then_branch.front(), then_branch.front().end());

    TF_ASSIGN_OR_RETURN(mlir::Operation * xla_recv,
                        lower_fn(send_mesh, dtensor_recv, output_type));
    builder.create<mlir::TF::YieldOp>(
        loc,
        /*operands=*/llvm::ArrayRef<mlir::Value>{xla_recv->getResult(0)});

    // Broadcast the received output to all GPU/TPU devices.
    mlir::Value if_output = recv_if->getResult(0);
    builder.setInsertionPointAfterValue(if_output);
    absl::flat_hash_set<std::string> reduced_dims;
    for (const auto& mesh_dim : recv_mesh.dims())
      reduced_dims.insert(mesh_dim.name);

    TF_ASSIGN_OR_RETURN(
        lowered_recv, EmitAllReduce(builder, recv_layout, reduced_dims, recv_if,
                                    kReduceOpAdd));

    if (need_i32_to_i64_upcast) {
      lowered_recv = builder.create<mlir::TF::CastOp>(
          loc, recv_type, lowered_recv->getResult(0));
    }

    // Replaces usages of DTensorRecv op with the broadcasted value.
    dtensor_recv.getOutput().replaceUsesWithIf(
        lowered_recv->getResult(0), [&](mlir::OpOperand& operand) {
          return !recv_if->isProperAncestor(operand.getOwner());
        });
    dtensor_recv.erase();
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

StatusOr<mlir::Operation*> LowerDTensorSendAndRecv(mlir::Operation* send_op,
                                                   mlir::Operation* recv_op) {
  auto dtensor_send = llvm::cast<mlir::TF::DTensorSend>(send_op);
  auto dtensor_recv = llvm::dyn_cast<mlir::TF::DTensorRecv>(recv_op);

  const Mesh recv_mesh = dtensor_recv.getMesh();
  TF_ASSIGN_OR_RETURN(
      std::optional<Mesh> send_mesh,
      ExtractDeviceMeshFromOp(
          send_op->getParentOfType<mlir::tf_device::ClusterOp>()));

  if (!send_mesh.has_value())
    return errors::InvalidArgument(
        "failed to get device ordinal as mesh for operation is not "
        "specified.");

  if (!send_mesh->is_tpu_mesh() && !recv_mesh.is_tpu_mesh()) {
    return errors::Unimplemented(
        "Multi-mesh tensor transfer between non-xla devices are not yet "
        "supported.");
  }

  const Layout recv_layout =
      Layout::ReplicatedOnMesh(recv_mesh, ValueRank(dtensor_recv.getOutput()));
  const Layout send_input_layout =
      Layout::ReplicatedOnMesh(*send_mesh, ValueRank(dtensor_send.getInput()));

  TF_ASSIGN_OR_RETURN(mlir::Operation * lowered_recv,
                      LowerDTensorRecvToXlaOp(dtensor_recv));
  dtensor_recv->replaceAllUsesWith(lowered_recv);
  dtensor_recv.erase();

  return LowerDTensorSendToXlaOp(send_input_layout, dtensor_send.getInput(),
                                 dtensor_send,
                                 /*send_from_device_zero=*/false);
}

}  // namespace dtensor
}  // namespace tensorflow
