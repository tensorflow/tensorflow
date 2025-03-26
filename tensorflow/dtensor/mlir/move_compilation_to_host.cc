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

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/device_utils.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORMOVECOMPILATIONTOHOST
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

// Prefix for send/recv key used for transferring compilation program key.
constexpr char kSendRecvKeyPrefix[] = "compilation_send_recv_key_";

// Identifies all StatefulPartitionedCallOps for executing computation for
// each mesh cluster and validate that at most one TPU computation exists.
mlir::LogicalResult IdentifyAndValidateMeshComputations(
    mlir::func::FuncOp function,
    std::map<Mesh, mlir::TF::StatefulPartitionedCallOp>* function_map) {
  for (auto dtensor_function :
       function.getOps<mlir::TF::StatefulPartitionedCallOp>()) {
    auto mesh_or = ExtractDeviceMeshFromOp(dtensor_function);
    if (!mesh_or.ok() || !mesh_or->has_value())
      return dtensor_function.emitOpError(
          "StatefulPartitionCall op must have `_mesh` attribute specified.");

    const Mesh& computation_mesh = mesh_or->value();
    if (function_map->count(computation_mesh))
      return dtensor_function.emitOpError(
          "Found DTensor function with duplicate mesh specification. There "
          "should be exactly 1 function for each mesh in computation cluster.");

    (*function_map)[computation_mesh] = dtensor_function;
  }

  int num_xla_meshes = 0;
  for (const auto& it : *function_map) {
    if (it.first.is_tpu_mesh()) num_xla_meshes += 1;
  }

  if (num_xla_meshes > 1)
    return function.emitOpError(
        "Multiple XLA computation clusters found. Only 1 XLA cluster for "
        "DTensor computation is supported for now.");

  return mlir::success();
}

// Creates Send/Recv ops to transfer TPUCompile program key from host
// computation to XLA computation.
mlir::LogicalResult CreateSendRecvOpsToTransferProgramKey(
    const Mesh& mesh, mlir::ModuleOp module, mlir::func::FuncOp function,
    mlir::OpBuilder::InsertPoint insertpoint,
    mlir::TF::_TPUCompileMlirOp compile_op,
    mlir::tf_device::LaunchOp compile_op_launch, int* num_send_recv,
    mlir::Value* program_key_output) {
  mlir::OpBuilder builder(module.getContext());
  mlir::Value compilation_key = *compile_op.getProgram().begin();
  absl::Span<const std::string> local_devices = mesh.local_devices();

  // Create tensor name mapping for each send/recv pair.
  llvm::SmallDenseMap<int, std::string> device_key_map;
  const int num_tpu_devices = local_devices.size();
  device_key_map.reserve(num_tpu_devices);
  for (int i = 0; i < num_tpu_devices; ++i) {
    std::string tensor_name = absl::StrCat(kSendRecvKeyPrefix, *num_send_recv);
    *num_send_recv += 1;
    device_key_map.try_emplace(i, std::move(tensor_name));
  }

  // Create send op to send TPU program key from host computation to XLA
  // computation.
  builder.setInsertionPointAfter(compile_op);
  for (int i = 0; i < num_tpu_devices; ++i) {
    const std::string& tensor_name = device_key_map[i];
    auto send = builder.create<mlir::TF::_HostSendOp>(
        compile_op->getLoc(), compilation_key, tensor_name,
        compile_op_launch.getDevice(),
        /*send_device_incarnation=*/0, local_devices[i]);
    send->setAttr("device", compile_op_launch.getDeviceAttr());
  }

  // Create Recv ops to receive program key from host to each xla device
  // computation.
  llvm::SmallVector<mlir::func::FuncOp, 4> compilation_key_functions;
  compilation_key_functions.reserve(num_tpu_devices);
  mlir::SymbolTable symbol_table(module);

  // For receiving TPU program key from host, `recv_device` attribute depends
  // on `device_id` argument and therefore cannot be known statically.
  // Therefore, we use tf.Case op to select correct receive op depending on
  // the device id value.
  for (int i = 0; i < num_tpu_devices; ++i) {
    auto func_type = mlir::FunctionType::get(
        builder.getContext(), llvm::ArrayRef<mlir::Type>{},
        llvm::ArrayRef<mlir::Type>{compilation_key.getType()});

    mlir::func::FuncOp recv_select_fn = mlir::func::FuncOp::create(
        compile_op.getLoc(),
        llvm::formatv("recv_compile_key_{0}_{1}", i, *num_send_recv).str(),
        func_type, llvm::ArrayRef<mlir::NamedAttribute>{});
    symbol_table.insert(recv_select_fn);
    *num_send_recv += 1;

    mlir::Block* fn_block = recv_select_fn.addEntryBlock();
    mlir::OpBuilder fn_builder = mlir::OpBuilder::atBlockEnd(fn_block);
    auto recv = fn_builder.create<mlir::TF::_HostRecvOp>(
        compile_op->getLoc(),
        mlir::cast<mlir::TensorType>(compilation_key.getType()),
        device_key_map[i], compile_op_launch.getDevice(),
        /*send_device_incarnation=*/0, local_devices[i]);
    recv->setAttr("device", builder.getStringAttr(local_devices[i]));

    fn_builder.create<mlir::func::ReturnOp>(recv_select_fn.getLoc(),
                                            recv.getTensor());

    compilation_key_functions.emplace_back(recv_select_fn);
  }

  // Create logic that receives program key.
  builder.restoreInsertionPoint(insertpoint);
  auto device_id = GetDeviceOrdinal(mesh, function.getLoc(), function, &builder,
                                    /*return_int64_type=*/false);
  if (!device_id.ok()) return function->emitOpError("Cannot get device id");

  llvm::SmallVector<mlir::Attribute, 4> symbols;
  for (auto& func : compilation_key_functions)
    symbols.push_back(mlir::SymbolRefAttr::get(func));

  // Create a TF::Case op that selects `values` based on `id`.
  auto program_key = builder.create<mlir::TF::CaseOp>(
      compile_op.getLoc(),
      /*output=*/llvm::SmallVector<mlir::Type, 4>{compilation_key.getType()},
      /*branch_index=*/*device_id,
      /*input=*/llvm::ArrayRef<mlir::Value>{},
      /*branches=*/builder.getArrayAttr(symbols),
      /*is_stateless=*/builder.getBoolAttr(false));
  *program_key_output = program_key.getResult(0);
  return mlir::success();
}

struct CompilationKeyRecvInfo {
  const Mesh& receiving_function_mesh;
  mlir::func::FuncOp receiving_function;
  mlir::OpBuilder::InsertPoint recv_insertion_point;
  mlir::Value program_key;
};

// Broadcasts compilation key across meshes specified by `recv_info`. The
// broadcasted compilation key is added to `program_key` of each vector
// element of `recv_info`.
mlir::LogicalResult SendRecvCompilationKey(
    const Mesh& host_mesh, mlir::ModuleOp module,
    mlir::TF::_TPUCompileMlirOp compile_op,
    mlir::tf_device::LaunchOp compile_launch_op,
    mlir::Operation* compilation_move_before, int* num_send_recv,
    llvm::SmallVectorImpl<CompilationKeyRecvInfo>* recv_info) {
  for (int i = 0; i < recv_info->size(); ++i) {
    CompilationKeyRecvInfo& info = (*recv_info)[i];
    // Create send/recv ops to transfer compilation key from receiving meshes.
    mlir::Value program_key;
    if (mlir::failed(CreateSendRecvOpsToTransferProgramKey(
            info.receiving_function_mesh, module, info.receiving_function,
            info.recv_insertion_point, compile_op, compile_launch_op,
            num_send_recv, &program_key)))
      return mlir::failure();

    info.program_key = program_key;
  }

  return mlir::success();
}

mlir::LogicalResult HandleCompilationOps(
    const llvm::SmallVectorImpl<
        mlir::TF::_XlaCompileMlirPlaceholderProgramKeyOp>& compilation_key_ops,
    std::map<Mesh, mlir::TF::StatefulPartitionedCallOp>& computation_map,
    mlir::ModuleOp module, int* num_send_recv) {
  // Identity XLA function and corresponding CPU functions to move compilation.
  const auto xla_mesh = llvm::find_if(
      computation_map, [](const auto& it) { return it.first.is_tpu_mesh(); });

  if (xla_mesh == computation_map.end()) {
    return module.emitOpError(
        "Found TPUCompilationKey op but XLA computation does not exist.");
  }

  mlir::func::FuncOp tpu_function = xla_mesh->second.func();
  mlir::func::FuncOp host_function;
  Mesh host_mesh;
  for (auto compilation_key : compilation_key_ops) {
    auto parent_function =
        compilation_key->getParentOfType<mlir::func::FuncOp>();

    if (!host_function) {
      host_function = parent_function;
      auto mesh_it = llvm::find_if(computation_map, [&](auto& it) {
        return it.second.getF() == host_function.getSymName();
      });
      if (mesh_it == computation_map.end())
        return compilation_key.emitOpError(
            "cannot find host mesh for TPU computation.");

      host_mesh = mesh_it->first;

    } else {
      // TODO(hongjunchoi): Handle the case when CopyToMesh is used with
      // special topology approach. In this case there will be 2 host
      // meshes/functions.
      if (host_function != parent_function)
        return compilation_key.emitOpError(
            "Found multiple TPU host mesh functions. There must be at most one "
            "TPU host function.");
    }
  }

  // Identify TPUCompileOp to host side mesh.
  llvm::SmallVector<mlir::TF::_TPUCompileMlirOp, 4> compile_ops;
  tpu_function.walk(
      [&](mlir::TF::_TPUCompileMlirOp op) { compile_ops.emplace_back(op); });

  const int num_compilations = compile_ops.size();
  if (num_compilations != 1)
    return tpu_function.emitOpError(llvm::formatv(
        "Expected exactly 1 compilation op for TPU computation. Found {0}",
        num_compilations));

  mlir::TF::_TPUCompileMlirOp compile_op = *compile_ops.begin();
  mlir::Operation& first_host_op = host_function.getBody().front().front();
  mlir::OpBuilder builder(&first_host_op);
  mlir::OpBuilder::InsertPoint host_insertion_point =
      builder.saveInsertionPoint();
  mlir::Operation* compilation_move_before = &first_host_op;

  // If host mesh has multiple local devices only conduct compilation for the
  // first host device by creating If Op to only compile for host with device
  // ordinal 0.
  if (host_mesh.local_device_ids().size() > 1) {
    auto device_ordinal_host = GetDeviceOrdinal(
        host_mesh, compile_op.getLoc(),
        first_host_op.getParentOfType<mlir::func::FuncOp>(), &builder);
    if (!device_ordinal_host.ok())
      return compile_op.emitOpError(
          llvm::formatv("error while creating TPU compilation logic. {0}",
                        device_ordinal_host.status().message()));

    mlir::Value predicate_host = builder.create<mlir::TF::EqualOp>(
        compile_op.getLoc(), *device_ordinal_host,
        CreateIntScalarConst(0, builder, compile_op.getLoc()),
        /*incompatible_shape_error=*/builder.getBoolAttr(true));

    // If op here contains send/recv and TPUCompile op that should not be pruned
    // away. Therefore, we explicitly set the op to be stateful.
    auto if_host = builder.create<mlir::TF::IfRegionOp>(
        compile_op.getLoc(), llvm::SmallVector<mlir::Type, 4>{}, predicate_host,
        /*is_stateless=*/builder.getBoolAttr(false),
        GetUniqueControlflowFnName("compilation_host_then", builder),
        GetUniqueControlflowFnName("compilation_host_else", builder));

    // Create empty else branch region.
    auto& host_else_branch = if_host.getElseBranch();
    host_else_branch.push_back(new mlir::Block);
    builder.setInsertionPointToEnd(&host_else_branch.front());
    builder.create<mlir::TF::YieldOp>(
        compile_op.getLoc(),
        /*operands=*/llvm::ArrayRef<mlir::Value>{});

    // Create then branch region with logic to compile TPU program and send
    // program key to all TPU devices.
    auto& host_then_branch = if_host.getThenBranch();
    host_then_branch.push_back(new mlir::Block);
    builder.setInsertionPointToEnd(&host_then_branch.front());
    auto yield = builder.create<mlir::TF::YieldOp>(
        compile_op.getLoc(),
        /*operands=*/llvm::ArrayRef<mlir::Value>{});
    compilation_move_before = yield;

    builder.setInsertionPointAfter(if_host);
    host_insertion_point = builder.saveInsertionPoint();
  }

  auto compile_launch_op =
      compile_op->getParentOfType<mlir::tf_device::LaunchOp>();

  // Move Compile op and compile succeeded assert ops to host function.
  compile_launch_op->moveBefore(compilation_move_before);

  for (mlir::Operation* user : compile_launch_op.getResult(0).getUsers())
    user->getParentOfType<mlir::tf_device::LaunchOp>()->moveBefore(
        compilation_move_before);

  // Send and receive compilation key across meshes.
  llvm::SmallVector<CompilationKeyRecvInfo, 4> compilation_key_recv_info;
  builder.setInsertionPointToStart(&tpu_function.front());
  auto device_insertion_point = builder.saveInsertionPoint();
  compilation_key_recv_info.emplace_back(CompilationKeyRecvInfo{
      xla_mesh->first, tpu_function, device_insertion_point, nullptr});

  compilation_key_recv_info.emplace_back(CompilationKeyRecvInfo{
      host_mesh, host_function, host_insertion_point, nullptr});

  if (mlir::failed(SendRecvCompilationKey(
          host_mesh, module, compile_op, compile_launch_op,
          compilation_move_before, num_send_recv, &compilation_key_recv_info)))
    return mlir::failure();

  // Replace usages of TPU program key in host and device meshes.
  mlir::Value device_program_key = compilation_key_recv_info[0].program_key;
  tpu_function.walk([&](mlir::Operation* op) {
    if (llvm::isa<mlir::TF::TPUExecuteOp,
                  mlir::TF::TPUExecuteAndUpdateVariablesOp>(op))
      op->setOperand(op->getNumOperands() - 1, device_program_key);
  });

  // Remove placeholder CompilationKey ops and replace it's usages with output
  // of TPUCompile op.
  mlir::Value host_program_key = compilation_key_recv_info[1].program_key;
  for (auto compilation_key_op : compilation_key_ops) {
    compilation_key_op.replaceAllUsesWith(host_program_key);
    compilation_key_op.erase();
  }
  return mlir::success();
}

// Pass to move TPUCompile/TPUCompileSucceededAssert op to host mesh computation
// and add necessary send/recv ops to transfer TPU program key to TPU device
// computation.
struct DTensorMoveCompilationToHost
    : public impl::DTensorMoveCompilationToHostBase<
          DTensorMoveCompilationToHost> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);
    auto module = getOperation();

    llvm::SmallVector<mlir::TF::_XlaCompileMlirPlaceholderProgramKeyOp, 4>
        compilation_key_ops;
    module.walk([&](mlir::TF::_XlaCompileMlirPlaceholderProgramKeyOp op) {
      compilation_key_ops.emplace_back(op);
    });

    if (compilation_key_ops.empty()) return;

    mlir::func::FuncOp main_func =
        module.lookupSymbol<mlir::func::FuncOp>("main");
    if (!main_func) return;

    std::map<Mesh, mlir::TF::StatefulPartitionedCallOp> computation_map;
    if (mlir::failed(
            IdentifyAndValidateMeshComputations(main_func, &computation_map)))
      return signalPassFailure();

    int num_send_recv = 0;
    if (mlir::failed(HandleCompilationOps(compilation_key_ops, computation_map,
                                          module, &num_send_recv)))
      return signalPassFailure();
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorMoveCompilationToHost() {
  return std::make_unique<DTensorMoveCompilationToHost>();
}

}  // namespace dtensor
}  // namespace tensorflow
