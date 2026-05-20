/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_constants.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf_ifrt_passes.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

using mlir::TF::AsyncIfrtCallOp;
using mlir::TF::IfrtCallOp;
using mlir::TF::SetStaticDimensionBoundsOp;

// Identifies operands of `IfrtCallOp` or `AsyncIfrtCallOp` that are defined by
// `SetStaticDimensionBoundsOp` and returns a map from operand index to the
// defining op.
template <typename OpT>
llvm::SmallDenseMap<size_t, SetStaticDimensionBoundsOp>
GetArgIdxToStaticShapeOpMap(OpT ifrt_call) {
  llvm::SmallDenseMap<size_t, SetStaticDimensionBoundsOp>
      arg_idx_to_static_shape_op;
  for (const auto& [i, arg] : llvm::enumerate(ifrt_call.getArgs())) {
    mlir::Value cur_arg = arg;
    // We only check `IdentityOp` (not `IdentityNOp`) because
    // `SetStaticDimensionBoundsOp` returns a single tensor.
    while (auto identity = cur_arg.getDefiningOp<mlir::TF::IdentityOp>()) {
      cur_arg = identity.getInput();
    }
    if (auto static_shape_op =
            cur_arg.getDefiningOp<SetStaticDimensionBoundsOp>()) {
      VLOG(2) << "Found SetStaticDimensionBoundsOp for arg " << i;
      arg_idx_to_static_shape_op[i] = static_shape_op;
    }
  }
  return arg_idx_to_static_shape_op;
}

// Replaces `old_op` with a new op with `updated_args` and
// `static_shape_args`. Returns the new op.
template <typename OpT>
OpT UpdateIfrtCallOp(OpT old_op,
                     const llvm::SmallVector<mlir::Value, 4>& updated_args,
                     const llvm::SmallVector<mlir::Value, 4>& static_shape_args,
                     mlir::OpBuilder& builder) {
  // Clone to a new op.
  builder.setInsertionPoint(old_op);
  auto new_op = mlir::cast<OpT>(builder.clone(*old_op));
  // Wire up inputs.
  llvm::SmallVector<mlir::Value, 4> new_operands;
  new_operands.append(updated_args.begin(), updated_args.end());
  new_operands.append(static_shape_args.begin(), static_shape_args.end());
  new_op->setOperands(new_operands);
  // This attr is to delimit the boundary between original arguments and static
  // shape arguments.
  new_op->setAttr("operandSegmentSizes",
                  builder.getDenseI32ArrayAttr(
                      {static_cast<int32_t>(updated_args.size()),
                       static_cast<int32_t>(static_shape_args.size())}));
  // Wire up outputs.
  old_op.replaceAllUsesWith(new_op.getResults());
  // Get rid of the old op.
  old_op.erase();
  return new_op;
}

// Updates TPUCompileMetadataProto in function attributes to include new
// arguments for static shapes. Each static shape is added as a new parameter
// argument with replicated sharding.
mlir::LogicalResult UpdateTpuCompileMetadata(
    mlir::func::FuncOp func_op,
    const llvm::SmallVector<mlir::Value, 4>& static_shape_args) {
  // Read the old metadata from the func op.
  auto metadata_attr =
      func_op->getAttrOfType<mlir::StringAttr>(kMetadataTextAttrName);
  if (!metadata_attr) return mlir::success();
  tensorflow::tpu::TPUCompileMetadataProto metadata;
  if (!tsl::protobuf::TextFormat::ParseFromString(
          metadata_attr.getValue().str(), &metadata)) {
    return func_op.emitError() << "Failed to parse " << kMetadataTextAttrName;
  }
  // Update metadata proto to include static shape args.
  for (const auto& static_shape_arg : static_shape_args) {
    auto* new_compile_arg = metadata.add_args();
    // NOTE: tf2hlo.cc strictly requires PARAMETER kind for all args.
    new_compile_arg->set_kind(
        tensorflow::tpu::TPUCompileMetadataProto::Arg::PARAMETER);
    tensorflow::DataType dtype;
    if (const auto status =
            tensorflow::ConvertToDataType(static_shape_arg.getType(), &dtype);
        !status.ok()) {
      return func_op.emitError() << "Failed to convert static shape type to "
                                    "TensorFlow DataType: "
                                 << status.message();
    }
    new_compile_arg->set_dtype(dtype);
    tensorflow::ConvertTypeToTensorShape(static_shape_arg.getType())
        .AsProto(new_compile_arg->mutable_shape());
    new_compile_arg->mutable_sharding()->set_type(xla::OpSharding::REPLICATED);
  }
  // Write the new metadata to the func op.
  std::string new_metadata_str;
  if (!tsl::protobuf::TextFormat::PrintToString(metadata, &new_metadata_str)) {
    return func_op.emitError()
           << "Failed to serialize " << kMetadataTextAttrName;
  }
  func_op->setAttr(
      kMetadataTextAttrName,
      mlir::StringAttr::get(func_op.getContext(), new_metadata_str));
  return mlir::success();
}

// Updates the `FuncOp` identified by `program_id` to incorporate
// `static_shape_args`. This includes setting `tf._static_shape_arg` attributes
// on original arguments, updating the function signature, adding block
// arguments, and updating the `TPUCompileMetadataProto`.
template <typename OpT>
mlir::LogicalResult UpdateCalleeFuncOp(
    mlir::ModuleOp module, OpT ifrt_call,
    const llvm::SmallVector<mlir::Value, 4>& static_shape_args,
    const llvm::SmallDenseMap<size_t, size_t>& arg_idx_to_static_shape_idx) {
  // Find the callee `FuncOp` by program ID.
  mlir::func::FuncOp func_op;
  // NOTE: IFRT program functions are generated by
  // `RewriteClusterToIfrtCallPass` and inserted into the module's symbol table,
  // so they are always top-level operations.
  for (auto f : module.getOps<mlir::func::FuncOp>()) {
    auto id_attr =
        f->getAttrOfType<mlir::IntegerAttr>("tfrt_ifrt_serving.program_id");
    // NOTE: Cast is used because this ID is uint64_t in `IfrtCallOp` but
    // signless (aka int64_t in MLIR's sense) in `FuncOp`.
    if (id_attr &&
        static_cast<uint64_t>(id_attr.getInt()) == ifrt_call.getProgramId()) {
      func_op = f;
      break;
    }
  }
  if (!func_op) {
    return ifrt_call.emitOpError() << "callee func with program_id "
                                   << ifrt_call.getProgramId() << " not found";
  }
  // Point original args to new static shape args.
  //
  // The `tf._static_shape_arg_idx` attribute on an original argument indicates
  // the index of the corresponding static shape argument in the updated
  // function signature. For example, if the original arguments are (arg0, arg1,
  // arg2) and only arg0 and arg2 have static shapes (ss0, ss2), the new
  // signature is (arg0, arg1, arg2, ss0, ss2). Then arg0 will have
  // `tf._static_shape_arg_idx = 3` and arg2 will have
  // `tf._static_shape_arg_idx = 4`.
  mlir::OpBuilder builder(func_op);
  for (const auto& [arg_idx, static_shape_idx] : arg_idx_to_static_shape_idx) {
    func_op.setArgAttr(
        arg_idx, "tf._static_shape_arg_idx",
        builder.getI32IntegerAttr(static_cast<int32_t>(static_shape_idx)));
  }
  // Update function type to include static shape args.
  const auto& old_function_type = func_op.getFunctionType();
  llvm::SmallVector<mlir::Type, 4> new_input_types(
      old_function_type.getInputs().begin(),
      old_function_type.getInputs().end());
  for (const auto& arg : static_shape_args) {
    new_input_types.push_back(arg.getType());
  }
  func_op.setType(mlir::FunctionType::get(module.getContext(), new_input_types,
                                          old_function_type.getResults()));
  // Append static shape args to the entry block.
  for (const auto& arg : static_shape_args) {
    func_op.getBody().front().addArgument(arg.getType(), func_op.getLoc());
  }
  return UpdateTpuCompileMetadata(func_op, static_shape_args);
}

template <typename OpT>
mlir::LogicalResult ProcessIfrtCall(
    OpT ifrt_call, mlir::ModuleOp module, mlir::OpBuilder& op_builder,
    llvm::DenseSet<uint64_t>& processed_programs) {
  llvm::SmallDenseMap<size_t, SetStaticDimensionBoundsOp>
      arg_idx_to_static_shape_op = GetArgIdxToStaticShapeOpMap(ifrt_call);
  if (arg_idx_to_static_shape_op.empty()) return mlir::success();

  VLOG(2) << "Found `IfrtCallOp` with " << arg_idx_to_static_shape_op.size()
          << " static shaped args. IfrtCallOp: " << ifrt_call;

  mlir::OperandRange old_args = ifrt_call.getArgs();
  llvm::SmallVector<mlir::Value, 4> updated_args;
  updated_args.resize(old_args.size());
  llvm::SmallVector<mlir::Value, 4> static_shape_args;
  llvm::SmallDenseMap<size_t, size_t> arg_idx_to_static_shape_idx;
  for (const auto& [i, arg] : llvm::enumerate(old_args)) {
    auto iter = arg_idx_to_static_shape_op.find(i);
    if (iter != arg_idx_to_static_shape_op.end()) {
      SetStaticDimensionBoundsOp static_shape_op = iter->second;
      arg_idx_to_static_shape_idx[i] =
          old_args.size() + static_shape_args.size();
      static_shape_args.push_back(static_shape_op.getStaticShape());
      updated_args[i] = static_shape_op.getInput();
    } else {
      updated_args[i] = arg;
    }
  }
  uint64_t program_id = ifrt_call.getProgramId();
  auto new_ifrt_call =
      UpdateIfrtCallOp(ifrt_call, updated_args, static_shape_args, op_builder);

  // Defensive coding behavior: Ensure we only update the callee FuncOp once
  // per program_id. In theory, it is highly unlikely that the same program
  // is called by both IfrtCallOp and AsyncIfrtCallOp (or multiple times with
  // different static shape bounds) in the same module, because the choice of
  // op type is usually controlled by a session-wide flag (e.g.,
  // enable_async_ifrt). However, protecting against it avoids duplicate
  // argument promotion in the callee signature.
  if (!processed_programs.contains(program_id)) {
    if (mlir::failed(UpdateCalleeFuncOp(module, new_ifrt_call,
                                        static_shape_args,
                                        arg_idx_to_static_shape_idx))) {
      return mlir::failure();
    }
    processed_programs.insert(program_id);
  }

  // Erase all `SetStaticDimensionBounds` ops.
  for (auto [_, op] : arg_idx_to_static_shape_op) {
    op.replaceAllUsesWith(op.getOperand(0));
    op.erase();
  }
  return mlir::success();
}

// Pass that propagates static shapes from `tf.SetStaticDimensionBoundsOp` to
// `IfrtCallOp` and callee `FuncOp`.
struct PropagateStaticShapesPass : public mlir::OperationPass<mlir::ModuleOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PropagateStaticShapesPass)

  PropagateStaticShapesPass()
      : mlir::OperationPass<mlir::ModuleOp>(
            mlir::TypeID::get<PropagateStaticShapesPass>()) {}
  PropagateStaticShapesPass(const PropagateStaticShapesPass& other) = default;

  llvm::StringRef getArgument() const override {
    return "propagate-static-shapes";
  }
  llvm::StringRef getDescription() const override {
    return "Propagates static shapes from tf.SetStaticDimensionBoundsOp to "
           "IfrtCallOp and callee FuncOp.";
  }
  llvm::StringRef getName() const override {
    return "PropagateStaticShapesPass";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<PropagateStaticShapesPass>(*this);
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder op_builder(module.getContext());

    llvm::DenseSet<uint64_t> processed_programs;

    // Collect all calls in the module, before any modifications.
    llvm::SmallVector<IfrtCallOp, 4> ifrt_calls;
    llvm::SmallVector<AsyncIfrtCallOp, 4> async_ifrt_calls;
    module.walk([&ifrt_calls](IfrtCallOp op) { ifrt_calls.push_back(op); });
    module.walk([&async_ifrt_calls](AsyncIfrtCallOp op) {
      async_ifrt_calls.push_back(op);
    });

    for (auto ifrt_call : ifrt_calls) {
      if (mlir::failed(ProcessIfrtCall(ifrt_call, module, op_builder,
                                       processed_programs))) {
        return signalPassFailure();
      }
    }
    for (auto ifrt_call : async_ifrt_calls) {
      if (mlir::failed(ProcessIfrtCall(ifrt_call, module, op_builder,
                                       processed_programs))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreatePropagateStaticShapesPass() {
  return std::make_unique<PropagateStaticShapesPass>();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
