/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/transforms/corert_converter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/attr_lowering_utils.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/attributes.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/types.h"  // from @tf_runtime
#include "tfrt/distributed_runtime/opdefs/kernels.h"  // from @tf_runtime

namespace tensorflow {

CoreRTConverter::CoreRTConverter(
    mlir::MLIRContext *context,
    const mlir::TF::SideEffectAnalysis::Info *side_effect_analysis)
    : builder_(context), side_effect_analysis_(*side_effect_analysis) {
  addConversion([](tfrt::compiler::ChainType type) { return type; });
  addConversion([](tfrt::corert::OpHandlerType type) { return type; });
  addConversion([](tfrt::dist::DistributedContextType type) { return type; });
  addConversion([](tfrt::corert::TensorHandleType type) { return type; });
  addConversion([=](mlir::TensorType type) -> llvm::Optional<mlir::Type> {
    // Ref types are not supported in both compiler and runtime.
    if (type.getElementType().isa<mlir::TF::TensorFlowRefType>())
      return llvm::None;
    return tensor_handle_type();
  });
  addConversion([=](mlir::Type type) -> llvm::Optional<mlir::Type> {
    if (type == builder_.getI1Type()) return type;
    return llvm::None;
  });
}

void CoreRTConverter::MaterializeDerivedAttributes(mlir::Operation *op) {
  if (auto interface = llvm::dyn_cast<mlir::DerivedAttributeOpInterface>(op)) {
    auto derived_attrs = interface.materializeDerivedAttributes();
    for (auto named_attr : derived_attrs) {
      op->setAttr(named_attr.getName(), named_attr.getValue());
    }
  }
}

mlir::ArrayAttr CoreRTConverter::CreateOpFuncAttrs(
    ArrayRef<NamedAttribute> attrs,
    llvm::SmallVector<mlir::StringAttr, 4> *func_attr_keys) {
  llvm::SmallVector<mlir::Attribute, 4> attr_array;
  for (auto key_and_value : attrs) {
    auto attr_key = key_and_value.getName();
    auto attr_value = key_and_value.getValue();
    if (!IsUnusedTfrtAttribute(attr_key) &&
        attr_value.isa<mlir::FlatSymbolRefAttr, mlir::SymbolRefAttr>()) {
      auto func_attr = attr_value.dyn_cast<mlir::FlatSymbolRefAttr>();
      auto converted = ConvertSymbolAttrToStringAttr(func_attr);
      mlir::StringAttr key = builder_.getStringAttr(attr_key.strref());
      attr_array.push_back(builder_.getArrayAttr({key, converted}));

      // Remove the attribute to avoid being converted again.
      func_attr_keys->push_back(attr_key);
    }
  }
  return builder_.getArrayAttr(attr_array);
}

// TODO(chky): Add support for multiple device instances.
llvm::Optional<ParseDeviceNameResult> CoreRTConverter::ParseDeviceName(
    llvm::StringRef device_name) const {
  std::string tf_device_name = device_name.str();

  if (tf_device_name.empty()) {
    return llvm::None;
  }

  ParseDeviceNameResult result;
  result.device_name = tf_device_name;

  // Parse the device name in format of the current tensorflow.
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(result.device_name, &parsed_name)) {
    return llvm::None;
  }
  if (!parsed_name.has_type) {
    return llvm::None;
  }
  result.device_type = parsed_name.type;

  result.op_handler_name = tf_device_name;

  return result;
}

llvm::Optional<ParseDeviceNameResult> CoreRTConverter::ParseDeviceName(
    mlir::Operation *op) const {
  auto device_attr = op->getAttr("device");
  if (!device_attr) {
    return llvm::None;
  }

  auto parsed_device_name =
      ParseDeviceName(device_attr.cast<mlir::StringAttr>().getValue());
  if (!parsed_device_name) op->emitWarning("failed to parse device name.");
  return parsed_device_name;
}

mlir::Value CoreRTConverter::ConvertOpHandler(
    mlir::Operation *op, llvm::StringRef op_handler_name,
    ConversionPatternRewriter *rewriter) {
  auto iter = op_handler_by_name_.find(op_handler_name);
  if (iter != op_handler_by_name_.end()) return iter->second;

  mlir::Block *block = op->getBlock();
  ConversionPatternRewriter::InsertionGuard insertion_guard(*rewriter);
  rewriter->setInsertionPointToStart(block);

  func::FuncOp func_op = op->getParentOfType<mlir::func::FuncOp>();
  mlir::Value in_chain = func_op.getArgument(0);
  auto get_op_handler_op = rewriter->create<tfrt::corert::GetOpHandler>(
      block->getParent()->getLoc(), op_handler_type(), in_chain,
      op_handler_name);
  op_handler_by_name_[op_handler_name] = get_op_handler_op.getResult();
  return get_op_handler_op.getResult();
}

mlir::Value CoreRTConverter::GetDistributedContext(
    mlir::Operation *op, mlir::ConversionPatternRewriter *rewriter) {
  mlir::func::FuncOp func_op = op->getParentOfType<mlir::func::FuncOp>();
  auto iter = distributed_context_by_func_.find(func_op.getOperation());
  if (iter != distributed_context_by_func_.end()) {
    return iter->second;
  }
  ConversionPatternRewriter::InsertionGuard insertion_guard(*rewriter);
  rewriter->setInsertionPoint(op);
  auto get_dist_ctx_op = rewriter->create<tfrt::dist::GetDistributedContextOp>(
      op->getLoc(), distributed_context_type());

  mlir::Value result = get_dist_ctx_op.result();
  distributed_context_by_func_[func_op.getOperation()] = result;
  return result;
}

mlir::Value CoreRTConverter::GetRemoteChainManager(
    mlir::Operation *op, mlir::ConversionPatternRewriter *rewriter) {
  mlir::func::FuncOp func_op = op->getParentOfType<mlir::func::FuncOp>();
  auto iter = remote_chain_mgr_by_func_.find(func_op.getOperation());
  if (iter != remote_chain_mgr_by_func_.end()) {
    return iter->second;
  }
  ConversionPatternRewriter::InsertionGuard insertion_guard(*rewriter);
  rewriter->setInsertionPoint(op);

  mlir::Type remote_chain_mgr_type =
      builder_.getType<::tfrt::dist::RemoteChainManagerType>();
  mlir::Value dist_ctx = GetDistributedContext(op, rewriter);
  auto create_mgr_op = rewriter->create<tfrt::dist::CreateRemoteChainManager>(
      op->getLoc(), remote_chain_mgr_type, dist_ctx);

  mlir::Value result = create_mgr_op.result();
  remote_chain_mgr_by_func_[func_op.getOperation()] = result;
  return result;
}

mlir::Value CoreRTConverter::GetLocalSideEffectChain(
    mlir::Operation *op, mlir::ConversionPatternRewriter *rewriter) {
  auto func_op = op->getParentOfType<mlir::func::FuncOp>();

  llvm::SmallVector<mlir::Operation *, 4> predecessors;
  if (llvm::isa<mlir::func::ReturnOp>(op)) {
    auto sinks = side_effect_analysis_.ControlSinks();
    predecessors.assign(sinks.begin(), sinks.end());
  } else {
    predecessors = side_effect_analysis_.DirectControlPredecessors(op);
  }

  llvm::SmallVector<mlir::Value, 2> chains;
  for (auto *pred : predecessors) {
    // TODO(chky): ReadVariableOp is removed in the pass and not converted.
    // Ideally, every side-effecting op should be converted to a
    // tfrt_fallback.executeop.seq op. The special rewrite logic of
    // ReadVariableOp should be done in a previous pass.
    if (auto chain = local_side_effect_chains_.lookup(pred))
      chains.push_back(chain);
  }

  // If there is no side-effect predecessor, then the input side-effect chain
  // is used.
  if (chains.empty()) return func_op.getArgument(0);

  if (chains.size() == 1) return chains[0];

  // If there are multiple side-effect predecessors, insert a merge_chains
  // kernel and return the merged chain.
  ConversionPatternRewriter::InsertionGuard insertion_guard(*rewriter);
  rewriter->setInsertionPoint(op);
  return rewriter->create<tfrt::compiler::MergeChainsOp>(op->getLoc(),
                                                         chain_type(), chains);
}

mlir::Value CoreRTConverter::GetTaskHandle(
    mlir::Operation *op, StringRef task_name,
    mlir::ConversionPatternRewriter *rewriter) {
  mlir::func::FuncOp func_op = op->getParentOfType<mlir::func::FuncOp>();
  llvm::StringMap<mlir::Value> &task_handle_by_name =
      task_handles_by_func_[func_op.getOperation()];
  auto iter = task_handle_by_name.find(task_name);
  if (iter != task_handle_by_name.end()) {
    return iter->second;
  }

  mlir::Value distributed_context = GetDistributedContext(op, rewriter);
  auto task_handle_op = rewriter->create<tfrt::dist::GetTaskHandleOp>(
      op->getLoc(), rewriter->getType<tfrt::dist::TaskHandleType>(),
      distributed_context, task_name);

  task_handle_by_name[task_name] = task_handle_op.getResult();
  return task_handle_op.getResult();
}

mlir::Value CoreRTConverter::GetRemoteSideEffectChain(
    mlir::Operation *op, StringRef remote_host,
    mlir::ConversionPatternRewriter *rewriter) {
  mlir::Value remote_chain_mgr = GetRemoteChainManager(op, rewriter);
  mlir::Value local_chain = GetLocalSideEffectChain(op, rewriter);
  mlir::Value task_handle = GetTaskHandle(op, remote_host, rewriter);
  mlir::Type remote_obj_id_ty =
      rewriter->getType<tfrt::dist::RemoteObjectIdType>();

  // Get the remote chain using the tfrt_dist.get_chain_for_task_handle op.
  auto get_chain_op = rewriter->create<tfrt::dist::GetChainForTaskHandleOp>(
      op->getLoc(), remote_obj_id_ty, local_chain, remote_chain_mgr,
      task_handle);
  return get_chain_op.getResult();
}

mlir::StringAttr CoreRTConverter::ConvertSymbolAttrToStringAttr(
    mlir::FlatSymbolRefAttr symbol_attr) {
  // Currently in TF graph to MLIR importing, a "0" is appended to the original
  // function name, so we pop it here. The renaming is for TF/XLA v1 bridge
  // use cases. Refer to b/142268695, b/141617294 for more context.
  //
  // In TFRT use cases, in almost every case "0" is the only literal
  // appended since TF Graph already guarantee function name uniqueness.
  // TODO(b/172092902): Investigate a better way to make the tf_func_name to
  // mlir_tf_func_name conversion reversible.
  auto func_name = symbol_attr.getValue().drop_back().str();

  return mlir::StringAttr::get(builder_.getContext(), func_name);
}

}  // namespace tensorflow
