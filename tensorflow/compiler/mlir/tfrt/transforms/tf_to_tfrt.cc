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

// This file implements lowering of TF dialect to TFRT CoreRuntime ExecuteOp.
// This lowering pass is heavily experimental and incomplete. External code
// should not depend on the code here. And please do not take example on it as
// "the path forward" for this.

#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.h"
#include "tensorflow/compiler/mlir/tfrt/analysis/tensor_array_side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tfrt/ir/gpu_ops.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/attr_lowering_utils.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/corert_converter.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/fallback_converter.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/gpu_passes.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tfrt_jitrt_stub.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/utils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/types.h"  // from @tf_runtime
#include "tfrt/test_kernels/opdefs/test_kernels.h"  // from @tf_runtime

namespace tensorflow {
namespace {

constexpr unsigned kFallbackBenefit = 1;
constexpr unsigned kCoreRTBenefit = 2;
constexpr char kGpuDeviceName[] =
    "/job:localhost/replica:0/task:0/device:GPU:0";
constexpr char kTFDeviceAttr[] = "tf.device";
constexpr char kTFRTDeviceAttr[] = "tfrt.device";
constexpr char kDeviceAttr[] = "device";
constexpr char kHostAttr[] = "host";
constexpr char kDeviceTypeTpu[] = "TPU";
constexpr int64_t kDefaultCheapCost = 1;

void getDependentConversionDialects(mlir::DialectRegistry &registry) {
  registry.insert<tfrt::corert::CoreRTDialect, mlir::func::FuncDialect,
                  tfrt::fallback_async::FallbackAsyncDialect,
                  tfrt::compiler::TFRTDialect>();

  RegisterJitRtDialects(registry);
}

mlir::Value GetFunctionInputChain(mlir::Operation *op) {
  auto func_op = op->getParentOfType<mlir::func::FuncOp>();
  return func_op.getArgument(0);
}

llvm::SmallVector<mlir::Value, 4> AddGpuVariableAndInputTensorTransferOps(
    mlir::Operation *op, llvm::SmallVector<mlir::Value, 4> operands,
    mlir::ConversionPatternRewriter &rewriter) {
  llvm::SmallVector<mlir::Value, 4> new_operands;
  assert(op->getOperands().size() == operands.size());
  for (int i = 0; i < op->getOperands().size(); ++i) {
    if (IsResultVariable(op->getOperand(i), operands[i])) {
      auto transfer_variable_op =
          rewriter.create<tfrt::gpu::MaybeTransferVariableOp>(
              op->getLoc(), rewriter.getType<tfrt::fallback::TFTensorType>(),
              mlir::ValueRange{operands[i]}, ArrayRef<mlir::NamedAttribute>());
      new_operands.push_back(transfer_variable_op);
    } else {
      auto transfer_to_device_op =
          rewriter.create<tfrt::gpu::TransferToDeviceOp>(
              op->getLoc(), rewriter.getType<tfrt::fallback::TFTensorType>(),
              mlir::ValueRange{operands[i]}, ArrayRef<mlir::NamedAttribute>());
      new_operands.push_back(transfer_to_device_op);
    }
  }
  return new_operands;
}

llvm::SmallVector<mlir::Value, 4> AddGpuTransferFromDeviceOps(
    mlir::Operation *op, llvm::SmallVector<mlir::Value, 4> results,
    mlir::ConversionPatternRewriter &rewriter) {
  assert(results.size() == op->getNumResults());
  llvm::SmallVector<mlir::Value, 4> new_results;
  for (int idx = 0; idx < results.size(); ++idx) {
    if (op->getResult(idx).use_empty()) {
      // If the result is not used, it is not transferred.
      new_results.push_back(results[idx]);
    } else {
      auto transfer_from_device_op =
          rewriter.create<tfrt::gpu::TransferFromDeviceOp>(
              op->getLoc(), rewriter.getType<tfrt::fallback::TFTensorType>(),
              results[idx]);
      new_results.push_back(transfer_from_device_op);
    }
  }
  return new_results;
}

// Convert TF dialect ops to tfrt_fallback.executeop for non-side-effecting ops
// and tfrt_fallback.executeop.seq for side-effecting ops.
//
// For example,
//
// %0 = "tf.MatMul"(%arg0, %arg1) {device = "cpu"} :
//    (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
//
// is converted to
//
// %result = tfrt_fallback.executeop device("cpu")
//    "tf.MatMul"(%arg0, %arg1) : 1
//
class FallbackExecuteOpConversion : public mlir::ConversionPattern {
 public:
  FallbackExecuteOpConversion(
      mlir::MLIRContext *context, CoreRTConverter *corert_converter,
      tfrt_compiler::FallbackConverter *fallback_converter,
      const mlir::SymbolTable *symbol_table,
      const tfrt_compiler::CostAnalysis *cost_analysis,
      bool tpu_lower_to_fallback, bool target_tpurt, bool use_bridge_for_gpu)
      : mlir::ConversionPattern(mlir::Pattern::MatchAnyOpTypeTag(),
                                kFallbackBenefit, context),
        corert_converter_(*corert_converter),
        fallback_converter_(*fallback_converter),
        symbol_table_(*symbol_table),
        cost_analysis_(*cost_analysis),
        tpu_lower_to_fallback_(tpu_lower_to_fallback),
        target_tpurt_(target_tpurt),
        use_bridge_for_gpu_(use_bridge_for_gpu) {}

  LogicalResult matchAndRewrite(
      mlir::Operation *op, ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter) const override {
    if (!UseFallback(op)) return failure();

    if (target_tpurt_ && IsTpuCompileAndExecuteOps(op)) return failure();

    corert_converter_.MaterializeDerivedAttributes(op);

    mlir::StringAttr device = op->getAttrOfType<mlir::StringAttr>(kDeviceAttr);
    if (!device || device.getValue().empty())
      return op->emitWarning("failed to find a non-empty 'device' attribute");
    op->removeAttr(kDeviceAttr);
    auto parsed_device_name =
        corert_converter_.ParseDeviceName(device.getValue());
    if (!parsed_device_name)
      return op->emitWarning("failed to parse the device name");

    // Convert the function (symbol) attributes to an array of string
    // attributes, which represents the function names.
    llvm::SmallVector<mlir::StringAttr, 4> func_attr_keys;

    // If the op is XlaLaunch on GPU, the function attribute will use the
    // function name in MLIR, instead of the original function name in the
    // function library, because the function could have been changed by bridge,
    // e.g., variable lifting. The new MLIR function will need to be exported to
    // the function library for runtime to use.
    bool use_mlir_func_name =
        parsed_device_name->device_type == DEVICE_GPU && use_bridge_for_gpu_ &&
        op->getName().getStringRef().str() == "tf.XlaLaunch";

    mlir::ArrayAttr op_func_attrs = corert_converter_.CreateOpFuncAttrs(
        symbol_table_, op->getAttrs(), &func_attr_keys, use_mlir_func_name);
    if (!op_func_attrs) {
      return op->emitWarning("failed to create func attributes.");
    }

    // Remove the function attributes, which have already been processed.
    for (const auto &key : func_attr_keys) op->removeAttr(key);

    Builder builder(op->getContext());
    mlir::ArrayAttr op_attrs = CreateTfrtOpAttrs(op->getAttrs(), builder);
    if (!op_attrs) return op->emitWarning("failed to lower attributes.");

    mlir::StringAttr op_name =
        rewriter.getStringAttr(op->getName().getStringRef());

    // Ops with _tpu_replicate attribute are TPU ops.
    bool is_tpu_op = op->hasAttr("_tpu_replicate") ||
                     llvm::isa<mlir::TF::TPUReplicatedInputOp,
                               mlir::TF::TPUReplicatedOutputOp>(op);

    if ((parsed_device_name->device_type == kDeviceTypeTpu &&
         !tpu_lower_to_fallback_) ||
        // Convert ops running on TPU to CoreRT dialect to prevent the creation
        // of tfrt_fallback_async.createop for them.
        // These ops will be encountered here only when using fallback to run
        // TPU models, in which case, these ops are assumed to be in a function
        // called by a TPUPartitionedCall op and will be compiled in
        // TPUPartitionedCall op via FunctionLibraryRuntime and not be processed
        // by BEFExecutor.
        //
        // We also avoid creating tfrt_fallback_async.createop for all GPU ops
        // except for tf.XlaLaunch. This is correct as long as we only run XLA
        // clusters on GPU and all other ops on CPU.
        is_tpu_op ||
        (parsed_device_name->device_type == DEVICE_GPU &&
         op->getName().getStringRef().str() != "tf.XlaLaunch")) {
      return ConvertToCoreRTExecuteOp(
          op, operands, parsed_device_name->op_handler_name, op_attrs,
          op_func_attrs, op_name, rewriter);
    }

    // For DEVICE_CPU, DEVICE_DEFAULT, and DEVICE_TPU_SYSTEM, we use fallback.
    // Note that TPU bridge should handle all ops that are required to be
    // executed on TPU. So if there are still ops that are placed on TPU at this
    // stage of lowering TF to TFRT, then these ops are supposed to be executed
    // on host.
    return ConvertToFallbackExecuteOp(op, operands, device, op_attrs,
                                      op_func_attrs, op_name,
                                      fallback_converter_, rewriter);
  }

 private:
  // Return true if this op can be lowered to fallback ops.
  bool UseFallback(mlir::Operation *op) const {
    if (!llvm::isa<mlir::TF::TensorFlowDialect>(op->getDialect())) return false;

    // Below is a blocklist of ops that should not go through CoreRTExecuteOp
    // conversion.
    // TODO(b/173017701): have a centralized place to hold the information
    // whether a TF op should be lowered to FallbackExecute op.
    return !llvm::isa<mlir::TF::_TfrtSetResourceOp,
                      mlir::TF::_TfrtGetResourceOp,
                      // Do not use fallback on the TPU fused
                      // compile_and_execute kernel.
                      mlir::TF::TPUCompileMlirAndExecuteOp,
                      // Specifically handle control flow ops.
                      mlir::TF::CaseOp, mlir::TF::IfOp, mlir::TF::WhileOp,
                      mlir::TF::StatefulPartitionedCallOp,
                      mlir::TF::PartitionedCallOp, mlir::TF::LegacyCallOp>(op);
  }

  bool IsTpuCompileAndExecuteOps(mlir::Operation *op) const {
    return llvm::isa<mlir::TF::_TPUCompileMlirOp,
                     mlir::TF::TPUCompileSucceededAssertOp,
                     mlir::TF::TPUExecuteOp>(op);
  }

  mlir::LogicalResult ConvertToFallbackExecuteOp(
      mlir::Operation *op, ValueRange operands, mlir::StringAttr device,
      mlir::ArrayAttr op_attrs, mlir::ArrayAttr op_func_attrs,
      mlir::StringAttr op_name,
      tfrt_compiler::FallbackConverter &fallback_converter,
      mlir::ConversionPatternRewriter &rewriter) const;

  mlir::LogicalResult ConvertToCoreRTExecuteOp(
      mlir::Operation *op, ValueRange operands, llvm::StringRef op_handler_name,
      mlir::ArrayAttr op_attrs, mlir::ArrayAttr op_func_attrs,
      mlir::StringAttr op_name,
      mlir::ConversionPatternRewriter &rewriter) const;

  CoreRTConverter &corert_converter_;
  tfrt_compiler::FallbackConverter &fallback_converter_;
  const mlir::SymbolTable &symbol_table_;
  const tfrt_compiler::CostAnalysis &cost_analysis_;
  bool tpu_lower_to_fallback_;
  bool target_tpurt_;
  // TODO(b/260915352): Remove the flag and default to using bridge.
  bool use_bridge_for_gpu_;
};

mlir::LogicalResult FallbackExecuteOpConversion::ConvertToFallbackExecuteOp(
    mlir::Operation *op, ValueRange operands, mlir::StringAttr device,
    mlir::ArrayAttr op_attrs, mlir::ArrayAttr op_func_attrs,
    mlir::StringAttr op_name,
    tfrt_compiler::FallbackConverter &fallback_converter,
    mlir::ConversionPatternRewriter &rewriter) const {
  llvm::SmallVector<Type, 4> result_types(
      op->getNumResults(), rewriter.getType<tfrt::fallback::TFTensorType>());

  // Convert the operands to tfrt_fallback.tf_tensor if needed.
  llvm::SmallVector<mlir::Value, 4> new_operands;
  if (mlir::failed(tfrt_compiler::ConvertFallbackOperands(
          op, device.getValue(), operands, &new_operands, rewriter)))
    return failure();

  auto fallback_key =
      rewriter.getI64IntegerAttr(fallback_converter.GetNextFallbackKey());

  // Query cost analysis to assign costs.
  IntegerAttr cost;
  auto parsed_device_name =
      corert_converter_.ParseDeviceName(device.getValue());
  bool is_gpu_op =
      parsed_device_name && parsed_device_name->device_type == DEVICE_GPU;
  if (is_gpu_op) {
    // For GPU ops, the host only needs to dispatch them to GPUs, which should
    // be relatively cheap for the host.
    cost = rewriter.getI64IntegerAttr(kDefaultCheapCost);
  } else {
    cost = rewriter.getI64IntegerAttr(cost_analysis_.GetCost(op));
  }

  // For now, we only consider GPU XLA clusters in the form of XlaLaunch for
  // simplicity. We could extend to support other GPU ops that cann't be XLAed.
  bool is_xla_launch_on_gpu =
      is_gpu_op && use_bridge_for_gpu_ &&
      op->getName().getStringRef().str() == "tf.XlaLaunch";
  if (is_xla_launch_on_gpu) {
    new_operands =
        AddGpuVariableAndInputTensorTransferOps(op, new_operands, rewriter);
  }

  if (mlir::isMemoryEffectFree(op)) {
    auto new_op = rewriter.create<tfrt::fallback_async::ExecuteOp>(
        op->getLoc(), result_types, new_operands, device, op_attrs,
        op_func_attrs, fallback_key, op_name, cost);
    fallback_converter.RegisterFallbackOp(new_op);
    rewriter.replaceOp(op, new_op.getResults());
  } else {
    auto in_chain = corert_converter_.GetLocalSideEffectChain(op, &rewriter);
    auto out_chain = in_chain;

    if (tfrt_compiler::IsTensorArrayOp(op)) {
      // If it is a tensor array op, we don't need to use
      // tfrt_fallback_async.executeop.seq because its operands/results already
      // take care of control dependencies.
      auto new_op = rewriter.create<tfrt::fallback_async::ExecuteOp>(
          op->getLoc(), result_types, new_operands, device, op_attrs,
          op_func_attrs, fallback_key, op_name, cost);
      fallback_converter.RegisterFallbackOp(new_op);
      rewriter.replaceOp(op, new_op.getResults());
    } else {
      // Create tfrt_fallback.executeop.seq if it is a side-effecting op.
      auto new_op = rewriter.create<tfrt::fallback_async::ExecuteOpSeq>(
          op->getLoc(), corert_converter_.chain_type(), result_types, in_chain,
          new_operands, device, op_attrs, op_func_attrs, fallback_key, op_name,
          cost);
      fallback_converter.RegisterFallbackOp(new_op);
      out_chain = new_op.getOutOpChain();
      if (is_xla_launch_on_gpu) {
        // TODO(b/262280565): Remove unnecessary data transfers when there are
        // multiple XLA clusters.
        auto results =
            AddGpuTransferFromDeviceOps(op, new_op.getResults(), rewriter);
        rewriter.replaceOp(op, results);
      } else {
        rewriter.replaceOp(op, new_op.getResults());
      }
    }

    // Register the converted op so that it can be retrieved by successors.
    corert_converter_.RegisterLocalSideEffectChain(op, out_chain);
  }

  return success();
}

mlir::LogicalResult FallbackExecuteOpConversion::ConvertToCoreRTExecuteOp(
    mlir::Operation *op, ValueRange operands, llvm::StringRef op_handler_name,
    mlir::ArrayAttr op_attrs, mlir::ArrayAttr op_func_attrs,
    mlir::StringAttr op_name, mlir::ConversionPatternRewriter &rewriter) const {
  llvm::SmallVector<Type, 4> result_types(
      op->getNumResults(), rewriter.getType<tfrt::corert::TensorHandleType>());

  // Convert the operands to tensorhandles if needed.
  llvm::SmallVector<mlir::Value, 4> new_operands;
  if (mlir::failed(tfrt_compiler::ConvertCoreRTOperands(
          op, operands, &new_operands, rewriter)))
    return failure();

  // Get the op handler, or create one if there does not exist one. Note that
  // ConvertOpHandler changes internal state so it can only be called if the
  // rewrite is guaranteed to succeed afterwards.
  auto op_handler =
      corert_converter_.ConvertOpHandler(op, op_handler_name, &rewriter);
  if (!op_handler) return failure();

  if (mlir::isMemoryEffectFree(op)) {
    auto new_op = rewriter.create<tfrt::corert::ExecuteOp>(
        op->getLoc(), result_types, op_handler, new_operands, op_attrs,
        op_func_attrs, op_name);
    rewriter.replaceOp(op, new_op.getResults());
  } else {
    // Create corert.executeop.seq if it is a side-effecting op.
    auto new_op = rewriter.create<tfrt::corert::ExecuteOpSeq>(
        op->getLoc(), corert_converter_.chain_type(), result_types, op_handler,
        corert_converter_.GetLocalSideEffectChain(op, &rewriter), new_operands,
        op_attrs, op_func_attrs, op_name);
    rewriter.replaceOp(op, new_op.getResults());

    // Register the converted op so that it can be retrieved by successors.
    corert_converter_.RegisterLocalSideEffectChain(op, new_op.getOutOpChain());
  }

  return success();
}

class FallbackConstOpConversion
    : public mlir::OpConversionPattern<mlir::TF::ConstOp> {
 public:
  FallbackConstOpConversion(mlir::MLIRContext *context,
                            CoreRTConverter *corert_converter)
      : mlir::OpConversionPattern<mlir::TF::ConstOp>(context),
        corert_converter_(*corert_converter) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::ConstOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    // Some data types are handled separately using a fast path.
    if (IsSupportedTfrtNumericDType(op.getDtype()) ||
        op.getDtype().isa<mlir::TF::StringType>())
      return failure();

    // For other data types that do not have a fast path (eg. quantized types),
    // we convert it to serialized tensor proto.

    tensorflow::TensorProto tensor_proto;
    auto status = ConvertToTensorProto(op.getValue(), &tensor_proto);
    if (!status.ok()) return op.emitError(tsl::NullTerminatedMessage(status));

    rewriter.replaceOpWithNewOp<tfrt::fallback_async::ConstTensorProtoOp>(
        op, rewriter.getType<tfrt::fallback::TFTensorType>(),
        tensor_proto.SerializeAsString());

    return success();
  }

 private:
  CoreRTConverter &corert_converter_;
};

class FallbackSetResourceOp
    : public mlir::OpConversionPattern<mlir::TF::_TfrtSetResourceOp> {
 public:
  FallbackSetResourceOp(mlir::MLIRContext *context,
                        CoreRTConverter *corert_converter)
      : mlir::OpConversionPattern<mlir::TF::_TfrtSetResourceOp>(context),
        corert_converter_(*corert_converter) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::_TfrtSetResourceOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::StringAttr device = op->getAttrOfType<mlir::StringAttr>(kDeviceAttr);
    if (!device || device.getValue().empty())
      return op->emitWarning("failed to find a non-empty 'device' attribute");

    // Currently the static resource is always on host CPU.
    //
    // TODO(chky): Support resource on other devices.
    llvm::SmallVector<mlir::Value, 4> new_operands;
    if (mlir::failed(tfrt_compiler::ConvertFallbackOperands(
            op, tfrt_compiler::GetDefaultCpuDeviceName(), adaptor.getOperands(),
            &new_operands, rewriter)))
      return failure();

    assert(new_operands.size() == 1);

    auto new_op = rewriter.create<tfrt::fallback_async::SetResourceOp>(
        op.getLoc(), corert_converter_.chain_type(),
        corert_converter_.GetLocalSideEffectChain(op, &rewriter),
        new_operands[0], device.getValue(), op.getIndex());

    // Register the converted op so that it can be retrieved by successors.
    corert_converter_.RegisterLocalSideEffectChain(op, new_op.getOutCh());

    rewriter.eraseOp(op);

    return success();
  }

 private:
  CoreRTConverter &corert_converter_;
};

class FallbackGetResourceOp
    : public mlir::OpConversionPattern<mlir::TF::_TfrtGetResourceOp> {
 public:
  FallbackGetResourceOp(mlir::MLIRContext *context,
                        CoreRTConverter *corert_converter)
      : mlir::OpConversionPattern<mlir::TF::_TfrtGetResourceOp>(context),
        corert_converter_(*corert_converter) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::_TfrtGetResourceOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::StringAttr device = op->getAttrOfType<mlir::StringAttr>(kDeviceAttr);
    if (!device || device.getValue().empty())
      return op->emitWarning("failed to find a non-empty 'device' attribute");

    llvm::SmallVector<mlir::Type, 4> result_types(
        op.getNumResults(), rewriter.getType<tfrt::fallback::TFTensorType>());

    auto ready_chain = rewriter.create<tfrt::compiler::NewChainOp>(
        op.getLoc(), rewriter.getType<tfrt::compiler::ChainType>());

    auto new_op = rewriter.create<tfrt::fallback_async::GetResourceOp>(
        op.getLoc(), corert_converter_.chain_type(), result_types, ready_chain,
        device.getValue(), op.getIndices());

    rewriter.replaceOp(op, new_op.getResults());

    return success();
  }

 private:
  CoreRTConverter &corert_converter_;
};

// Convert a tf_device.remote_run op to a tfrt_dist.remote_execute_func op.
//
// For example,
//
// %result = tf_device.remote_run "/job:worker/replica:0/task:0"
//   @remote_func(%arg0) : (tensor<i32>) -> (tensor<i32>)
//
// is converted to
//
// %0 = tfrt_dist.get_distributed_context
// %1 = tfrt_dist.get_task_handle %0
//  {task_name = "/job:worker/replica:0/task:0"}
// %2 = tfrt_dist.test_create_remote_chain_manager %0
// %3 = tfrt_dist.get_chain_for_task_handle %in_chain, %2, %1
// %out_op_chain, %results:2 = tfrt_dist.remote_execute_func[%in_chain, %0, %1]
//  @remote_func(%3, %arg1): (!tfrt_dist.remote_object_id, !corert.tensorhandle)
//  -> (!tfrt_dist.remote_object_id, !corert.tensorhandle)
// %4 = tfrt_dist.set_chain_for_task_handle %out_op_chain, %2, %1, %results#0
//
class TFDeviceRemoteRunOpConversion
    : public mlir::OpConversionPattern<tf_device::RemoteRunOp> {
 public:
  TFDeviceRemoteRunOpConversion(mlir::MLIRContext *context,
                                mlir::TypeConverter *type_converter,
                                CoreRTConverter *corert_converter)
      : mlir::OpConversionPattern<tf_device::RemoteRunOp>(context,
                                                          kCoreRTBenefit),
        type_converter_(*type_converter),
        corert_converter_(*corert_converter) {}

  LogicalResult matchAndRewrite(
      tf_device::RemoteRunOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    SymbolTable symtab(module);
    func::FuncOp callee = symtab.lookup<func::FuncOp>(op.getCallee());
    if (!callee) {
      op.emitOpError("callee function ") << op.getCallee() << " is not found";
      return failure();
    }
    StringAttr host = callee->getAttrOfType<StringAttr>(kHostAttr);
    if (!host) {
      op.emitOpError("callee function ")
          << op.getCallee() << " should have the host attribute";
      return failure();
    }

    llvm::SmallVector<mlir::Value, 4> arguments;

    for (mlir::Value argument : op.getCalleeArgs()) {
      arguments.push_back(argument);
    }

    llvm::SmallVector<mlir::Type, 4> result_types;
    // The first result of the remote function should be a remote chain which
    // is added to the function signature when it is lowered from TF dialect to
    // TFRT dialect.
    for (mlir::Type type : op.getResultTypes()) {
      (void)type_converter_.convertType(type, result_types);
    }

    return success();
  }

 private:
  mlir::TypeConverter &type_converter_;
  CoreRTConverter &corert_converter_;
};

// Lowers a tf.BatchFunction to tfrt_fallback.batch_function.
class FallbackBatchFunctionOpConversion
    : public mlir::OpConversionPattern<mlir::TF::BatchFunctionOp> {
 public:
  FallbackBatchFunctionOpConversion(mlir::MLIRContext *context,
                                    CoreRTConverter *corert_converter)
      : mlir::OpConversionPattern<mlir::TF::BatchFunctionOp>(context,
                                                             kFallbackBenefit),
        corert_converter_(*corert_converter) {}

  LogicalResult matchAndRewrite(
      mlir::TF::BatchFunctionOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    corert_converter_.MaterializeDerivedAttributes(op);

    llvm::SmallVector<NamedAttribute, 12> attr_array;
    for (auto &key_and_value : op->getAttrs()) {
      StringRef name = key_and_value.getName();
      if (name == "_output_shapes" || name == "f") {
        continue;
      }
      attr_array.push_back(key_and_value);
    }

    Builder builder(op.getContext());
    ArrayAttr op_attrs = CreateTfrtOpAttrs(attr_array, builder);
    if (!op_attrs) return op.emitWarning("failed to lower attributes.");

    mlir::StringAttr device = op->getAttrOfType<mlir::StringAttr>(kDeviceAttr);
    if (!device || device.getValue().empty())
      return op->emitWarning("failed to find a non-empty 'device' attribute");
    auto parsed_device_name =
        corert_converter_.ParseDeviceName(device.getValue());
    if (!parsed_device_name)
      return op->emitWarning("failed to parse the device name");

    llvm::SmallVector<mlir::Value, 4> new_operands;
    if (mlir::failed(tfrt_compiler::ConvertFallbackOperands(
            op, device.getValue(), adaptor.getOperands(), &new_operands,
            rewriter)))
      return failure();

    llvm::SmallVector<Type, 4> result_types(
        op->getNumResults(), rewriter.getType<tfrt::fallback::TFTensorType>());

    auto new_op = rewriter.create<tfrt::fallback_async::BatchFunctionOp>(
        op.getLoc(), result_types, new_operands, device, op.getFAttr(),
        op_attrs);
    rewriter.replaceOp(op, new_op.getResults());
    return success();
  }

 private:
  CoreRTConverter &corert_converter_;
};

// Lower a tf.Const op that creates a string tensor to a native
// corert.create_string_tensor op.
class CoreRTConstDenseTensorOpConversion
    : public mlir::OpConversionPattern<mlir::TF::ConstOp> {
 public:
  CoreRTConstDenseTensorOpConversion(mlir::MLIRContext *context,
                                     CoreRTConverter *corert_converter)
      : mlir::OpConversionPattern<mlir::TF::ConstOp>(context, kCoreRTBenefit),
        corert_converter_(*corert_converter) {}

  LogicalResult matchAndRewrite(
      mlir::TF::ConstOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!IsSupportedTfrtNumericDType(op.getDtype())) return failure();

    // Only CPU ops can be lowered using this conversion. If there is no device
    // assignment, this op is treated as a CPU op and can be lowered.
    if (auto parsed_device_name = corert_converter_.ParseDeviceName(op))
      if (parsed_device_name->device_type != DEVICE_CPU) return failure();

    auto new_op = rewriter.create<tfrt::corert::ConstDenseTensorOp>(
        op.getLoc(), corert_converter_.tensor_handle_type(),
        op.getValue().cast<DenseElementsAttr>());
    rewriter.replaceOp(op, new_op->getResult(0));
    return success();
  }

 private:
  CoreRTConverter &corert_converter_;
};

// Convert the FuncOp with the following changes to meet TFRT's requirements:
// 1) Convert types for the arguments and the results.
// 2) Add a chain to the arguments.
// 3) Add a chain to the results for side-effects.
// 4) If any argument has a tf.device attribute, change the attribute name
//    to tfrt.device.
// 5) If any result has a tf.device attribute, change the attribute name
//    to tfrt.device.
//
// The input chain is used to signal visibility of all side-effects before
// calling this function. The output chain is used to signal visibility of all
// side-effects of this function.
class TFRTFuncOpSignatureConversion
    : public mlir::OpConversionPattern<mlir::func::FuncOp> {
 public:
  TFRTFuncOpSignatureConversion(mlir::MLIRContext *ctx,
                                mlir::TypeConverter *type_converter)
      : OpConversionPattern(ctx), type_converter_(*type_converter) {}

  LogicalResult matchAndRewrite(
      mlir::func::FuncOp func_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    mlir::FunctionType type = func_op.getFunctionType();

    // Convert the original function arguments.
    mlir::TypeConverter::SignatureConversion converted_signature(
        type.getNumInputs());
    // Add a chain as the first input.
    converted_signature.addInputs(
        rewriter.getType<tfrt::compiler::ChainType>());

    // Convert the original function results.
    SmallVector<Type, 1> converted_results;
    // Add a chain as the first result.
    converted_results.push_back(rewriter.getType<tfrt::compiler::ChainType>());

    if (failed(type_converter_.convertSignatureArgs(type.getInputs(),
                                                    converted_signature)) ||
        failed(type_converter_.convertTypes(type.getResults(),
                                            converted_results)) ||
        failed(rewriter.convertRegionTypes(&func_op.getBody(), type_converter_,
                                           &converted_signature))) {
      return failure();
    }

    llvm::SmallVector<mlir::DictionaryAttr, 4> arg_attrs;
    // The first input, which is a chain added by this pass, has no attribute.
    arg_attrs.emplace_back();
    func_op.getAllArgAttrs(arg_attrs);
    // If any argument has a tf.device attribute, change the attribute name to
    // tfrt.device.
    for (auto &arg_attr : arg_attrs) {
      mlir::NamedAttrList arg_attr_list(arg_attr);
      if (Attribute device = arg_attr_list.erase(kTFDeviceAttr)) {
        arg_attr_list.set(kTFRTDeviceAttr, device);
        arg_attr = arg_attr_list.getDictionary(device.getContext());
      }
    }
    arg_attrs.resize(converted_signature.getConvertedTypes().size());

    // The first result, which is a chain added by this pass, has no attribute.
    llvm::SmallVector<mlir::DictionaryAttr, 4> result_attrs;
    result_attrs.emplace_back();
    func_op.getAllResultAttrs(result_attrs);
    // If any result has a tf.device attribute, change the attribute name to
    // tfrt.device.
    for (auto &result_attr : result_attrs) {
      mlir::NamedAttrList result_attr_list(result_attr);
      if (Attribute device = result_attr_list.erase(kTFDeviceAttr)) {
        result_attr_list.set(kTFRTDeviceAttr, device);
        result_attr = result_attr_list.getDictionary(device.getContext());
      }
    }
    result_attrs.resize(converted_results.size());

    // Update the function signature in-place.
    rewriter.updateRootInPlace(func_op, [&] {
      func_op.setType(mlir::FunctionType::get(
          func_op.getContext(), converted_signature.getConvertedTypes(),
          converted_results));
      func_op.setAllArgAttrs(arg_attrs);
      func_op.setAllResultAttrs(result_attrs);
    });

    return success();
  }

 private:
  mlir::TypeConverter &type_converter_;
};

// Lower a tf.Const op that creates a string tensor to a native
// corert.create_string_tensor op.
class CoreRTConstStringTensorOpConversion
    : public mlir::OpConversionPattern<mlir::TF::ConstOp> {
 public:
  CoreRTConstStringTensorOpConversion(mlir::MLIRContext *context,
                                      CoreRTConverter *corert_converter)
      : mlir::OpConversionPattern<mlir::TF::ConstOp>(context, kCoreRTBenefit),
        corert_converter_(*corert_converter) {}

  LogicalResult matchAndRewrite(
      mlir::TF::ConstOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {  // NOLINT
    if (!op.getDtype().isa<mlir::TF::StringType>()) return failure();

    DenseStringElementsAttr attr =
        op.getValue().cast<DenseStringElementsAttr>();

    llvm::SmallVector<Attribute, 4> values;
    values.reserve(attr.getNumElements());
    for (const auto &element : attr.getRawStringData())
      values.push_back(rewriter.getStringAttr(
          llvm::StringRef(element.data(), element.size())));

    // Create the shape attribute from the tensor shape.
    ArrayRef<int64_t> shape = op.getValue().getShapedType().getShape();
    llvm::SmallVector<mlir::Attribute, 4> dims;
    dims.reserve(shape.size());
    auto i64_type = rewriter.getIntegerType(64);
    for (auto dim : shape)
      dims.push_back(rewriter.getIntegerAttr(i64_type, dim));

    auto new_op = rewriter.create<tfrt::corert::ConstStringTensorOp>(
        op.getLoc(), corert_converter_.tensor_handle_type(),
        rewriter.getArrayAttr(dims), rewriter.getArrayAttr(values));

    rewriter.replaceOp(op, new_op.getResult());

    return success();
  }

 private:
  CoreRTConverter &corert_converter_;
};

LogicalResult ConvertFunctionCallOperands(
    mlir::Operation *op, ValueRange operands,
    llvm::SmallVectorImpl<mlir::Value> *new_operands,
    mlir::ConversionPatternRewriter &rewriter, bool func_use_fallback_tensor) {
  if (func_use_fallback_tensor) {
    // TODO(b/182232457): Support other devices.
    return tfrt_compiler::ConvertFallbackOperands(
        op, tfrt_compiler::GetDefaultCpuDeviceName(), operands, new_operands,
        rewriter);
  } else {
    return tfrt_compiler::ConvertCoreRTOperands(op, operands, new_operands,
                                                rewriter);
  }
}

// Convert TF call ops (eg. StatefulPartitionedCall) to tfrt.call.
template <typename CallOp>
class TFRTCallOpConversion : public mlir::OpConversionPattern<CallOp> {
 public:
  TFRTCallOpConversion(mlir::MLIRContext *context,
                       mlir::TypeConverter *type_converter,
                       CoreRTConverter *corert_converter,
                       bool func_use_fallback_tensor)
      : mlir::OpConversionPattern<CallOp>(context),
        type_converter_(*type_converter),
        corert_converter_(*corert_converter),
        func_use_fallback_tensor_(func_use_fallback_tensor) {}

  LogicalResult matchAndRewrite(
      CallOp op, typename CallOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (auto xla_must_compile =
            op->template getAttrOfType<mlir::BoolAttr>("_XlaMustCompile");
        xla_must_compile && xla_must_compile.getValue()) {
      return mlir::failure();
    }

    auto callee =
        op.getCallableForCallee().template dyn_cast<mlir::SymbolRefAttr>();
    if (!callee) return failure();

    llvm::SmallVector<mlir::Value, 4> new_operands;
    new_operands.push_back(
        corert_converter_.GetLocalSideEffectChain(op, &rewriter));
    // Currently the converted functions always use !corert.tensorhandle types
    // for tensor arguments and results.
    //
    // TODO(b/175881042): We should avoid the tensor conversion here if the
    // operand is !tfrt_fallback.tf_tensor, and it is also used as fallback
    // tensor inside the callee function.
    if (mlir::failed(ConvertFunctionCallOperands(op, adaptor.getOperands(),
                                                 &new_operands, rewriter,
                                                 func_use_fallback_tensor_)))
      return failure();

    llvm::SmallVector<mlir::Type, 4> result_types;
    result_types.push_back(rewriter.getType<tfrt::compiler::ChainType>());
    for (auto type : op.getOperation()->getResultTypes()) {
      if (failed(type_converter_.convertType(type, result_types)))
        return failure();
    }

    auto new_op = rewriter.create<tfrt::compiler::CallOp>(
        op.getLoc(), result_types, callee.getRootReference().getValue(),
        new_operands);
    rewriter.replaceOp(op, new_op.getResults().drop_front());

    if (!mlir::isMemoryEffectFree(op)) {
      // Register the converted op so that it can be retrieved by successors.
      // TODO(chky): Add OpTraits or OpInterface, rather than assume first
      // result is a chain.
      corert_converter_.RegisterLocalSideEffectChain(op, new_op.getResult(0));
    }

    return success();
  }

 private:
  mlir::TypeConverter &type_converter_;
  CoreRTConverter &corert_converter_;
  bool func_use_fallback_tensor_;
};

// Convert func ReturnOp to tfrt.return.
//
// TODO(chky): conversion to tfrt kernels should come from a common tf_to_tfrt
// library.
class TFRTReturnOpConversion
    : public mlir::OpConversionPattern<mlir::func::ReturnOp> {
 public:
  TFRTReturnOpConversion(mlir::MLIRContext *context,
                         CoreRTConverter *corert_converter,
                         bool func_use_fallback_tensor)
      : mlir::OpConversionPattern<mlir::func::ReturnOp>(context),
        corert_converter_(*corert_converter),
        func_use_fallback_tensor_(func_use_fallback_tensor) {}

  LogicalResult matchAndRewrite(
      mlir::func::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value, 2> new_operands;

    // Currently in mlir::TF::SideEffectAnalysis, all terminator ops are treated
    // as side-effect ops and they have predecessors but not successors.
    //
    // TODO(chky): ReturnOp has no side effect. Once the special handling in
    // mlir::TF::SideEffectAnalysis is removed, the chains should come from
    // side-effecting ops with no successors in the function.
    new_operands.push_back(
        corert_converter_.GetLocalSideEffectChain(op, &rewriter));
    if (mlir::failed(ConvertFunctionCallOperands(op, adaptor.getOperands(),
                                                 &new_operands, rewriter,
                                                 func_use_fallback_tensor_)))
      return failure();

    rewriter.replaceOpWithNewOp<tfrt::compiler::ReturnOp>(op, new_operands);
    return success();
  }

 private:
  CoreRTConverter &corert_converter_;
  bool func_use_fallback_tensor_;
};

// Convert tf.Case op to tfrt.Case.
//
// TF dialect:
// %outputs = "tf.Case"(%arg, ...) { branches = [@branch0, @branch1], ...}
//
// lowered TFRT CoreRT dialect:
// %idx_int = corert.tensorhandle_to_int32 %idx
// %out_chain, %outputs = tfrt.case %idx_int [@branch0, @branch1] (%chain, ...)
class TFRTCaseOpConversion : public mlir::OpConversionPattern<TF::CaseOp> {
 public:
  TFRTCaseOpConversion(mlir::MLIRContext *context,
                       mlir::TypeConverter *type_converter,
                       CoreRTConverter *corert_converter,
                       bool func_use_fallback_tensor)
      : mlir::OpConversionPattern<TF::CaseOp>(context),
        type_converter_(*type_converter),
        corert_converter_(*corert_converter),
        func_use_fallback_tensor_(func_use_fallback_tensor) {}

  LogicalResult matchAndRewrite(
      TF::CaseOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    mlir::ArrayAttr branches = op.getBranches();

    llvm::SmallVector<mlir::Type, 4> result_types;
    result_types.push_back(corert_converter_.chain_type());
    for (mlir::Type type : op->getResultTypes()) {
      if (failed(type_converter_.convertType(type, result_types)))
        return failure();
    }

    llvm::SmallVector<mlir::Value, 4> branch_operands;
    branch_operands.push_back(
        corert_converter_.GetLocalSideEffectChain(op, &rewriter));
    if (mlir::failed(ConvertFunctionCallOperands(
            op, adaptor.getOperands().drop_front(), &branch_operands, rewriter,
            func_use_fallback_tensor_)))
      return failure();

    mlir::Value index_operand = adaptor.getOperands()[0];
    // TODO(b/182233401): Support TF tensor; remove the conversion op here.
    if (index_operand.getType().isa<tfrt::fallback::TFTensorType>()) {
      // TODO(b/182232457): Support other devices.
      index_operand =
          rewriter
              .create<
                  tfrt::fallback_async::FallbackTensorToCoreRTTensorHandleOp>(
                  op.getLoc(),
                  rewriter.getType<tfrt::corert::TensorHandleType>(),
                  adaptor.getOperands()[0],
                  tfrt_compiler::GetDefaultCpuDeviceName())
              .getResult(0);
    }
    if (!index_operand.getType().isa<tfrt::corert::TensorHandleType>())
      return op.emitError(
          "branch index operand is expected to be a TensorHandle.");
    mlir::Value index_value =
        rewriter.create<tfrt::corert::TensorHandleToIntOp>(
            op.getLoc(), rewriter.getI32Type(), index_operand);

    auto new_op = rewriter.create<tfrt::compiler::CaseOp>(
        op.getLoc(), result_types, index_value, branches, branch_operands);

    rewriter.replaceOp(op, new_op.getBranchOutputs().drop_front());
    return success();
  }

 private:
  mlir::TypeConverter &type_converter_;
  CoreRTConverter &corert_converter_;
  bool func_use_fallback_tensor_;
};

static mlir::Value GetPredicate(mlir::Operation *op, mlir::Value cond_operand,
                                mlir::ConversionPatternRewriter &rewriter) {
  if (!cond_operand.getType().isa<tfrt::fallback::TFTensorType>()) {
    cond_operand = tfrt_compiler::ConvertCoreRTTensorHandleToFallbackTensor(
        op->getLoc(), tfrt_compiler::GetDefaultCpuDeviceName(), cond_operand,
        rewriter);
    if (!cond_operand) {
      op->emitWarning("failed to convert the cond operand to fallback tensor.");
      return {};
    }
  }

  return rewriter.create<tfrt::fallback_async::PredicateOp>(
      op->getLoc(), rewriter.getI1Type(), cond_operand);
}

class TFRTCondOpConversion : public mlir::OpConversionPattern<mlir::TF::IfOp> {
 public:
  TFRTCondOpConversion(mlir::MLIRContext *context,
                       mlir::TypeConverter *type_converter,
                       CoreRTConverter *corert_converter,
                       bool func_use_fallback_tensor)
      : mlir::OpConversionPattern<TF::IfOp>(context),
        type_converter_(*type_converter),
        corert_converter_(*corert_converter),
        func_use_fallback_tensor_(func_use_fallback_tensor) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::IfOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::FlatSymbolRefAttr then_branch = op.getThenBranchAttr();
    mlir::FlatSymbolRefAttr else_branch = op.getElseBranchAttr();

    llvm::SmallVector<Type, 4> result_types;
    result_types.push_back(rewriter.getType<tfrt::compiler::ChainType>());
    for (mlir::Type type : op.getOperation()->getResultTypes()) {
      if (failed(type_converter_.convertType(type, result_types)))
        return failure();
    }

    // Convert the cond tensor to a boolean value so that it can be used by
    // tfrt.cond kernel.
    auto bool_cond = GetPredicate(op, adaptor.getOperands()[0], rewriter);
    if (!bool_cond) return failure();

    llvm::SmallVector<mlir::Value, 4> new_operands;
    // The first arg of converted branch functions should be !tfrt.chain.
    new_operands.push_back(
        corert_converter_.GetLocalSideEffectChain(op, &rewriter));

    if (mlir::failed(ConvertFunctionCallOperands(
            op, adaptor.getOperands().drop_front(), &new_operands, rewriter,
            func_use_fallback_tensor_)))
      return failure();

    auto new_op = rewriter.create<tfrt::compiler::CondOp>(
        op.getLoc(), result_types, bool_cond, then_branch, else_branch,
        new_operands);

    // The first result is a !tfrt.chain.
    rewriter.replaceOp(op, new_op.getResults().drop_front(1));

    if (!mlir::isMemoryEffectFree(op)) {
      // Register the converted op so that it can be retrieved by successors.
      // TODO(chky): Add OpTraits or OpInterface, rather than assume first
      // result is a chain.
      corert_converter_.RegisterLocalSideEffectChain(op, new_op.getResult(0));
    }

    return success();
  }

 private:
  mlir::TypeConverter &type_converter_;
  CoreRTConverter &corert_converter_;
  bool func_use_fallback_tensor_;
};

// Convert TF WhileOp to tfrt.while. tfrt.while use a boolean condition and has
// slightly different semantics from tf.While for performance and generality.
// The pseudo code of tfrt.while is as follows:
//
//  while(cond) {
//    outputs, cond = body(inputs)
//    inputs = outputs
//  }
//  return outputs
//
// So we need to insert extra convertion kernels and merge functions when
// lowering tf.While to tfrt.while.
//
//  %result = tf.While(%arg) {cond = @original_cond_fn, body =
//  @original_body_fn}
//
// is converted to
//
//  func @new_pred_fn(%ch, %arg) {
//    %ch0, %cond_tensor = tfrt.call @original_cond_fn(%ch, %arg)
//    %cond_bool = tfrt_fallback_async.predicate %cond_tensor
//    tfrt.return %ch0, %cond_bool
//  }
//
//  func @new_while_body(%ch, %arg) {
//    %ch0, %result = tfrt.call @original_body_fn(%ch, %arg)
//    %ch1, %cond_bool = tfrt.call @new_pred_fn(%ch0, %result)
//    tfrt.return %ch1, %result, %cond_bool
//  }
//
//  %ch0, %first_iter_cond = tfrt.call @new_pred_fn(%ch, %arg)
//  %ch1, %result = tfrt.while %first_iter_cond @new_while_body(%ch0, %arg)
//
class TFRTWhileOpConversion
    : public mlir::OpConversionPattern<mlir::TF::WhileOp> {
 public:
  TFRTWhileOpConversion(mlir::MLIRContext *context,
                        mlir::TypeConverter *type_converter,
                        CoreRTConverter *corert_converter,
                        mlir::SymbolTable *symbol_table,
                        const tfrt_compiler::TensorArraySideEffectAnalysis
                            *tensor_array_side_effect_analysis,
                        bool func_use_fallback_tensor,
                        bool enable_while_parallel_iterations)
      : mlir::OpConversionPattern<TF::WhileOp>(context),
        type_converter_(*type_converter),
        corert_converter_(*corert_converter),
        symbol_table_(*symbol_table),
        tensor_array_side_effect_analysis_(*tensor_array_side_effect_analysis),
        func_use_fallback_tensor_(func_use_fallback_tensor),
        enable_while_parallel_iterations_(enable_while_parallel_iterations) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::WhileOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::FlatSymbolRefAttr cond_fn = op.getCondAttr();
    mlir::FlatSymbolRefAttr body_fn = op.getBodyAttr();

    llvm::SmallVector<Type, 4> while_arg_result_types;
    // Insert a chain for side effects as the first argument/result.
    while_arg_result_types.push_back(
        rewriter.getType<tfrt::compiler::ChainType>());
    for (mlir::Type type : op.getOperation()->getResultTypes()) {
      if (failed(type_converter_.convertType(type, while_arg_result_types)))
        return failure();
    }

    // Convert the operands to either fallback tensor or corert tensors as
    // specified in the option.
    llvm::SmallVector<mlir::Value, 4> new_operands;
    if (mlir::failed(ConvertFunctionCallOperands(op, adaptor.getOperands(),
                                                 &new_operands, rewriter,
                                                 func_use_fallback_tensor_)))
      return failure();

    // Create the predicate function that calls the original cond function and
    // in addition convert the result to a boolean value.
    mlir::func::FuncOp pred_fn =
        GetPredicateFunction(op, cond_fn, while_arg_result_types, rewriter);
    if (!pred_fn) return failure();

    auto in_chain = corert_converter_.GetLocalSideEffectChain(op, &rewriter);
    auto out_chain = in_chain;

    bool has_at_most_tensor_array_effect = HasAtMostTensorArrayEffect(op);

    // Prepare the arguments to call the pred function for the first iteration.
    llvm::SmallVector<mlir::Value, 4> pred_args;
    pred_args.push_back(
        has_at_most_tensor_array_effect ? GetFunctionInputChain(op) : in_chain);
    pred_args.append(new_operands.begin(), new_operands.end());

    // Insert a call op to call the pred function for the first iteration.
    auto call_pred_fn = rewriter.create<tfrt::compiler::CallOp>(
        op.getLoc(), pred_fn.getFunctionType().getResults(),
        pred_fn.getSymName(), pred_args);

    auto pred_chain = call_pred_fn.getResult(0);
    auto first_iteration_bool_cond = call_pred_fn.getResult(1);

    mlir::func::FuncOp new_body_fn = GetWhileBodyFunction(
        op, body_fn, pred_fn, while_arg_result_types, rewriter);

    // Use the pred chain as the chain to the while body. The rest args should
    // be the same as the pred_args.
    auto &while_args = pred_args;
    while_args[0] = pred_chain;

    int64_t parallel_iterations =
        enable_while_parallel_iterations_ ? op.getParallelIterations() : 1;

    auto new_op = rewriter.create<tfrt::compiler::WhileOp>(
        op.getLoc(), while_arg_result_types, first_iteration_bool_cond,
        while_args, new_body_fn.getSymName(), parallel_iterations);

    rewriter.replaceOp(op, new_op.getResults().drop_front());

    if (!has_at_most_tensor_array_effect) out_chain = new_op.getResult(0);

    if (!mlir::isMemoryEffectFree(op)) {
      // Register the converted op so that it can be retrieved by successors.
      // TODO(chky): Add OpTraits or OpInterface, rather than assume first
      // result is a chain.
      corert_converter_.RegisterLocalSideEffectChain(op, out_chain);
    }
    return success();
  }

 private:
  bool HasAtMostTensorArrayEffect(mlir::TF::WhileOp op) const {
    return tensor_array_side_effect_analysis_.HasAtMostTensorArrayEffect(
               op.cond_function()) &&
           tensor_array_side_effect_analysis_.HasAtMostTensorArrayEffect(
               op.body_function());
  }

  mlir::func::FuncOp GetPredicateFunction(
      mlir::TF::WhileOp op, mlir::FlatSymbolRefAttr cond_fn,
      mlir::TypeRange arg_types,
      mlir::ConversionPatternRewriter &rewriter) const;

  mlir::func::FuncOp GetWhileBodyFunction(
      mlir::TF::WhileOp op, mlir::FlatSymbolRefAttr body_fn,
      mlir::func::FuncOp pred_fn, mlir::TypeRange arg_types,
      mlir::ConversionPatternRewriter &rewriter) const;

  mlir::TypeConverter &type_converter_;
  CoreRTConverter &corert_converter_;
  mlir::SymbolTable &symbol_table_;
  const tfrt_compiler::TensorArraySideEffectAnalysis
      &tensor_array_side_effect_analysis_;
  bool func_use_fallback_tensor_;
  bool enable_while_parallel_iterations_;
};

// Create the pred function that contains a call to the original cond function
// and a predicate kernel that converts the cond tensor to a boolean value. eg.
//
// func @pred_fn(%ch, %arg) {
//  %ch0, %cond_tensor = tfrt.call @original_cond_fn(%ch, %arg)
//  %cond_bool = tfrt_fallback_async.predicate %cond_tensor
//  return %ch0, %cond_bool
// }
//
mlir::func::FuncOp TFRTWhileOpConversion::GetPredicateFunction(
    mlir::TF::WhileOp op, mlir::FlatSymbolRefAttr cond_fn,
    mlir::TypeRange arg_types,
    mlir::ConversionPatternRewriter &rewriter) const {
  std::string pred_fn_name = cond_fn.getValue().str() + "/tfrt_predicate";

  if (auto pred_fn = symbol_table_.lookup<mlir::func::FuncOp>(pred_fn_name)) {
    return pred_fn;
  }

  auto func_op = op->getParentOfType<mlir::func::FuncOp>();

  mlir::ConversionPatternRewriter::InsertionGuard insertion_guard(rewriter);
  rewriter.setInsertionPointAfter(func_op);

  std::array<mlir::Type, 2> pred_result_types = {
      rewriter.getType<tfrt::compiler::ChainType>(), rewriter.getI1Type()};

  auto func_type = rewriter.getFunctionType(arg_types, pred_result_types);

  auto pred_fn =
      rewriter.create<mlir::func::FuncOp>(op.getLoc(), pred_fn_name, func_type);

  auto *block = pred_fn.addEntryBlock();
  rewriter.setInsertionPointToStart(block);

  // There are at least two arguments, with the first being !tfrt.chain and the
  // second being either !tfrt_fallback.tf_tensor or !corert.tensorhandle.
  // cond_fn must have two results. The first of them must also be !tfrt.chain
  // and the second must have the same tensor type as arguments. So we can just
  // use the first two arg types as the result types.
  assert(arg_types.size() >= 2);
  auto cond_result_types = arg_types.take_front(2);

  auto call_cond_fn = rewriter.create<tfrt::compiler::CallOp>(
      op.getLoc(), cond_result_types, cond_fn, block->getArguments());

  auto chain = call_cond_fn.getResult(0);
  auto cond = call_cond_fn.getResult(1);

  auto bool_cond = GetPredicate(op, cond, rewriter);
  if (!bool_cond) return {};

  llvm::SmallVector<mlir::Value, 2> results = {chain, bool_cond};

  rewriter.create<tfrt::compiler::ReturnOp>(op.getLoc(), results);

  symbol_table_.insert(pred_fn);

  return pred_fn;
}

// Create the new while body function that contains a call to original while
// body and then a call to the pred function. eg.
//
// func @new_while_body(%ch, %arg) {
//   %ch0, %result = tfrt.call @original_body(%ch, %arg)
//   %ch1, %cond_bool = tfrt.call @pred_function(%ch0, %arg)
//   tfrt.return %ch1, %result, %cond_bool
// }
//
mlir::func::FuncOp TFRTWhileOpConversion::GetWhileBodyFunction(
    mlir::TF::WhileOp op, mlir::FlatSymbolRefAttr original_body_fn,
    mlir::func::FuncOp pred_fn, mlir::TypeRange arg_types,
    mlir::ConversionPatternRewriter &rewriter) const {
  int64_t parallel_iterations =
      enable_while_parallel_iterations_ ? op.getParallelIterations() : 1;

  std::string body_fn_name = original_body_fn.getValue().str() + "/tfrt_body_" +
                             absl::StrCat(parallel_iterations);

  if (auto body_fn = symbol_table_.lookup<mlir::func::FuncOp>(body_fn_name)) {
    return body_fn;
  }

  auto func_op = op->getParentOfType<mlir::func::FuncOp>();

  mlir::ConversionPatternRewriter::InsertionGuard insertion_guard(rewriter);
  rewriter.setInsertionPointAfter(func_op);

  auto while_result_types = arg_types;

  llvm::SmallVector<mlir::Type, 4> body_result_types(arg_types.begin(),
                                                     arg_types.end());

  // The last result of the while body function is the boolean condition.
  body_result_types.push_back(rewriter.getI1Type());
  auto func_type = rewriter.getFunctionType(arg_types, body_result_types);
  auto body_fn =
      rewriter.create<mlir::func::FuncOp>(op.getLoc(), body_fn_name, func_type);
  if (parallel_iterations > 1) {
    // Disable stream merging by setting cost threshold to 1. The key to
    // parallelize while iterations is to execute iteration index handling (e.g.
    // ++i; i < max_iterations) in parallel to loop bodies. Since iteration
    // index handling is usually cheap when this while loop is parallelizable,
    // we don't want to merge it with loop bodies into one stream of inline
    // execution. The quickest way to achieve this is to disable stream merging
    // for loop functions. The potential downside is stream merging won't be
    // applied to other part of the loop body, so there might be excessive
    // threading overhead.
    //
    // TODO(chky): Consider a better way of parallizing while ops, so that we
    // can have stream merging applied within the loop body. One option is to
    // perform compiler transformation to extract iteration index handling logic
    // out of the loop body and convert it to a parallel_map-like op.
    body_fn->setAttr("tfrt.cost_threshold", rewriter.getI64IntegerAttr(1));
  }

  auto *block = body_fn.addEntryBlock();
  rewriter.setInsertionPointToStart(block);

  // Insert a call to the original body function.
  auto call_original_body_fn = rewriter.create<tfrt::compiler::CallOp>(
      op.getLoc(), while_result_types, original_body_fn, block->getArguments());

  // Insert a call to the pred function, which contains a call to the original
  // cond function and the predicate kernel that converts the tensor to boolean
  // value.
  auto call_pred_fn = rewriter.create<tfrt::compiler::CallOp>(
      op.getLoc(), pred_fn.getFunctionType().getResults(), pred_fn.getSymName(),
      call_original_body_fn.getResults());

  auto pred_chain = call_pred_fn.getResult(0);

  llvm::SmallVector<mlir::Value, 4> body_results;
  // The first result should be the chain returned from the pred function.
  body_results.push_back(pred_chain);
  // Then comes the results from the orignal while body excluding the first
  // chain (which is replaced by the `pred_chain`).
  for (auto res : llvm::drop_begin(call_original_body_fn.getResults())) {
    body_results.push_back(res);
  }
  // The last result should be the boolean value converted from the condition.
  auto bool_cond = call_pred_fn.getResult(1);
  body_results.push_back(bool_cond);

  rewriter.create<tfrt::compiler::ReturnOp>(op.getLoc(), body_results);

  symbol_table_.insert(body_fn);

  return body_fn;
}

// Helper function for specifying legal dialects for conversion to CoreRT.
void SetUpTFToTFRTConversionLegality(mlir::ConversionTarget *target,
                                     mlir::TypeConverter *func_type_converter,
                                     mlir::Type chain_type) {
  target->addLegalDialect<tfrt::corert::CoreRTDialect>();
  target->addLegalDialect<tfrt::fallback_async::FallbackAsyncDialect>();
  target->addLegalDialect<tfrt::compiler::TFRTDialect>();
  target->addLegalDialect<tfrt::test::TestDialect>();
  target->addIllegalDialect<TF::TensorFlowDialect>();
  target->addIllegalDialect<tf_device::TensorFlowDeviceDialect>();
  target->addDynamicallyLegalOp<mlir::func::FuncOp>([func_type_converter,
                                                     chain_type](
                                                        func::FuncOp op) {
    auto func_type = op.getFunctionType();
    if (func_type.getNumInputs() == 0 || func_type.getInput(0) != chain_type)
      return false;
    if (func_type.getNumResults() == 0 || func_type.getResult(0) != chain_type)
      return false;

    return func_type_converter->isSignatureLegal(op.getFunctionType()) &&
           func_type_converter->isLegal(&op.getBody());
  });
}

// Helper function for inserting TF dialect to TFRT dialect op conversion
// patterns.
void PopulateTFToTFRTConversionPatterns(
    mlir::MLIRContext *context, mlir::RewritePatternSet *patterns,
    CoreRTConverter *corert_converter,
    tfrt_compiler::FallbackConverter *fallback_converter,
    mlir::SymbolTable *symbol_table,
    const tfrt_compiler::CostAnalysis *cost_analysis,
    const tfrt_compiler::TensorArraySideEffectAnalysis
        *tensor_array_side_effect_analysis,
    bool func_use_fallback_tensor, bool enable_while_parallel_iterations,
    bool tpu_lower_to_fallback, bool target_tpurt, bool use_bridge_for_gpu) {
  // By default, we lower all TF ops to fallback ops.
  patterns->add<FallbackExecuteOpConversion>(
      context, corert_converter, fallback_converter, symbol_table,
      cost_analysis, tpu_lower_to_fallback, target_tpurt, use_bridge_for_gpu);
  patterns->add<FallbackConstOpConversion, FallbackSetResourceOp,
                FallbackGetResourceOp>(context, corert_converter);

  // For control flow ops, we handle them according to the option.
  mlir::TypeConverter *func_type_converter;
  if (func_use_fallback_tensor) {
    func_type_converter = fallback_converter;
  } else {
    func_type_converter = corert_converter;
  }
  patterns->add<TFRTFuncOpSignatureConversion>(context, func_type_converter);
  patterns->add<TFRTReturnOpConversion>(context, corert_converter,
                                        func_use_fallback_tensor);
  patterns->add<TFRTWhileOpConversion>(
      context, func_type_converter, corert_converter, symbol_table,
      tensor_array_side_effect_analysis, func_use_fallback_tensor,
      enable_while_parallel_iterations);
  patterns->add<TFRTCallOpConversion<mlir::TF::StatefulPartitionedCallOp>,
                TFRTCallOpConversion<mlir::TF::PartitionedCallOp>,
                TFRTCallOpConversion<mlir::TF::LegacyCallOp>,
                TFRTCaseOpConversion, TFRTCondOpConversion>(
      context, func_type_converter, corert_converter, func_use_fallback_tensor);

  // For tf.BatchFunction, we need a special fallback op to batch a BEF
  // function.
  patterns->add<FallbackBatchFunctionOpConversion>(context, corert_converter);

  // Below patterns are preferred over fallback lowering as we want to use
  // CoreRT interface for native kernels. This is only temporary and it will
  // refactored to use SSOT interface.

  // Here we use specialized patterns for tf.Const on CPU as it is incorrect to
  // use ExecuteOp pattern to convert string tensor attribute.
  patterns->add<CoreRTConstStringTensorOpConversion,
                CoreRTConstDenseTensorOpConversion>(context, corert_converter);
}

// Lower TF dialect MLIR to TFRT dialect.
class TfToTfrtConversionPass
    : public mlir::PassWrapper<TfToTfrtConversionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    getDependentConversionDialects(registry);

    if (target_tpurt_) RegisterTPUDialects(&registry);
    if (target_gpu_) RegisterGpuDialects(&registry);
  }

  llvm::StringRef getArgument() const final { return "tf-to-tfrt"; }
  llvm::StringRef getDescription() const final {
    return "Convert Tensorflow dialect (generated from tf.function) to TFRT "
           "dialect.";
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TfToTfrtConversionPass)

  TfToTfrtConversionPass() = default;
  explicit TfToTfrtConversionPass(const TfrtPipelineOptions &options) {
    target_tpurt_ = options.target_tpurt;
    tpu_use_core_selector_ = options.tpu_use_core_selector;
    tpu_use_bundled_transfer_ = options.tpu_use_bundled_transfer;
    tpu_lower_to_fallback_ = options.tpu_lower_to_fallback;
    tpu_transfer_result_to_host_ = options.tpu_transfer_result_to_host;
    use_tpu_host_allocator_for_inputs_ =
        options.use_tpu_host_allocator_for_inputs;
    tpu_allow_unpadded_batch_ = options.tpu_allow_unpadded_batch;
    cost_threshold_ = options.cost_threshold;
    upper_cost_threshold_ = options.upper_cost_threshold;
    merge_inter_dependent_streams_ = options.merge_inter_dependent_streams;
    func_use_fallback_tensor_ = options.func_use_fallback_tensor;
    enable_while_parallel_iterations_ =
        options.enable_while_parallel_iterations;
    target_gpu_ = options.target_gpu;
    use_bridge_for_gpu_ = options.use_bridge_for_gpu;
  }
  TfToTfrtConversionPass(const TfToTfrtConversionPass &) {}

  mlir::LogicalResult runOnFunction(
      mlir::func::FuncOp func,
      const mlir::TF::SideEffectAnalysis::Info &side_effect_analysis,
      const tfrt_compiler::TensorArraySideEffectAnalysis
          &tensor_array_side_effect_analysis,
      tfrt_compiler::FallbackConverter &fallback_converter,
      mlir::SymbolTable &symbol_table) {
    auto &context = getContext();
    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&getContext());
    CoreRTConverter corert_converter(&context, &side_effect_analysis);
    tfrt_compiler::CostAnalysis cost_analysis(func);

    if (target_tpurt_)
      AddTPUTargetDialectAndPatterns(
          &target, &patterns, &context, &corert_converter, &fallback_converter,
          TfrtTpuExecuteOpConversionOptions{
              tpu_use_core_selector_, tpu_use_bundled_transfer_,
              tpu_transfer_result_to_host_, use_tpu_host_allocator_for_inputs_,
              tpu_allow_unpadded_batch_},
          tpu_lower_to_fallback_);

    if (target_gpu_) {
      AddGpuTargetDialectAndPatterns(&context, &target, &patterns);
    }

    mlir::TypeConverter *func_type_converter;
    if (func_use_fallback_tensor_) {
      func_type_converter = &fallback_converter;
    } else {
      func_type_converter = &corert_converter;
    }
    SetUpTFToTFRTConversionLegality(&target, func_type_converter,
                                    corert_converter.chain_type());

    PopulateJitRtConversionPatterns(&target, &context, &patterns,
                                    &corert_converter);

    PopulateTFToTFRTConversionPatterns(
        &context, &patterns, &corert_converter, &fallback_converter,
        &symbol_table, &cost_analysis, &tensor_array_side_effect_analysis,
        func_use_fallback_tensor_, enable_while_parallel_iterations_,
        tpu_lower_to_fallback_, target_tpurt_, use_bridge_for_gpu_);

    return mlir::applyPartialConversion(func, target, std::move(patterns));
  }

  void runOnOperation() override {
    auto module = getOperation();
    const auto &side_effect_analysis =
        getAnalysis<mlir::TF::SideEffectAnalysis>();

    tfrt_compiler::TensorArraySideEffectAnalysis
        tensor_array_side_effect_analysis(module);
    tfrt_compiler::FallbackConverter fallback_converter(&getContext());

    mlir::SymbolTable symbol_table(module);

    auto func_op_range = module.getOps<mlir::func::FuncOp>();
    llvm::SmallVector<mlir::func::FuncOp, 4> func_ops(func_op_range.begin(),
                                                      func_op_range.end());
    for (auto func : func_ops) {
      if (!func.isExternal()) {
        if (mlir::failed(runOnFunction(
                func, side_effect_analysis.GetAnalysisForFunc(func),
                tensor_array_side_effect_analysis, fallback_converter,
                symbol_table))) {
          signalPassFailure();
          return;
        }

        ChainDanglingValuesinFunction(func);
      }
    }

    CreateFallbackInitializationFunction(module, fallback_converter);

    // Set cost threshold as a module attribute. It will be used later in the
    // runtime to decide whether certain execution is cheap enough to be inline
    // executed.
    mlir::Builder builder(module);
    module->setAttr("tfrt.cost_threshold",
                    builder.getI64IntegerAttr(cost_threshold_));
    module->setAttr("tfrt.upper_cost_threshold",
                    builder.getI64IntegerAttr(upper_cost_threshold_));
    module->setAttr("tfrt.merge_inter_dependent_streams",
                    builder.getBoolAttr(merge_inter_dependent_streams_));
  }

 private:
  // Chain all dangling values (ie. values with no users) together and merge it
  // with the first returned chain. This merged chain can be used to signal the
  // completion of all execution including side-effets.
  void ChainDanglingValuesinFunction(mlir::func::FuncOp func_op) {
    auto &block = func_op.front();

    llvm::SmallVector<mlir::Value, 2> dangling_values;

    // Find dangling function arguments.
    for (auto arg : block.getArguments()) {
      if (arg.use_empty()) {
        dangling_values.push_back(arg);
      }
    }

    // Find dangling values produced by ops in the function.
    for (auto &op : block) {
      for (mlir::Value result : op.getResults()) {
        if (result.use_empty()) {
          dangling_values.push_back(result);
        }
      }
    }

    if (dangling_values.empty()) return;

    // Merge these dangling values with the first returned chain.
    auto return_op =
        llvm::cast<tfrt::compiler::ReturnOp>(block.getTerminator());
    auto chain = return_op->getOperand(0);
    assert(chain.getType().isa<tfrt::compiler::ChainType>());
    dangling_values.push_back(chain);

    mlir::OpBuilder builder(return_op);

    auto new_chain = builder.create<tfrt::compiler::MergeChainsOp>(
        return_op->getLoc(), builder.getType<tfrt::compiler::ChainType>(),
        dangling_values);

    return_op->setOperand(0, new_chain);
  }

  void CreateFallbackInitializationFunction(
      mlir::ModuleOp module,
      tfrt_compiler::FallbackConverter &fallback_converter) {
    mlir::OpBuilder builder(&module.getBodyRegion());

    auto chain_type = builder.getType<tfrt::compiler::ChainType>();

    auto func_op = builder.create<mlir::func::FuncOp>(
        module.getLoc(), "_tfrt_fallback_init",
        mlir::FunctionType::get(module.getContext(), /*inputs=*/chain_type,
                                /*results=*/chain_type));

    auto *block = func_op.addEntryBlock();
    builder.setInsertionPointToStart(block);

    mlir::Value chain_value = block->getArgument(0);

    // Create operations for all fallback kernels in the module.
    for (auto *op : fallback_converter.GetFallbackOps()) {
      auto create_op = builder.create<tfrt::fallback_async::CreateOp>(
          func_op.getLoc(), chain_type, chain_value);

      create_op->setAttrs(op->getAttrs());
      create_op->setAttr("num_args", builder.getI64IntegerAttr(GetNumArgs(op)));

      chain_value = create_op;
    }

    chain_value =
        CreateJitRtFallbackCompileKernel(builder, module, chain_value);

    builder.create<tfrt::compiler::ReturnOp>(func_op.getLoc(), chain_value);
  }

  int64_t GetNumArgs(mlir::Operation *fallback_op) {
    if (auto execute_op =
            llvm::dyn_cast<tfrt::fallback_async::ExecuteOp>(fallback_op)) {
      return execute_op.getArgs().size();
    } else if (auto execute_op_seq =
                   llvm::dyn_cast<tfrt::fallback_async::ExecuteOpSeq>(
                       fallback_op)) {
      return execute_op_seq.getArgs().size();
    } else if (auto execute_op_allocator =
                   llvm::dyn_cast<tfrt::fallback_async::ExecuteOpWithAllocator>(
                       fallback_op)) {
      return execute_op_allocator.getArgs().size();
    } else if (auto execute_op_seq_allocator = llvm::dyn_cast<
                   tfrt::fallback_async::ExecuteOpSeqWithAllocator>(
                   fallback_op)) {
      return execute_op_seq_allocator.getArgs().size();
    }
    llvm_unreachable("invalid fallback op type");
  }

  Option<bool> target_tpurt_{*this, "target-tpurt",
                             llvm::cl::desc("Target TPURT dialect if true."),
                             llvm::cl::init(false)};

  Option<bool> tpu_use_core_selector_{
      *this, "tpu-use-core-selector",
      llvm::cl::desc("If true, use ServingCoreSelector to pick TPU core. "
                     "Otherwise, use the assigned core."),
      llvm::cl::init(true)};

  Option<bool> tpu_use_bundled_transfer_{
      *this, "tpu-use-bundled-transfer",
      llvm::cl::desc("If true, use BundledTransferToTpuOp to transfer "
                     "variables and input tensors to TPU."),
      llvm::cl::init(true)};

  Option<bool> tpu_lower_to_fallback_{
      *this, "tpu-lower-to-fallback",
      llvm::cl::desc("If true, lower a TF op that's placed on TPU device "
                     "to be executed by tfrt_fallback.execute. Note that this "
                     "applies to ops other than TPU compile and execute ops."),
      llvm::cl::init(true)};

  // TODO(b/194081364): remove this option once we unify servo TPU serving
  // result transfer behavior.
  Option<bool> tpu_transfer_result_to_host_{
      *this, "tpu-transfer-result-to-host",
      llvm::cl::desc("If true, transfer the result of tpurt.execute from TPU "
                     "to host."),
      llvm::cl::init(true)};

  Option<bool> use_tpu_host_allocator_for_inputs_{
      *this, "use-tpu-host-allocator-for-inputs",
      llvm::cl::desc("If true, fallback executeops that produce inputs to tpu "
                     "program will use tpu host allocator."),
      llvm::cl::init(false)};

  Option<TfrtCompileOptions::TpuAllowUnpaddedBatch> tpu_allow_unpadded_batch_{
      *this, "tpu-allow-unpadded-batch",
      llvm::cl::desc("To allow unpadded batch for TPU execution."),
      llvm::cl::values(
          clEnumValN(TfrtCompileOptions::TpuAllowUnpaddedBatch::kDisabled,
                     "disabled", "Disable this feature."),
          clEnumValN(TfrtCompileOptions::TpuAllowUnpaddedBatch::kAuto, "auto",
                     "Enable this feature when in-graph batching is detected."),
          clEnumValN(TfrtCompileOptions::TpuAllowUnpaddedBatch::kEnforced,
                     "enforced", "Force to enable this feature.")),
      llvm::cl::init(TfrtCompileOptions::TpuAllowUnpaddedBatch::kDisabled)};

  Option<bool> target_gpu_{
      *this, "target-gpu",
      llvm::cl::desc("If true, target GPU compiler passes."),
      llvm::cl::init(false)};

  // TODO(b/260915352): Remove the flag and default to using bridge.
  Option<bool> use_bridge_for_gpu_{
      *this, "use-bridge-for-gpu",
      llvm::cl::desc("If true, GPU bridge is used."), llvm::cl::init(false)};

  Option<uint64_t> cost_threshold_{
      *this, "tfrt-cost-threshold",
      llvm::cl::desc(
          "The cost threshold to decide whether a sequence of operations is "
          "cheap, and then whether it can be executed inline."),
      llvm::cl::init(1)};

  Option<int64_t> upper_cost_threshold_{
      *this, "tfrt-upper-cost-threshold",
      llvm::cl::desc(
          "The threshold to limit the merging of dependent sequence."),
      llvm::cl::init(-1)};

  Option<bool> merge_inter_dependent_streams_{
      *this, "tfrt-merge-inter-dependent-streams",
      llvm::cl::desc("If true, streams with inter data depenedencies will be "
                     "preferred to be merged for inline execution."),
      llvm::cl::init(false)};

  Option<bool> func_use_fallback_tensor_{
      *this, "func-use-fallback-tensor",
      llvm::cl::desc(
          "If true, use TF tensor as input/output types in func (and other "
          "control flow) ops."),
      llvm::cl::init(false)};

  Option<bool> enable_while_parallel_iterations_{
      *this, "enable-while-parallel-iterations",
      llvm::cl::desc("If true, tf.While op will be parallelized. This is "
                     "currently experimental."),
      llvm::cl::init(false)};
};
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTfToTfrtConversionPass(const TfrtPipelineOptions &options) {
  return std::make_unique<TfToTfrtConversionPass>(options);
}

// -------------------------------------------------------------------------- //
void CreateTfToTfrtPipeline(mlir::OpPassManager &pm,
                            const TfrtPipelineOptions &options) {
  pm.addPass(CreateTfToTfrtConversionPass(options));

  pm.addPass(CreateRemoveDeviceAttributePass());

  // Run optimizer on the MLIR module in CoreRT dialect.
  if (options.enable_optimizer) {
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createInlinerPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        tfrt_compiler::CreateInsertFallbackTensorCopyPass());
  }
}

static void CreateTfExecutorToTfrtPipelineHelper(
    mlir::OpPassManager &pm, const TfrtPipelineOptions &options) {
  CreateTFExecutorToTFPreInvariantOptimizationPipelineHelper(pm, options);
  CreateTFExecutorToTFInvariantOptimizationPipelineHelper(pm, options);
  CreateTfToTfrtPipeline(pm, options);
}

// If verbose logging is on, dump the output of each pass to a file directory,
// set via env var TF_DUMP_GRAPH_PREFIX. e.g.:
// export TF_DUMP_GRAPH_PREFIX=/tmp/mlir
Status CreateTfExecutorToTfrtPipeline(mlir::PassManager &pm,
                                      const TfrtPipelineOptions &options) {
  TF_RETURN_IF_ERROR(
      CreateTFExecutorToTFPreInvariantOptimizationPipeline(pm, options));
  CreateTFExecutorToTFInvariantOptimizationPipelineHelper(pm, options);
  CreateTfToTfrtPipeline(pm, options);
  return OkStatus();
}

static mlir::PassRegistration<TfToTfrtConversionPass> tf_to_tfrt_pass;

static mlir::PassPipelineRegistration<TfrtPipelineOptions> tf_pipeline(
    "tf-executor-to-tfrt-pipeline",
    "Convert Tensorflow Executor dialect to TFRT dialect and "
    "also apply necessary optimization passes.",
    CreateTfExecutorToTfrtPipelineHelper);

}  // namespace tensorflow
