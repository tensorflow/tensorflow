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
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/tf_to_mlrt.h"

#include <stdint.h>

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "google/protobuf/text_format.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h.inc"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tfrt/constants.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_ops.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_tpu_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/execute_op_registry.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/tpu_conversion_patterns.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/util.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/utils.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner_cache.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace mlrt_compiler {
namespace {

constexpr char kXlaLaunchOp[] = "XlaLaunch";

mlir::Value CreateCustomDevice(mlir::Location loc, llvm::StringRef device_name,
                               mlir::ConversionPatternRewriter &rewriter) {
  if (device_name == kTpuHostDevice) {
    return rewriter.create<tf_mlrt_tpu::GetTpuHostDeviceOp>(
        loc, rewriter.getType<tf_mlrt::TFDeviceType>());
  }

  return nullptr;
}

class FuncOpSignatureConversion final
    : public mlir::OpConversionPattern<mlir::func::FuncOp> {
 public:
  explicit FuncOpSignatureConversion(
      mlir::MLIRContext *context, mlir::TypeConverter *type_converter,
      const llvm::DenseMap<llvm::StringRef, llvm::SmallVector<mlir::Type>>
          *function_call_site_input_types)
      : mlir::OpConversionPattern<mlir::func::FuncOp>(context),
        type_converter_(*type_converter),
        function_call_site_input_types_(*function_call_site_input_types) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::func::FuncOp func_op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto it = function_call_site_input_types_.find(func_op.getName());
    if (it == function_call_site_input_types_.end()) {
      return mlir::failure();
    }
    const llvm::SmallVector<mlir::Type> &call_site_input_types = it->second;

    mlir::FunctionType func_type = func_op.getFunctionType();
    DCHECK_EQ(func_type.getNumInputs(), call_site_input_types.size());

    mlir::TypeConverter::SignatureConversion converted_signature(
        func_type.getNumInputs());
    for (const auto &[index, value] : llvm::enumerate(call_site_input_types)) {
      converted_signature.addInputs(index, value);
    }

    // Update the function signature in-place.
    rewriter.modifyOpInPlace(func_op, [&] {
      func_op.setType(mlir::FunctionType::get(
          func_op.getContext(), converted_signature.getConvertedTypes(),
          func_type.getResults()));
    });

    // Update the entry block
    if (rewriter.applySignatureConversion(&func_op.getBody().front(),
                                          converted_signature,
                                          &type_converter_) == nullptr) {
      return mlir::failure();
    }

    return mlir::success();
  }

 private:
  mlir::TypeConverter &type_converter_;
  const llvm::DenseMap<llvm::StringRef, llvm::SmallVector<mlir::Type>>
      &function_call_site_input_types_;
};

// Convert tf_mlrt::AsyncWhile's signature to tf_mlrt::TFTensorType
class TFAsyncWhileOpConversion
    : public mlir::OpConversionPattern<tf_mlrt::TFAsyncWhileOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      tf_mlrt::TFAsyncWhileOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto new_op = rewriter.create<tf_mlrt::AsyncWhileOp>(
        op.getLoc(), op.getResultTypes(), adaptor.getOperands(),
        op->getAttrs());
    rewriter.replaceOp(op, new_op.getResults());
    return mlir::success();
  }
};

class TFAwaitOpConversion final
    : public mlir::OpConversionPattern<tf_mlrt::TFAwaitOp> {
 public:
  explicit TFAwaitOpConversion(mlir::MLIRContext *context)
      : mlir::OpConversionPattern<tf_mlrt::TFAwaitOp>(context) {}

  mlir::LogicalResult matchAndRewrite(
      tf_mlrt::TFAwaitOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto new_op = rewriter.create<tf_mlrt::AwaitOp>(
        op->getLoc(), rewriter.getType<tf_mlrt::TFTensorType>(),
        adaptor.getFuture());
    rewriter.replaceOp(op, new_op.getResult());
    return mlir::success();
  }
};

class TFPromiseOpConversion final
    : public mlir::OpConversionPattern<tf_mlrt::TFPromiseOp> {
 public:
  explicit TFPromiseOpConversion(mlir::MLIRContext *context)
      : mlir::OpConversionPattern<tf_mlrt::TFPromiseOp>(context) {}

  mlir::LogicalResult matchAndRewrite(
      tf_mlrt::TFPromiseOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    if (llvm::isa<::mlrt::compiler::FutureType>(
            adaptor.getTensor().getType())) {
      auto new_op = rewriter.create<tf_mlrt::PromiseFutureOp>(
          op->getLoc(), adaptor.getPromise(), adaptor.getTensor());
      rewriter.replaceOp(op, new_op->getResults());

    } else {
      auto new_op = rewriter.create<tf_mlrt::PromiseOp>(
          op->getLoc(), adaptor.getPromise(), adaptor.getTensor());
      rewriter.replaceOp(op, new_op->getResults());
    }
    return mlir::success();
  }
};

// Convert tf_mlrt::MapFn's signature to tf_mlrt::TFTensorType
class TFMapFnOpConversion
    : public mlir::OpConversionPattern<tf_mlrt::TFMapFnOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      tf_mlrt::TFMapFnOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> result_types;
    result_types.resize(op->getResultTypes().size(),
                        rewriter.getType<tf_mlrt::TFTensorType>());

    auto new_op = rewriter.create<tf_mlrt::MapFnOp>(
        op.getLoc(), result_types, adaptor.getOperands(), op->getAttrs());
    rewriter.replaceOp(op, new_op.getResult());
    return mlir::success();
  }
};

// Convert TF call ops (eg. StatefulPartitionedCall) to call.
template <typename TFCallOp>
class TFCallOpConversion : public mlir::OpConversionPattern<TFCallOp> {
 public:
  TFCallOpConversion(mlir::MLIRContext *context,
                     mlir::TypeConverter *type_converter)
      : mlir::OpConversionPattern<TFCallOp>(context),
        type_converter_(*type_converter) {}

  mlir::LogicalResult matchAndRewrite(
      TFCallOp op, typename TFCallOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    if (auto xla_must_compile =
            op->template getAttrOfType<mlir::BoolAttr>("_XlaMustCompile");
        xla_must_compile && xla_must_compile.getValue()) {
      return mlir::failure();
    }

    auto callee =
        op.getCallableForCallee().template dyn_cast<mlir::SymbolRefAttr>();
    if (!callee) return mlir::failure();

    llvm::SmallVector<mlir::Type, 4> result_types;
    for (auto type : op.getOperation()->getResultTypes()) {
      if (failed(type_converter_.convertType(type, result_types)))
        return mlir::failure();
    }

    auto new_op = rewriter.create<mlir::func::CallOp>(
        op.getLoc(), result_types, callee.getRootReference().getValue(),
        adaptor.getOperands());
    rewriter.replaceOp(op, new_op.getResults());
    return mlir::success();
  }

 private:
  mlir::TypeConverter &type_converter_;
};

// Convert tf.Case op to mlrt.Case.
//
// TF dialect:
// %outputs = "tf.Case"(%idx_tensor, %arg, ...) { branches = [@branch0,
// @branch1],
// ...}
//
// lowered MLRT dialect:
// %branch_idx = tf_mlrt.tensor_to_int32(%idx_tensor)
// %outputs = mlrt.case %branch_idx [@branch0, @branch1] (%arg, ...)
class CaseOpConversion : public mlir::OpConversionPattern<mlir::TF::CaseOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::CaseOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ArrayAttr branches = op.getBranches();

    llvm::SmallVector<mlir::Type> result_types;
    result_types.resize(op->getResultTypes().size(),
                        rewriter.getType<tf_mlrt::TFTensorType>());

    auto index_operand = rewriter.create<tf_mlrt::TensorToIntOp>(
        op.getLoc(), rewriter.getI32Type(), adaptor.getBranchIndex());

    auto new_op = rewriter.create<mlrt::compiler::CaseOp>(
        op.getLoc(), result_types, index_operand.getResult(), branches,
        adaptor.getInput());

    rewriter.replaceOp(op, new_op.getResults());
    return mlir::success();
  }
};

class AsyncOpConversion
    : public mlir::OpConversionPattern<mlrt::compiler::AsyncOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  // Hook for derived classes to implement combined matching and rewriting.
  mlir::LogicalResult matchAndRewrite(
      mlrt::compiler::AsyncOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlrt::compiler::AsyncOp>(
        op, op->getResultTypes(), adaptor.getOperands(), op.getCallee());
    return mlir::success();
  }
};

// SetResourceOpConversion lowers a TF SetResource op to a tf_mlrt.set_resource
// op.
class SetResourceOpConversion final
    : public mlir::OpConversionPattern<mlir::TF::_TfrtSetResourceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::_TfrtSetResourceOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tf_mlrt::SetResourceOp>(op, adaptor.getArg(),
                                                        op.getIndex());
    return mlir::success();
  }
};

// GetResourceOpConversion lowers a TF GetResource op to a tf_mlrt.get_resource
// op.
class GetResourceOpConversion final
    : public mlir::OpConversionPattern<mlir::TF::_TfrtGetResourceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::_TfrtGetResourceOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> result_types(
        op.getNumResults(), rewriter.getType<tf_mlrt::TFTensorType>());
    auto new_op = rewriter.create<tf_mlrt::GetResourceOp>(
        op->getLoc(), result_types, op.getIndices());
    rewriter.replaceOp(op, new_op->getResults());
    return mlir::success();
  }
};

// Convert tf_mlrt.TFIfrtLoadVariableOp to tf_mlrt.IfrtLoadVariableOp
class TFIfrtLoadVariableOpConversion
    : public mlir::OpConversionPattern<tf_mlrt::TFIfrtLoadVariableOp> {
 public:
  TFIfrtLoadVariableOpConversion(mlir::MLIRContext *context,
                                 mlir::TypeConverter *type_converter)
      : mlir::OpConversionPattern<tf_mlrt::TFIfrtLoadVariableOp>(context),
        type_converter_(*type_converter) {}

  mlir::LogicalResult matchAndRewrite(
      tf_mlrt::TFIfrtLoadVariableOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type, 4> result_types;
    for (auto type : op->getResultTypes()) {
      if (failed(type_converter_.convertType(type, result_types)))
        return mlir::failure();
    }

    auto new_op = rewriter.create<tf_mlrt::IfrtLoadVariableOp>(
        op.getLoc(), result_types, adaptor.getOperands()[0],
        op.getUsedByHostAttr());
    rewriter.replaceOp(op, new_op);

    return mlir::success();
  }

 private:
  mlir::TypeConverter &type_converter_;
};

// Convert tf.IfrtRestoreVariableOp to tf_mlrt.IfrtRestoreVariableOp
class IfrtRestoreVariableOpConversion
    : public mlir::OpConversionPattern<mlir::TF::IfrtRestoreVariableOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::IfrtRestoreVariableOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto new_op = rewriter.create<tf_mlrt::IfrtRestoreVariableOp>(
        op.getLoc(), adaptor.getOperands()[0], adaptor.getOperands()[1],
        adaptor.getOperands()[2],
        adaptor.getOperands().slice(3, adaptor.getOperands().size() - 3),
        op.getRestoredDtypes(), op.getTruncateInCast());
    rewriter.replaceOp(op, new_op);

    return mlir::success();
  }
};

std::optional<std::string> DecodeLongName(mlir::Location loc) {
  if (auto name_loc = mlir::dyn_cast<mlir::NameLoc>(loc)) {
    return name_loc.getName().str();
  }

  if (auto fused_loc = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    std::string fused_name;
    for (auto l : fused_loc.getLocations()) {
      if (auto n = DecodeLongName(l)) {
        fused_name += *n;
      }
    }
    return fused_name;
  }

  return std::nullopt;
}

std::string GetNodeName(mlir::Operation *op) {
  auto name = [&]() -> std::string {
    if (auto name = DecodeLongName(op->getLoc())) {
      return *std::move(name);
    }

    return op->getName().stripDialect().str();
  }();

  for (char &c : name) {
    if (c == ':') c = '/';
  }
  return name;
}

void CanonicalizeFunctionNameInNodeDef(const mlir::SymbolTable &symbol_table,
                                       NodeDef &node_def) {
  for (auto &p : *node_def.mutable_attr()) {
    if (p.second.has_func()) {
      auto *func = p.second.mutable_func();
      if (auto n = CanonicalizeTensorflowFunctionName(
              symbol_table, func->name(),
              /*use_mlir_func_name=*/false)) {
        func->set_name(*n);
      }
    }

    if (p.second.has_list() && p.second.list().func_size() > 0) {
      for (auto &func : *p.second.mutable_list()->mutable_func()) {
        if (auto n = CanonicalizeTensorflowFunctionName(
                symbol_table, func.name(),
                /*use_mlir_func_name=*/false)) {
          func.set_name(*n);
        }
      }
    }
  }
}

class ExecuteOpConversion final : public mlir::ConversionPattern {
 public:
  ExecuteOpConversion(mlir::MLIRContext *context,
                      const mlir::SymbolTable *symbol_table,
                      mlir::TypeConverter *type_converter,
                      ExecuteOpRegistry *execute_op_registry,
                      tfrt_stub::OpKernelRunnerCache *op_kernel_cache,
                      const tfrt_stub::FallbackState *fallback_state)
      : mlir::ConversionPattern(*type_converter,
                                mlir::Pattern::MatchAnyOpTypeTag(),
                                /*benefit=*/1, context),
        symbol_table_(*symbol_table),
        execute_op_registry_(*execute_op_registry),
        op_kernel_cache_(*op_kernel_cache),
        fallback_state_(*fallback_state) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter) const override {
    // TODO(b/173017701): Avoid fallback for ops within XLA GPU clusters.
    if (!UseFallback(op)) return mlir::failure();

    if (auto const_op = llvm::dyn_cast<mlir::TF::ConstOp>(op)) {
      tensorflow::TensorProto tensor_proto;
      auto status = ConvertToTensorProto(const_op.getValue(), &tensor_proto);
      if (!status.ok())
        return const_op.emitError(absl::StatusMessageAsCStr(status));

      rewriter.replaceOpWithNewOp<tf_mlrt::ConstOp>(
          op, rewriter.getType<tf_mlrt::TFTensorType>(),
          tensor_proto.SerializeAsString());
      return mlir::success();
    }

    // The assign_op_key pass should have ran.
    if (!op->hasAttr(tensorflow::tfrt_compiler::kOpKeyAttrName))
      return op->emitError("does not have op_key defined");

    std::string node_name = GetNodeName(op);

    uint32_t execute_key = op->getAttrOfType<mlir::IntegerAttr>(
                                 tensorflow::tfrt_compiler::kOpKeyAttrName)
                               .getInt();

    absl::StrAppend(&node_name, "_", execute_key);

    auto statusor_node_def = tensorflow::ConvertTFDialectOpToNodeDef(
        op, node_name, /*ignore_unregistered_attrs=*/false);
    if (!statusor_node_def.ok())
      return op->emitWarning("failed to export NodeDef.");
    auto &node_def = **statusor_node_def;

    CanonicalizeFunctionNameInNodeDef(symbol_table_, node_def);

    std::string node_def_text;
    google::protobuf::TextFormat::PrintToString(node_def, &node_def_text);

    mlir::Value device;
    if (auto custom_device =
            op->getAttrOfType<mlir::StringAttr>(kTfMlrtCustomDevice)) {
      device =
          CreateCustomDevice(op->getLoc(), custom_device.getValue(), rewriter);
      if (!device) return op->emitWarning("Failed to create custom device.");
    }

    mlir::Operation *new_op = nullptr;

    auto create_async_execute_ops = [&]() -> mlir::LogicalResult {
      llvm::SmallVector<mlir::Type, 4> result_types(
          op->getNumResults(), rewriter.getType<mlrt::compiler::FutureType>());
      if (device) {
        new_op = rewriter.replaceOpWithNewOp<tf_mlrt::AsyncExecuteOpWithDevice>(
            op, result_types, device, operands, node_def_text, execute_key);
      } else {
        new_op = rewriter.replaceOpWithNewOp<tf_mlrt::AsyncExecuteOp>(
            op, result_types, operands, node_def_text, execute_key);
      }
      if (mlir::failed(
              execute_op_registry_.RegisterExecuteOp(new_op, execute_key))) {
        return op->emitWarning("Fail to register async op");
      }
      return mlir::success();
    };

    // TODO(b/300999257): check whether to clean up for AoT mockGpu case later

    if (node_def.op() == kXlaLaunchOp) {
      // XlaLaunch Op an AsyncOpKernel, we lower it to tf_mlrt.async_executeop,
      // which return !mlrt.futures. These results will be converted as
      // necessary through the target materialization hook in the type
      // converter.
      if (mlir::failed(create_async_execute_ops())) {
        return mlir::failure();
      }
    } else {
      auto op_kernel_runner = op_kernel_cache_.GetOrCreate(
          tfrt::Location(nullptr, execute_key), node_def.op(),
          node_def.device(), op->getNumOperands(),
          [&](tensorflow::AttrValueMap *attr_value_map) {
            *attr_value_map = node_def.attr();
            return absl::OkStatus();
          },
          fallback_state_.device_manager(),
          fallback_state_.process_function_library_runtime());
      // TODO(290630314): Use LOG_IF when absl logging is available
      if (!op_kernel_runner.ok()) {
        std::cerr << op_kernel_runner.status() << "\n";
      }

      if (op_kernel_runner.ok() && (*op_kernel_runner)->IsAsync()) {
        // If it is an AsyncOpKernel, we lower it to tf_mlrt.async_executeop,
        // which return !mlrt.futures. These results will be converted as
        // necessary through the target materialization hook in the type
        // converter.
        if (mlir::failed(create_async_execute_ops())) {
          return mlir::failure();
        }
      } else {
        // Otherwise, lower to tf_mlrt.executeop.
        llvm::SmallVector<mlir::Type, 4> result_types(
            op->getNumResults(), rewriter.getType<tf_mlrt::TFTensorType>());
        if (device) {
          new_op = rewriter.replaceOpWithNewOp<tf_mlrt::ExecuteOpWithDevice>(
              op, result_types, device, operands, node_def_text, execute_key);
        } else {
          new_op = rewriter.replaceOpWithNewOp<tf_mlrt::ExecuteOp>(
              op, result_types, operands, node_def_text, execute_key);
        }

        if (op_kernel_runner.ok()) {
          // Only register this executeop if its opkernel can be created.
          // Otherwise, it is an unused op so we don't need to create them at
          // runtime.
          if (mlir::failed(execute_op_registry_.RegisterExecuteOp(
                  new_op, execute_key))) {
            return op->emitWarning("Fail to register sync op");
          }
        }
      }
    }

    return mlir::success();
  }

 private:
  const mlir::SymbolTable &symbol_table_;
  ExecuteOpRegistry &execute_op_registry_;
  tfrt_stub::OpKernelRunnerCache &op_kernel_cache_;
  const tfrt_stub::FallbackState &fallback_state_;
};

mlir::Value GetPredicate(mlir::Operation *op, mlir::Value cond_operand,
                         mlir::ConversionPatternRewriter &rewriter) {
  return rewriter.create<tf_mlrt::PredicateOp>(
      op->getLoc(), rewriter.getI1Type(), cond_operand);
}

class CondOpConversion : public mlir::OpConversionPattern<mlir::TF::IfOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::IfOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::FlatSymbolRefAttr then_branch = op.getThenBranchAttr();
    mlir::FlatSymbolRefAttr else_branch = op.getElseBranchAttr();

    llvm::SmallVector<mlir::Type, 4> result_types(
        op.getNumResults(), rewriter.getType<tf_mlrt::TFTensorType>());

    auto bool_cond = GetPredicate(op, adaptor.getCond(), rewriter);

    auto new_op = rewriter.create<mlrt::compiler::CondOp>(
        op.getLoc(), result_types, bool_cond, adaptor.getInput(), then_branch,
        else_branch);

    rewriter.replaceOp(op, new_op.getResults());

    return mlir::success();
  }
};

// Convert TF WhileOp to mlrt.while.
// The pseudo code of mlrt.while is as follows:
//
//  while(cond) {
//    outputs, cond = body(inputs)
//    inputs = outputs
//  }
//  return outputs, cond
//
// So we need to insert extra conversion kernels and merge functions when
// lowering tf.While to mlrt.while.
//
//  %result = tf.While(%arg) {cond = @original_cond_fn, body =
//  @original_body_fn}
//
// is converted to
//
//  func @new_pred_fn(%arg) {
//    %cond_tensor = func.call @original_cond_fn(%arg)
//    %cond_bool = mlrt.predicate %cond_tensor
//    return %cond_bool
//  }
//
//  func @new_while_body(%arg) {
//    %result = func.call @original_body_fn(%arg)
//    %cond_bool = func.call @new_pred_fn(%result)
//    return%result, %cond_bool
//  }
//
//  %first_iter_cond = func.call @new_pred_fn(%arg)
//  %result = mlrt.while %first_iter_cond @new_while_body(%arg)
//
class WhileOpConversion : public mlir::OpConversionPattern<mlir::TF::WhileOp> {
 public:
  WhileOpConversion(mlir::MLIRContext *context,
                    mlir::TypeConverter *type_converter,
                    mlir::SymbolTable *symbol_table)
      : mlir::OpConversionPattern<mlir::TF::WhileOp>(*type_converter, context),
        symbol_table_(*symbol_table) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::WhileOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::FlatSymbolRefAttr cond_fn = op.getCondAttr();
    mlir::FlatSymbolRefAttr body_fn = op.getBodyAttr();

    // Create the predicate function that calls the original cond function and
    // in addition convert the result to a boolean value.
    mlir::func::FuncOp pred_fn = GetPredicateFunction(
        op, cond_fn, adaptor.getOperands().getTypes(), rewriter);
    if (!pred_fn) return mlir::failure();

    // Insert a call op to call the pred function for the first iteration.
    auto call_pred_fn = rewriter.create<mlir::func::CallOp>(
        op.getLoc(), pred_fn.getFunctionType().getResults(),
        pred_fn.getSymName(), adaptor.getOperands());

    if (!call_pred_fn) return mlir::failure();

    // Create the new while body function.
    mlir::func::FuncOp new_body_fn = GetWhileBodyFunction(
        op, body_fn, pred_fn, adaptor.getOperands().getTypes(), rewriter);

    // mlrt.while returns one more additional boolean value than tf.while.
    llvm::SmallVector<mlir::Type, 4> while_result_types(
        adaptor.getOperands().getTypes().begin(),
        adaptor.getOperands().getTypes().end());  // = while_arg_types;
    while_result_types.push_back(rewriter.getI1Type());
    auto new_op = rewriter.create<mlrt::compiler::WhileOp>(
        op.getLoc(), while_result_types, call_pred_fn.getResult(0),
        adaptor.getOperands(), new_body_fn.getSymName());

    rewriter.replaceOp(op, new_op.getResults().drop_back());

    return mlir::success();
  }

 private:
  mlir::func::FuncOp GetPredicateFunction(
      mlir::TF::WhileOp op, mlir::FlatSymbolRefAttr cond_fn,
      mlir::TypeRange arg_types,
      mlir::ConversionPatternRewriter &rewriter) const;

  mlir::func::FuncOp GetWhileBodyFunction(
      mlir::TF::WhileOp op, mlir::FlatSymbolRefAttr body_fn,
      mlir::func::FuncOp pred_fn, mlir::TypeRange arg_types,
      mlir::ConversionPatternRewriter &rewriter) const;

  mlir::SymbolTable &symbol_table_;
};

// Create the pred function that contains a call to the original cond function
// and a predicate kernel that converts the cond tensor to a boolean value. eg.
//
// func @pred_fn( %arg) {
//  %cond_tensor = tf_mlrt.call @original_cond_fn(%arg)
//  %cond_bool = tf_mlrt.predicate %cond_tensor
//  return %cond_bool
// }
//
mlir::func::FuncOp WhileOpConversion::GetPredicateFunction(
    mlir::TF::WhileOp op, mlir::FlatSymbolRefAttr cond_fn,
    mlir::TypeRange arg_types,
    mlir::ConversionPatternRewriter &rewriter) const {
  std::string pred_fn_name =
      absl::StrCat(cond_fn.getValue().str(), "/tf_mlrt_predicate");

  if (auto pred_fn = symbol_table_.lookup<mlir::func::FuncOp>(pred_fn_name)) {
    return pred_fn;
  }

  auto func_op = op->getParentOfType<mlir::func::FuncOp>();

  mlir::ConversionPatternRewriter::InsertionGuard insertion_guard(rewriter);
  rewriter.setInsertionPointAfter(func_op);

  auto func_type = rewriter.getFunctionType(arg_types, {rewriter.getI1Type()});

  auto pred_fn =
      rewriter.create<mlir::func::FuncOp>(op.getLoc(), pred_fn_name, func_type);

  auto *block = pred_fn.addEntryBlock();
  rewriter.setInsertionPointToStart(block);

  auto call_cond_fn = rewriter.create<mlir::func::CallOp>(
      op.getLoc(), arg_types.take_front(), cond_fn, block->getArguments());
  mlir::Value bool_cond = GetPredicate(op, call_cond_fn.getResult(0), rewriter);
  rewriter.create<mlir::func::ReturnOp>(op.getLoc(), bool_cond);

  symbol_table_.insert(pred_fn);

  return pred_fn;
}

// Create the new while body function that contains a call to original while
// body and then a call to the pred function. eg.
//
// func @while_body(%arg) {
//   %result = mlrt.call @original_body(%arg)
//   %cond_bool = mlrt.call @pred_function(%arg)
//   mlrt.return %result, %cond_bool
// }
//
mlir::func::FuncOp WhileOpConversion::GetWhileBodyFunction(
    mlir::TF::WhileOp op, mlir::FlatSymbolRefAttr original_body_fn,
    mlir::func::FuncOp pred_fn, mlir::TypeRange arg_types,
    mlir::ConversionPatternRewriter &rewriter) const {
  std::string body_fn_name =
      absl::StrCat(original_body_fn.getValue().str(), "/tf_mlrt_body");

  if (auto body_fn = symbol_table_.lookup<mlir::func::FuncOp>(body_fn_name)) {
    return body_fn;
  }

  auto func_op = op->getParentOfType<mlir::func::FuncOp>();

  mlir::ConversionPatternRewriter::InsertionGuard insertion_guard(rewriter);
  rewriter.setInsertionPointAfter(func_op);

  llvm::SmallVector<mlir::Type, 4> body_result_types(arg_types.begin(),
                                                     arg_types.end());
  // The last result of the while body function is the boolean condition.
  body_result_types.push_back(rewriter.getI1Type());

  auto func_type = rewriter.getFunctionType(arg_types, body_result_types);
  auto body_fn =
      rewriter.create<mlir::func::FuncOp>(op.getLoc(), body_fn_name, func_type);

  auto *block = body_fn.addEntryBlock();
  rewriter.setInsertionPointToStart(block);

  // Insert a call to the original body function.
  // The returned result type is also the original argument types.
  auto call_original_body_fn = rewriter.create<mlir::func::CallOp>(
      op.getLoc(), arg_types, original_body_fn, block->getArguments());

  // Insert a call to the pred function, which contains a call to the original
  // cond function and the predicate kernel that converts the tensor to boolean
  // value.
  auto call_pred_fn = rewriter.create<mlir::func::CallOp>(
      op.getLoc(), pred_fn.getFunctionType().getResults(), pred_fn.getSymName(),
      call_original_body_fn.getResults());

  llvm::SmallVector<mlir::Value, 4> body_results =
      call_original_body_fn.getResults();

  // The last result should be the boolean value converted from the condition.
  auto bool_cond = call_pred_fn.getResult(0);
  body_results.push_back(bool_cond);

  rewriter.create<mlir::func::ReturnOp>(op.getLoc(), body_results);

  symbol_table_.insert(body_fn);

  return body_fn;
}

class BatchFunctionOpConversion
    : public mlir::OpConversionPattern<mlir::TF::BatchFunctionOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::TF::BatchFunctionOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    std::string node_name = GetNodeName(op);

    auto statusor_node_def = tensorflow::ConvertTFDialectOpToNodeDef(
        op, node_name, /*ignore_unregistered_attrs=*/true);
    if (!statusor_node_def.ok())
      return op->emitWarning("failed to export NodeDef.");
    const auto &node_def = **statusor_node_def;

    std::string node_def_text;
    google::protobuf::TextFormat::PrintToString(node_def, &node_def_text);

    llvm::SmallVector<mlir::Type, 4> result_types(
        op->getNumResults(), rewriter.getType<mlrt::compiler::FutureType>());

    rewriter.replaceOpWithNewOp<tf_mlrt::BatchFunctionOp>(
        op, result_types, adaptor.getOperands(), node_def.device(),
        op.getFAttr(), node_def_text);

    return mlir::success();
  }
};

void CreateFallbackInitializationFunction(
    mlir::ModuleOp module, ExecuteOpRegistry &execute_op_registry) {
  mlir::OpBuilder builder(&module.getBodyRegion());

  auto func_op = builder.create<mlir::func::FuncOp>(
      module.getLoc(), "_tfrt_fallback_init",
      mlir::FunctionType::get(module.getContext(), /*inputs=*/{},
                              /*outputs=*/{}));

  auto *block = func_op.addEntryBlock();
  builder.setInsertionPointToStart(block);

  // Create operations for all fallback kernels in the module.
  for (const auto &[op_index, op] :
       llvm::enumerate(execute_op_registry.GetExecuteOps())) {
    if (op) {
      // There might be unused ops, and we don't need to create them at runtime.
      //
      // TODO(chky, deqiangc): Clean up unused ops before hand.
      builder.create<tf_mlrt::CreateOp>(
          func_op.getLoc(), /*resultTypes=*/mlir::TypeRange{},
          /*operands=*/mlir::ValueRange{}, op->getAttrs());
    }
  }

  builder.create<mlir::func::ReturnOp>(func_op.getLoc());
}

// Move the tf_mlrt.await ops to right before their first uses to avoid
// unnecessary blocking.
void MoveAwaitOpToFirstUse(mlir::Block &block) {
  llvm::SmallVector<tf_mlrt::AwaitOp> await_ops;
  for (auto &op : block) {
    if (auto await_op = llvm::dyn_cast<tf_mlrt::AwaitOp>(&op)) {
      await_ops.push_back(await_op);
    }
  }

  for (auto op : await_ops) {
    auto result = op.getResult();
    if (result.use_empty()) continue;

    mlir::Operation *first_user = *result.user_begin();
    for (auto *user : result.getUsers()) {
      if (user->isBeforeInBlock(first_user)) {
        first_user = user;
      }
    }

    op->moveBefore(first_user);
  }
}

const tfrt_stub::FallbackState &GetDefaultFallbackState() {
  static const auto *const fallback_state = []() {
    tensorflow::SessionOptions session_options;
    tensorflow::FunctionDefLibrary fdef_lib;
    auto fallback_state =
        tfrt_stub::FallbackState::Create(session_options, fdef_lib).value();
    return fallback_state.release();
  }();

  return *fallback_state;
}

// The conversion pass that is run before 'tf-mlrt-parallelization' passes. The
// parallelization pass changes the graph content, so any rewrite/conversion
// that depends on the graph instead of individual ops should be done before
// parallelization.
class TfToMlrtPreParallelizationConversionPass
    : public mlir::PassWrapper<TfToMlrtPreParallelizationConversionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  TfToMlrtPreParallelizationConversionPass() = default;
  explicit TfToMlrtPreParallelizationConversionPass(
      const TfrtPipelineOptions &options) {
    // This is needed to progating user configs into this pass.
    options_.copyOptionValuesFrom(options);
  }
  TfToMlrtPreParallelizationConversionPass(
      const TfToMlrtPreParallelizationConversionPass &other) {}
  TfToMlrtPreParallelizationConversionPass &operator=(
      const TfToMlrtPreParallelizationConversionPass &) = delete;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TfToMlrtPreParallelizationConversionPass)

 private:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlrt::compiler::MlrtDialect>();
    registry.insert<tensorflow::tf_mlrt::TensorflowMlrtDialect>();
    registry.insert<mlir::func::FuncDialect>();

    RegisterTpuDialect(registry);
  }

  llvm::StringRef getArgument() const final {
    return "pre-parallel-tf-to-mlrt";
  }
  llvm::StringRef getDescription() const final {
    return "pre-parallel-tf-to-mlrt";
  }

  mlir::LogicalResult initialize(mlir::MLIRContext *context) override {
    if (use_tpu_host_allocator_for_inputs_.hasValue()) {
      options_.use_tpu_host_allocator_for_inputs =
          use_tpu_host_allocator_for_inputs_;
    }

    return mlir::success();
  }

  mlir::LogicalResult runOnFunction(mlir::func::FuncOp func) {
    auto &context = getContext();
    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&getContext());
    target.addLegalDialect<mlrt::compiler::MlrtDialect,
                           tensorflow::tf_mlrt::TensorflowMlrtDialect,
                           mlir::TF::TensorFlowDialect>();
    PopulateTpuPreParallelizationConversionPatterns(target, patterns, options_);

    return mlir::applyPartialConversion(func, target, std::move(patterns));
  }

  void runOnOperation() override {
    auto module = getOperation();

    for (auto func : module.getOps<mlir::func::FuncOp>()) {
      if (mlir::failed(runOnFunction(func))) {
        signalPassFailure();
        return;
      }
    }
  }

  Option<bool> use_tpu_host_allocator_for_inputs_{
      *this, "use-tpu-host-allocator-for-inputs",
      llvm::cl::desc("If true, fallback executeops that produce inputs to tpu "
                     "program will use tpu host allocator."),
      llvm::cl::init(false)};

  TfrtPipelineOptions options_;
};

class TfToMlrtConversionPass
    : public mlir::PassWrapper<TfToMlrtConversionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  TfToMlrtConversionPass()
      : TfToMlrtConversionPass({}, &GetDefaultFallbackState()) {}
  explicit TfToMlrtConversionPass(
      const TfrtPipelineOptions &options,
      const tfrt_stub::FallbackState *fallback_state)
      : fallback_state_(*fallback_state) {
    // This is needed to progating user configs into this pass.
    options_.copyOptionValuesFrom(options);
  }
  TfToMlrtConversionPass(const TfToMlrtConversionPass &other)
      : fallback_state_(other.fallback_state_) {}
  TfToMlrtConversionPass &operator=(const TfToMlrtConversionPass &) = delete;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TfToMlrtConversionPass)

 private:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlrt::compiler::MlrtDialect>();
    registry.insert<tensorflow::tf_mlrt::TensorflowMlrtDialect>();
    registry.insert<mlir::func::FuncDialect>();

    RegisterTpuDialect(registry);
  }

  llvm::StringRef getArgument() const final { return "tf-to-mlrt"; }
  llvm::StringRef getDescription() const final { return "tf-to-mlrt"; }

  mlir::LogicalResult initialize(mlir::MLIRContext *context) override {
    // TODO(b/285064425): See if this and below are the right way to
    // accommodate other dialects.
    type_converter_.addConversion([](mlir::Type type) { return type; });
    type_converter_.addConversion(
        [=](mlir::TensorType type) -> std::optional<mlir::Type> {
          // Ref types are not supported in both compiler and runtime.
          if (mlir::isa<mlir::TF::TensorFlowRefType>(type.getElementType()))
            return std::nullopt;
          return tf_mlrt::TFTensorType::get(context);
        });

    auto future_to_tensor_materialization =
        [](mlir::OpBuilder &builder, mlir::Type desired_type,
           mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1) return mlir::Value();

      if (mlir::isa<mlrt::compiler::FutureType>(inputs[0].getType())) {
        if (mlir::isa<tf_mlrt::TFTensorType>(desired_type)) {
          return builder.create<tf_mlrt::AwaitOp>(loc, desired_type, inputs[0]);
        }

        return mlir::Value();
      }

      return inputs[0];
    };

    type_converter_.addTargetMaterialization(future_to_tensor_materialization);
    type_converter_.addArgumentMaterialization(
        future_to_tensor_materialization);
    type_converter_.addSourceMaterialization(
        [](mlir::OpBuilder &builder, mlir::Type result_type,
           mlir::ValueRange inputs,
           mlir::Location loc) -> std::optional<mlir::Value> {
          return builder
              .create<mlir::UnrealizedConversionCastOp>(loc, result_type,
                                                        inputs)
              .getResult(0);
        });

    if (use_tpu_host_allocator_for_inputs_.hasValue()) {
      options_.use_tpu_host_allocator_for_inputs =
          use_tpu_host_allocator_for_inputs_;
    }

    return mlir::success();
  }

  void runOnOperation() override {
    auto module = getOperation();
    mlir::SymbolTable symbol_table(module);

    // Use llvm::make_early_inc_range instead of the stock range from
    // module.getOps because conversions such as WhileOpConversion could insert
    // new functions into the module ops list causing the stock range to not
    // able to find next OP correctly.
    for (auto func :
         llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
      if (mlir::failed(runOnFunction(func, symbol_table))) {
        signalPassFailure();
        return;
      }
    }

    // Some mlrt kernels such as tf_mlrt_tpu.CompileAndExecute produce futures,
    // but function invoked by mlrt execute op are not aware of these changes.
    // We add a post process to fix up this caller-callee mismatch.
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
      CollectFunctionCallSiteInputTypes(func);
    }
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
      if (mlir::failed(PostProcessFunctionSignature(func, symbol_table))) {
        signalPassFailure();
        return;
      }
      // Move the tf_mlrt.await ops to right before their first uses to avoid
      // unnecessary blocking.
      MoveAwaitOpToFirstUse(func.getBlocks().front());
    }

    CreateFallbackInitializationFunction(module, execute_op_registry_);

    module.walk([&](mlir::UnrealizedConversionCastOp op) {
      op->replaceAllUsesWith(op->getOperands());
      op->erase();
    });

    AddAwaitOpToUnusedFutures(module);
  }

  void AddAwaitOpToUnusedFutures(mlir::ModuleOp module) {
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
      llvm::SmallVector<mlir::Value> unused_futures;

      auto is_unused_future = [](mlir::Value result) {
        return llvm::isa<::mlrt::compiler::FutureType>(result.getType()) &&
               result.use_empty();
      };

      for (auto arg : func.getArguments()) {
        if (is_unused_future(arg)) {
          unused_futures.push_back(arg);
        }
      }

      for (auto &op : func.getBody().front()) {
        for (mlir::Value result : op.getResults()) {
          if (is_unused_future(result)) {
            unused_futures.push_back(result);
          }
        }
      }

      if (!unused_futures.empty()) {
        auto builder =
            mlir::OpBuilder::atBlockTerminator(&func.getBody().front());
        builder.create<::mlrt::compiler::AwaitAllControlOp>(func.getLoc(),
                                                            unused_futures);
      }
    }
  }

  mlir::LogicalResult PostProcessFunctionSignature(
      mlir::func::FuncOp func, mlir::SymbolTable &symbol_table) {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [this](mlir::func::FuncOp func) {
          // By default, we assume callers are well behaved.
          if (function_call_site_input_types_.find(func.getName()) ==
              function_call_site_input_types_.end()) {
            return true;
          }
          DCHECK_EQ(function_call_site_input_types_.at(func.getName()).size(),
                    func.getFunctionType().getInputs().size());

          for (auto [expected_input_type, call_site_type] :
               llvm::zip(func.getFunctionType().getInputs(),
                         function_call_site_input_types_.at(func.getName()))) {
            if (expected_input_type != call_site_type) {
              return false;
            }
          }
          return true;
        });

    patterns.add<FuncOpSignatureConversion>(&getContext(), &type_converter_,
                                            &function_call_site_input_types_);

    return mlir::applyPartialConversion(func, target, std::move(patterns));
  }

  void CollectFunctionCallSiteInputTypes(mlir::func::FuncOp func) {
    func.walk([&function_call_site_input_types =
                   function_call_site_input_types_](
                  mlir::Operation *op) mutable {
      // Only collect the call-site input types when a function is invoked
      // by async op. This is the only known case that the previous pass
      // may left un-match types between call-site and callee.
      if (auto async_op = llvm::dyn_cast<mlrt::compiler::AsyncOp>(op)) {
        function_call_site_input_types[async_op.getCallee()
                                           .getLeafReference()] =
            llvm::SmallVector<mlir::Type>(async_op.getOperandTypes().begin(),
                                          async_op.getOperandTypes().end());
      }
    });
  }

  mlir::LogicalResult runOnFunction(mlir::func::FuncOp func,
                                    mlir::SymbolTable &symbol_table) {
    auto &context = getContext();
    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&getContext());
    target.addLegalDialect<mlrt::compiler::MlrtDialect,
                           tensorflow::tf_mlrt::TensorflowMlrtDialect>();
    target.addIllegalDialect<mlir::TF::TensorFlowDialect>();

    target.addIllegalOp<tf_mlrt::TFAsyncWhileOp>();
    target.addIllegalOp<tf_mlrt::TFIfrtLoadVariableOp>();
    target.addIllegalOp<tf_mlrt::TFAwaitOp>();
    target.addIllegalOp<tf_mlrt::TFPromiseOp>();
    target.addIllegalOp<tf_mlrt::TFMapFnOp>();

    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [this](mlir::func::FuncOp op) {
          return type_converter_.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [this](mlir::func::ReturnOp op) {
          for (auto operand : op.getOperands()) {
            if (!type_converter_.isLegal(operand.getType())) return false;
          }
          return true;
        });
    target.addDynamicallyLegalOp<mlrt::compiler::AsyncOp>(
        [this](mlrt::compiler::AsyncOp op) {
          for (auto operand : op.getOperands()) {
            if (!type_converter_.isLegal(operand.getType())) return false;
          }
          return true;
        });
    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [this](mlir::func::CallOp op) {
          for (auto operand : op.getOperands()) {
            if (!type_converter_.isLegal(operand.getType())) {
              return false;
            }
          }
          return true;
        });

    // LINT.IfChange
    // Order the list of added ops alphabetically.
    patterns.add<WhileOpConversion>(&context, &type_converter_, &symbol_table);
    patterns.add<AsyncOpConversion, GetResourceOpConversion,
                 SetResourceOpConversion, IfrtRestoreVariableOpConversion,
                 TFAwaitOpConversion, TFPromiseOpConversion>(&context);
    patterns.add<BatchFunctionOpConversion, CaseOpConversion, CondOpConversion,
                 TFAsyncWhileOpConversion, TFMapFnOpConversion>(type_converter_,
                                                                &context);
    patterns.add<ExecuteOpConversion>(&context, &symbol_table, &type_converter_,
                                      &execute_op_registry_, &op_kernel_cache_,
                                      &fallback_state_);
    patterns.add<TFIfrtLoadVariableOpConversion,
                 TFCallOpConversion<mlir::TF::PartitionedCallOp>,
                 TFCallOpConversion<mlir::TF::StatefulPartitionedCallOp>,
                 TFCallOpConversion<mlir::TF::LegacyCallOp>>(&context,
                                                             &type_converter_);
    // LINT.ThenChange(util.cc)

    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, type_converter_);
    mlir::populateReturnOpTypeConversionPattern(patterns, type_converter_);

    PopulateTpuConversionPatterns(target, patterns, type_converter_,
                                  execute_op_registry_, options_);

    return mlir::applyPartialConversion(func, target, std::move(patterns));
  }

  Option<bool> use_tpu_host_allocator_for_inputs_{
      *this, "use-tpu-host-allocator-for-inputs",
      llvm::cl::desc("If true, fallback executeops that produce inputs to tpu "
                     "program will use tpu host allocator."),
      llvm::cl::init(false)};

  TfrtPipelineOptions options_;
  mlir::TypeConverter type_converter_;
  ExecuteOpRegistry execute_op_registry_;
  tfrt_stub::OpKernelRunnerCache op_kernel_cache_;
  const tfrt_stub::FallbackState &fallback_state_;

  // True input argument types for a given function at call site.
  llvm::DenseMap<llvm::StringRef, llvm::SmallVector<mlir::Type>>
      function_call_site_input_types_;
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTfToMlrtPreParallelizationConversionPass(
    const TfrtPipelineOptions &options) {
  return std::make_unique<TfToMlrtPreParallelizationConversionPass>(options);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTfToMlrtConversionPass(const TfrtPipelineOptions &options,
                             const tfrt_stub::FallbackState *fallback_state) {
  return std::make_unique<TfToMlrtConversionPass>(options, fallback_state);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTfToMlrtConversionPass(const TfrtPipelineOptions &options) {
  return CreateTfToMlrtConversionPass(options, &GetDefaultFallbackState());
}

}  // namespace mlrt_compiler
}  // namespace tensorflow
