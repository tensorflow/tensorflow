/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/transforms/constant_fold.h"

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"

namespace mlir {
namespace TF {

static bool IsOk(const tensorflow::Status& s) {
  if (s.ok()) return true;
  VLOG(2) << s.error_message();
  return false;
}

#define RETURN_FAILURE_IF_ERROR(expr) \
  if (!IsOk(expr)) {                  \
    return mlir::failure();           \
  }

// Implements a TF specific policy on when constant folding is allowed.
// Policy:
//
// Disable constant folding if operands size is greater than a certain
// threshold (`kOperandsSizeThreshold`).
//
// Otherwise, allow folding if we do not know the shape of an operand or
// result i.e., one of these values has non-static shape. If we know all the
// shapes, find the total size of the operands and results. Folding of the op is
// allowed if one of the following conditions are met:
// 1. size of results is less than a certain threshold
// (`kResultsSizeThreshold`), or
// 2. size of results is within a factor (`kSizeFactor`) of size of operands, or
// TODO(b/157226221): Look into other heuristics for constant fold policy.
static bool ShouldBeFolded(Operation* inst) {
  bool has_unknown_shape = false;
  auto get_size = [&](TypeRange types) {
    int64_t size = 0;
    for (auto t : types) {
      auto tensor_type = t.cast<TensorType>();
      // Ignore types with undefined bit widths.
      if (!tensor_type.getElementType().isIntOrFloat()) continue;
      if (!tensor_type.hasStaticShape()) {
        has_unknown_shape = true;
        return size;
      }
      size += tensor_type.getNumElements() *
              tensor_type.getElementType().getIntOrFloatBitWidth();
    }
    return size;
  };

  int64_t results_size = get_size(inst->getResultTypes());
  int64_t operands_size = get_size(inst->getOperandTypes());

  constexpr int kSizeFactor = 2;
// TODO(b/233827625): Remove TF_DISABLE_CONSTANT_FOLDING macro.
#ifdef TF_DISABLE_CONSTANT_FOLDING
  constexpr int64_t kResultsSizeThreshold = 0;
#else
  constexpr int64_t kResultsSizeThreshold = (1 << 23);  // 1 MB
#endif
  constexpr int64_t kOperandsSizeThreshold = (1 << 30);  // 128 MB

  return (operands_size <= kOperandsSizeThreshold) &&
         (has_unknown_shape || (results_size <= kResultsSizeThreshold) ||
          (results_size <= kSizeFactor * operands_size));
}

static const tensorflow::tfrt_stub::FallbackState& GetDefaultFallbackState() {
  static const auto* const fallback_state = []() {
    tensorflow::SessionOptions session_options;
    tensorflow::FunctionDefLibrary fdef_lib;
    auto fallback_state =
        tensorflow::tfrt_stub::FallbackState::CreateWithCpuDevice(
            session_options, fdef_lib)
            .value();
    return fallback_state.release();
  }();

  return *fallback_state;
}

static std::function<void(std::function<void()>)>* GetDefaultRunner() {
  static auto* const default_runner =
      new std::function<void(std::function<void()>)>(
          [](const std::function<void()>& f) { f(); });
  return default_runner;
}

static mlir::LogicalResult EvaluateOperation(
    mlir::Operation* inst, llvm::ArrayRef<mlir::ElementsAttr> operands,
    llvm::SmallVectorImpl<mlir::Attribute>* results) {
  // If any operand is nullptr returns true for a failure.
  // TODO(b/120678030): remove this constraint if we find operators can be
  // evaluated with some unknown operands.
  if (std::any_of(operands.begin(), operands.end(),
                  [](mlir::Attribute operand) { return !operand; })) {
    VLOG(1) << "Can't evaluate since not all operands are constant.";
    return mlir::failure();
  }

  // Builds TF operation and sets all the attributes.
  std::string node_name = "unnamed";
  if (auto attr = inst->getAttrOfType<mlir::StringAttr>("name")) {
    node_name = std::string(attr.getValue());
  }
  auto node_def_or = tensorflow::ConvertTFDialectOpToNodeDef(
      inst, node_name.c_str(), /*ignore_unregistered_attrs=*/true);
  RETURN_FAILURE_IF_ERROR(node_def_or.status());
  const auto& node_def = node_def_or.value();

  const auto& fallback_state = GetDefaultFallbackState();

  // Explicitly set device to Host CPU instead of the device present in device
  // attribute of the MLIR op. The assigned device might be remote, not
  // available during compilation or compilation only device for on demand
  // execution which may create a recursion if used for constant folding.
  constexpr char kHostCpu[] = "/job:localhost/replica:0/task:0/CPU:0";

  auto statusor_runner = tensorflow::tfrt_stub::OpKernelRunner::Create(
      node_def->op(), node_def->name(), kHostCpu, operands.size(),
      [&](tensorflow::AttrValueMap* attr_value_map) {
        *attr_value_map = node_def->attr();
        return tensorflow::OkStatus();
      },
      fallback_state.device_manager(),
      fallback_state.process_function_library_runtime());
  RETURN_FAILURE_IF_ERROR(statusor_runner.status());
  const auto& runner = *statusor_runner;

  VLOG(1) << "Start to evaluate node: " << node_def->DebugString();

  std::vector<tensorflow::Tensor> inputs;

  // Adds inputs to the TF operation.
  for (const auto operand : operands) {
    tensorflow::Tensor tensor;
    RETURN_FAILURE_IF_ERROR(tensorflow::ConvertToTensor(operand, &tensor));
    inputs.push_back(std::move(tensor));
  }

  std::vector<tensorflow::TensorValue> input_values;
  for (auto& tensor : inputs) {
    input_values.emplace_back();
    input_values.back().tensor = &tensor;
  }

  tensorflow::OpKernelContext::Params params;
  params.inputs = input_values;
  params.device = runner.device();
  params.op_kernel = runner.op_kernel();
  // Still use original device's resource_manager.
  params.resource_manager = runner.resource_manager();
  params.input_alloc_attrs = runner.input_alloc_attrs();
  params.output_attr_array = runner.output_alloc_attrs().data();
  // Following two parameters are used to support executing tf.data via
  // fallback.
  params.function_library = runner.function_library_runtime();
  params.runner = GetDefaultRunner();

  // Executes the TF operation.
  tensorflow::OpKernelContext op_kernel_context(&params);
  runner.Run(&op_kernel_context);
  RETURN_FAILURE_IF_ERROR(op_kernel_context.status());

  // Converts the outputs to MLIR attributes.
  mlir::Builder builder(inst->getContext());

  for (int i = 0; i < op_kernel_context.num_outputs(); ++i) {
    DCHECK(op_kernel_context.mutable_output(i));
    auto attr_or = tensorflow::ConvertTensor(
        *op_kernel_context.mutable_output(i), &builder);
    RETURN_FAILURE_IF_ERROR(attr_or.status());
    results->push_back(attr_or.value());
  }

  VLOG(1) << "Evaluate node " << node_name << " successfully!";

  return mlir::success();
}

LogicalResult ConstantFoldFallbackHook(
    Operation* inst, ArrayRef<Attribute> operands,
    SmallVectorImpl<OpFoldResult>& results) {  // NOLINT
  // Instructions with side effects should not be constant folded to preserve
  // the original semantics. Ops that have no side effect and zero results but
  // could be folded should have a custom folder instead of relying on the
  // TensorFlow folding hook.
  if (inst->getNumResults() == 0 ||
      inst->hasTrait<OpTrait::TF::NoConstantFold>() ||
      inst->getNumRegions() != 0 || !isMemoryEffectFree(inst))
    return failure();

  // If any of the result types are variants, don't try to constant fold them.
  // This creates opaque variant constants which lose information and would
  // require "raising" later.
  for (auto type : inst->getResultTypes()) {
    if (auto tensor_type = type.dyn_cast<TensorType>()) {
      if (tensor_type.getElementType().isa<VariantType>()) {
        return failure();
      }
    }
  }

  // If all the results are empty and has numerical element types, set results
  // to empty elements attribute. This is restricted to the numerical element
  // types as the DenseElementsAttr only supports numerical and string types.
  // TODO(hinsu): Handle ops that have one of the results empty for constant
  // propagation.
  bool has_empty_numerical_results =
      llvm::all_of(inst->getResultTypes(), [](Type ty) {
        ShapedType shaped_ty = ty.cast<ShapedType>();
        Type element_ty = shaped_ty.getElementType();
        return shaped_ty.hasStaticShape() && shaped_ty.getNumElements() == 0 &&
               element_ty.isIntOrFloat();
      });
  if (has_empty_numerical_results &&
      // TODO(jpienaar): Remove this once some unmodeled op behavior is
      // addressed.
      inst->isRegistered()) {
    for (Type ty : inst->getResultTypes()) {
      auto shaped_ty = ty.cast<ShapedType>();
      results.push_back(
          DenseElementsAttr::get(shaped_ty, llvm::ArrayRef<Attribute>()));
    }
    return success();
  }

  // Do not execute function calls.
  if (llvm::isa<TF::WhileOp, TF::CaseOp, TF::IfOp, CallOpInterface>(inst)) {
    return failure();
  }

  // Determine if we should attempt to fold this operation by considering the
  // size/size increase due to folding.
  if (!ShouldBeFolded(inst)) return failure();

  // Returns directly if any of the operands is not an elements attributes.
  if (std::any_of(operands.begin(), operands.end(), [](Attribute attr) {
        return !attr || !attr.isa<ElementsAttr>();
      }))
    return failure();

  SmallVector<ElementsAttr, 4> inputs;
  inputs.reserve(operands.size());
  for (auto input : operands) {
    inputs.push_back(input.cast<ElementsAttr>());
  }

  // Avoid overlapping folds with the same context.
  // TODO(jpienaar): Avoid using global context & mutex here.
  static auto* mu = new tensorflow::mutex();
  tensorflow::mutex_lock l(*mu);
  SmallVector<Attribute, 8> constants;
  LogicalResult status = EvaluateOperation(inst, inputs, &constants);
  results.assign(constants.begin(), constants.end());
  return status;
}

static bool init_hooks = ([] () {
  TensorFlowDialect::RegisterConstantFoldHook(ConstantFoldFallbackHook);
}(), true);

}  // namespace TF
}  // namespace mlir
