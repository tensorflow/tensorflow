/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/transforms/constant_fold_utils.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/translate_utils.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"

namespace mlir {
namespace TF {

using tensorflow::tfrt_stub::FallbackState;
using tensorflow::tfrt_stub::OpKernelRunner;

static bool IsOk(const absl::Status& s) {
  if (s.ok()) return true;
  VLOG(2) << s.message();
  return false;
}

#define RETURN_FAILURE_IF_ERROR(expr) \
  if (!IsOk(expr)) {                  \
    return mlir::failure();           \
  }

bool CanBeFolded(Operation* inst) {
  // Instructions with side effects should not be constant folded to preserve
  // the original semantics. Ops that have no side effect and zero results but
  // could be folded should have a custom folder instead of relying on the
  // TensorFlow folding hook.
  if (inst == nullptr || inst->getNumResults() == 0 ||
      inst->hasTrait<OpTrait::TF::NoConstantFold>() ||
      inst->getNumRegions() != 0 || !isMemoryEffectFree(inst)) {
    return false;
  }

  // If any of the result types are variants, don't try to constant fold them.
  // This creates opaque variant constants which lose information and would
  // require "raising" later.
  for (const Type type : inst->getResultTypes()) {
    if (const TensorType tensor_type = mlir::dyn_cast<TensorType>(type)) {
      if (mlir::isa<VariantType>(tensor_type.getElementType())) {
        return false;
      }
    }
  }

  // Operations that execute function calls shouldn't be constant folded.
  if (llvm::isa<TF::WhileOp, TF::CaseOp, TF::IfOp, CallOpInterface>(inst)) {
    return false;
  }

  return true;
}

static const FallbackState& GetDefaultFallbackState() {
  static const auto* const fallback_state = []() {
    tensorflow::SessionOptions session_options;
    tensorflow::FunctionDefLibrary fdef_lib;
    auto fallback_state =
        FallbackState::CreateWithCpuDevice(session_options, fdef_lib).value();
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

LogicalResult EvaluateOperation(Operation* inst,
                                llvm::ArrayRef<ElementsAttr> operands,
                                llvm::SmallVector<Attribute>& results) {
  // If any operand is nullptr returns true for a failure.
  // TODO(b/120678030): remove this constraint if we find operators can be
  // evaluated with some unknown operands.
  if (std::any_of(operands.begin(), operands.end(),
                  [](Attribute operand) { return !operand; })) {
    VLOG(1) << "Can't evaluate since not all operands are constant.";
    return failure();
  }

  // Builds TF operation and sets all the attributes.
  std::string node_name = "unnamed";
  if (const StringAttr attr = inst->getAttrOfType<StringAttr>("name")) {
    node_name = std::string(attr.getValue());
  }
  absl::StatusOr<std::unique_ptr<tensorflow::NodeDef>> node_def =
      tensorflow::ConvertTFDialectOpToNodeDef(
          inst, node_name.c_str(), /*ignore_unregistered_attrs=*/true);
  RETURN_FAILURE_IF_ERROR(node_def.status());

  const FallbackState& fallback_state = GetDefaultFallbackState();

  // Explicitly set device to Host CPU instead of the device present in device
  // attribute of the MLIR op. The assigned device might be remote, not
  // available during compilation or compilation only device for on demand
  // execution which may create a recursion if used for constant folding.
  std::string host_cpu = tensorflow::DeviceNameUtils::FullName(
      /*job=*/"localhost", /*replica=*/0, /*task=*/0, /*type=*/"CPU", /*id=*/0);

  absl::StatusOr<OpKernelRunner> runner = OpKernelRunner::Create(
      node_def->get()->op(), node_def->get()->name(), host_cpu, operands.size(),
      [&](tensorflow::AttrValueMap* attr_value_map) {
        *attr_value_map = node_def->get()->attr();
        return absl::OkStatus();
      },
      fallback_state.device_manager(),
      fallback_state.process_function_library_runtime());
  RETURN_FAILURE_IF_ERROR(runner.status());

  VLOG(1) << "Start to evaluate node: " << node_def->get()->DebugString();

  std::vector<tensorflow::Tensor> inputs;

  // Adds inputs to the TF operation.
  for (const ElementsAttr& operand : operands) {
    tensorflow::Tensor tensor;
    RETURN_FAILURE_IF_ERROR(tensorflow::ConvertToTensor(operand, &tensor));
    inputs.push_back(std::move(tensor));
  }

  std::vector<tensorflow::TensorValue> input_values;
  for (tensorflow::Tensor& tensor : inputs) {
    input_values.emplace_back();
    input_values.back().tensor = &tensor;
  }

  tensorflow::OpKernelContext::Params params;
  params.inputs = input_values;
  params.device = runner->device();
  params.op_kernel = runner->op_kernel();

  // Still use original device's resource_manager.
  params.resource_manager = runner->resource_manager();
  params.input_alloc_attrs = runner->input_alloc_attrs();
  params.output_attr_array = runner->output_alloc_attrs().data();

  // Following two parameters are used to support executing tf.data via
  // fallback.
  params.function_library = runner->function_library_runtime();
  params.runner = GetDefaultRunner();

  // Executes the TF operation.
  tensorflow::OpKernelContext op_kernel_context(&params);
  runner->Run(&op_kernel_context);
  RETURN_FAILURE_IF_ERROR(op_kernel_context.status());

  // Converts the outputs to MLIR attributes.
  Builder builder(inst->getContext());

  for (int i = 0; i < op_kernel_context.num_outputs(); ++i) {
    DCHECK(op_kernel_context.mutable_output(i));
    absl::StatusOr<ElementsAttr> result_attr = tensorflow::ConvertTensor(
        *op_kernel_context.mutable_output(i), &builder);
    RETURN_FAILURE_IF_ERROR(result_attr.status());
    results.push_back(result_attr.value());
  }

  VLOG(1) << "Evaluate node " << node_name << " successfully!";

  return success();
}

}  // namespace TF
}  // namespace mlir
