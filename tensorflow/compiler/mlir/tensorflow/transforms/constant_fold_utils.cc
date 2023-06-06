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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/c/tf_status.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/tsl/platform/mem.h"

namespace mlir {
namespace TF {

using tensorflow::tfrt_stub::FallbackState;
using tensorflow::tfrt_stub::OpKernelRunner;

static bool IsOk(const tensorflow::Status& s) {
  if (s.ok()) return true;
  VLOG(2) << s.message();
  return false;
}

#define RETURN_FAILURE_IF_ERROR(expr) \
  if (!IsOk(expr)) {                  \
    return mlir::failure();           \
  }

TFE_Context* GetContextForConstantFold() {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  // Only initialize single CPU.
  tensorflow::ConfigProto config_proto;
  // This is conceptually equal to what we do in python/eager/context.py but
  // with all GPU/TPU devices ignored and CPU only set to 1.
  (*config_proto.mutable_device_count())["CPU"] = 1;
  config_proto.add_device_filters("/device:CPU:*");
  // Limit the thread pool size. Without this, TF by default creates as many
  // threads as the number of CPUs (`port::MaxParallelism()`). This can be
  // expensive since this TFE context persists the entire program execution.
  config_proto.set_inter_op_parallelism_threads(2);
  std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> config(
      TF_NewBuffer(), TF_DeleteBuffer);
  DCHECK(config->data == nullptr);

  // Copy config_proto into config.
  {
    const size_t proto_size = config_proto.ByteSizeLong();
    void* buf = tsl::port::Malloc(proto_size);
    if (buf == nullptr) {
      LOG(ERROR) << "Failed to allocate memory to serialize ConfigProto "
                    "while creating context options for constant folding";
      return nullptr;
    }
    if (!config_proto.SerializeWithCachedSizesToArray(
            static_cast<uint8_t*>(buf))) {
      tsl::port::Free(buf);
      LOG(ERROR) << "Unable to serialize ConfigProto while creating context "
                    "options for constant folding";
      return nullptr;
    }
    config->data = buf;
    config->length = proto_size;
    config->data_deallocator = [](void* data, size_t length) {
      tsl::port::Free(data);
    };
  }

  TFE_ContextOptionsSetConfig(opts.get(), config->data, config->length,
                              status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    LOG(ERROR) << "Failed to set context options for constant folding: "
               << status.get();
    return nullptr;
  }

  // Input tensors are placed on the host CPU so use the explicit device
  // policy to fail if no CPU kernels are available for the op.
  TFE_ContextOptionsSetDevicePlacementPolicy(opts.get(),
                                             TFE_DEVICE_PLACEMENT_EXPLICIT);
  auto ctx = TFE_NewContext(opts.get(), status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    LOG(ERROR) << "Failed to create context for constant folding: "
               << status.get();
    return nullptr;
  }
  return ctx;
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
    if (const TensorType tensor_type = type.dyn_cast<TensorType>()) {
      if (tensor_type.getElementType().isa<VariantType>()) {
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
        return tensorflow::OkStatus();
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
