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

#include "tensorflow/compiler/mlir/tensorflow/utils/eval_util.h"

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

using gtl::MakeCleanup;

#define RETURN_FAILURE_IF_ERROR(expr) \
  if (!IsOk(expr)) {                  \
    return mlir::failure();           \
  }

static bool IsOk(const TF_Status* s) {
  if (TF_GetCode(s) == TF_OK) return true;
  VLOG(2) << TF_Message(s);
  return false;
}

static bool IsOk(const Status& s) {
  if (s.ok()) return true;
  VLOG(2) << s.error_message();
  return false;
}

// Update node_def's device attribute (if any) to use a local device, that is
// /job:localhost/replica:0/task:0/{DEVICE_TYPE}:{DEVICE_ID}.
// This is because EvaluateOperation only has access to local devices but the
// given node may carry a device assignment to a remote device. In that case,
// evaluation would fail even if we have a device of same type locally. By
// altering device assignment to a local one, we could successfully evaluate in
// that case.
void ForceUseLocalhostDevice(NodeDef* node_def) {
  DeviceNameUtils::ParsedName parsed_name;

  if (!DeviceNameUtils::ParseFullName(node_def->device(), &parsed_name)) return;

  if (parsed_name.has_job) parsed_name.job = "localhost";
  if (parsed_name.has_replica) parsed_name.replica = 0;
  if (parsed_name.has_task) parsed_name.task = 0;

  *node_def->mutable_device() =
      DeviceNameUtils::ParsedNameToString(parsed_name);
}

mlir::LogicalResult EvaluateOperation(
    mlir::Operation* inst, llvm::ArrayRef<mlir::ElementsAttr> operands,
    TFE_Context* context, llvm::SmallVectorImpl<mlir::Attribute>* results) {
  if (!context) {
    VLOG(1) << "Can't evaluate with null context.";
    return mlir::failure();
  }
  // If any operand is nullptr returns true for a failure.
  // TODO(b/120678030): remove this constraint if we find operators can be
  // evaluated with some unknown operands.
  if (std::any_of(operands.begin(), operands.end(),
                  [](mlir::Attribute operand) { return !operand; })) {
    VLOG(1) << "Can't evaluate since not all operands are constant.";
    return mlir::failure();
  }

  TF_Status* status = TF_NewStatus();
  auto clean_status = MakeCleanup([status] { TF_DeleteStatus(status); });

  // Builds TF operation and sets all the attributes.
  std::string node_name = "unnamed";
  if (auto attr = inst->getAttrOfType<mlir::StringAttr>("name")) {
    node_name = attr.getValue();
  }
  auto node_def_or = ConvertTFDialectOpToNodeDef(
      inst, node_name.c_str(), /*ignore_unregistered_attrs=*/true);
  RETURN_FAILURE_IF_ERROR(node_def_or.status());
  const auto& node_def = node_def_or.ValueOrDie();

  ForceUseLocalhostDevice(node_def.get());

  TFE_Op* op = TFE_NewOp(context, node_def->op().c_str(), status);
  RETURN_FAILURE_IF_ERROR(status);
  auto clean_op = MakeCleanup([op] { TFE_DeleteOp(op); });
  TFE_OpSetDevice(op, node_def->device().c_str(), status);
  RETURN_FAILURE_IF_ERROR(status);
  for (const auto& attr : node_def->attr()) {
    SetOpAttrValueScalar(context, op, attr.second, attr.first.c_str(), status);
    RETURN_FAILURE_IF_ERROR(status);
  }

  VLOG(1) << "Start to evaluate node: " << node_def->DebugString();

  // Adds inputs to the TF operation.
  for (const auto operand : operands) {
    Tensor tensor;
    RETURN_FAILURE_IF_ERROR(ConvertToTensor(operand, &tensor));
    TF_Tensor* tf_tensor = TF_TensorFromTensor(tensor, status);
    RETURN_FAILURE_IF_ERROR(status);
    auto clean_tensor =
        MakeCleanup([tf_tensor] { TF_DeleteTensor(tf_tensor); });
    TFE_TensorHandle* input_handle = TFE_NewTensorHandle(tf_tensor, status);
    RETURN_FAILURE_IF_ERROR(status);
    auto clean_input_handle =
        MakeCleanup([input_handle] { TFE_DeleteTensorHandle(input_handle); });
    TFE_OpAddInput(op, input_handle, status);
    RETURN_FAILURE_IF_ERROR(status);
  }

  // Executes the TF operation.
  int num_outputs = inst->getNumResults();
  absl::InlinedVector<TFE_TensorHandle*, 2> outputs(num_outputs);
  TFE_Execute(op, outputs.data(), &num_outputs, status);
  RETURN_FAILURE_IF_ERROR(status);
  auto clean_outputs = MakeCleanup([&outputs] {
    for (TFE_TensorHandle* tensor_handle : outputs) {
      TFE_DeleteTensorHandle(tensor_handle);
    }
  });

  // Converts the outputs to MLIR attributes.
  mlir::Builder builder(inst->getContext());
  for (TFE_TensorHandle* tensor_handle : outputs) {
    TF_Tensor* tf_tensor = TFE_TensorHandleResolve(tensor_handle, status);
    RETURN_FAILURE_IF_ERROR(status);
    auto clean_tensor =
        MakeCleanup([tf_tensor] { TF_DeleteTensor(tf_tensor); });
    Tensor tensor;
    RETURN_FAILURE_IF_ERROR(TF_TensorToTensor(tf_tensor, &tensor));
    auto attr_or = ConvertTensor(tensor, &builder);
    RETURN_FAILURE_IF_ERROR(attr_or.status());
    results->push_back(attr_or.ValueOrDie());
  }

  VLOG(1) << "Evaluate node " << node_name << " successfully!";

  return mlir::success();
}

#undef RETURN_FAILURE_IF_ERROR
}  // namespace tensorflow
