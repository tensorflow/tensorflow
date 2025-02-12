/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/placement_utils.h"

#include <variant>

#include "absl/status/status.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/custom_device.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace eager {

// These ops are not pinnable since they generate data. It can be slower to
// generate and then copy the data instead of just generating the data on the
// device directly.
static bool IsPinnableOp(absl::string_view op_name) {
  static const gtl::FlatSet<string>* unpinnable_ops = new gtl::FlatSet<string>({
      "RandomUniform",
      "RandomUniformInt",
      "RandomStandardNormal",
      "StatelessRandomUniform",
      "StatelessRandomUniformInt",
      "StatelessRandomUniformFullInt",
      "StatelessRandomNormal",
  });

  // XRT ops refer to per-device handles that are not safe to move between
  // devices.
  return unpinnable_ops->find(string(op_name)) == unpinnable_ops->end() &&
         !absl::StartsWith(op_name, "XRT");
}
// Validate if the remote device with the given incarnation is valid in the
// remote device manager of the current eager context.
static absl::Status ValidateTensorHandleRemoteDevice(
    EagerContext* ctx, int64_t device_incarnation) {
  if (ctx->remote_device_mgr()->ContainsDevice(device_incarnation)) {
    return absl::OkStatus();
  }
  return errors::InvalidArgument(
      "Resource input tensor contains an invalid device. This might happen "
      "when the client has connected to a different cluster, or some remote "
      "workers have been restarted.");
}

bool IsColocationExempt(absl::string_view op_name) {
  const auto& exempt_ops = InputColocationExemptionRegistry::Global()->Get();
  return exempt_ops.find(string(op_name)) != exempt_ops.end();
}

bool IsFunction(absl::string_view op_name) {
  const OpDef* op_def = nullptr;
  absl::Status s = OpDefForOp(string(op_name), &op_def);
  if (!s.ok()) {
    if (!absl::IsNotFound(s)) {
      LOG(WARNING) << "Looking up OpDef failed with error: " << s;
    }
    // Cannot find OpDef, it is a function.
    return true;
  }
  return false;
}

absl::Status MaybePinSmallOpsToCpu(
    bool* result, absl::string_view op_name,
    absl::Span<ImmediateExecutionTensorHandle* const> args,
    absl::string_view cpu_device_name) {
  if (IsFunction(op_name) || IsColocationExempt(op_name) ||
      !IsPinnableOp(op_name)) {
    *result = false;
    return absl::OkStatus();
  }

  // Ops without inputs are usually ops that generate a tensor in some way and
  // usually require being present on whatever device they are scheduled on
  // - for e.g. VarHandleOp or _Recv).
  if (args.empty()) {
    *result = false;
    return absl::OkStatus();
  }

  int i = 0;
  for (auto* arg : args) {
    absl::Status s;
    const char* device_name = arg->DeviceName(&s);
    DataType dtype = arg->DataType();
    TF_RETURN_IF_ERROR(s);

    DVLOG(2) << "for op " << op_name << " input " << i << " "
             << DataTypeString(dtype) << " input device = " << device_name;

    // Input is on CPU.
    if (device_name != cpu_device_name) {
      *result = false;
      return absl::OkStatus();
    }

    if (dtype != DataType::DT_INT32 && dtype != DataType::DT_INT64) {
      *result = false;
      return absl::OkStatus();
    }

    int64_t num_elements;
    TF_RETURN_IF_ERROR(arg->NumElements(&num_elements));
    if (num_elements > 64) {
      *result = false;
      return absl::OkStatus();
    }
    i++;
  }

  // TODO(nareshmodi): Is it possible there is no int32/int64 CPU kernel for
  // an op, but there is a GPU kernel?
  DVLOG(1) << "Forcing op " << op_name
           << " to be on the CPU since all input tensors have an "
              "int32/int64 dtype, and are small (less than 64 elements).";
  *result = true;
  return absl::OkStatus();
}

absl::Status MaybePinToResourceDevice(Device** device,
                                      const EagerOperation& op) {
  if (op.colocation_exempt()) {
    return absl::OkStatus();
  }
  EagerContext& ctx = op.EagerContext();
  const absl::InlinedVector<TensorHandle*, 4>* inputs;
  TF_RETURN_IF_ERROR(op.TensorHandleInputs(&inputs));
  Device* op_device = op.Device() == kVariantDeviceNull
                          ? ctx.HostCPU()
                          : std::get<Device*>(op.Device());
  for (int i = 0; i < inputs->size(); ++i) {
    TensorHandle* tensor_handle = (*inputs)[i];
    if (tensor_handle->dtype == DT_RESOURCE) {
      if (tensor_handle->resource_remote_device_incarnation() != 0) {
        TF_RETURN_IF_ERROR(ValidateTensorHandleRemoteDevice(
            &ctx, tensor_handle->resource_remote_device_incarnation()));
      }
      Device* resource_device = tensor_handle->resource_device();
      DVLOG(2) << "for op " << op.Name() << " input " << i << " "
               << DataTypeString(tensor_handle->dtype)
               << " input device = " << resource_device->name()
               << ", op device = " << op_device->name();
      // We check for `op->Device() == nullptr` because it can be later
      // interpreted as unspecified device and a different device can
      // be selected based on device priority. If any input to an op
      // is a resource we must pin it to prevent different device selection.
      // TODO(iga): null device can mean "unspecified" or "CPU". Clean this up.
      if (resource_device != op_device || op.Device() == kVariantDeviceNull) {
        DVLOG(1) << (resource_device != op_device ? "Changing " : "Setting ")
                 << "device of operation " << op.Name() << " to "
                 << resource_device->name() << " because input #" << i
                 << " is a resource in this device.";
        *device = resource_device;
        return absl::OkStatus();
        // No point in looking at other inputs. If there are other resources,
        // they must have the same device and we already declared the op to be
        // ineligible for CPU pinning.
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace eager
}  // namespace tensorflow
