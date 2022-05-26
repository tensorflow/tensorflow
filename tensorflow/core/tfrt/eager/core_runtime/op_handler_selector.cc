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

#include "tensorflow/core/tfrt/eager/core_runtime/op_handler_selector.h"

#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/placement_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/tfrt/eager/core_runtime/op_handler_registry.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/device.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

EagerOpHandlerSelector::EagerOpHandlerSelector(CoreRuntime* core_runtime,
                                               EagerContext* eager_context,
                                               OpHandler* fallback_op_handler,
                                               bool pin_small_ops_to_cpu)
    : core_runtime_(core_runtime),
      eager_context_(eager_context),
      cpu_device_(core_runtime->GetHostContext()->GetHostDevice()),
      cpu_op_handler_(core_runtime_->GetOpHandler(cpu_device_.name())),
      fallback_op_handler_(fallback_op_handler),
      pin_small_ops_to_cpu_(pin_small_ops_to_cpu) {
  assert(cpu_op_handler_);
  assert(fallback_op_handler_);
}

EagerOpHandlerSelector::~EagerOpHandlerSelector() {}

Status EagerOpHandlerSelector::SelectFromArguments(
    const ImmediateExecutionOperation& op, OpHandler** op_handler) {
  // If the op contains resource handle, place the op on the device of the
  // resource.
  // TODO(tfrt-devs): Unify this logic with MaybePinToResourceDevice in Eager
  // runtime.
  for (int i = 0; i < op.GetInputs().size(); i++) {
    auto& handle = op.GetInputs()[i];
    Status s;
    if (handle->DataType() == tensorflow::DT_RESOURCE) {
      auto device_name = handle->DeviceName(&s);
      TF_RETURN_IF_ERROR(s);
      *op_handler = core_runtime_->GetOpHandler(device_name);
      if (*op_handler != nullptr) {
        DVLOG(1) << "Setting device of operation " << op.Name() << " to "
                 << device_name << " because input #" << i
                 << " is a resource in this device.";
        return ::tensorflow::OkStatus();
      }
    }
  }

  // Pin the op to cpu op handler if it is a small ops and all its inputs
  // are on cpu already.
  if (pin_small_ops_to_cpu_) {
    bool pin_to_cpu;
    TF_RETURN_IF_ERROR(tensorflow::eager::MaybePinSmallOpsToCpu(
        &pin_to_cpu, op.Name(), op.GetInputs(),
        {cpu_device_.name().data(), cpu_device_.name().size()}));
    if (pin_to_cpu) {
      *op_handler = cpu_op_handler_;
      return ::tensorflow::OkStatus();
    }
  }

  // Note: The output op_handler is nullptr.
  return ::tensorflow::OkStatus();
}

Status EagerOpHandlerSelector::SelectFromNodeDef(
    const ImmediateExecutionOperation& op, const NodeDef* ndef,
    OpHandler** op_handler) {
  const auto& requested_device = op.DeviceName();

  // TODO(fishx): Use TFRT native op registry to select op handler.

  // TODO(fishx): Add a cache for following device placement using current TF.
  // Use EagerContext from current tf to select op handler for this op.
  tensorflow::DeviceNameUtils::ParsedName device_parsed_name;
  if (!tensorflow::DeviceNameUtils::ParseFullName(requested_device,
                                                  &device_parsed_name)) {
    return tensorflow::errors::InvalidArgument("Failed to parse device name: ",
                                               requested_device);
  }

  tensorflow::Device* device;
  TF_RETURN_IF_ERROR(
      eager_context_->SelectDevice(device_parsed_name, *ndef, &device));

  *op_handler = core_runtime_->GetOpHandler(device->name());

  if (!(*op_handler)) *op_handler = fallback_op_handler_;

  return ::tensorflow::OkStatus();
}

}  // namespace tf
}  // namespace tfrt
