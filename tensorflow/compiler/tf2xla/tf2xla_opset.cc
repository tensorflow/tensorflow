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

#include "tensorflow/compiler/tf2xla/tf2xla_opset.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def.pb.h"

namespace tensorflow {

const int SUPPORTED_DEVICES_NUM = 2;
static const char* const SUPPORTED_DEVICES[SUPPORTED_DEVICES_NUM] = {
    DEVICE_GPU_XLA_JIT, DEVICE_CPU_XLA_JIT};

bool IsSupportedBackend(absl::string_view device_name) {
  for (int i = 0; i < SUPPORTED_DEVICES_NUM; i++) {
    if (SUPPORTED_DEVICES[i] == device_name) return true;
  }
  return false;
}

absl::Status RegisterBackends(absl::string_view device_name) {
  if (!IsSupportedBackend(device_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat(device_name, " is not supported.  Supported devices are ",
                     absl::StrJoin(SUPPORTED_DEVICES, ", ")));
  }
  // All backends need to be registered before DeviceKernels is called
  // because it calls RegisterCompilationKernels which will only run 1x,
  // meaning if a device is registered afterwards the ops for that device
  // will not be included.
  auto op_filter = [](KernelDef* kdef) {
    if (kdef->op() == "Const") {
      AddDtypeToKernelDefConstraint("dtype", DT_STRING, kdef);
    }
    if (kdef->op() == "Assert") {
      AddDtypeToKernelDefConstraint("T", DT_STRING, kdef);
    }
    return true;
  };

  // Backends might already be registered due to preprocesser macros defined
  // in xla_op_registery.h so this first checks to see if they are registered
  // already because re-registering the same device will cause a failure.
  if (!XlaOpRegistry::IsBackendRegistered(DEVICE_GPU_XLA_JIT)) {
    static auto gpu_backend =
        XlaBackendRegistrar(DEVICE_GPU_XLA_JIT, kGpuAllTypes, op_filter);
  }
  if (!XlaOpRegistry::IsBackendRegistered(DEVICE_CPU_XLA_JIT)) {
    static auto cpu_backend =
        XlaBackendRegistrar(DEVICE_CPU_XLA_JIT, kCpuAllTypes, op_filter);
  }
  if (!XlaOpRegistry::IsBackendRegistered(std::string(device_name))) {
    return absl::InternalError(
        absl::StrCat(device_name, " is not registered."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::string>> GetRegisteredXlaOpsForDevice(
    absl::string_view device_name) {
  auto status = RegisterBackends(device_name);
  if (!status.ok()) return status;

  std::vector<const KernelDef*> kernel_defs =
      XlaOpRegistry::DeviceKernels(std::string(device_name), true);
  std::vector<std::string> op_names;
  op_names.reserve(kernel_defs.size());
  for (const auto& kernel_def : kernel_defs) {
    op_names.push_back(kernel_def->op());
  }
  std::sort(op_names.begin(), op_names.end());
  return op_names;
}
}  // namespace tensorflow
