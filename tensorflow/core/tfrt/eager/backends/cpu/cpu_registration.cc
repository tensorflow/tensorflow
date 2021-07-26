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

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.h"
#include "tensorflow/core/tfrt/eager/core_runtime/op_handler_registry.h"
#include "tfrt/cpu/core_runtime/cpu_op_handler.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime

namespace tfrt {
namespace tf {
namespace cpu {

using ::tensorflow::DeviceMgr;

static void RegisterCpuOpHandler(CoreRuntime* core_runtime,
                                 ResourceContext* resource_context,
                                 const DeviceMgr* device_mgr) {
  for (auto& device : device_mgr->ListDevices()) {
    auto& parsed_name = device->parsed_name();
    assert(parsed_name.has_id && parsed_name.has_type);
    if (parsed_name.type == "CPU") {
      auto cpu = core_runtime->GetHostContext()
                     ->GetDeviceManager()
                     ->GetDeviceRef<CpuDevice>(device->name());
      auto expected_fallback_op_handler =
          tensorflow::tfd::CreateRuntimeFallbackOpHandler(core_runtime,
                                                          device->name());
      assert(expected_fallback_op_handler);

      auto expected_cpu_op_handler = ::tfrt::CreateCpuOpHandler(
          core_runtime, std::move(cpu), expected_fallback_op_handler.get());
      assert(expected_cpu_op_handler);

      expected_cpu_op_handler.get()->AddImplicitConversion(
          tensorflow::tfd::RuntimeFallbackTensor::kTensorType,
          DenseHostTensor::kTensorType);
      expected_cpu_op_handler.get()->AddImplicitConversion(
          tensorflow::tfd::RuntimeFallbackTensor::kTensorType,
          AnyScalarHostTensor::kTensorType);
      expected_cpu_op_handler.get()->AddImplicitConversion(
          tensorflow::tfd::RuntimeFallbackTensor::kTensorType,
          StringHostTensor::kTensorType);

      core_runtime->RegisterOpHandler(device->name(),
                                      expected_cpu_op_handler.get());
      VLOG(1) << "Registered OpHandler for CPU device: " << device->name();

      // TODO(fishx): Remove this when lowering pass can use full device name.
      if (parsed_name.id == 0) {
        core_runtime->RegisterOpHandler("cpu", expected_cpu_op_handler.get());
      }
    }
  }
}

static OpHandlerRegistration register_cpu(RegisterCpuOpHandler);

}  // namespace cpu
}  // namespace tf
}  // namespace tfrt
