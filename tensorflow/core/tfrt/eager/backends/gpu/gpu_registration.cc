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

#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.h"
#include "tensorflow/core/tfrt/eager/core_runtime/op_handler_registry.h"
#include "tfrt/gpu/core_runtime/gpu_op_handler.h"  // from @tf_runtime
#include "tfrt/gpu/device/device.h"  // from @tf_runtime
#include "tfrt/gpu/device/device_util.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tfrt {
namespace tf {
namespace gpu {

using ::tfrt::CoreRuntime;

static void RegisterGpuOpHandler(CoreRuntime* core_runtime,
                                 ResourceContext* resource_context,
                                 const DeviceMgr* device_mgr) {
  for (auto& device : device_mgr->ListDevices()) {
    auto& parsed_name = device->parsed_name();
    assert(parsed_name.has_id && parsed_name.has_type);
    if (parsed_name.type == "GPU") {
      // Please see the difference between tf_device_id and platform_device_id
      // here in tensorflow/core/common_runtime/device/device_id.h
      tensorflow::TfDeviceId tf_device_id(parsed_name.id);
      tensorflow::PlatformDeviceId platform_device_id;
      tensorflow::Status s = tensorflow::GpuIdManager::TfToPlatformDeviceId(
          tf_device_id, &platform_device_id);
      if (!s.ok()) {
        LOG(ERROR) << "Failed to convert gpu device [" << device->name()
                   << "] to platform device id due to error: " << s.message();
        continue;
      }
      auto gpu = tfrt::gpu::GetOrCreateGpuDevice(
          device->name(), platform_device_id.value(),
          core_runtime->GetHostContext());
      if (!gpu) {
        LOG(ERROR) << "Failed to create gpu device [" << device->name()
                   << "]. Error: " << StrCat(gpu.takeError());
        continue;
      }
      LOG(INFO) << "Found a GPU device: " << device->name();
      auto expected_fallback_op_handler =
          tensorflow::tfd::CreateRuntimeFallbackOpHandler(core_runtime,
                                                          device->name());
      assert(expected_fallback_op_handler);

      auto expected_gpu_op_handler =
          ::tfrt::gpu::CreateGpuOpHandler(core_runtime, std::move(gpu.get()),
                                          expected_fallback_op_handler.get());
      assert(expected_gpu_op_handler);

      core_runtime->RegisterOpHandler(device->name(),
                                      expected_gpu_op_handler.get());

      // TODO(fishx): Remove this when lowering pass can use full device name.
      if (parsed_name.id == 0) {
        core_runtime->RegisterOpHandler("gpu", expected_gpu_op_handler.get());
      }
    }
  }
}

static OpHandlerRegistration register_gpu(RegisterGpuOpHandler);

}  // namespace gpu
}  // namespace tf
}  // namespace tfrt
