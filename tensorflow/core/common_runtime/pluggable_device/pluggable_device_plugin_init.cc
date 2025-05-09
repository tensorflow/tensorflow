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

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_plugin_init.h"

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorflow/c/experimental/grappler/grappler_internal.h"
#include "tensorflow/c/experimental/pluggable_profiler/pluggable_profiler_internal.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor_internal.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/pjrt_device_context.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_api.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_factory.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

static absl::Status InitDeviceModule(stream_executor::SEInitPluginFn init_fn) {
  if (init_fn == nullptr) {
    VLOG(1) << "Device module init function not found.";
    return absl::OkStatus();
  }

  string device_type, platform_name;
  TF_RETURN_IF_ERROR(stream_executor::InitStreamExecutorPlugin(
      init_fn, &device_type, &platform_name));

  DeviceFactory::Register(
      device_type,
      std::make_unique<PluggableDeviceFactory>(device_type, platform_name),
      /*priority=*/220, /*is_pluggable_device=*/true);

  TF_RETURN_IF_ERROR(CopyTensor::Register(
      DeviceType(device_type), DeviceType(device_type),
      PluggableDeviceUtil::DeviceToDeviceCopy,
      /*is_pluggable_device=*/true));  // Register the Copy tensor.

  VLOG(1) << "Successfully initialized Device module.";
  return absl::OkStatus();
}

typedef const PJRT_Api* (*PjrtApiInitFn)();
static absl::Status InitNextPluggableDeviceModule(TFNPDInitPluginFn init_fn,
                                                  PjrtApiInitFn init_pjrt_fn) {
  if (init_fn == nullptr) {
    VLOG(1) << "Next pluggable device init function not found.";
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(auto init_params, InitNextPluggableDevicePlugin(init_fn));
  std::string device_type(init_params.device_type);
  std::string compilation_device_name(init_params.compilation_device_name);
  int priority = init_params.priority;
  bool is_pluggable_device = init_params.is_pluggable_device;
  // Loads the PJRT plugin.
  // TODO(b/265301627): use LoadPjrtPlugin when it supports windows.
  if (init_pjrt_fn == nullptr) {
    VLOG(1) << "PJRT plugin init function not found for " << device_type;
    return absl::OkStatus();
  }
  TF_RETURN_IF_ERROR(pjrt::SetPjrtApi(device_type, init_pjrt_fn()));
  TF_ASSIGN_OR_RETURN(bool is_pjrt_plugin_initialized,
                      pjrt::IsPjrtPluginInitialized(device_type));
  if (!is_pjrt_plugin_initialized) {
    TF_RETURN_IF_ERROR(pjrt::InitializePjrtPlugin(device_type));
  }
  DeviceFactory::Register(device_type,
                          std::make_unique<NextPluggableDeviceFactory>(
                              device_type, compilation_device_name),
                          priority, is_pluggable_device);
  if (init_params.use_pjrt_on_demand_compile) {
    // XlaCompileOnDemand op compiles a TensorFlow op to a PjRtExecutable and
    // runs it.
    auto& pjrt_rollout_config = GetXlaOpsCommonFlags()->tf_xla_use_device_api;
    pjrt_rollout_config.AllowForDeviceInXlaCompileOnDemand(
        DeviceType(device_type));
    CHECK(  // Crash OK
        pjrt_rollout_config.IsEnabledInXlaCompileOnDemandForDevice(
            DeviceType(device_type)))
        << "Using Device API (PjRt) for 'on-demand' mode needs to be turned on "
           "by setting the '--tf_xla_use_device_api_for_compile_on_demand' "
           "flag in the `TF_XLA_FLAGS` environment variable.";

    static XlaDeviceOpRegistrations* registrations = RegisterXlaDeviceKernels(
        device_type.c_str(), compilation_device_name.c_str());
    (void)registrations;

    VLOG(1) << "Registered XlaCompileOnDemand op for device_type: "
            << device_type;
  }
  TF_RETURN_IF_ERROR(CopyTensor::Register(
      DeviceType(device_type), DeviceType(device_type), PjRtDeviceToDeviceCopy,
      /*is_pluggable_device=*/true));  // Register the Copy tensor.
  VLOG(1) << "Successfully initialized NextPluggableDevice module.";
  return absl::OkStatus();
}

typedef void (*TFKernelInitFn)();
static absl::Status InitKernelModule(TFKernelInitFn init_fn) {
  if (init_fn == nullptr) {
    VLOG(1) << "Kernel module init function not found.";
    return absl::OkStatus();
  }

  init_fn();

  VLOG(1) << "Successfully initialized Kernel module.";
  return absl::OkStatus();
}

static absl::Status InitGraphModule(grappler::TFInitGraphPluginFn init_fn) {
  if (init_fn == nullptr) {
    VLOG(1) << "Graph module init function not found.";
    return absl::OkStatus();
  }
  TF_RETURN_IF_ERROR(grappler::InitGraphPlugin(init_fn));

  VLOG(1) << "Successfully initialized Graph module.";
  return absl::OkStatus();
}

static absl::Status InitProfilerModule(profiler::TFInitProfilerFn init_fn) {
  if (init_fn == nullptr) {
    VLOG(1) << "Profiler module init function not found.";
    return absl::OkStatus();
  }

  TF_RETURN_IF_ERROR(profiler::InitPluginProfiler(init_fn));

  VLOG(1) << "Successfully initialized Profiler module";
  return absl::OkStatus();
}

absl::Status FindSymbol(void* dso_handle, const char* name,
                        void** output_symbol) {
  static tensorflow::Env* env = tensorflow::Env::Default();
  absl::Status status =
      env->GetSymbolFromLibrary(dso_handle, name, output_symbol);
  if (absl::IsNotFound(status)) {
    *output_symbol = nullptr;
    return absl::OkStatus();
  }
  return status;
}

absl::Status RegisterPluggableDevicePlugin(void* dso_handle) {
  if (dso_handle == nullptr) {
    return absl::InvalidArgumentError(
        "RegisterPluggableDevicePlugin called with null dso_handle");
  }

  // Step 1 Find InitDevice functions.
  PluggableDeviceInit_Api api;
  TF_RETURN_IF_ERROR(
      FindSymbol(dso_handle, "SE_InitPlugin", &api.init_plugin_fn));
  TF_RETURN_IF_ERROR(
      FindSymbol(dso_handle, "TFNPD_InitPlugin", &api.init_np_plugin_fn));
  TF_RETURN_IF_ERROR(
      FindSymbol(dso_handle, "GetPjrtApi", &api.get_pjrt_api_fn));
  // Step 2 Find InitKernel function.
  TF_RETURN_IF_ERROR(
      FindSymbol(dso_handle, "TF_InitKernel", &api.init_kernel_fn));
  // Step 3 Find InitGraph function.
  TF_RETURN_IF_ERROR(
      FindSymbol(dso_handle, "TF_InitGraph", &api.init_graph_fn));
  // Step 4 Find InitProfiler function.
  TF_RETURN_IF_ERROR(
      FindSymbol(dso_handle, "TF_InitProfiler", &api.init_profiler_fn));
  return RegisterPluggableDevicePlugin(&api);
}

absl::Status RegisterPluggableDevicePlugin(const PluggableDeviceInit_Api* api) {
  if (api == nullptr) {
    VLOG(1) << "PluggableDevice_Api is null";
    return absl::OkStatus();
  }

  // All modules are optional. Only return an error when a module is found but
  // has issues in loading / initializing.
  // Step 1 Init Device Module.
  TF_RETURN_IF_ERROR(InitDeviceModule(
      reinterpret_cast<stream_executor::SEInitPluginFn>(api->init_plugin_fn)));
  TF_RETURN_IF_ERROR(InitNextPluggableDeviceModule(
      reinterpret_cast<TFNPDInitPluginFn>(api->init_np_plugin_fn),
      reinterpret_cast<PjrtApiInitFn>(api->get_pjrt_api_fn)));

  // Step 2 Init Kernel Module.
  TF_RETURN_IF_ERROR(
      InitKernelModule(reinterpret_cast<TFKernelInitFn>(api->init_kernel_fn)));

  // Step 3 Init Graph Module.
  TF_RETURN_IF_ERROR(InitGraphModule(
      reinterpret_cast<grappler::TFInitGraphPluginFn>(api->init_graph_fn)));

  // Step 4 Init Profiler Module.
  TF_RETURN_IF_ERROR(InitProfilerModule(
      reinterpret_cast<profiler::TFInitProfilerFn>(api->init_profiler_fn)));

  return absl::OkStatus();
}

}  // namespace tensorflow
