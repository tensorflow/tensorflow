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

#include <memory>

#include "tensorflow/c/experimental/grappler/grappler_internal.h"
#include "tensorflow/c/experimental/pluggable_profiler/pluggable_profiler_internal.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor_internal.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_api.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_api.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_factory.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/tsl/platform/errors.h"

namespace tensorflow {

static Status InitDeviceModule(void* dso_handle) {
  void* dso_symbol;
  tensorflow::Env* env = tensorflow::Env::Default();
  Status status =
      env->GetSymbolFromLibrary(dso_handle, "SE_InitPlugin", &dso_symbol);

  if (errors::IsNotFound(status)) {
    VLOG(1) << "Device module not found.";
    return OkStatus();
  } else if (status != OkStatus()) {
    return status;
  }
  auto init_fn = reinterpret_cast<stream_executor::SEInitPluginFn>(dso_symbol);

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
  return OkStatus();
}

static Status InitNextPluggableDeviceModule(void* dso_handle) {
  void* dso_symbol;
  tensorflow::Env* env = tensorflow::Env::Default();

  // Loads the next pluggable device.
  Status status =
      env->GetSymbolFromLibrary(dso_handle, "TFNPD_InitPlugin", &dso_symbol);
  if (errors::IsNotFound(status)) {
    VLOG(1) << "Next pluggable device module not found.";
    return OkStatus();
  } else if (status != OkStatus()) {
    return status;
  }
  auto init_fn = reinterpret_cast<TFNPDInitPluginFn>(dso_symbol);
  string device_type, compilation_device_name;
  TF_RETURN_IF_ERROR(InitNextPluggableDevicePlugin(init_fn, &device_type,
                                                   &compilation_device_name));

  // Loads the PJRT plugin.
  // TODO(b/265301627): use LoadPjrtPlugin when it supports windows.
  status = env->GetSymbolFromLibrary(dso_handle, "GetPjrtApi", &dso_symbol);
  if (errors::IsNotFound(status)) {
    VLOG(1) << "Loading PJRT plugin failed for " << device_type << ": "
            << status.error_message();
    return OkStatus();
  } else if (!status.ok()) {
    return status;
  }
  auto init_pjrt_fn = reinterpret_cast<pjrt::PjrtApiInitFn>(dso_symbol);
  TF_RETURN_IF_ERROR(pjrt::InitPjrtPlugin(init_pjrt_fn, device_type));

  // TODO(b/265303775): consider let NextPluggableDevice decide the priority in
  // TFNPDInitPluginFn.
  DeviceFactory::Register(device_type,
                          std::make_unique<NextPluggableDeviceFactory>(
                              device_type, compilation_device_name),
                          /*priority=*/200, /*is_pluggable_device=*/false);

  VLOG(1) << "Successfully initialized NextPluggableDevice module.";
  return OkStatus();
}

static Status InitGraphModule(void* dso_handle) {
  void* dso_symbol;
  tensorflow::Env* env = tensorflow::Env::Default();
  Status status =
      env->GetSymbolFromLibrary(dso_handle, "TF_InitGraph", &dso_symbol);

  if (errors::IsNotFound(status)) {
    VLOG(1) << "Graph module not found.";
    return OkStatus();
  } else if (status != OkStatus()) {
    return status;
  }
  auto init_fn = reinterpret_cast<grappler::TFInitGraphPluginFn>(dso_symbol);
  TF_RETURN_IF_ERROR(grappler::InitGraphPlugin(init_fn));

  VLOG(1) << "Successfully initialized Graph module.";
  return OkStatus();
}

typedef void (*TFKernelInitFn)();
static Status InitKernelModule(void* dso_handle) {
  void* dso_symbol;
  tensorflow::Env* env = tensorflow::Env::Default();
  Status status =
      env->GetSymbolFromLibrary(dso_handle, "TF_InitKernel", &dso_symbol);

  if (errors::IsNotFound(status)) {
    VLOG(1) << "Kernel module not found.";
    return OkStatus();
  } else if (status != OkStatus()) {
    return status;
  }

  auto init_fn = reinterpret_cast<TFKernelInitFn>(dso_symbol);
  init_fn();

  VLOG(1) << "Successfully initialized Kernel module.";
  return OkStatus();
}

static Status InitProfilerModule(void* dso_handle) {
  void* dso_symbol;
  tensorflow::Env* env = tensorflow::Env::Default();

  Status status =
      env->GetSymbolFromLibrary(dso_handle, "TF_InitProfiler", &dso_symbol);

  if (errors::IsNotFound(status)) {
    VLOG(1) << "Profiler module not found.";
    return OkStatus();
  } else if (status != OkStatus()) {
    return status;
  }

  auto init_fn = reinterpret_cast<profiler::TFInitProfilerFn>(dso_symbol);
  TF_RETURN_IF_ERROR(profiler::InitPluginProfiler(init_fn));

  VLOG(1) << "Successfully initialized Profiler module";
  return OkStatus();
}

Status RegisterPluggableDevicePlugin(void* dso_handle) {
  // All modules are optional. Only return an error when a module is found but
  // has issues in loading / initializing.
  // Step 1 Init Device Module.
  TF_RETURN_IF_ERROR(InitDeviceModule(dso_handle));
  TF_RETURN_IF_ERROR(InitNextPluggableDeviceModule(dso_handle));

  // Step 2 Init Kernel Module.
  TF_RETURN_IF_ERROR(InitKernelModule(dso_handle));

  // Step 3 Init Graph Module.
  TF_RETURN_IF_ERROR(InitGraphModule(dso_handle));

  // Step 4 Init Profiler Module.
  TF_RETURN_IF_ERROR(InitProfilerModule(dso_handle));

  return OkStatus();
}

}  // namespace tensorflow
