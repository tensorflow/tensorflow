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

#include "tensorflow/c/experimental/grappler/grappler_internal.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor_internal.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

static Status InitDeviceAndGraphModule(void* dso_handle) {
  void* dso_symbol_se;
  void* dso_symbol_graph;
  tensorflow::Env* env = tensorflow::Env::Default();

  Status status_se =
      env->GetSymbolFromLibrary(dso_handle, "SE_InitPlugin", &dso_symbol_se);
  Status status_graph =
      env->GetSymbolFromLibrary(dso_handle, "TF_InitGraph", &dso_symbol_graph);

  // Raise error if neither device nor graph is found.
  if (errors::IsNotFound(status_se) && errors::IsNotFound(status_graph)) {
    return errors::NotFound(status_se.ToString() + " " +
                            status_graph.ToString());
  }

  if (status_se == Status::OK()) {
    auto init_fn =
        reinterpret_cast<stream_executor::SEInitPluginFn>(dso_symbol_se);

    string device_type, platform_name;
    TF_RETURN_IF_ERROR(stream_executor::InitStreamExecutorPlugin(
        init_fn, &device_type, &platform_name));

    DeviceFactory::Register(
        device_type, new PluggableDeviceFactory(device_type, platform_name),
        /*priority=*/220, /*is_pluggable_device=*/true);

    TF_RETURN_IF_ERROR(CopyTensor::Register(
        DeviceType(device_type), DeviceType(device_type),
        PluggableDeviceUtil::DeviceToDeviceCopy,
        /*is_pluggable_device=*/true));  // Register the Copy tensor.
  }

  if (status_graph == Status::OK()) {
    auto init_fn =
        reinterpret_cast<grappler::TFInitGraphPluginFn>(dso_symbol_graph);
    TF_RETURN_IF_ERROR(grappler::InitGraphPlugin(init_fn));
  }

  return Status::OK();
}

typedef void (*TFKernelInitFn)();
static Status InitKernelModule(void* dso_handle) {
  void* dso_symbol;
  tensorflow::Env* env = tensorflow::Env::Default();

  TF_RETURN_IF_ERROR(
      env->GetSymbolFromLibrary(dso_handle, "TF_InitKernel", &dso_symbol));
  auto init_fn = reinterpret_cast<TFKernelInitFn>(dso_symbol);
  init_fn();
  return Status::OK();
}

Status RegisterPluggableDevicePlugin(void* dso_handle) {
  // Step 1 Init Device/Graph Module
  TF_RETURN_IF_ERROR(InitDeviceAndGraphModule(dso_handle));

  // Step 2 Init Kernel Module
  TF_RETURN_IF_ERROR(InitKernelModule(dso_handle));

  return Status::OK();
}

}  // namespace tensorflow
