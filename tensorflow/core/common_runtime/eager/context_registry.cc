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

#include "tensorflow/core/common_runtime/eager/context_registry.h"

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/core/common_runtime/eager/context.h"

namespace tensorflow {

ContextRegistry* GlobalContextRegistry() {
  static ContextRegistry* registry = new ContextRegistry;
  return registry;
}

ImmediateExecutionContext* CreateEagerContext(const TFE_ContextOptions* opts) {
  std::vector<std::unique_ptr<tensorflow::Device>> devices;
  auto status = tensorflow::DeviceFactory::AddDevices(
      opts->session_options.options, "/job:localhost/replica:0/task:0",
      &devices);
  if (!status.ok()) return nullptr;
  std::unique_ptr<tensorflow::DeviceMgr> device_mgr(
      new tensorflow::DynamicDeviceMgr(std::move(devices)));

  Rendezvous* r = new IntraProcessRendezvous(device_mgr.get());

  EagerContext* eager_context = new EagerContext(
      opts->session_options.options,
      static_cast<ContextDevicePlacementPolicy>(opts->device_placement_policy),
      opts->async, device_mgr.release(),
      /*device_mgr_owned*/ true, r,
      /*cluster_flr=*/nullptr,
      /*collective_executor_mgr=*/nullptr,
      /*run_eager_op_as_function=*/opts->run_eager_op_as_function,
      /*jit_compile_rewrite=*/opts->jit_compile_rewrite);
  return reinterpret_cast<ImmediateExecutionContext*>(eager_context);
}

// Register TF eager context creator by default.
void RegisterEagerContext() {
  auto* context_registry = GlobalContextRegistry();
  auto eager_context_creator = [](const TFE_ContextOptions* opts) {
    return CreateEagerContext(opts);
  };
  context_registry->Register("eager", eager_context_creator);
}

REGISTER_EAGER_CONTEXT_CREATOR(EagerContext, "eager", CreateEagerContext);

}  // namespace tensorflow
