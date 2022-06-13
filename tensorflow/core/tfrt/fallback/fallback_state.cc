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
#include "tensorflow/core/tfrt/fallback/fallback_state.h"

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace tfrt_stub {

StatusOr<std::unique_ptr<FallbackState>> FallbackState::Create(
    const SessionOptions &session_options,
    const tensorflow::FunctionDefLibrary &fdef_lib) {
  // Create devices.
  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));

  return std::make_unique<FallbackState>(session_options, std::move(devices),
                                         fdef_lib);
}

FallbackState::FallbackState(const SessionOptions &session_options,
                             std::vector<std::unique_ptr<Device>> devices,
                             const tensorflow::FunctionDefLibrary &fdef_lib)
    : session_options_(session_options),
      device_manager_(std::move(devices)),
      func_lib_def_(OpRegistry::Global(), fdef_lib),
      pflr_(&device_manager_, session_options.env, &session_options.config,
            TF_GRAPH_DEF_VERSION, &func_lib_def_,
            session_options.config.graph_options().optimizer_options(),
            /*thread_pool=*/nullptr, /*parent=*/nullptr,
            /*session_metadata=*/nullptr,
            Rendezvous::Factory{
                [](const int64, const DeviceMgr *device_mgr, Rendezvous **r) {
                  *r = new IntraProcessRendezvous(device_mgr);
                  return OkStatus();
                }}) {
  for (auto *d : device_manager_.ListDevices()) {
    device_set_.AddDevice(d);
  }

  // client_device is the device for feed and fetch tensors.
  device_set_.set_client_device(device_manager_.HostCPU());
}

StatusOr<std::unique_ptr<GraphExecutionState>>
FallbackState::CreateGraphExecutionState(GraphDef graph_def) const {
  // Create GraphExecutionState which contains the preprocessed graph including
  // device information. The following code is adapted from
  // http://cs?q=tensorflow/core/common_runtime/direct_session.cc:427%20at_cl:352783230

  GraphExecutionStateOptions options;
  options.device_set = &device_set_;
  options.session_options = &session_options_;
  options.session_handle = "tfrt_fallback_handle";

  std::unique_ptr<GraphExecutionState> execution_state;
  TF_RETURN_IF_ERROR(GraphExecutionState::MakeForBaseGraph(
      std::move(graph_def), options, &execution_state));
  return execution_state;
}

}  // namespace tfrt_stub
}  // namespace tensorflow
