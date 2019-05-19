/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/flex/delegate_data.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/status.h"

namespace tflite {
namespace flex {
DelegateData::DelegateData() {}

DelegateData::~DelegateData() {
  if (eager_context_) eager_context_->Unref();
}

tensorflow::Status DelegateData::Prepare(
    const tensorflow::SessionOptions& session_options) {
  if (eager_context_) {
    return tensorflow::Status();
  }

  std::vector<std::unique_ptr<tensorflow::Device>> devices;

  TF_RETURN_IF_ERROR(tensorflow::DeviceFactory::AddDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));

  std::unique_ptr<tensorflow::DeviceMgr> device_mgr =
      absl::make_unique<tensorflow::DeviceMgr>(std::move(devices));
  // Note that Rendezvous is ref-counted so it will be automatically deleted.
  tensorflow::Rendezvous* rendezvous =
      new tensorflow::IntraProcessRendezvous(device_mgr.get());
  eager_context_ = new tensorflow::EagerContext(
      session_options,
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      /*async=*/false, std::move(device_mgr), rendezvous);
  return tensorflow::Status();
}

}  // namespace flex
}  // namespace tflite
