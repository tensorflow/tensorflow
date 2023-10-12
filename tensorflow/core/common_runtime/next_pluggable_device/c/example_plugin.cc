/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/next_pluggable_device/c/example_plugin.h"

#include <string>

#include "tensorflow/core/platform/logging.h"
#include "tsl/platform/env.h"
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime

namespace example_plugin {

void TFNPD_DeviceEventAwait(TFNPD_DeviceEvent* event, TF_Status* status) {
  tfrt::RCReference<tfrt::AsyncValue> av_event = event->event;
  tfrt::Await(av_event);
  CHECK(av_event->IsAvailable());  // Crash OK.
  if (av_event->IsError()) {
    TF_SetStatus(status, TF_INTERNAL,
                 std::string(av_event->GetError().message()).c_str());
  } else {
    TF_SetStatus(status, TF_OK, "");
  }
}

bool TFNPD_DeviceEventIsReady(TFNPD_DeviceEvent* event) {
  return event->event->IsAvailable();
}

void TFNPD_DeviceEventAndThen(TFNPD_DeviceEvent* event, void (*callback)(void*),
                              void* callback_arg) {
  event->event->AndThen(
      [callback, callback_arg]() { (*callback)(callback_arg); });
}

void TFNPD_DeviceEventDelete(TFNPD_DeviceEvent* event) { delete event; }

TFNPD_DeviceEvent* CreateDeviceEventAndSetAvailable(tfrt::HostContext* host,
                                                    bool set_as_error) {
  // Create an AsyncValueRef as the event. Type is not important here.
  auto av_ref = tfrt::MakeUnconstructedAsyncValueRef<bool>();
  TFNPD_DeviceEvent* event = new TFNPD_DeviceEvent();
  event->event = av_ref.CopyRCRef();

  // Sleep for two seconds and set the async value available.
  tfrt::EnqueueWork(host, [av_ref = av_ref.CopyRef(), set_as_error] {
    LOG(INFO) << "Sleep for 2 seconds...";
    tsl::Env::Default()->SleepForMicroseconds(2 * 1000 * 1000);
    LOG(INFO) << "Slept for 2 seconds. Set the event to be available.";
    if (set_as_error) {
      av_ref.SetError("ERROR");
    } else {
      av_ref.emplace(true);
    }
  });
  return event;
}

}  // namespace example_plugin

const TFNPD_Api example_plugin_api = {
    /*struct_size=*/TFNPD_Api_STRUCT_SIZE,
    /*priv=*/nullptr,

    /*TFNPD_NewDeviceEvent=*/nullptr,
    /*TFNPD_DeviceEventAwait=*/example_plugin::TFNPD_DeviceEventAwait,
    /*TFNPD_DeviceEventIsReady=*/example_plugin::TFNPD_DeviceEventIsReady,
    /*TFNPD_DeviceEventAndThen=*/example_plugin::TFNPD_DeviceEventAndThen,
    /*TFNPD_DeviceEventDelete=*/example_plugin::TFNPD_DeviceEventDelete,
};

const TFNPD_Api* GetExamplePluginApi() { return &example_plugin_api; }
