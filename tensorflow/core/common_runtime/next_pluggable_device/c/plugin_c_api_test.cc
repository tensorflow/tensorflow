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

#include "tensorflow/core/common_runtime/next_pluggable_device/c/plugin_c_api.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/example_plugin.h"
#include "tensorflow/core/platform/status.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace {

struct CallbackParams {
  std::function<void(const tensorflow::Status&)> callback;
  tensorflow::Status status;
  const TFNPD_Api* api;
  TFNPD_DeviceEvent* event;

  ~CallbackParams() {
    // Explicitly call the deletion API to free the event.
    api->TFNPD_DeviceEventDelete(event);
  }
};

// This function is passed to the AndThen C API. The AndThen C API waits for
// the event to become ready, and invokes this function. The implementation of
// this function is not exposed to the plugin.
void InvokeCallbackFn(void* arg) {
  CallbackParams* params = reinterpret_cast<CallbackParams*>(arg);
  params->callback(params->status);
  // Explicitly delete the params after callback is done.
  delete params;
}

class PluginEventTestFixture : public testing::Test {
 protected:
  PluginEventTestFixture() {
    api_ = GetExamplePluginApi();
    auto diag_handler = [](const tfrt::DecodedDiagnostic& diag) {
      LOG(ERROR) << diag.message();
    };
    std::unique_ptr<tfrt::ConcurrentWorkQueue> work_queue =
        tfrt::CreateMultiThreadedWorkQueue(
            /*num_threads=*/4, /*num_blocking_threads=*/4);
    std::unique_ptr<tfrt::HostAllocator> host_allocator =
        tfrt::CreateMallocAllocator();

    host_ = std::make_unique<tfrt::HostContext>(
        diag_handler, std::move(host_allocator), std::move(work_queue));

    status_ = TF_NewStatus();
  }

  ~PluginEventTestFixture() override { TF_DeleteStatus(status_); }

  std::unique_ptr<tfrt::HostContext> host_;
  const TFNPD_Api* api_;
  TF_Status* status_;
};

TEST_F(PluginEventTestFixture, TestAwait) {
  std::unique_ptr<TFNPD_DeviceEvent> event;
  event.reset(example_plugin::CreateDeviceEventAndSetAvailable(host_.get()));
  // Event should be available after two seconds.
  EXPECT_FALSE(api_->TFNPD_DeviceEventIsReady(event.get()));
  api_->TFNPD_DeviceEventAwait(event.get(), status_);
  EXPECT_TRUE(api_->TFNPD_DeviceEventIsReady(event.get()));
  EXPECT_EQ(TF_GetCode(status_), TF_OK);
}

TEST_F(PluginEventTestFixture, TestAwaitWithError) {
  std::unique_ptr<TFNPD_DeviceEvent> event;
  event.reset(
      example_plugin::CreateDeviceEventAndSetAvailable(host_.get(),
                                                       /*set_as_error=*/true));
  // Event should be available after two seconds.
  EXPECT_FALSE(api_->TFNPD_DeviceEventIsReady(event.get()));
  api_->TFNPD_DeviceEventAwait(event.get(), status_);
  EXPECT_TRUE(api_->TFNPD_DeviceEventIsReady(event.get()));
  EXPECT_EQ(TF_GetCode(status_), TF_INTERNAL);
  EXPECT_STREQ(TF_Message(status_), "ERROR");
}

TEST_F(PluginEventTestFixture, TestInvokeCallback) {
  auto result_avref = tfrt::MakeUnconstructedAsyncValueRef<int>();
  std::string tennis_goat = "Sampras";

  auto done = [result_avref = result_avref.CopyRef(),
               &tennis_goat](const tensorflow::Status& status) {
    result_avref.emplace(42);
    LOG(INFO) << "Invoking status callback. Tennis goat is: "
              << status.error_message();
    tennis_goat = status.error_message();
  };

  TFNPD_DeviceEvent* event =
      example_plugin::CreateDeviceEventAndSetAvailable(host_.get());

  tensorflow::Status status(absl::StatusCode::kInternal, "Federer");

  // CallbackParams stores the "done" callback function passed in by TF, and
  // status, which is "done"'s arg. We need to add another indirection since we
  // can only cast a lambda without captures to be a function pointer.
  CallbackParams* params =
      new CallbackParams{std::move(done), status, api_, event};

  api_->TFNPD_DeviceEventAndThen(event, &InvokeCallbackFn,
                                 /*callback_arg=*/params);

  // The test fixture can be deleted before the closure in AndThen() is
  // finished. Move the host context here to extend the lifetime.
  result_avref.AndThen([result_avref = result_avref.CopyRef(), tennis_goat,
                        host = std::move(host_)] {
    EXPECT_EQ(result_avref.get(), 42);
    LOG(INFO) << "Tennis goat: " << tennis_goat;
    EXPECT_EQ(tennis_goat, "Federer");
  });
}

}  // namespace
