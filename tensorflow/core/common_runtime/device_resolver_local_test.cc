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
#include "tensorflow/core/common_runtime/device_resolver_local.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

#define NUM_DEVS 3

class DeviceResolverLocalTest : public ::testing::Test {
 protected:
  DeviceResolverLocalTest() {
    SessionOptions options;
    string task_name = "/job:localhost/replica:0/task:0";
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", NUM_DEVS});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(options, task_name, &devices));
    device_mgr_.reset(new DeviceMgr(std::move(devices)));
    drl_.reset(new DeviceResolverLocal(device_mgr_.get()));
  }

  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<DeviceResolverLocal> drl_;
};

TEST_F(DeviceResolverLocalTest, GetDeviceLocalitiesKnown) {
  CollectiveParams cp;
  std::vector<DeviceLocality> localities;
  cp.instance.device_names.push_back(
      "/job:localhost/replica:0/task:0/device:CPU:1");
  cp.instance.device_names.push_back(
      "/job:localhost/replica:0/task:0/device:CPU:2");
  Notification note;
  Status status;
  drl_->GetDeviceLocalitiesAsync(cp.instance, &localities,
                                 [&note, &status](const Status& s) {
                                   status = s;
                                   note.Notify();
                                 });
  note.WaitForNotification();
  TF_EXPECT_OK(status);
  EXPECT_EQ(2, localities.size());
}

TEST_F(DeviceResolverLocalTest, GetDeviceLocalitiesUnknown) {
  CollectiveParams cp;
  std::vector<DeviceLocality> localities;
  // In some builds there may be 1 GPU, but there should never be 9.
  cp.instance.device_names.push_back(
      "/job:localhost/replica:0/task:0/device:GPU:9");
  Notification note;
  Status status;
  drl_->GetDeviceLocalitiesAsync(cp.instance, &localities,
                                 [&note, &status](const Status& s) {
                                   status = s;
                                   note.Notify();
                                 });
  note.WaitForNotification();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(0, localities.size());
}

}  // namespace
}  // namespace tensorflow
