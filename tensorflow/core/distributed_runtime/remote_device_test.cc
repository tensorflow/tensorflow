/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/remote_device.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_testlib.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

const char* const kSession = "remote_session";

class RemoteDeviceTest : public ::testing::Test {
 protected:
  string remote_name_;
  std::unique_ptr<WorkerCacheInterface> worker_cache_;
  std::unique_ptr<WorkerInterface> wi_;
  std::vector<Device*> devices_;
  std::unique_ptr<test::TestCluster> cluster_;

  RemoteDeviceTest() {
    SessionOptions options;
    (*options.config.mutable_device_count())["CPU"] = 2;
    TF_CHECK_OK(test::TestCluster::MakeTestCluster(options, 1, &cluster_));
    const string& hostport = cluster_->targets()[0];
    GrpcChannelSpec spec;
    spec.AddHostPortsJob("localhost", {hostport});
    worker_cache_.reset(
        NewGrpcWorkerCache(NewGrpcChannelCache(spec, NewHostPortGrpcChannel)));
    remote_name_ = "/job:localhost/replica:0/task:0";
    wi_.reset(worker_cache_->CreateWorker(remote_name_));
  }

  void SetUp() override {
    Notification n;
    NewRemoteDevices(Env::Default(), worker_cache_.get(), remote_name_,
                     [&n, this](const Status& s, std::vector<Device*>* found) {
                       TF_CHECK_OK(s);
                       devices_ = *found;
                       n.Notify();
                     });
    n.WaitForNotification();
    EXPECT_EQ(devices_.size(), 2);
    std::sort(devices_.begin(), devices_.end(), [](Device* a, Device* b) {
      return a->name().compare(b->name()) < 0;
    });
  }

  void TearDown() override {
    for (auto d : devices_) delete d;
  }
};

TEST_F(RemoteDeviceTest, GetStatus) {
  // We know what the testlib's fake server does.
  EXPECT_EQ(devices_[0]->name(), strings::StrCat(remote_name_, "/cpu:0"));
  EXPECT_EQ(devices_[0]->attributes().device_type(),
            DeviceType(DEVICE_CPU).type());
  EXPECT_EQ(devices_[0]->attributes().memory_limit(), 256 << 20);
  EXPECT_EQ(devices_[1]->name(), strings::StrCat(remote_name_, "/cpu:1"));
  EXPECT_EQ(devices_[1]->attributes().memory_limit(), 256 << 20);
}

}  // namespace tensorflow
