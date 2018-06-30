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
#include <stdlib.h>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
#define NUM_DEVS 3

class RpcCollectiveExecutorMgrTest : public ::testing::Test {
 protected:
  RpcCollectiveExecutorMgrTest() {
    string task_name = "/job:localhost/replica:0/task:0";
    SessionOptions options;
    options.config.mutable_experimental()->set_collective_group_leader(
        task_name);
    WorkerCacheInterface* worker_cache = nullptr;
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", NUM_DEVS});
    TF_CHECK_OK(DeviceFactory::AddDevices(options, task_name, &devices_));
    device_mgr_.reset(new DeviceMgr(devices_));
    std::unique_ptr<DeviceResolverDistributed> dr(new DeviceResolverDistributed(
        device_mgr_.get(), worker_cache, task_name));
    std::unique_ptr<CollectiveParamResolverDistributed> cpr(
        new CollectiveParamResolverDistributed(options.config,
                                               device_mgr_.get(), dr.get(),
                                               worker_cache, task_name));
    // This CME is the group leader.
    cme_.reset(new RpcCollectiveExecutorMgr(options.config, device_mgr_.get(),
                                            std::move(dr), std::move(cpr),
                                            worker_cache, task_name));
  }

  std::unique_ptr<RpcCollectiveExecutorMgr> cme_;
  std::vector<Device*> devices_;
  std::unique_ptr<DeviceMgr> device_mgr_;
};

TEST_F(RpcCollectiveExecutorMgrTest, FindOrCreate) {
  CollectiveExecutor::Handle* h =
      new CollectiveExecutor::Handle(cme_->FindOrCreate(1), true);
  EXPECT_TRUE(h->get());
  CollectiveExecutor::Handle* h2 =
      new CollectiveExecutor::Handle(cme_->FindOrCreate(1), true);
  EXPECT_EQ(h->get(), h2->get());
  CollectiveExecutor* ce = h->get();
  delete h;
  delete h2;
  CollectiveExecutor* ce2 = cme_->FindOrCreate(1);
  EXPECT_EQ(ce, ce2);
  ce2->Unref();
  cme_->Cleanup(1);
}

TEST_F(RpcCollectiveExecutorMgrTest, NextStepId) {
  int64 x = cme_->NextStepId(7);
  EXPECT_EQ(x, CollectiveExecutor::kInvalidId);
  // Calling Refresh should generate a valid id.
  {
    Notification note;
    Status status;
    cme_->RefreshStepIdSequenceAsync(7,
                                     [this, &status, &note](const Status& s) {
                                       status = s;
                                       note.Notify();
                                     });
    EXPECT_TRUE(status.ok());
  }
  x = cme_->NextStepId(7);
  EXPECT_NE(x, CollectiveExecutor::kInvalidId);
  // Should keep returning same number.
  EXPECT_EQ(x, cme_->NextStepId(7));
  EXPECT_EQ(x, cme_->NextStepId(7));
  // Retire on a different graph_key should have no effect.
  cme_->RetireStepId(6, x);
  EXPECT_EQ(x, cme_->NextStepId(7));
  // Retire on same graph_key should advance.
  cme_->RetireStepId(7, x);
  int64 y = cme_->NextStepId(7);
  EXPECT_EQ((x + 1) & (((1uLL << 56) - 1) | (1uLL << 56)), y);
  // Calling refresh should jump to a different point in the random space.
  {
    Notification note;
    Status status;
    cme_->RefreshStepIdSequenceAsync(7,
                                     [this, &status, &note](const Status& s) {
                                       status = s;
                                       note.Notify();
                                     });

    note.WaitForNotification();
    EXPECT_TRUE(status.ok());
  }
  int64 z = cme_->NextStepId(7);
  // z should not be equal to or a successor of y.
  EXPECT_NE(y, z);
  EXPECT_GT(llabs(y - z), 3);
}

}  // namespace tensorflow
