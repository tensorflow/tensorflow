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
#include "tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h"

#include <stdlib.h>

#include <string>
#include <vector>

#include "absl/synchronization/notification.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/nccl/collective_communicator.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/worker.pb.h"
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
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(options, task_name, &devices));
    device_mgr_ = std::make_unique<StaticDeviceMgr>(std::move(devices));
    std::unique_ptr<DeviceResolverDistributed> dr(
        new DeviceResolverDistributed(device_mgr_.get()));
    std::unique_ptr<CollectiveParamResolverDistributed> cpr(
        new CollectiveParamResolverDistributed(
            options.config, device_mgr_.get(), dr.get(),
            /*nccl_communicator*/ nullptr, worker_cache, task_name));
    // This CME is the group leader.
    cme_.reset(new RpcCollectiveExecutorMgr(
        options.config, device_mgr_.get(), std::move(dr), std::move(cpr),
        MaybeCreateNcclCommunicator(options.config), worker_cache, task_name));
  }

  std::unique_ptr<RpcCollectiveExecutorMgr> cme_;
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
  int64_t x = cme_->NextStepId(7);
  EXPECT_EQ(x, CollectiveExecutor::kInvalidId);
  // Calling Refresh should generate a valid id.
  {
    absl::Notification note;
    absl::Status status;
    cme_->RefreshStepIdSequenceAsync(
        7, [this, &status, &note](const absl::Status& s) {
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
  int64_t y = cme_->NextStepId(7);
  EXPECT_EQ((x + 1) & (((1uLL << 56) - 1) | (1uLL << 56)), y);
  // Calling refresh should jump to a different point in the random space.
  {
    absl::Notification note;
    absl::Status status;
    cme_->RefreshStepIdSequenceAsync(
        7, [this, &status, &note](const absl::Status& s) {
          status = s;
          note.Notify();
        });

    note.WaitForNotification();
    EXPECT_TRUE(status.ok());
  }
  int64_t z = cme_->NextStepId(7);
  // z should not be equal to or a successor of y.
  EXPECT_NE(y, z);
  EXPECT_GT(llabs(y - z), 3);
}

TEST_F(RpcCollectiveExecutorMgrTest, GetStepSequence) {
  int64_t x = cme_->NextStepId(3);
  EXPECT_EQ(x, CollectiveExecutor::kInvalidId);
  int64_t y = cme_->NextStepId(4);
  EXPECT_EQ(y, CollectiveExecutor::kInvalidId);
  GetStepSequenceRequest request;
  GetStepSequenceResponse response;
  request.add_graph_key(3);
  request.add_graph_key(4);
  {
    absl::Notification note;
    absl::Status status;
    cme_->GetStepSequenceAsync(&request, &response,
                               [this, &status, &note](const absl::Status& s) {
                                 status = s;
                                 note.Notify();
                               });
    note.WaitForNotification();
    EXPECT_TRUE(status.ok());
  }
  ASSERT_EQ(2, response.step_sequence_size());
  std::unordered_map<int64_t, int64_t> values;
  for (const auto& ss : response.step_sequence()) {
    values[ss.graph_key()] = ss.next_step_id();
  }
  EXPECT_NE(values[3], CollectiveExecutor::kInvalidId);
  EXPECT_NE(values[4], CollectiveExecutor::kInvalidId);
  // Re-get, should be same values.
  response.Clear();
  {
    absl::Notification note;
    absl::Status status;
    cme_->GetStepSequenceAsync(&request, &response,
                               [this, &status, &note](const absl::Status& s) {
                                 status = s;
                                 note.Notify();
                               });
    note.WaitForNotification();
    EXPECT_TRUE(status.ok());
  }
  ASSERT_EQ(2, response.step_sequence_size());
  for (const auto& ss : response.step_sequence()) {
    EXPECT_EQ(values[ss.graph_key()], ss.next_step_id());
  }
}

}  // namespace tensorflow
