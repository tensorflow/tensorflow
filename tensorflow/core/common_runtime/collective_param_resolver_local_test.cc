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
#include "tensorflow/core/common_runtime/collective_executor_mgr.h"

#include "tensorflow/core/common_runtime/collective_param_resolver_local.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

#define NUM_DEVS 3

class CollectiveParamResolverLocalTest : public ::testing::Test {
 protected:
  CollectiveParamResolverLocalTest() {
    ConfigProto cp;
    SessionOptions options;
    string task_name = "/job:localhost/replica:0/task:0";
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", NUM_DEVS});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(options, task_name, &devices));
    device_mgr_.reset(new DeviceMgr(std::move(devices)));
    drl_.reset(new DeviceResolverLocal(device_mgr_.get()));
    prl_.reset(new CollectiveParamResolverLocal(device_mgr_.get(), drl_.get(),
                                                task_name));
  }

  void RunCompleteDefaultRanking(
      const CollectiveParams& shared_cp,
      const std::vector<DeviceLocality>& localities,
      const std::vector<int32>& gpu_ring_order,
      const std::vector<string>& expected_device_order) {
    CollectiveParams cp;
    cp.instance.device_names = shared_cp.instance.device_names;
    CollectiveParamResolverLocal::InstanceRec ir;
    {
      mutex_lock l(ir.out_mu);
      ir.shared.name = shared_cp.name;
      ir.shared.group = shared_cp.group;
      ir.shared.instance = shared_cp.instance;
      if (!gpu_ring_order.empty()) {
        ir.shared.instance.gpu_ring_order = "";
        for (int i = 0; i < static_cast<int32>(gpu_ring_order.size() - 1);
             ++i) {
          ir.shared.instance.gpu_ring_order = strings::StrCat(
              ir.shared.instance.gpu_ring_order, gpu_ring_order[i], ",");
        }
        ir.shared.instance.gpu_ring_order = strings::StrCat(
            ir.shared.instance.gpu_ring_order, gpu_ring_order.back());
      }
      VLOG(2) << "gpu_ring_order " << ir.shared.instance.gpu_ring_order;
      prl_->CompleteDefaultRanking(nullptr, &cp, &ir, localities);
      EXPECT_EQ(ir.shared.instance.device_names, expected_device_order);
    }
  }

  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<DeviceResolverLocal> drl_;
  std::unique_ptr<CollectiveParamResolverLocal> prl_;
};

TEST_F(CollectiveParamResolverLocalTest, CompleteDefaultRanking) {
  constexpr int kNumGpus = 8;
  CollectiveParams cp;
  std::vector<DeviceLocality> localities(kNumGpus);
  cp.name = "PRLTest";
  cp.group.device_type = DeviceType("GPU");
  cp.group.num_tasks = 1;
  cp.group.group_size = kNumGpus;
  cp.instance.instance_key = 5;
  cp.instance.type = REDUCTION_COLLECTIVE;
  cp.instance.data_type = DataType(DT_FLOAT);
  std::unordered_set<int> clique1 = {0, 1, 6, 7};
  for (int gpu_idx = 0; gpu_idx < kNumGpus; ++gpu_idx) {
    cp.instance.task_names.push_back("/job:localhost/replica:0/task:0");
    cp.instance.device_names.push_back(strings::StrCat(
        "/job:localhost/replica:0/task:0/device:GPU:", gpu_idx));
    DeviceLocality* locality = &localities[gpu_idx];
    // Build localities so that 0,1,6,7 and 2,3,4,5 form 2 strongly connected
    // components.  Across components, connect 3 and 7.
    for (int link_idx = 0; link_idx < kNumGpus; ++link_idx) {
      if (gpu_idx == link_idx) continue;
      bool gpu_in_clique1 = clique1.find(gpu_idx) != clique1.end();
      bool link_in_clique1 = clique1.find(link_idx) != clique1.end();
      if ((gpu_in_clique1 && link_in_clique1) ||
          (!gpu_in_clique1 && !link_in_clique1)) {
        LocalLinks* links = locality->mutable_links();
        InterconnectLink* ilink = links->add_link();
        ilink->set_device_id(link_idx);
        ilink->set_strength(2);
      } else if ((gpu_idx == 3 && link_idx == 7) ||
                 (gpu_idx == 7 && link_idx == 3)) {
        LocalLinks* links = locality->mutable_links();
        InterconnectLink* ilink = links->add_link();
        ilink->set_device_id(link_idx);
        ilink->set_strength(1);
      }
    }
  }
  RunCompleteDefaultRanking(cp, localities, {1, 3, 5, 7, 6, 4, 2, 0},
                            {
                                "/job:localhost/replica:0/task:0/device:GPU:1",
                                "/job:localhost/replica:0/task:0/device:GPU:3",
                                "/job:localhost/replica:0/task:0/device:GPU:5",
                                "/job:localhost/replica:0/task:0/device:GPU:7",
                                "/job:localhost/replica:0/task:0/device:GPU:6",
                                "/job:localhost/replica:0/task:0/device:GPU:4",
                                "/job:localhost/replica:0/task:0/device:GPU:2",
                                "/job:localhost/replica:0/task:0/device:GPU:0",
                            });
  RunCompleteDefaultRanking(cp, localities, {7, 6, 5, 4, 3, 2, 1, 0},
                            {
                                "/job:localhost/replica:0/task:0/device:GPU:7",
                                "/job:localhost/replica:0/task:0/device:GPU:6",
                                "/job:localhost/replica:0/task:0/device:GPU:5",
                                "/job:localhost/replica:0/task:0/device:GPU:4",
                                "/job:localhost/replica:0/task:0/device:GPU:3",
                                "/job:localhost/replica:0/task:0/device:GPU:2",
                                "/job:localhost/replica:0/task:0/device:GPU:1",
                                "/job:localhost/replica:0/task:0/device:GPU:0",
                            });
  // With no gpu_ring_order passed, automatic link detection should kick in.
  // Starting at dev 0, the best order would be: 0,1,6,7,3,2,4,5
  RunCompleteDefaultRanking(cp, localities, {},
                            {
                                "/job:localhost/replica:0/task:0/device:GPU:0",
                                "/job:localhost/replica:0/task:0/device:GPU:1",
                                "/job:localhost/replica:0/task:0/device:GPU:6",
                                "/job:localhost/replica:0/task:0/device:GPU:7",
                                "/job:localhost/replica:0/task:0/device:GPU:3",
                                "/job:localhost/replica:0/task:0/device:GPU:2",
                                "/job:localhost/replica:0/task:0/device:GPU:4",
                                "/job:localhost/replica:0/task:0/device:GPU:5",
                            });
}

TEST_F(CollectiveParamResolverLocalTest, CompleteParamsReduction1Task) {
  CollectiveParams cps[NUM_DEVS];
  Status statuses[NUM_DEVS];
  Notification note[NUM_DEVS];
  for (int i = 0; i < NUM_DEVS; ++i) {
    CollectiveParams* cp = &cps[i];
    cp->group.group_key = 1;
    cp->group.group_size = 3;
    cp->group.device_type = DeviceType("CPU");
    cp->group.num_tasks = 1;
    cp->instance.instance_key = 7;
    cp->instance.type = REDUCTION_COLLECTIVE;
    cp->instance.data_type = DataType(DT_FLOAT);
    cp->instance.shape = TensorShape({5});
    cp->instance.device_names.push_back(
        strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i));
    cp->instance.impl_details.subdiv_offsets.push_back(0);
    cp->is_source = false;
    Env::Default()->SchedClosure([this, i, cp, &note, &statuses]() {
      prl_->CompleteParamsAsync(cp->instance.device_names[0], cp,
                                nullptr /*CancellationManager*/,
                                [this, &statuses, &note, i](const Status& s) {
                                  statuses[i] = s;
                                  note[i].Notify();
                                });
    });
  }
  for (int i = 0; i < NUM_DEVS; ++i) {
    note[i].WaitForNotification();
  }
  for (int i = 0; i < NUM_DEVS; ++i) {
    TF_ASSERT_OK(statuses[i]);
    ASSERT_EQ(cps[i].instance.device_names.size(), 3);
    for (int j = 0; j < NUM_DEVS; ++j) {
      EXPECT_EQ(
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", j),
          cps[i].instance.device_names[j]);
      EXPECT_TRUE(cps[i].task.is_local[j]);
    }
    EXPECT_EQ(cps[i].instance.impl_details.subdiv_source_rank.size(), 0);
    EXPECT_FALSE(cps[i].is_source);
    EXPECT_EQ(cps[i].default_rank, i);
    EXPECT_TRUE(cps[i].instance.same_num_devices_per_task);
  }
}

void InitializeCollectiveParamsForBroadcast(int instance_key, int device_idx,
                                            bool is_source,
                                            CollectiveParams* cp) {
  cp->group.group_key = 1;
  cp->group.group_size = 3;
  cp->group.device_type = DeviceType("CPU");
  cp->group.num_tasks = 1;
  cp->instance.instance_key = instance_key;
  cp->instance.type = BROADCAST_COLLECTIVE;
  cp->instance.data_type = DataType(DT_FLOAT);
  cp->instance.shape = TensorShape({5});
  cp->instance.device_names.push_back(strings::StrCat(
      "/job:localhost/replica:0/task:0/device:CPU:", device_idx));
  cp->instance.impl_details.subdiv_offsets.push_back(0);
  cp->is_source = is_source;
}

TEST_F(CollectiveParamResolverLocalTest, CompleteParamsBroadcast1Task) {
  constexpr int kInstanceKey = 5;
  CollectiveParams cps[NUM_DEVS];
  Status statuses[NUM_DEVS];
  Notification note[NUM_DEVS];
  for (int i = 0; i < NUM_DEVS; ++i) {
    CollectiveParams* cp = &cps[i];
    InitializeCollectiveParamsForBroadcast(kInstanceKey, i, i == 1, cp);
    Env::Default()->SchedClosure([this, i, cp, &note, &statuses]() {
      prl_->CompleteParamsAsync(cp->instance.device_names[0], cp,
                                nullptr /*CancellationManager*/,
                                [&statuses, &note, i](const Status& s) {
                                  statuses[i] = s;
                                  note[i].Notify();
                                });
    });
  }
  for (int i = 0; i < NUM_DEVS; ++i) {
    note[i].WaitForNotification();
  }
  for (int i = 0; i < NUM_DEVS; ++i) {
    TF_ASSERT_OK(statuses[i]);
    ASSERT_EQ(cps[i].instance.device_names.size(), 3);
    for (int j = 0; j < NUM_DEVS; ++j) {
      EXPECT_EQ(
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", j),
          cps[i].instance.device_names[j]);
      EXPECT_TRUE(cps[i].task.is_local[j]);
    }
    EXPECT_EQ(cps[i].is_source, (i == 1));
    EXPECT_EQ(cps[i].default_rank, i);
    EXPECT_TRUE(cps[i].instance.same_num_devices_per_task);
  }
}

// If we don't mark any participant in a broadcast as the source, we essentially
// create a collective group with only broadcast recvs.  In that case, we should
// get an internal error from param resolution.
TEST_F(CollectiveParamResolverLocalTest, CompleteParamsBroadcastForgotSender) {
  constexpr int kInstanceKey = 8;
  CollectiveParams cps[NUM_DEVS];
  Status statuses[NUM_DEVS];
  Notification note[NUM_DEVS];
  for (int i = 0; i < NUM_DEVS; ++i) {
    CollectiveParams* cp = &cps[i];
    InitializeCollectiveParamsForBroadcast(kInstanceKey, i, false, cp);
    Env::Default()->SchedClosure([this, i, cp, &note, &statuses]() {
      prl_->CompleteParamsAsync(cp->instance.device_names[0], cp,
                                nullptr /*CancellationManager*/,
                                [&statuses, &note, i](const Status& s) {
                                  statuses[i] = s;
                                  note[i].Notify();
                                });
    });
  }
  for (int i = 0; i < NUM_DEVS; ++i) {
    note[i].WaitForNotification();
  }
  for (int i = 0; i < NUM_DEVS; ++i) {
    EXPECT_EQ(statuses[i].code(), error::INTERNAL);
    EXPECT_EQ(statuses[i].error_message(),
              strings::StrCat(
                  "Instance ", kInstanceKey,
                  " found no source for broadcast.  This could mean that there"
                  " were group_size=",
                  NUM_DEVS, " BcastRecvs but no BcastSend."));
  }
}

}  // namespace tensorflow
