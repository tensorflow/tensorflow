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
    TF_CHECK_OK(DeviceFactory::AddDevices(options, task_name, &devices_));
    device_mgr_.reset(new DeviceMgr(devices_));
    drl_.reset(new DeviceResolverLocal(device_mgr_.get()));
    prl_.reset(new CollectiveParamResolverLocal(device_mgr_.get(), drl_.get(),
                                                task_name));
  }

  void GenSubdivPerms(const string& device, int source_rank,
                      CollectiveParams* cp) {
    CollectiveParamResolverLocal::GenerateSubdivPerms(device, source_rank, cp);
  }

  std::vector<Device*> devices_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<DeviceResolverLocal> drl_;
  std::unique_ptr<CollectiveParamResolverLocal> prl_;
};

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
    EXPECT_EQ(cps[i].subdiv_rank[0], i);
    EXPECT_EQ(cps[i].instance.impl_details.subdiv_source_rank.size(), 0);
    EXPECT_FALSE(cps[i].is_source);
    EXPECT_EQ(cps[i].default_rank, i);
    EXPECT_TRUE(cps[i].instance.same_num_devices_per_task);
  }
}

TEST_F(CollectiveParamResolverLocalTest, CompleteParamsBroadcast1Task) {
  CollectiveParams cps[NUM_DEVS];
  Status statuses[NUM_DEVS];
  Notification note[NUM_DEVS];
  for (int i = 0; i < NUM_DEVS; ++i) {
    CollectiveParams* cp = &cps[i];
    cp->group.group_key = 1;
    cp->group.group_size = 3;
    cp->group.device_type = DeviceType("CPU");
    cp->group.num_tasks = 1;
    cp->instance.instance_key = 3;
    cp->instance.type = BROADCAST_COLLECTIVE;
    cp->instance.data_type = DataType(DT_FLOAT);
    cp->instance.shape = TensorShape({5});
    cp->instance.device_names.push_back(
        strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i));
    cp->instance.impl_details.subdiv_offsets.push_back(0);
    cp->is_source = (i == 1);
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
    ASSERT_GT(cps[i].subdiv_rank.size(), 0);
    EXPECT_EQ(cps[i].subdiv_rank[0], i);
    ASSERT_GT(cps[i].instance.impl_details.subdiv_source_rank.size(), 0);
    EXPECT_EQ(cps[i].instance.impl_details.subdiv_source_rank[0], 1);
    EXPECT_EQ(cps[i].is_source, (i == 1));
    EXPECT_EQ(cps[i].default_rank, i);
    EXPECT_TRUE(cps[i].instance.same_num_devices_per_task);
  }
}

TEST_F(CollectiveParamResolverLocalTest, GenerateSubdivPerms) {
  static const int kNumDevsPerTask = 8;
  static const int kNumTasks = 3;
  static const int kNumDevs = kNumDevsPerTask * kNumTasks;
  CollectiveParams cp;
  std::vector<string> device_names;
  std::vector<string> task_names;
  cp.group.group_key = 1;
  cp.group.group_size = kNumDevs;
  cp.group.device_type = DeviceType("GPU");
  cp.group.num_tasks = kNumTasks;
  cp.instance.instance_key = 3;
  cp.instance.type = REDUCTION_COLLECTIVE;
  cp.instance.data_type = DataType(DT_FLOAT);
  cp.instance.shape = TensorShape({5});
  cp.instance.impl_details.subdiv_offsets.push_back(0);
  cp.is_source = false;
  for (int i = 0; i < kNumDevs; ++i) {
    int task_id = i / kNumDevsPerTask;
    int dev_id = i % kNumDevsPerTask;
    string task_name = strings::StrCat("/job:worker/replica:0/task:", task_id);
    task_names.push_back(task_name);
    string device_name = strings::StrCat(task_name, "/device:GPU:", dev_id);
    device_names.push_back(device_name);
    cp.instance.task_names.push_back(task_name);
    cp.instance.device_names.push_back(device_name);
  }

  int test_rank = 0;
  cp.default_rank = test_rank;
  cp.instance.impl_details.subdiv_offsets = {0, 4};
  GenSubdivPerms(cp.instance.device_names[test_rank], 0, &cp);
  std::vector<int> expected_0 = {0,  1,  2,  3,  4,  5,  6,  7,
                                 8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23};
  std::vector<int> expected_1 = {4, 5, 6,  7,  0,  1,  2,  3,  12, 13, 14, 15,
                                 8, 9, 10, 11, 20, 21, 22, 23, 16, 17, 18, 19};
  for (int i = 0; i < kNumDevs; ++i) {
    EXPECT_EQ(expected_0[i],
              cp.instance.impl_details.subdiv_permutations[0][i]);
    EXPECT_EQ(expected_1[i],
              cp.instance.impl_details.subdiv_permutations[1][i]);
  }
  EXPECT_EQ(0, cp.subdiv_rank[0]);
  EXPECT_EQ(4, cp.subdiv_rank[1]);

  test_rank = 3;
  cp.default_rank = test_rank;
  cp.instance.impl_details.subdiv_offsets = {3, -3};
  cp.instance.impl_details.subdiv_permutations.clear();
  GenSubdivPerms(cp.instance.device_names[test_rank], 0, &cp);
  expected_0 = {3,  4, 5, 6,  7,  0,  1,  2,  11, 12, 13, 14,
                15, 8, 9, 10, 19, 20, 21, 22, 23, 16, 17, 18};
  expected_1 = {4, 3,  2,  1,  0,  7,  6,  5,  12, 11, 10, 9,
                8, 15, 14, 13, 20, 19, 18, 17, 16, 23, 22, 21};
  for (int i = 0; i < kNumDevs; ++i) {
    EXPECT_EQ(expected_0[i],
              cp.instance.impl_details.subdiv_permutations[0][i]);
    EXPECT_EQ(expected_1[i],
              cp.instance.impl_details.subdiv_permutations[1][i]);
  }
  EXPECT_EQ(0, cp.subdiv_rank[0]);
  EXPECT_EQ(1, cp.subdiv_rank[1]);
}

}  // namespace tensorflow
