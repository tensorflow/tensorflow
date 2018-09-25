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
    EXPECT_EQ(cps[i].is_source, (i == 1));
    EXPECT_EQ(cps[i].default_rank, i);
    EXPECT_TRUE(cps[i].instance.same_num_devices_per_task);
  }
}

}  // namespace tensorflow
