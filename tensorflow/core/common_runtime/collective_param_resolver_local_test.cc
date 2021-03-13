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
#include "tensorflow/core/common_runtime/collective_param_resolver_local.h"

#include <atomic>

#include "tensorflow/core/common_runtime/collective_executor_mgr.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

#define NUM_DEVS 3

class CollectiveParamResolverLocalTest : public ::testing::Test {
 protected:
  CollectiveParamResolverLocalTest() {
    ConfigProto cp;
    SessionOptions options;
    task_name_ = "/job:localhost/replica:0/task:0";
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", NUM_DEVS});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(options, task_name_, &devices));
    device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));
    drl_.reset(new DeviceResolverLocal(device_mgr_.get()));
    ResetParamResolver();
  }

  void ResetParamResolver() {
    ConfigProto cp;
    prl_.reset(new CollectiveParamResolverLocal(cp, device_mgr_.get(),
                                                drl_.get(), task_name_));
  }

  void RunCompleteDefaultRanking(
      CollGroupParams group, const std::vector<DeviceAttributes>& attributes,
      const std::vector<int32>& gpu_ring_order,
      const std::vector<string>& expected_device_order) {
    if (!gpu_ring_order.empty()) {
      group.gpu_ring_order = "";
      for (int i = 0; i < static_cast<int32>(gpu_ring_order.size() - 1); ++i) {
        group.gpu_ring_order =
            strings::StrCat(group.gpu_ring_order, gpu_ring_order[i], ",");
      }
      group.gpu_ring_order =
          strings::StrCat(group.gpu_ring_order, gpu_ring_order.back());
    }
    VLOG(2) << "gpu_ring_order " << group.gpu_ring_order;
    prl_->CompleteDefaultRanking(attributes, &group);
    EXPECT_EQ(group.device_names, expected_device_order);
  }

  DeviceAttributes GetDeviceAttributes(const string& device_name) {
    Device* device = nullptr;
    TF_CHECK_OK(device_mgr_->LookupDevice(device_name, &device));
    return device->attributes();
  }

  string task_name_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<DeviceResolverLocal> drl_;
  std::unique_ptr<CollectiveParamResolverLocal> prl_;
};

TEST_F(CollectiveParamResolverLocalTest, CompleteDefaultRanking) {
  constexpr int kNumGpus = 8;
  CollGroupParams group;
  std::vector<DeviceAttributes> attributes(kNumGpus);
  group.device_type = DeviceType("GPU");
  group.num_tasks = 1;
  group.group_size = kNumGpus;
  std::unordered_set<int> clique1 = {0, 1, 6, 7};
  for (int gpu_idx = 0; gpu_idx < kNumGpus; ++gpu_idx) {
    group.task_names.push_back("/job:localhost/replica:0/task:0");
    group.device_names.push_back(strings::StrCat(
        "/job:localhost/replica:0/task:0/device:GPU:", gpu_idx));
    DeviceLocality locality;
    // Build localities so that 0,1,6,7 and 2,3,4,5 form 2 strongly connected
    // components.  Across components, connect 3 and 7.
    for (int link_idx = 0; link_idx < kNumGpus; ++link_idx) {
      if (gpu_idx == link_idx) continue;
      bool gpu_in_clique1 = clique1.find(gpu_idx) != clique1.end();
      bool link_in_clique1 = clique1.find(link_idx) != clique1.end();
      if ((gpu_in_clique1 && link_in_clique1) ||
          (!gpu_in_clique1 && !link_in_clique1)) {
        LocalLinks* links = locality.mutable_links();
        InterconnectLink* ilink = links->add_link();
        ilink->set_device_id(link_idx);
        ilink->set_strength(2);
      } else if ((gpu_idx == 3 && link_idx == 7) ||
                 (gpu_idx == 7 && link_idx == 3)) {
        LocalLinks* links = locality.mutable_links();
        InterconnectLink* ilink = links->add_link();
        ilink->set_device_id(link_idx);
        ilink->set_strength(1);
      }
    }
    *attributes[gpu_idx].mutable_locality() = locality;
  }
  RunCompleteDefaultRanking(group, attributes, {1, 3, 5, 7, 6, 4, 2, 0},
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
  RunCompleteDefaultRanking(group, attributes, {7, 6, 5, 4, 3, 2, 1, 0},
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
  RunCompleteDefaultRanking(group, attributes, {},
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
  CollectiveParams* cps[NUM_DEVS];
  Status statuses[NUM_DEVS];
  Notification note[NUM_DEVS];
  for (int i = 0; i < NUM_DEVS; ++i) {
    cps[i] = new CollectiveParams();
    CollectiveParams* cp = cps[i];
    cp->group.group_key = 1;
    cp->group.group_size = 3;
    cp->group.device_type = DeviceType("CPU");
    cp->group.num_tasks = 1;
    cp->instance.instance_key = 7;
    cp->instance.type = REDUCTION_COLLECTIVE;
    cp->instance.data_type = DataType(DT_FLOAT);
    cp->instance.shape = TensorShape({5});
    cp->instance.impl_details.subdiv_offsets.push_back(0);
    cp->is_source = false;
    Env::Default()->SchedClosure([this, i, cp, &note, &statuses]() {
      string device =
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
      prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp,
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
    ASSERT_EQ(cps[i]->group.device_names.size(), 3);
    for (int j = 0; j < NUM_DEVS; ++j) {
      EXPECT_EQ(
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", j),
          cps[i]->group.device_names[j]);
      EXPECT_TRUE(cps[i]->task.is_local[j]);
    }
    EXPECT_EQ(cps[i]->instance.impl_details.subdiv_source_rank.size(), 0);
    EXPECT_FALSE(cps[i]->is_source);
    EXPECT_EQ(cps[i]->default_rank, i);
    EXPECT_TRUE(cps[i]->group.same_num_devices_per_task);
    cps[i]->Unref();
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
  cp->instance.impl_details.subdiv_offsets.push_back(0);
  cp->is_source = is_source;
}

TEST_F(CollectiveParamResolverLocalTest, CompleteParamsBroadcast1Task) {
  constexpr int kInstanceKey = 5;
  CollectiveParams* cps[NUM_DEVS];
  Status statuses[NUM_DEVS];
  Notification note[NUM_DEVS];
  for (int i = 0; i < NUM_DEVS; ++i) {
    cps[i] = new CollectiveParams();
    CollectiveParams* cp = cps[i];
    InitializeCollectiveParamsForBroadcast(kInstanceKey, i, i == 1, cp);
    Env::Default()->SchedClosure([this, i, cp, &note, &statuses]() {
      string device =
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
      prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp,
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
    ASSERT_EQ(cps[i]->group.device_names.size(), 3);
    for (int j = 0; j < NUM_DEVS; ++j) {
      EXPECT_EQ(
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", j),
          cps[i]->group.device_names[j]);
      EXPECT_TRUE(cps[i]->task.is_local[j]);
    }
    EXPECT_EQ(cps[i]->is_source, (i == 1));
    EXPECT_EQ(cps[i]->default_rank, i);
    EXPECT_TRUE(cps[i]->group.same_num_devices_per_task);
    cps[i]->Unref();
  }
}

// If we don't mark any participant in a broadcast as the source, we essentially
// create a collective group with only broadcast recvs.  In that case, we should
// get an internal error from param resolution.
TEST_F(CollectiveParamResolverLocalTest, CompleteParamsBroadcastForgotSender) {
  constexpr int kInstanceKey = 8;
  CollectiveParams* cps[NUM_DEVS];
  Status statuses[NUM_DEVS];
  Notification note[NUM_DEVS];
  for (int i = 0; i < NUM_DEVS; ++i) {
    cps[i] = new CollectiveParams();
    CollectiveParams* cp = cps[i];
    InitializeCollectiveParamsForBroadcast(kInstanceKey, i, false, cp);
    Env::Default()->SchedClosure([this, i, cp, &note, &statuses]() {
      string device =
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
      prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp,
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
    cps[i]->Unref();
  }
}

CollectiveParams* MakeCollectiveParams(int group_key, int instance_key,
                                       bool is_source) {
  auto* cp = new CollectiveParams();
  cp->group.group_key = group_key;
  cp->group.group_size = NUM_DEVS;
  cp->group.device_type = DeviceType("CPU");
  cp->group.num_tasks = 1;
  cp->instance.instance_key = instance_key;
  // CompleteInstanceLocal only waits for the group for broadcasts.
  // Testing with broadcasts yields better coverage.
  cp->instance.type = BROADCAST_COLLECTIVE;
  cp->is_source = is_source;
  return cp;
}

TEST_F(CollectiveParamResolverLocalTest, AbortPendingGroup) {
  CancellationManager cancel_mgr;
  std::vector<CollectiveParams*> cp(NUM_DEVS - 1);
  BlockingCounter start(NUM_DEVS - 1);
  BlockingCounter done(NUM_DEVS - 1);
  for (int i = 0; i < NUM_DEVS - 1; ++i) {
    Env::Default()->SchedClosure([this, i, &cancel_mgr, &cp, &start, &done] {
      string device =
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
      cp[i] = MakeCollectiveParams(/*group_key*/ 100, /*instance_key*/ 100,
                                   /*is_source*/ i == 0);
      prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp[i], &cancel_mgr,
                                [&done, cp = cp[i]](const Status& s) {
                                  EXPECT_EQ(s.code(), error::ABORTED);
                                  EXPECT_EQ(s.error_message(), "__aborted__");
                                  done.DecrementCount();
                                  cp->Unref();
                                });
      start.DecrementCount();
    });
  }
  start.Wait();
  prl_->StartAbort(Status(error::ABORTED, "__aborted__"));
  done.Wait();
}

TEST_F(CollectiveParamResolverLocalTest, AbortPendingInstance) {
  CancellationManager cancel_mgr;
  std::vector<CollectiveParams*> cp(NUM_DEVS);
  int group_key = 100;
  int instance_key = 100;
  // First do a normal CompleteParamsAsync to complete the group;
  {
    BlockingCounter done(NUM_DEVS);
    for (int i = 0; i < NUM_DEVS; ++i) {
      Env::Default()->SchedClosure([this, group_key, instance_key, i,
                                    &cancel_mgr, &cp, &done] {
        string device =
            strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
        cp[i] = MakeCollectiveParams(group_key, instance_key,
                                     /*is_source*/ i == 0);
        prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp[i],
                                  &cancel_mgr,
                                  [&done, cp = cp[i]](const Status& s) {
                                    EXPECT_EQ(s.code(), error::OK);
                                    done.DecrementCount();
                                    cp->Unref();
                                  });
      });
    }
    done.Wait();
  }
  BlockingCounter start(NUM_DEVS - 1);
  BlockingCounter done(NUM_DEVS - 1);
  for (int i = 0; i < NUM_DEVS - 1; ++i) {
    Env::Default()->SchedClosure([this, group_key, instance_key, i, &cancel_mgr,
                                  &cp, &start, &done] {
      string device =
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
      cp[i] = MakeCollectiveParams(group_key, instance_key + 1,
                                   /*is_source*/ i == 0);
      prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp[i], &cancel_mgr,
                                [&done, cp = cp[i]](const Status& s) {
                                  EXPECT_EQ(s.code(), error::ABORTED);
                                  EXPECT_EQ(s.error_message(), "__aborted__");
                                  done.DecrementCount();
                                  cp->Unref();
                                });
      start.DecrementCount();
    });
  }
  start.Wait();
  prl_->StartAbort(Status(error::ABORTED, "__aborted__"));
  done.Wait();
}

TEST_F(CollectiveParamResolverLocalTest, CompleteParamsAfterAbortion) {
  CancellationManager cancel_mgr;
  int group_key = 100;
  int instance_key = 100;
  // First do a normal CompleteParamsAsync to complete the group;
  {
    std::vector<CollectiveParams*> cp(NUM_DEVS);
    BlockingCounter done(NUM_DEVS);
    for (int i = 0; i < NUM_DEVS; ++i) {
      Env::Default()->SchedClosure([this, group_key, instance_key, i,
                                    &cancel_mgr, &cp, &done] {
        string device =
            strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
        cp[i] = MakeCollectiveParams(group_key, instance_key,
                                     /*is_source*/ i == 0);
        prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp[i],
                                  &cancel_mgr,
                                  [&done, cp = cp[i]](const Status& s) {
                                    EXPECT_EQ(s.code(), error::OK);
                                    done.DecrementCount();
                                    cp->Unref();
                                  });
      });
    }
    done.Wait();
  }
  prl_->StartAbort(Status(error::ABORTED, "__aborted__"));

  auto complete_params = [this, &cancel_mgr](int group_key, int instance_key) {
    string device = "/job:localhost/replica:0/task:0/device:CPU:0";
    Notification done;
    auto* cp = MakeCollectiveParams(group_key, instance_key,
                                    /*is_source*/ true);
    core::ScopedUnref unref(cp);
    prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp, &cancel_mgr,
                              [&done](const Status& s) {
                                EXPECT_EQ(s.code(), error::ABORTED);
                                EXPECT_EQ(s.error_message(), "__aborted__");
                                done.Notify();
                              });
    done.WaitForNotification();
  };
  // It should error without waiting for the all following combinations:
  // - existing group, existing instance
  complete_params(group_key, instance_key);
  // - existing group, new instance
  complete_params(group_key, instance_key + 1);
  // - new group, new instance
  complete_params(group_key + 1, instance_key + 1);
}

TEST_F(CollectiveParamResolverLocalTest, AbortNormalCompleteParamsAsync) {
  // The concurrent nature makes it hard to test abortion, which can happen at
  // any moment. We don't have good options to inject control points into the
  // code to explicitly test every possible scenarios, so we run the test for
  // many times to have a better chance to cover different cases.
  CancellationManager cancel_mgr;
  std::atomic<int64> num_ok{0};
  for (int cnt = 0; cnt < 100; ++cnt) {
    // Launching threads that keep doing CompleteInstanceLocal.
    BlockingCounter done(NUM_DEVS);
    for (int i = 0; i < NUM_DEVS; ++i) {
      string device =
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
      Env::Default()->SchedClosure(
          [this, i, device, &num_ok, &cancel_mgr, &done] {
            int key = 100;
            while (true) {
              Status status;
              Notification n;
              auto* cp =
                  MakeCollectiveParams(/* group_key*/ key, /*instance_key*/ key,
                                       /*is_source*/ i == 0);
              prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp,
                                        &cancel_mgr,
                                        [&status, &n](const Status& s) {
                                          status = s;
                                          n.Notify();
                                        });
              n.WaitForNotification();
              cp->Unref();
              // The status should be either OK or the aborted status.
              if (!status.ok()) {
                EXPECT_EQ(status.code(), error::ABORTED);
                EXPECT_EQ(status.error_message(), "__aborted__");
                done.DecrementCount();
                return;
              }
              ++num_ok;
              ++key;
            }
          });
    }
    // Introduce a random delay up to 50ms, so that we're more likely to abort
    // on different code points each time.
    int64 delay_ms = random::New64() % 50000;
    Env::Default()->SleepForMicroseconds(delay_ms);
    prl_->StartAbort(Status(error::ABORTED, "__aborted__"));
    done.Wait();
    ResetParamResolver();
  }
  // There should be at least a few successes, otherwise the delay may be too
  // short and may not cover certain stages of param resolution.
  EXPECT_GT(num_ok.load(), 50);
}

}  // namespace tensorflow
