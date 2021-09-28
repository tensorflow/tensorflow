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

#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/test_collective_executor_mgr.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/test_utils.h"
#include "tensorflow/core/distributed_runtime/worker.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

static std::unique_ptr<Device> NewDevice(const string& type,
                                         const string& name) {
  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {}
    Status Sync() override { return Status::OK(); }
    Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  attr.mutable_locality()->set_numa_node(3);  // a non-default value
  attr.set_incarnation(random::New64());
  return absl::make_unique<FakeDevice>(attr);
}

class FakeCache : public TestWorkerCache {
 public:
  // Override the Locality methods to actually pass through to the
  // worker.
  bool GetDeviceLocalityNonBlocking(const string& device,
                                    DeviceLocality* locality) override {
    return false;
  }

  void GetDeviceLocalityAsync(const string& device, DeviceLocality* locality,
                              StatusCallback done) override {
    string task_name;
    string dev_part;
    if (!DeviceNameUtils::SplitDeviceName(device, &task_name, &dev_part)) {
      done(errors::Internal("failed to parse device name"));
      return;
    }
    auto it = workers_.find(task_name);
    if (it == workers_.end()) {
      done(errors::Internal("failed to find worker ", task_name));
      return;
    }
    WorkerInterface* wi = it->second;
    GetStatusRequest req;
    GetStatusResponse resp;
    Status status = wi->GetStatus(&req, &resp);
    if (!status.ok()) {
      done(status);
      return;
    }
    for (const auto& it : resp.device_attributes()) {
      if (it.name() == device) {
        *locality = it.locality();
        done(Status::OK());
        return;
      }
    }
    done(errors::Internal("device not found: ", device));
  }
};

class FakeNcclCommunicator : public NcclCommunicatorInterface {
 public:
  // We only need to define GenerateCommunicatorKey().
  string GenerateCommunicatorKey() override { return "mock-communicator-key"; }

  void Enqueue(std::shared_ptr<CollectiveContext> col_ctx,
               StatusCallback done) override {
    done(Status::OK());
  }

  void StartAbort(const Status& s) override {}
};

class DeviceResDistTest : public ::testing::Test {
 public:
  ~DeviceResDistTest() override {
    for (auto& name_param : cp_) {
      name_param.second->Unref();
    }
  }

 protected:
  void DefineWorkers(int num_workers, int num_devices,
                     const string& device_type, bool nccl) {
    for (int w = 0; w < num_workers; ++w) {
      string name = strings::StrCat("/job:worker/replica:0/task:", w);
      DefineWorker(name, device_type, num_devices, nccl);
    }
  }

  void DefineWorker(const string& worker_name, const string& device_type,
                    int num_devices, bool nccl) {
    ConfigProto config;
    config.mutable_experimental()->set_collective_group_leader(
        "/job:worker/replica:0/task:0");
    config.mutable_experimental()->set_collective_nccl(nccl);

    std::vector<std::unique_ptr<Device>> devices;
    for (int i = 0; i < num_devices; ++i) {
      devices.push_back(NewDevice(
          device_type,
          strings::StrCat(worker_name, "/device:", device_type, ":", i)));
    }
    device_mgrs_[worker_name] =
        absl::make_unique<StaticDeviceMgr>(std::move(devices));
    std::vector<string>* dv = &dev_by_task_[worker_name];
    dv->clear();
    for (auto* d : device_mgrs_[worker_name]->ListDevices()) {
      dv->push_back(d->name());
    }
    dev_resolvers_[worker_name] = absl::make_unique<DeviceResolverDistributed>(
        device_mgrs_[worker_name].get());
    cp_resolvers_[worker_name] =
        absl::make_unique<CollectiveParamResolverDistributed>(
            config, device_mgrs_[worker_name].get(),
            dev_resolvers_[worker_name].get(), &nccl_communicator_, &wc_,
            worker_name);
    auto worker_env = absl::make_unique<WorkerEnv>();
    worker_env->env = Env::Default();
    worker_env->local_devices = device_mgrs_[worker_name]->ListDevices();
    worker_env->device_mgr = device_mgrs_[worker_name].get();
    worker_env->collective_executor_mgr =
        absl::make_unique<TestCollectiveExecutorMgr>(
            cp_resolvers_[worker_name].get(), /*rma=*/nullptr);
    workers_[worker_name] = absl::make_unique<Worker>(worker_env.get());
    worker_envs_[worker_name] = std::move(worker_env);
    wc_.AddWorker(worker_name, workers_[worker_name].get());
  }

  void DefineCollectiveParams(int num_workers, int num_devices,
                              const string& device_type,
                              CollectiveType coll_type = REDUCTION_COLLECTIVE,
                              int source_rank = 0) {
    for (int wi = 0; wi < num_workers; ++wi) {
      string task_name = strings::StrCat("/job:worker/replica:0/task:", wi);
      for (int di = 0; di < num_devices; ++di) {
        int idx = wi * num_devices + di;
        string device_name =
            strings::StrCat(task_name, "/device:", device_type, ":", di);
        cp_[device_name] =
            CreateCollectiveParams(num_workers, num_devices, device_type,
                                   coll_type, idx == source_rank);
      }
    }
  }

  CollectiveParams* CreateCollectiveParams(int num_workers, int num_devices,
                                           const string& device_type,
                                           CollectiveType coll_type,
                                           bool is_source) {
    const int kGroupKey = 5;
    const int kInstanceKey = 3;
    auto* cp = new CollectiveParams();
    cp->is_source = is_source;
    cp->group.group_key = kGroupKey;
    cp->group.group_size = num_workers * num_devices;
    cp->group.device_type = DeviceType(device_type);
    cp->group.num_tasks = num_workers;
    cp->instance.instance_key = kInstanceKey;
    cp->instance.type = coll_type;
    cp->instance.data_type = DT_FLOAT;
    cp->instance.shape = TensorShape({64});
    cp->instance.impl_details.subdiv_offsets.push_back(0);
    return cp;
  }

  void IssueRequests(int num_workers, int num_devices) {
    {
      mutex_lock l(mu_);
      num_done_ = 0;
    }
    int group_size = num_workers * num_devices;
    for (int wi = 0; wi < num_workers; ++wi) {
      string task_name = strings::StrCat("/job:worker/replica:0/task:", wi);
      for (int di = 0; di < num_devices; ++di) {
        string device_name = strings::StrCat(task_name, "/device:CPU:", di);
        IssueRequest(task_name, device_name, group_size);
      }
    }
  }

  void IssueRequest(const string& task_name, const string& device_name,
                    int group_size) {
    Device* device = nullptr;
    TF_CHECK_OK(device_mgrs_[task_name]->LookupDevice(device_name, &device));
    CollectiveParams* cp = cp_[device_name];
    CollectiveParamResolverDistributed* cp_res = cp_resolvers_[task_name].get();
    CHECK(cp_res);
    cp_res->CompleteParamsAsync(
        device->attributes(), cp, &cm_,
        [this, device_name, group_size](const Status& s) {
          status_[device_name] = s;
          {
            mutex_lock l(mu_);
            ++num_done_;
            if (num_done_ == group_size) {
              done_.notify_all();
            }
          }
        });
  }

  void ValidateCollectiveParams(int num_workers, int num_devices) {
    int device_count = num_workers * num_devices;
    {
      mutex_lock l(mu_);
      if (num_done_ < device_count) {
        done_.wait(l);
      }
    }
    // Verify that all cp_ values get the same set of task and device
    // names, with unique default_rank in the expected order.
    const int dev_count = num_workers * num_devices;
    string dev0 = "/job:worker/replica:0/task:0/device:CPU:0";
    for (int wi = 0; wi < num_workers; ++wi) {
      string task_name = strings::StrCat("/job:worker/replica:0/task:", wi);
      for (int di = 0; di < num_devices; ++di) {
        string device_name = strings::StrCat(task_name, "/device:CPU:", di);
        int idx = wi * num_devices + di;
        TF_ASSERT_OK(status_[device_name]);
        EXPECT_EQ(cp_[device_name]->default_rank, idx);
        EXPECT_EQ(cp_[device_name]->group.members.size(), dev_count);
        EXPECT_EQ(cp_[device_name]->group.members[idx].device.name(),
                  device_name);
        EXPECT_EQ(cp_[device_name]->group.members[idx].task, task_name);
        ValidateDeviceResolver(*cp_[device_name], task_name);
        if (idx > 0) {
          EXPECT_EQ(cp_[dev0]->group.runtime_details.communicator_key,
                    cp_[device_name]->group.runtime_details.communicator_key);
          for (int i = 0; i < dev_count; ++i) {
            EXPECT_EQ(cp_[dev0]->group.members[i].device.name(),
                      cp_[device_name]->group.members[i].device.name());
            EXPECT_EQ(cp_[dev0]->group.members[i].task,
                      cp_[device_name]->group.members[i].task);
          }
        }
      }
    }
  }

  void ValidateDeviceResolver(const CollectiveParams& cp, const string& task) {
    for (const CollGroupMember& member : cp.group.members) {
      DeviceAttributes attributes;
      TF_ASSERT_OK(dev_resolvers_[task]->GetDeviceAttributes(
          member.device.name(), &attributes));
    }
  }

  void RestartWorker(int worker_idx, int num_workers, int num_devices,
                     const string& device_type, bool nccl,
                     CollectiveType coll_type = REDUCTION_COLLECTIVE,
                     bool is_source = false) {
    string worker_name =
        strings::StrCat("/job:worker/replica:0/task:", worker_idx);
    DefineWorker(worker_name, device_type, num_devices, nccl);
    for (int i = 0; i < num_devices; ++i) {
      string device_name =
          strings::StrCat(worker_name, "/device:", device_type, ":", i);
      if (cp_.find(device_name) != cp_.end()) {
        cp_[device_name]->Unref();
      }
      cp_[device_name] = CreateCollectiveParams(
          num_workers, num_devices, device_type, coll_type, is_source);
      status_.erase(device_name);
    }
  }

  FakeCache wc_;
  FakeNcclCommunicator nccl_communicator_;
  CancellationManager cm_;
  // Below are keyed by task names.
  absl::flat_hash_map<string, std::unique_ptr<DeviceMgr>> device_mgrs_;
  absl::flat_hash_map<string, std::unique_ptr<DeviceResolverDistributed>>
      dev_resolvers_;
  absl::flat_hash_map<string,
                      std::unique_ptr<CollectiveParamResolverDistributed>>
      cp_resolvers_;
  absl::flat_hash_map<string, std::vector<string>> dev_by_task_;
  absl::flat_hash_map<string, std::unique_ptr<WorkerEnv>> worker_envs_;
  absl::flat_hash_map<string, std::unique_ptr<Worker>> workers_;
  // Below are keyed by device names;
  absl::flat_hash_map<string, CollectiveParams*> cp_;
  absl::flat_hash_map<string, Status> status_;
  mutex mu_;
  int num_done_ TF_GUARDED_BY(mu_);
  condition_variable done_;
};

TEST_F(DeviceResDistTest, Workers1Devices1) {
  const int num_workers = 1;
  const int num_devices = 1;
  DefineWorkers(num_workers, num_devices, "CPU", /*nccl*/ false);
  DefineCollectiveParams(num_workers, num_devices, "CPU");
  IssueRequests(num_workers, num_devices);
  ValidateCollectiveParams(num_workers, num_devices);
}

TEST_F(DeviceResDistTest, Workers2Devices2) {
  const int num_workers = 2;
  const int num_devices = 2;
  DefineWorkers(num_workers, num_devices, "CPU", /*nccl*/ false);
  DefineCollectiveParams(num_workers, num_devices, "CPU");
  IssueRequests(num_workers, num_devices);
  ValidateCollectiveParams(num_workers, num_devices);
}

TEST_F(DeviceResDistTest, DifferentIncarnation) {
  const int num_workers = 2;
  const int num_devices = 1;
  DefineWorkers(num_workers, num_devices, "CPU", /*nccl*/ false);
  DefineCollectiveParams(num_workers, num_devices, "CPU");
  IssueRequests(num_workers, num_devices);
  RestartWorker(1, num_workers, num_devices, "CPU", /*nccl*/ false);
  const string task_name = "/job:worker/replica:0/task:1";
  const string device_name = absl::StrCat(task_name, "/device:CPU:0");
  IssueRequest(task_name, device_name, num_workers * num_devices);
  EXPECT_TRUE(errors::IsFailedPrecondition(status_[device_name]));
}

TEST_F(DeviceResDistTest, BroadcastSourceRank0) {
  const int num_workers = 2;
  const int num_devices = 2;
  const int source_rank = 0;
  DefineWorkers(num_workers, num_devices, "CPU", /*nccl*/ false);
  DefineCollectiveParams(num_workers, num_devices, "CPU", BROADCAST_COLLECTIVE,
                         source_rank);
  IssueRequests(num_workers, num_devices);
  ValidateCollectiveParams(num_workers, num_devices);
}

TEST_F(DeviceResDistTest, BroadcastSourceRank3) {
  const int num_workers = 2;
  const int num_devices = 2;
  const int source_rank = 3;
  DefineWorkers(num_workers, num_devices, "CPU", /*nccl*/ false);
  DefineCollectiveParams(num_workers, num_devices, "CPU", BROADCAST_COLLECTIVE,
                         source_rank);
  IssueRequests(num_workers, num_devices);
  ValidateCollectiveParams(num_workers, num_devices);
}

TEST_F(DeviceResDistTest, Workers4Devices3) {
  const int num_workers = 4;
  const int num_devices = 3;
  DefineWorkers(num_workers, num_devices, "CPU", true);
  DefineCollectiveParams(num_workers, num_devices, "CPU");
  IssueRequests(num_workers, num_devices);
  ValidateCollectiveParams(num_workers, num_devices);
}

}  // namespace
}  // namespace tensorflow
