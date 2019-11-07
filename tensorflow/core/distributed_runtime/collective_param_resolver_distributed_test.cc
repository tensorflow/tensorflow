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

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/test_utils.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
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
  return absl::make_unique<FakeDevice>(attr);
}

class FakeWorker : public TestWorkerInterface {
 public:
  FakeWorker(const string& name, DeviceMgr* dev_mgr,
             CollectiveParamResolverDistributed* cpres)
      : name_(name), device_mgr_(dev_mgr), param_resolver_(cpres) {}

  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response, bool fail_fast,
                      StatusCallback done) override {
    std::vector<DeviceAttributes> dev_attr;
    device_mgr_->ListDeviceAttributes(&dev_attr);
    for (const auto& da : dev_attr) {
      *response->add_device_attributes() = da;
    }
    done(Status::OK());
  }

  void CompleteGroupAsync(CallOptions* opts,
                          const CompleteGroupRequest* request,
                          CompleteGroupResponse* response,
                          StatusCallback done) override {
    param_resolver_->CompleteGroupAsync(request, response, &cm_, done);
  }

  void CompleteInstanceAsync(CallOptions* ops,
                             const CompleteInstanceRequest* request,
                             CompleteInstanceResponse* response,
                             StatusCallback done) override {
    param_resolver_->CompleteInstanceAsync(request, response, &cm_, done);
  }

 private:
  string name_;
  DeviceMgr* device_mgr_;
  CancellationManager cm_;
  CollectiveParamResolverDistributed* param_resolver_;
};

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

class DeviceResDistTest : public ::testing::Test {
 protected:
  DeviceResDistTest() {}

  ~DeviceResDistTest() override {
    for (DeviceMgr* dm : device_mgrs_) {
      delete dm;
    }
    for (auto it : dev_resolvers_) {
      delete it.second;
    }
    for (auto it : cp_resolvers_) {
      delete it.second;
    }
    for (FakeWorker* w : workers_) {
      delete w;
    }
  }

  void DefineWorkers(int num_workers, int num_devices,
                     const string& device_type, bool nccl) {
    ConfigProto config;
    for (int w = 0; w < num_workers; ++w) {
      string name = strings::StrCat("/job:worker/replica:0/task:", w);
      if (w == 0) {
        config.mutable_experimental()->set_collective_group_leader(name);
        if (nccl) {
          config.mutable_experimental()->set_collective_nccl(true);
        }
      }
      DefineWorker(config, name, device_type, num_devices);
    }
  }

  void DefineWorker(const ConfigProto& config, const string& worker_name,
                    const string& device_type, int num_devices) {
    std::vector<std::unique_ptr<Device>> devices;
    for (int i = 0; i < num_devices; ++i) {
      devices.push_back(NewDevice(
          device_type,
          strings::StrCat(worker_name, "/device:", device_type, ":", i)));
    }
    DeviceMgr* dev_mgr = new StaticDeviceMgr(std::move(devices));
    device_mgrs_.push_back(dev_mgr);
    std::vector<string>* dv = &dev_by_task_[worker_name];
    for (auto* d : dev_mgr->ListDevices()) {
      dv->push_back(d->name());
    }
    DeviceResolverDistributed* dev_res =
        new DeviceResolverDistributed(dev_mgr, &wc_, worker_name);
    dev_resolvers_[worker_name] = dev_res;
    CollectiveParamResolverDistributed* cp_res =
        new CollectiveParamResolverDistributed(config, dev_mgr, dev_res, &wc_,
                                               worker_name);
    cp_resolvers_[worker_name] = cp_res;
    FakeWorker* fw = new FakeWorker(worker_name, dev_mgr, cp_res);
    workers_.push_back(fw);
    wc_.AddWorker(worker_name, fw);
  }

  void DefineCollectiveParams(int num_workers, int num_devices) {
    const int kGroupKey = 5;
    const int kInstanceKey = 3;
    for (int wi = 0; wi < num_workers; ++wi) {
      string task_name = strings::StrCat("/job:worker/replica:0/task:", wi);
      for (int di = 0; di < num_devices; ++di) {
        string device_name = strings::StrCat(task_name, "/device:CPU:", di);
        cp_.push_back(CollectiveParams());
        CollectiveParams& cp = cp_.back();
        cp.group.group_key = kGroupKey;
        cp.group.group_size = num_workers * num_devices;
        cp.group.device_type = DEVICE_CPU;
        cp.group.num_tasks = num_workers;
        cp.instance.instance_key = kInstanceKey;
        cp.instance.type = REDUCTION_COLLECTIVE;
        cp.instance.data_type = DT_FLOAT;
        cp.instance.shape = TensorShape({64});
        cp.instance.impl_details.subdiv_offsets.push_back(0);
      }
    }
  }

  void IssueRequests(int num_workers, int num_devices) {
    const int device_count = num_workers * num_devices;
    {
      mutex_lock l(mu_);
      num_done_ = 0;
    }
    cp_.resize(device_count);
    status_.resize(device_count);
    int idx = 0;
    for (int wi = 0; wi < num_workers; ++wi) {
      for (int di = 0; di < num_devices; ++di) {
        IssueRequest(num_workers, num_devices, idx);
        ++idx;
      }
    }
  }

  void IssueRequest(int num_workers, int num_devices, int idx) {
    int device_count = num_workers * num_devices;
    int wi = idx / num_devices;
    int di = idx % num_devices;
    string task_name = strings::StrCat("/job:worker/replica:0/task:", wi);
    string device_name = strings::StrCat(task_name, "/device:CPU:", di);
    while (idx >= cp_.size()) {
      status_.resize(idx + 1);
      cp_.resize(idx + 1);
    }
    CollectiveParams* cp = &cp_[idx];
    CollectiveParamResolverDistributed* cp_res = cp_resolvers_[task_name];
    CHECK(cp_res);
    cp_res->CompleteParamsAsync(device_name, cp, &cm_,
                                [this, idx, device_count](const Status& s) {
                                  status_[idx] = s;
                                  {
                                    mutex_lock l(mu_);
                                    ++num_done_;
                                    if (num_done_ == device_count) {
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
    for (int wi = 0; wi < num_workers; ++wi) {
      string task_name = strings::StrCat("/job:worker/replica:0/task:", wi);
      for (int di = 0; di < num_devices; ++di) {
        string device_name = strings::StrCat(task_name, "/device:CPU:", di);
        int idx = wi * num_devices + di;
        TF_ASSERT_OK(status_[idx]);
        EXPECT_EQ(cp_[idx].default_rank, idx);
        EXPECT_EQ(cp_[idx].instance.device_names.size(), dev_count);
        EXPECT_EQ(cp_[idx].instance.device_names[idx], device_name);
        EXPECT_EQ(cp_[idx].instance.task_names[idx], task_name);
        if (idx > 0) {
          EXPECT_EQ(cp_[0].group.runtime_details.communicator_key,
                    cp_[idx].group.runtime_details.communicator_key);
          for (int i = 0; i < dev_count; ++i) {
            EXPECT_EQ(cp_[0].instance.device_names[i],
                      cp_[idx].instance.device_names[i]);
            EXPECT_EQ(cp_[0].instance.task_names[i],
                      cp_[idx].instance.task_names[i]);
          }
        }
      }
    }
  }

  FakeCache wc_;
  CancellationManager cm_;
  std::vector<DeviceMgr*> device_mgrs_;
  std::unordered_map<string, DeviceResolverDistributed*> dev_resolvers_;
  std::unordered_map<string, CollectiveParamResolverDistributed*> cp_resolvers_;
  std::unordered_map<string, std::vector<string>> dev_by_task_;
  std::vector<FakeWorker*> workers_;
  std::vector<CollectiveParams> cp_;
  std::vector<Status> status_;
  mutex mu_;
  int num_done_ GUARDED_BY(mu_);
  condition_variable done_;
};

TEST_F(DeviceResDistTest, Workers1Devices1) {
  const int num_workers = 1;
  const int num_devices = 1;
  DefineWorkers(num_workers, num_devices, "CPU", false);
  DefineCollectiveParams(num_workers, num_devices);
  IssueRequests(num_workers, num_devices);
  ValidateCollectiveParams(num_workers, num_devices);
}

TEST_F(DeviceResDistTest, Workers2Devices2) {
  const int num_workers = 2;
  const int num_devices = 2;
  DefineWorkers(num_workers, num_devices, "CPU", false);
  DefineCollectiveParams(num_workers, num_devices);
  IssueRequests(num_workers, num_devices);
  ValidateCollectiveParams(num_workers, num_devices);
}

#ifndef GOOGLE_CUDA
namespace {
// A mock NcclReducer for testing group runtime details initialization with CPU
// builds.  The only meaningful function in this class is
// `InitializeCollectiveGroupRuntimeDetails`.
class MockNcclReducer : public CollectiveImplementationInterface {
 public:
  MockNcclReducer() = default;

  Status InitializeCollectiveParams(CollectiveParams*) override {
    return Status::OK();
  }
  Status InitializeCollectiveContext(CollectiveContext*) override {
    return Status::OK();
  }
  Status InitializeCollectiveGroupRuntimeDetails(
      CollGroupRuntimeDetails* col_group_runtime_details) override {
    col_group_runtime_details->communicator_key = "mock-communicator-key";
    return Status::OK();
  }
  void Run(StatusCallback done) override {}
};
}  // namespace

REGISTER_COLLECTIVE(NcclReduce, MockNcclReducer);
#endif

TEST_F(DeviceResDistTest, Workers4Devices3) {
  const int num_workers = 4;
  const int num_devices = 3;
  DefineWorkers(num_workers, num_devices, "CPU", true);
  DefineCollectiveParams(num_workers, num_devices);
  IssueRequests(num_workers, num_devices);
  ValidateCollectiveParams(num_workers, num_devices);
}

}  // namespace
}  // namespace tensorflow
