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

#include "tensorflow/core/distributed_runtime/collective_rma_distributed.h"

#include "google/protobuf/any.pb.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/test_utils.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/transport_options.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

// The only interesting method on CollectiveRemoteAccessDistributed
// that's not on CollectiveRemoteAccessLocal is RecvFromPeer which
// issues a RecvBufAsync call against a WorkerInterface.  That's all
// that's tested here.  Note that RecvFromPeer can do a
// DeviceResolverInterface::GetDeviceLocalityAsync call in preparation
// for the RecvBufAsync.

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

static int64 kStepId = 123;

class FakeWorker : public TestWorkerInterface {
 public:
  FakeWorker(const string& name, DeviceMgr* dev_mgr,
             DeviceResolverDistributed* dres)
      : name_(name),
        device_mgr_(dev_mgr),
        device_resolver_(dres),
        buf_rendezvous_(kStepId, dev_mgr) {}

  // Direct access to a BufRendezvous that holds whatever the remote
  // worker is supposed to have.
  BufRendezvous* buf_rendezvous() { return &buf_rendezvous_; }

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

  void RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                    RecvBufResponse* response, StatusCallback done) override {
    opts->SetCancelCallback([this]() {
      // Within this test the call is satisfied by a process-local
      // BufRendezvous table. In real application the BufRendezvous
      // would be on the other side of a network hop, so call
      // BufRendezvous::StartAbort() from a separate thread to be
      // more consistent with that situation and avoid mutex deadlock.
      SchedClosure([this]() {
        Env::Default()->SleepForMicroseconds(100);
        buf_rendezvous_.StartAbort(errors::Internal("Cancelled"));
      });
    });
    VLOG(2) << "ConsumeBuf key=" << request->buf_rendezvous_key()
            << " src_device=" << request->src_device()
            << " src_incarnation=" << request->src_incarnation();
    buf_rendezvous_.ConsumeBuf(
        request->buf_rendezvous_key(), request->src_device(),
        request->src_incarnation(),
        [opts, response, done](const Status& s, BufRendezvous::Hook* h) {
          if (s.ok()) {
            opts->ClearCancelCallback();
            // Since this is not really RDMA into pre-allocated memory send the
            // bytes in the response.
            RecvBufRespExtra extra;
            int64 num_bytes = h->prod_value->TotalBytes();
            extra.add_tensor_content(string(
                reinterpret_cast<const char*>(DMAHelper::base(h->prod_value)),
                num_bytes));
            response->mutable_transport_options()->PackFrom(extra);
          }
          done(s);
          if (h) BufRendezvous::DoneWithHook(h);
        });
  }

 private:
  string name_;
  DeviceMgr* device_mgr_;
  DeviceResolverDistributed* device_resolver_;
  BufRendezvous buf_rendezvous_;
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

class CollRMADistTest : public ::testing::Test {
 protected:
  CollRMADistTest()
      : work_queue_(
            std::make_shared<UnboundedWorkQueue>(Env::Default(), "test")) {}

  ~CollRMADistTest() override {
    for (DeviceMgr* dm : device_mgrs_) {
      delete dm;
    }
    for (auto it : dev_resolvers_) {
      delete it.second;
    }
    for (FakeWorker* w : workers_) {
      delete w;
    }
  }

  void SetUp() override {
    const int num_workers = 2;
    const int num_devices = 1;
    string device_type = "CPU";
    string dev0_worker_name;
    for (int w = 0; w < num_workers; ++w) {
      string name = strings::StrCat("/job:worker/replica:0/task:", w);
      if (w == 0) {
        dev0_worker_name = name;
      }
      DefineWorker(name, device_type, num_devices);
    }
    // All tests simulate requests from worker 0 to worker 1.
    rma_.reset(new CollectiveRemoteAccessDistributed(
        device_mgrs_[0], dev_resolvers_[dev0_worker_name], work_queue_, &wc_,
        kStepId));

    const int kNumElts = 8;
    expected_value_ = Tensor(DT_FLOAT, {kNumElts});
    to_tensor_ = Tensor(DT_FLOAT, {kNumElts});
    auto exp_alias = expected_value_.flat<float>();
    auto to_alias = to_tensor_.flat<float>();
    for (int i = 0; i < kNumElts; ++i) {
      exp_alias(i) = i;
      to_alias(i) = -1;
    }
  }

  void DefineWorker(const string& worker_name, const string& device_type,
                    int num_devices) {
    std::vector<std::unique_ptr<Device>> devices;
    for (int i = 0; i < num_devices; ++i) {
      devices.push_back(NewDevice(
          device_type,
          strings::StrCat(worker_name, "/device:", device_type, ":", i)));
    }
    DeviceMgr* dev_mgr = new StaticDeviceMgr(std::move(devices));
    device_mgrs_.push_back(dev_mgr);
    std::vector<string>* dv = &dev_by_task_[worker_name];
    dv->clear();
    for (auto d : dev_mgr->ListDevices()) {
      dv->push_back(d->name());
    }
    DeviceResolverDistributed* dev_res =
        new DeviceResolverDistributed(dev_mgr, &wc_, worker_name);
    dev_resolvers_[worker_name] = dev_res;
    FakeWorker* fw = new FakeWorker(worker_name, dev_mgr, dev_res);
    workers_.push_back(fw);
    wc_.AddWorker(worker_name, fw);
  }

  void RestartWorker(const string& worker_name, const string& device_type,
                     int num_devices) {
    auto it = dev_resolvers_.find(worker_name);
    if (it != dev_resolvers_.end()) {
      delete it->second;
      dev_resolvers_.erase(it);
    }
    DefineWorker(worker_name, device_type, num_devices);
  }

  void ValidateResultTensor() {
    ASSERT_EQ(expected_value_.NumElements(), to_tensor_.NumElements());
    for (int i = 0; i < to_tensor_.NumElements(); ++i) {
      EXPECT_FLOAT_EQ(expected_value_.flat<float>()(i),
                      to_tensor_.flat<float>()(i));
    }
  }

  FakeCache wc_;
  CancellationManager cm_;
  std::vector<DeviceMgr*> device_mgrs_;
  std::unordered_map<string, DeviceResolverDistributed*> dev_resolvers_;
  std::unordered_map<string, std::vector<string>> dev_by_task_;
  std::shared_ptr<UnboundedWorkQueue> work_queue_;
  std::vector<FakeWorker*> workers_;
  std::unique_ptr<CollectiveRemoteAccessDistributed> rma_;
  mutex mu_;
  int num_done_ GUARDED_BY(mu_);
  condition_variable done_;
  Tensor expected_value_;
  Tensor to_tensor_;
  CallOptions opts_;
  DeviceLocality device_locality_;
  AllocatorAttributes alloc_attr_;
};

TEST_F(CollRMADistTest, ProdFirstOK) {
  Notification consumer_note;
  Notification producer_note;
  Status consumer_status;
  Status producer_status;
  FakeWorker* wi = workers_[1];
  const string kBufKey = "fake_buf_key";
  wi->buf_rendezvous()->ProvideBuf(
      kBufKey, nullptr /*device*/, nullptr /*dev_ctx*/, &expected_value_,
      AllocatorAttributes(),
      [&producer_note, &producer_status](const Status& s) {
        producer_status.Update(s);
        producer_note.Notify();
      });
  Device* dst_device = nullptr;
  string dev_name = "CPU:0";
  TF_EXPECT_OK(device_mgrs_[0]->LookupDevice(dev_name, &dst_device));
  DeviceContext* to_device_ctx = nullptr;
  rma_->RecvFromPeer(
      "/job:worker/replica:0/task:1/device:" + dev_name,  // peer_dev
      "/job:worker/replica:0/task:1",                     // peer_task
      false,                                              // peer_is_local
      kBufKey, dst_device, to_device_ctx, alloc_attr_, &to_tensor_,
      device_locality_, 0 /*dev_to_dev_stream_index*/,
      [&consumer_status, &consumer_note](const Status& s) {
        consumer_status = s;
        consumer_note.Notify();
      });
  consumer_note.WaitForNotification();
  TF_EXPECT_OK(consumer_status);
  producer_note.WaitForNotification();
  TF_EXPECT_OK(producer_status);
  ValidateResultTensor();
}

TEST_F(CollRMADistTest, ConsFirstOK) {
  Notification consumer_note;
  Notification producer_note;
  Status consumer_status;
  Status producer_status;
  FakeWorker* wi = workers_[1];
  const string kBufKey = "fake_buf_key";
  Device* dst_device = nullptr;
  string dev_name = "CPU:0";
  TF_EXPECT_OK(device_mgrs_[0]->LookupDevice(dev_name, &dst_device));
  DeviceContext* to_device_ctx = nullptr;
  rma_->RecvFromPeer(
      "/job:worker/replica:0/task:1/device:" + dev_name,  // peer_dev
      "/job:worker/replica:0/task:1",                     // peer_task
      false,                                              // peer_is_local
      kBufKey, dst_device, to_device_ctx, alloc_attr_, &to_tensor_,
      device_locality_, 0 /*dev_to_dev_stream_index*/,
      [&consumer_status, &consumer_note](const Status& s) {
        consumer_status = s;
        consumer_note.Notify();
      });
  wi->buf_rendezvous()->ProvideBuf(
      kBufKey, nullptr /*device*/, nullptr /*dev_ctx*/, &expected_value_,
      AllocatorAttributes(),
      [&producer_note, &producer_status](const Status& s) {
        producer_status.Update(s);
        producer_note.Notify();
      });
  consumer_note.WaitForNotification();
  TF_EXPECT_OK(consumer_status);
  producer_note.WaitForNotification();
  TF_EXPECT_OK(producer_status);
  ValidateResultTensor();
}

TEST_F(CollRMADistTest, ConsFirstAbort) {
  Notification consumer_note;
  Status consumer_status;
  const string kBufKey = "fake_buf_key";
  Device* dst_device = nullptr;
  string dev_name = "CPU:0";
  TF_EXPECT_OK(device_mgrs_[0]->LookupDevice(dev_name, &dst_device));
  DeviceContext* to_device_ctx = nullptr;
  rma_->RecvFromPeer(
      "/job:worker/replica:0/task:1/device:" + dev_name,  // peer_dev
      "/job:worker/replica:0/task:1",                     // peer_task
      false,                                              // peer_is_local
      kBufKey, dst_device, to_device_ctx, alloc_attr_, &to_tensor_,
      device_locality_, 0 /*dev_to_dev_stream_index*/,
      [&consumer_status, &consumer_note](const Status& s) {
        consumer_status = s;
        consumer_note.Notify();
      });
  rma_->StartAbort(errors::Internal("Deliberate Failure"));
  consumer_note.WaitForNotification();
  EXPECT_EQ(consumer_status.error_message(), "Cancelled");
}

TEST_F(CollRMADistTest, WorkerRestart) {
  Notification consumer_note;
  Notification producer_note;
  Status consumer_status;
  Status producer_status;
  FakeWorker* wi = workers_[1];
  const string buf_key = "fake_buf_key";
  Device* dst_device = nullptr;
  string dev_name = "CPU:0";
  TF_EXPECT_OK(device_mgrs_[0]->LookupDevice(dev_name, &dst_device));
  DeviceContext* to_device_ctx = nullptr;
  rma_->RecvFromPeer(
      "/job:worker/replica:0/task:1/device:" + dev_name,  // peer_dev
      "/job:worker/replica:0/task:1",                     // peer_task
      false,                                              // peer_is_local
      buf_key, dst_device, to_device_ctx, alloc_attr_, &to_tensor_,
      device_locality_, 0 /*dev_to_dev_stream_index*/,
      [&consumer_status, &consumer_note](const Status& s) {
        consumer_status = s;
        consumer_note.Notify();
      });
  wi->buf_rendezvous()->ProvideBuf(
      buf_key, nullptr /*device*/, nullptr /*dev_ctx*/, &expected_value_,
      AllocatorAttributes(),
      [&producer_note, &producer_status](const Status& s) {
        producer_status.Update(s);
        producer_note.Notify();
      });
  consumer_note.WaitForNotification();
  TF_EXPECT_OK(consumer_status);
  producer_note.WaitForNotification();
  TF_EXPECT_OK(producer_status);
  ValidateResultTensor();

  // Restart task 1 and check that recv from task 1 to task 0 fails.
  RestartWorker("/job:worker/replica:0/task:1", "CPU", 1);
  Notification post_restart_note;
  rma_->RecvFromPeer(
      "/job:worker/replica:0/task:1/device:" + dev_name,  // peer_dev
      "/job:worker/replica:0/task:1",                     // peer_task
      false,                                              // peer_is_local
      buf_key, dst_device, to_device_ctx, alloc_attr_, &to_tensor_,
      device_locality_, 0 /*dev_to_dev_stream_index*/,
      [&consumer_status, &post_restart_note](const Status& s) {
        consumer_status = s;
        post_restart_note.Notify();
      });
  post_restart_note.WaitForNotification();
  EXPECT_TRUE(errors::IsFailedPrecondition(consumer_status));
}

}  // namespace
}  // namespace tensorflow
