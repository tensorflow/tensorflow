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

#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"

#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/test_utils.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// string -> Tensor<string>
Tensor V(const string& content) {
  Tensor tensor(DT_STRING, TensorShape({}));
  tensor.scalar<tstring>()() = content;
  return tensor;
}

// Tensor<string> -> string
string V(const Tensor& tensor) {
  CHECK_EQ(tensor.dtype(), DT_STRING);
  CHECK(TensorShapeUtils::IsScalar(tensor.shape()));
  return tensor.scalar<tstring>()();
}

Rendezvous::ParsedKey MakeKey(const string& s) {
  Rendezvous::ParsedKey key;
  CHECK(Rendezvous::ParseKey(s, &key).ok());
  return key;
}

namespace {
// A dummy worker interface implementation that simply triggers the callback
// with OK status for RecvTensor request.
class DummyWorker : public TestWorkerInterface {
 public:
  void RecvTensorAsync(CallOptions* opts, const RecvTensorRequest* request,
                       TensorResponse* response, StatusCallback done) override {
    SchedClosure([done = std::move(done)]() {
      // Simulate a random delay for RPC. This is needed to fill the entire
      // object buffer in `RpcRecvTensorFreeList` and trigger the destruction of
      // RPC call objects.
      const int64_t t_us = random::New64() % 100 * 1000;
      Env::Default()->SleepForMicroseconds(t_us);
      done(absl::OkStatus());
    });
  }
};

// Fake cache implementation for WorkerEnv.
class DummyWorkerCache : public WorkerCacheInterface {
  void ListWorkers(std::vector<string>* workers) const override {}
  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) const override {}
  WorkerInterface* GetOrCreateWorker(const string& target) override {
    if (dummy_remote_worker_ == nullptr) {
      // Ownership transferred to WorkerFreeList
      dummy_remote_worker_ = new DummyWorker;
    }
    return dummy_remote_worker_;
  }
  absl::Status GetEagerClientCache(
      std::unique_ptr<eager::EagerClientCache>* eager_client_cache) override {
    return errors::Unimplemented("Unimplemented.");
  }
  absl::Status GetCoordinationClientCache(
      std::unique_ptr<CoordinationClientCache>* coord_client_cache) override {
    return errors::Unimplemented("Unimplemented.");
  }
  bool GetDeviceLocalityNonBlocking(const string& device,
                                    DeviceLocality* locality) override {
    return false;
  }
  void GetDeviceLocalityAsync(const string& device, DeviceLocality* locality,
                              StatusCallback done) override {}

 private:
  DummyWorker* dummy_remote_worker_ = nullptr;
};

static Device* CreateDevice(const char* type, const char* name) {
  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {}
    absl::Status Sync() override { return absl::OkStatus(); }
    Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  return new FakeDevice(attr);
}

static DeviceMgr* CreateDeviceMgr() {
  std::unique_ptr<Device> d0(
      CreateDevice("CPU", "/job:mnist/replica:1/task:2/cpu:1"));
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  return new StaticDeviceMgr(std::move(devices));
}
}  // namespace

class RpcRendezvousMgrTest : public ::testing::Test {
 protected:
  RpcRendezvousMgrTest()
      : cache_(new DummyWorkerCache),
        worker_session_("rpc_session", "/job:mnist/replica:1/task:2",
                        std::unique_ptr<WorkerCacheInterface>(cache_),
                        std::unique_ptr<DeviceMgr>(CreateDeviceMgr()),
                        std::unique_ptr<GraphMgr>(), nullptr,
                        [](WorkerSession* worker_session, bool called,
                           DeviceMgr* remote_device_mgr) { return nullptr; }),
        rmgr_(&env) {
    env.env = Env::Default();
  }

  DummyWorkerCache* cache_;  // Managed by worker_session.
  WorkerEnv env;

  WorkerSession worker_session_;
  RpcRendezvousMgr rmgr_;
};

TEST_F(RpcRendezvousMgrTest, LocalSendRecv) {
  const int64_t step_id = 123;
  const Rendezvous::ParsedKey key = MakeKey(Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0)));
  {
    tsl::core::RefCountPtr<RemoteRendezvous> rendez = rmgr_.Find(step_id);
    TF_ASSERT_OK(rendez->Initialize(&worker_session_));
    Rendezvous::Args args;
    TF_ASSERT_OK(rendez->Send(key, args, V("peach"), false));
  }
  {
    Tensor val(DT_FLOAT);
    bool val_dead = false;
    TF_ASSERT_OK(rmgr_.RecvLocal(step_id, key, &val, &val_dead));
    EXPECT_EQ(V(val), "peach");
  }
  rmgr_.Cleanup(step_id);
}

TEST_F(RpcRendezvousMgrTest, LocalAbort) {
  const Rendezvous::ParsedKey key = MakeKey(Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0)));
  {  // Explicit Abort().
    const int64_t step_id = 123;
    tsl::core::RefCountPtr<RemoteRendezvous> rendez = rmgr_.Find(step_id);
    SchedClosure([this, rendez = rendez.GetNewRef()]() {
      env.env->SleepForMicroseconds(100 * 1000);
      rendez->StartAbort(errors::Aborted(""));
    });
    Tensor val(DT_STRING);
    bool val_dead = false;
    Rendezvous::Args args;
    TF_ASSERT_OK(rendez->Initialize(&worker_session_));
    EXPECT_TRUE(absl::IsAborted(rendez->Recv(key, args, &val, &val_dead)));
  }
  {  // Cleanup causes Abort().
    const int64_t step_id = 321;
    tsl::core::RefCountPtr<RemoteRendezvous> rendez = rmgr_.Find(step_id);
    SchedClosure([this, step_id]() {
      env.env->SleepForMicroseconds(100 * 1000);
      rmgr_.Cleanup(step_id);
    });
    Tensor val(DT_STRING);
    bool val_dead = false;
    Rendezvous::Args args;
    TF_ASSERT_OK(rendez->Initialize(&worker_session_));
    EXPECT_TRUE(absl::IsAborted(rendez->Recv(key, args, &val, &val_dead)));
  }
}

TEST_F(RpcRendezvousMgrTest, LocalCancel) {
  const Rendezvous::ParsedKey key = MakeKey(Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0)));
  auto* cm = new CancellationManager();
  const int64_t step_id = 123;
  tsl::core::RefCountPtr<RemoteRendezvous> rendez = rmgr_.Find(step_id);
  Notification n;
  SchedClosure([this, cm, &n]() {
    env.env->SleepForMicroseconds(100 * 1000);
    cm->StartCancel();
    n.Notify();
  });
  Tensor val(DT_STRING);
  bool val_dead = false;
  Rendezvous::Args args;
  args.cancellation_manager = cm;
  TF_ASSERT_OK(rendez->Initialize(&worker_session_));
  EXPECT_TRUE(absl::IsCancelled(rendez->Recv(key, args, &val, &val_dead)));
  n.WaitForNotification();
  delete cm;
}

TEST_F(RpcRendezvousMgrTest, CancelAfterReceived) {
  const Rendezvous::ParsedKey key = MakeKey(Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0)));
  auto* cm = new CancellationManager();
  const int64_t step_id = 123;
  tsl::core::RefCountPtr<RemoteRendezvous> rendez = rmgr_.Find(step_id);
  Notification n;
  SchedClosure([this, rendez = rendez.get(), key, cm, &n]() {
    env.env->SleepForMicroseconds(100 * 1000);
    TF_ASSERT_OK(rendez->Send(key, Rendezvous::Args(), V("peach"), false));
    cm->StartCancel();
    n.Notify();
  });
  Tensor val(DT_STRING);
  bool val_dead = false;
  Rendezvous::Args args;
  args.cancellation_manager = cm;
  TF_ASSERT_OK(rendez->Initialize(&worker_session_));
  TF_ASSERT_OK(rendez->Recv(key, args, &val, &val_dead));
  EXPECT_EQ(V(val), "peach");
  n.WaitForNotification();
  delete cm;
}

namespace {
class DummyDeviceContext : public DeviceContext {
 public:
  explicit DummyDeviceContext(int stream_id) : stream_id_(stream_id) {}
  ~DummyDeviceContext() override {}
  int stream_id() const { return stream_id_; }

 private:
  const int stream_id_;
};
}  // namespace

TEST_F(RpcRendezvousMgrTest, TransferDummyDeviceContext) {
  DummyDeviceContext* dc = new DummyDeviceContext(123);

  const int64_t step_id = 123;
  const Rendezvous::ParsedKey key = MakeKey(Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0)));
  {
    tsl::core::RefCountPtr<RemoteRendezvous> rendez = rmgr_.Find(step_id);
    Rendezvous::Args args;
    args.device_context = dc;
    TF_ASSERT_OK(rendez->Initialize(&worker_session_));
    TF_ASSERT_OK(rendez->Send(key, args, V("peach"), false));
  }
  {
    Notification n;
    rmgr_.RecvLocalAsync(
        step_id, key,
        [&n](const absl::Status& s, const Rendezvous::Args send_args,
             const Rendezvous::Args recv_args, const Tensor& val,
             bool is_dead) {
          auto send_dev_context =
              static_cast<DummyDeviceContext*>(send_args.device_context);
          CHECK_EQ(123, send_dev_context->stream_id());
          CHECK_EQ(V(val), "peach");
          n.Notify();
        });
    n.WaitForNotification();
  }
  rmgr_.Cleanup(step_id);
  dc->Unref();
}

TEST_F(RpcRendezvousMgrTest, RemoteRecvOne) {
  const int64_t step_id = 123;
  const Rendezvous::ParsedKey key = MakeKey(Rendezvous::CreateKey(
      "/job:worker/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0)));
  {
    tsl::core::RefCountPtr<RemoteRendezvous> rendez = rmgr_.Find(step_id);
    TF_ASSERT_OK(rendez->Initialize(&worker_session_));
    Rendezvous::Args args;

    Tensor val(DT_STRING);
    bool val_dead = false;

    TF_ASSERT_OK(rendez->Recv(key, args, &val, &val_dead));
  }
  rmgr_.Cleanup(step_id);
}

TEST_F(RpcRendezvousMgrTest, RemoteRecvAsyncMany) {
  const int64_t step_id = 123;
  const Rendezvous::ParsedKey key = MakeKey(Rendezvous::CreateKey(
      "/job:worker/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0)));
  {
    tsl::core::RefCountPtr<RemoteRendezvous> rendez = rmgr_.Find(step_id);
    TF_ASSERT_OK(rendez->Initialize(&worker_session_));
    Rendezvous::Args args;

    // Send a large number of async RPC requests to fill up the buffer in
    // `RpcRecvTensorFreeList`, in order to test deleting RPC call objects.
    int num_requests = 10000;
    Tensor val(DT_STRING);
    mutex mu_;
    absl::Status status = absl::OkStatus();
    BlockingCounter counter(num_requests);

    for (int i = 0; i < num_requests; i++) {
      rendez->RecvAsync(key, args,
                        [&mu_, &status, &counter](const absl::Status& s,
                                                  const Rendezvous::Args&,
                                                  const Rendezvous::Args&,
                                                  const Tensor&, const bool) {
                          {
                            mutex_lock l(mu_);
                            status.Update(s);
                          }
                          counter.DecrementCount();
                        });
    }
    counter.Wait();
    TF_ASSERT_OK(status);
  }
  rmgr_.Cleanup(step_id);
}

}  // namespace tensorflow
