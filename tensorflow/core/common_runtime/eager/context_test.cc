/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/context.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/common_runtime/composite_device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context_distributed_manager.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tsl/platform/refcount.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "absl/cleanup/cleanup.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/distributed_runtime/test_utils.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#endif  // !IS_MOBILE_PLATFORM

namespace tensorflow {
namespace {

using ::testing::HasSubstr;

typedef FunctionDefHelper FDH;

// Return a fake device.
static Device* CreateDevice(const std::string& type, int n) {
  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {}
    absl::Status Sync() override { return absl::OkStatus(); }
    Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name("/job:localhost/replica:0/task:0/device:" + type + ":" +
                std::to_string(n));
  attr.set_device_type(type);
  return new FakeDevice(attr);
}

class EagerContextTest : public ::testing::Test {
 public:
  EagerContext* context() { return context_.get(); }

  void InitContext(const SessionOptions& opts,
                   ContextDevicePlacementPolicy policy, bool async = false,
                   tsl::core::RefCountPtr<Rendezvous> rendezvous = nullptr) {
    ASSERT_EQ(context_, nullptr);
    if (!device_manager_) {
      InitDeviceManager();
    }
    context_ = core::RefCountPtr<EagerContext>(new EagerContext(
        opts, policy, async, device_manager_.get(),
        /*device_mgr_owned=*/false, std::move(rendezvous),
        /*cluster_flr=*/nullptr, /*collective_executor_mgr=*/nullptr,
        /*run_eager_op_as_function=*/true));
  }

 protected:
  void InitDeviceManager() {
    ASSERT_EQ(device_manager_, nullptr);
    device_manager_ = std::make_unique<DynamicDeviceMgr>();
    std::vector<std::unique_ptr<Device>> added_devices;
    added_devices.emplace_back(CreateDevice(DEVICE_CPU, 0));
    added_devices.emplace_back(CreateDevice(DEVICE_CPU, 1));
    added_devices.emplace_back(CreateDevice(DEVICE_GPU, 0));
    added_devices.emplace_back(CreateDevice(DEVICE_GPU, 1));
    added_devices.emplace_back(CreateDevice(DEVICE_TPU, 0));

    TF_CHECK_OK(device_manager_->AddDevices(std::move(added_devices)));
  }

  std::unique_ptr<DynamicDeviceMgr> device_manager_;
  core::RefCountPtr<EagerContext> context_;
};

TEST_F(EagerContextTest, CompositeDevice) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  std::vector<std::string> underlying_devices = {
      "/job:worker/replica:0/task:0/device:CPU:0",
      "/job:worker/replica:0/task:0/device:CPU:1"};
  CompositeDevice* composite_device_0 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(underlying_devices,
                                                      /*device_name=*/"",
                                                      &composite_device_0));
  EXPECT_EQ(composite_device_0->name(),
            "/job:localhost/replica:0/task:0/device:COMPOSITE:0");
  CompositeDevice* device = nullptr;
  TF_EXPECT_OK(context()->FindCompositeDeviceFromName(
      "/job:localhost/replica:0/task:0/device:COMPOSITE:0", &device));
  EXPECT_EQ(device, composite_device_0);
  CompositeDevice* composite_device_1 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(underlying_devices,
                                                      /*device_name=*/"",
                                                      &composite_device_1));
  EXPECT_EQ(composite_device_1, composite_device_0);
  underlying_devices.push_back("/job:worker/replica:0/task:0/device:CPU:2");
  CompositeDevice* composite_device_2 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(underlying_devices,
                                                      /*device_name=*/"",
                                                      &composite_device_2));
  EXPECT_EQ(composite_device_2->name(),
            "/job:localhost/replica:0/task:0/device:COMPOSITE:1");
  TF_EXPECT_OK(context()->FindCompositeDeviceFromName(
      "/job:localhost/replica:0/task:0/device:COMPOSITE:1", &device));
  EXPECT_EQ(device, composite_device_2);

  EXPECT_TRUE(absl::IsNotFound(context()->FindCompositeDeviceFromName(
      "/job:localhost/replica:0/task:0/device:COMPOSITE:2", &device)));
}

TEST_F(EagerContextTest, CompositeDeviceWithGivenName) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const std::vector<std::string> underlying_devices_0 = {
      "/job:worker/replica:0/task:0/device:CPU:0",
      "/job:worker/replica:0/task:0/device:CPU:1"};
  const std::string composite_device_name =
      "/job:worker1/replica:0/task:0/device:COMPOSITE:5";
  // Create a CompositeDevice with the given name.
  CompositeDevice* composite_device_0 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(
      underlying_devices_0, composite_device_name, &composite_device_0));
  EXPECT_EQ(composite_device_0->name(), composite_device_name);

  CompositeDevice* device = nullptr;
  TF_EXPECT_OK(
      context()->FindCompositeDeviceFromName(composite_device_name, &device));
  EXPECT_EQ(device, composite_device_0);

  std::vector<std::string> underlying_devices_1 = {
      "/job:worker/replica:0/task:0/device:CPU:1",
      "/job:worker/replica:0/task:0/device:CPU:2"};
  // Find a CompositeDevice with the given name.
  CompositeDevice* composite_device_1 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(
      underlying_devices_1, composite_device_0->name(), &composite_device_1));
  EXPECT_EQ(composite_device_1, composite_device_0);
}

TEST_F(EagerContextTest, AddFunctionDef) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const Tensor kTwo = test::AsScalar<int64_t>(2);
  const FunctionDef x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two));
}

TEST_F(EagerContextTest, AddFunctionDefRepeatSame) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const Tensor kTwo = test::AsScalar<int64_t>(2);
  const FunctionDef x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two));
  const FunctionDef x_times_two_copy = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two_copy));
}

TEST_F(EagerContextTest, AddFunctionDefRepeatDifferent) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const Tensor kTwo = test::AsScalar<int64_t>(2);
  const FunctionDef x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two));
  const Tensor kThree = test::AsScalar<int64_t>(3);
  // Same function name but body is different. This should error out.
  const FunctionDef x_times_two_copy = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kThree}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  absl::Status s = context()->AddFunctionDef(x_times_two_copy);
  EXPECT_FALSE(s.ok());
}

TEST_F(EagerContextTest, FunctionErrorRecovery) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT, /*async=*/true);
  const FunctionDef assert_and_identity = FDH::Define(
      // Name
      "AssertAndIdentity",
      // Args
      {"x: float", "condition: bool"},
      // Return values
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"assert"},
           "Assert",
           {"condition", "x"},
           {{"T", std::vector<DataType>{DT_FLOAT}}}},
          {{"y"},
           "Identity",
           {"x"},
           {{"T", DT_FLOAT}},
           /*dep=*/{"assert"}},
      });
  absl::Status s = context()->AddFunctionDef(assert_and_identity);
  auto fail_op = ImmediateOpPtr(context()->CreateOperation());
  TF_ASSERT_OK(fail_op->Reset("AssertAndIdentity",
                              "/job:localhost/replica:0/task:0/device:CPU:0"));
  Tensor float_tensor = test::AsScalar<float>(3.0);
  auto input_float = core::RefCountPtr<ImmediateExecutionTensorHandle>(
      context()->CreateLocalHandleFromTFTensor(
          float_tensor, context()->HostCPUName().c_str()));
  Tensor bool_tensor_false = test::AsScalar<bool>(false);
  auto input_bool_false = core::RefCountPtr<ImmediateExecutionTensorHandle>(
      context()->CreateLocalHandleFromTFTensor(
          bool_tensor_false, context()->HostCPUName().c_str()));
  TF_ASSERT_OK(fail_op->AddInput(input_float.get()));
  TF_ASSERT_OK(fail_op->AddInput(input_bool_false.get()));
  std::vector<AbstractTensorHandle*> retvals(1);
  int num_retvals = retvals.size();
  StatusGroup op_and_sync_status;
  op_and_sync_status.Update(
      fail_op->Execute(absl::MakeSpan(retvals), &num_retvals));
  op_and_sync_status.Update(context()->SyncExecutors());
  ASSERT_THAT(op_and_sync_status.as_summary_status().message(),
              HasSubstr("assertion failed"));
  if (retvals[0] != nullptr) {
    retvals[0]->Unref();
    retvals[0] = nullptr;
  }

  Tensor bool_tensor_true = test::AsScalar<bool>(true);
  auto input_bool_true = core::RefCountPtr<ImmediateExecutionTensorHandle>(
      context()->CreateLocalHandleFromTFTensor(
          bool_tensor_true, context()->HostCPUName().c_str()));
  auto success_op = ImmediateOpPtr(context()->CreateOperation());
  TF_ASSERT_OK(success_op->Reset(
      "AssertAndIdentity", "/job:localhost/replica:0/task:0/device:CPU:0"));
  TF_ASSERT_OK(success_op->AddInput(input_float.get()));
  TF_ASSERT_OK(success_op->AddInput(input_bool_true.get()));
  // A second run of the function should work, despite the previous failure.
  TF_ASSERT_OK(success_op->Execute(absl::MakeSpan(retvals), &num_retvals));
  TF_ASSERT_OK(context()->SyncExecutors());
  retvals[0]->Unref();
  retvals[0] = nullptr;
}

TEST_F(EagerContextTest, XlaCompileDeviceType) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT, /*async=*/true);
  const Tensor kTwo = test::AsScalar<int64_t>(2);
  const FunctionDef x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: int64"},
      // Return values
      {"y: int64"}, {},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"y"}, "Mul", {"x", "two"}, {{"T", DT_INT64}}},
      });

  absl::Status s = context()->AddFunctionDef(x_times_two);
  context()->SetJitCompileRewrite(true);
  auto op = ImmediateOpPtr(context()->CreateOperation());
  TF_ASSERT_OK(
      op->Reset("XTimesTwo", "/job:localhost/replica:0/task:0/device:CPU:0"));
  Tensor int_tensor = test::AsScalar<int64_t>(3);
  auto input_int = core::RefCountPtr<ImmediateExecutionTensorHandle>(
      context()->CreateLocalHandleFromTFTensor(
          int_tensor, context()->HostCPUName().c_str()));
  TF_ASSERT_OK(op->AddInput(input_int.get()));
  std::vector<AbstractTensorHandle*> retvals(1);
  int num_retvals = retvals.size();
  TF_ASSERT_OK(op->Execute(absl::MakeSpan(retvals), &num_retvals));
  retvals[0]->Unref();
  retvals[0] = nullptr;
}

TEST_F(EagerContextTest, LocalRendezvousCreation) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  auto rendezvous_creator = context()->RendezvousFactory();

  // Create a new rendezvous instance.
  // Initially its ref-count is 2:
  // one added upon rendezvous creation, the other one added by EagerContext.
  tsl::core::RefCountPtr<Rendezvous> rendezvous_1;
  TF_ASSERT_OK(rendezvous_creator(1, nullptr, &rendezvous_1));
  EXPECT_EQ(rendezvous_1->RefCount(), 1);

  // Create another rendezvous instance with the same step-id.
  // This would add one more ref-count to the existing rendezvous insteance
  // insted of creating a new instance.
  tsl::core::RefCountPtr<Rendezvous> rendezvous_2;
  TF_ASSERT_OK(rendezvous_creator(1, nullptr, &rendezvous_2));
  EXPECT_EQ(rendezvous_2->RefCount(), 2);

  // Caller releases rendezvous-1.
  rendezvous_1.reset();
  EXPECT_EQ(rendezvous_2->RefCount(), 1);

  // Caller releases rendezvous-2.
  tsl::core::WeakPtr<Rendezvous> weak2{rendezvous_2.get()};
  rendezvous_2.reset();
  EXPECT_EQ(weak2.GetNewRef(), nullptr);
}

void TestGlobalRendezvous(EagerContext* context, bool reuse_global_rendezvous) {
  auto rendezvous_creator = context->RendezvousFactory(reuse_global_rendezvous);
  tsl::core::RefCountPtr<Rendezvous> rendezvous_1;
  TF_ASSERT_OK(rendezvous_creator(-1, nullptr, &rendezvous_1));
  EXPECT_EQ(rendezvous_1->RefCount(), 2);
  tsl::core::RefCountPtr<Rendezvous> rendezvous_2;
  TF_ASSERT_OK(rendezvous_creator(-1, nullptr, &rendezvous_2));
  EXPECT_EQ(rendezvous_2->RefCount(), 3);

  // Global rendezvous's ref-count should be back to 1 after resetting.
  context->ResetGlobalRendezvousForFunction();

  tsl::core::RefCountPtr<Rendezvous> rendezvous_3;
  TF_ASSERT_OK(rendezvous_creator(-1, nullptr, &rendezvous_3));
  EXPECT_EQ(rendezvous_3->RefCount(), 2);
}

TEST_F(EagerContextTest, GlobalRendezvousCreation) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);

  TestGlobalRendezvous(context(), false);
}

TEST_F(EagerContextTest, ReuseGlobalRendezvous) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);

  TestGlobalRendezvous(context(), true);
}

TEST_F(EagerContextTest, IsTensorFlowExecutorThreadTest) {
  {
    std::unique_ptr<Thread> t(tsl::Env::Default()->StartThread(
        ThreadOptions(), "tf_test",
        []() { EXPECT_TRUE(internal::IsTensorFlowExecutorThread()); }));
  }
  {
    std::unique_ptr<Thread> t(tsl::Env::Default()->StartThread(
        ThreadOptions(), "other_name",
        []() { EXPECT_FALSE(internal::IsTensorFlowExecutorThread()); }));
  }
}

class NotificationRendezvous : public Rendezvous {
 public:
  explicit NotificationRendezvous(absl::Notification* n, int64_t* thread_id)
      : n_(n), destructor_thread_id_(thread_id) {}
  ~NotificationRendezvous() override {
    if (destructor_thread_id_) {
      *destructor_thread_id_ = tsl::Env::Default()->GetCurrentThreadId();
    }
    n_->Notify();
  }

  absl::Status Send(const ParsedKey& key, const Args& args, const Tensor& val,
                    const bool is_dead) override {
    return absl::OkStatus();
  }
  void RecvAsync(const ParsedKey& key, const Args& args,
                 DoneCallback done) override {}
  void StartAbort(const absl::Status& status) override {}

 private:
  absl::Notification* n_;
  int64_t* destructor_thread_id_ = nullptr;
};

TEST_F(EagerContextTest, ReleaseOnTfThread) {
  absl::Notification n;
  int64_t rendezvous_destructor_thread_id = 0;
  int64_t tf_thread_id = 0;
  auto rendezvous = core::RefCountPtr<Rendezvous>(
      new NotificationRendezvous(&n, &rendezvous_destructor_thread_id));
  InitDeviceManager();
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT, false,
              std::move(rendezvous));

  {
    std::unique_ptr<Thread> t(
        tsl::Env::Default()->StartThread(ThreadOptions(), "tf_test", [&]() {
          tf_thread_id = tsl::Env::Default()->GetCurrentThreadId();
          // Drop the last reference to context_. Because the thread is named
          // "tf_test", EagerContext::Release() will detach a thread to execute
          // Unref().
          context_.release()->Release();
        }));
  }

  // Wait deterministically for the detached thread to finish the deletion
  // by waiting for the EagerContext's rendezvous member to be destroyed.
  n.WaitForNotification();

  EXPECT_NE(rendezvous_destructor_thread_id, tf_thread_id);
}

#if !defined(IS_MOBILE_PLATFORM)
class AliveTestWorker : public TestWorkerInterface {
 public:
  explicit AliveTestWorker(absl::Status status) : status_(std::move(status)) {}
  void GetStatusAsync(CallOptions* opts, const GetStatusRequest* request,
                      GetStatusResponse* response, bool fail_fast,
                      StatusCallback done) override {
    done(status_);
  }

 private:
  absl::Status status_;
};

class ReleaseCountingWorkerCache : public TestWorkerCache {
 public:
  void ReleaseWorker(const std::string& target,
                     WorkerInterface* worker) override {
    ++release_count_;
    last_released_target_ = target;
    last_released_worker_ = worker;
  }

  int release_count() const { return release_count_; }
  const std::string& last_released_target() const {
    return last_released_target_;
  }
  WorkerInterface* last_released_worker() const {
    return last_released_worker_;
  }

 private:
  int release_count_ = 0;
  std::string last_released_target_;
  WorkerInterface* last_released_worker_ = nullptr;
};

class FakeServer : public ServerInterface {
 public:
  explicit FakeServer(WorkerCacheInterface* worker_cache,
                      DeviceMgr* local_device_mgr) {
    master_env_.worker_cache = worker_cache;
    worker_env_.device_mgr = local_device_mgr;
  }

  void InstallSessionMgr(std::unique_ptr<SessionMgr> session_mgr) {
    session_mgr_ = std::move(session_mgr);
    worker_env_.session_mgr = session_mgr_.get();
  }

  absl::Status Start() override { return absl::OkStatus(); }
  absl::Status Stop() override { return absl::OkStatus(); }
  absl::Status Join() override { return absl::OkStatus(); }
  const std::string target() const override { return "grpc://localhost:0"; }
  WorkerEnv* worker_env() override { return &worker_env_; }
  MasterEnv* master_env() override { return &master_env_; }
  absl::Status UpdateServerDef(const ServerDef& server_def) override {
    return absl::OkStatus();
  }
  absl::Status AddMasterEagerContextToEagerService(
      uint64_t context_id, EagerContext* context) override {
    return absl::OkStatus();
  }
  absl::Status SetCoordinationServiceAgentInstance(
      tsl::CoordinationServiceAgent* agent) override {
    return absl::OkStatus();
  }
  absl::Status StopCoordinationService() override { return absl::OkStatus(); }

 private:
  WorkerEnv worker_env_;
  MasterEnv master_env_;
  std::unique_ptr<SessionMgr> session_mgr_;  // Destroyed first.
};

TEST_F(EagerContextTest, CheckRemoteAliveWithoutServer) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  EagerContextDistributedManager distributed_mgr(context());
  bool is_alive = true;
  absl::Status s = distributed_mgr.CheckRemoteAlive(
      "/job:worker/replica:0/task:0", &is_alive);
  EXPECT_EQ(s.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_FALSE(is_alive);
}

TEST_F(EagerContextTest, CheckRemoteAliveUsesSessionMgrCache) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const std::string task = "/job:worker/replica:0/task:0";
  AliveTestWorker alive_worker(absl::OkStatus());
  TestWorkerCache stale_cache;  // Has no workers; must not be used.
  auto server = std::make_unique<FakeServer>(/*worker_cache=*/nullptr,
                                             context()->local_device_mgr());
  FakeServer* raw_server = server.get();
  TF_ASSERT_OK(context()->StoreCollectiveOpsServer(
      std::move(server), context()->local_device_mgr(),
      /*rpc_collective_executor_mgr=*/nullptr));
  auto teardown = absl::MakeCleanup([this, raw_server] {
    context_.reset();
    delete raw_server;
  });
  EagerContextDistributedManager distributed_mgr(context());

  bool is_alive = true;
  EXPECT_EQ(distributed_mgr.CheckRemoteAlive(task, &is_alive).code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_FALSE(is_alive);

  auto* updated = new ReleaseCountingWorkerCache;
  updated->AddWorker(task, &alive_worker);
  auto session_mgr = std::make_unique<SessionMgr>(
      raw_server->worker_env(), "/job:localhost/replica:0/task:0",
      std::make_unique<ReleaseCountingWorkerCache>(),
      /*worker_cache_factory=*/nullptr,
      /*coordination_handler=*/nullptr);
  session_mgr->ResetDefaultWorkerCache(updated);
  raw_server->InstallSessionMgr(std::move(session_mgr));
  raw_server->master_env()->worker_cache = &stale_cache;

  TF_ASSERT_OK(distributed_mgr.CheckRemoteAlive(task, &is_alive));
  EXPECT_TRUE(is_alive);
  EXPECT_EQ(updated->release_count(), 1);
  EXPECT_EQ(updated->last_released_worker(), &alive_worker);
}

TEST_F(EagerContextTest, CheckRemoteAliveReleasesWorker) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  ReleaseCountingWorkerCache worker_cache;
  AliveTestWorker alive_worker(absl::OkStatus());
  AliveTestWorker dead_worker(absl::UnavailableError("worker down"));
  const std::string alive_task = "/job:worker/replica:0/task:0";
  const std::string dead_task = "/job:worker/replica:0/task:1";
  worker_cache.AddWorker(alive_task, &alive_worker);
  worker_cache.AddWorker(dead_task, &dead_worker);

  auto server = std::make_unique<FakeServer>(&worker_cache,
                                             context()->local_device_mgr());
  FakeServer* raw_server = server.get();
  TF_ASSERT_OK(context()->StoreCollectiveOpsServer(
      std::move(server), context()->local_device_mgr(),
      /*rpc_collective_executor_mgr=*/nullptr));
  // EagerContext::~EagerContext releases server_ without deleting it because
  // GrpcServer does not support clean shutdown; destroy context_ first and then
  // delete FakeServer on all exit paths so leak checks stay clean.
  auto teardown = absl::MakeCleanup([this, raw_server] {
    context_.reset();
    delete raw_server;
  });

  EagerContextDistributedManager distributed_mgr(context());
  bool is_alive = false;
  TF_ASSERT_OK(distributed_mgr.CheckRemoteAlive(alive_task, &is_alive));
  EXPECT_TRUE(is_alive);
  EXPECT_EQ(worker_cache.release_count(), 1);
  EXPECT_EQ(worker_cache.last_released_target(), alive_task);
  EXPECT_EQ(worker_cache.last_released_worker(), &alive_worker);

  TF_ASSERT_OK(distributed_mgr.CheckRemoteAlive(dead_task, &is_alive));
  EXPECT_FALSE(is_alive);
  EXPECT_EQ(worker_cache.release_count(), 2);
  EXPECT_EQ(worker_cache.last_released_target(), dead_task);
  EXPECT_EQ(worker_cache.last_released_worker(), &dead_worker);

  is_alive = true;
  absl::Status unknown = distributed_mgr.CheckRemoteAlive(
      "/job:worker/replica:0/task:9", &is_alive);
  EXPECT_EQ(unknown.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(is_alive);
  EXPECT_EQ(worker_cache.release_count(), 2);
}
#endif  // !IS_MOBILE_PLATFORM

}  // namespace
}  // namespace tensorflow
