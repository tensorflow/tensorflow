/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/coordination/coordination_service.h"

#include <string>
#include <utility>

#include "absl/synchronization/notification.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/distributed_runtime/test_utils.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"

namespace tensorflow {
namespace {
constexpr int kHeartbeatTimeoutMs = 5 * 1000;  // 5 seconds
constexpr char kCoordinationServiceType[] = "standalone";

class TestCoordinationClient : public CoordinationClient {
 public:
  TestCoordinationClient() = default;

  Status GetStatus() {
    mutex_lock l(mu_);
    return status_;
  }

  void RegisterWorkerAsync(CallOptions* opts,
                           const RegisterWorkerRequest* request,
                           RegisterWorkerResponse* response,
                           StatusCallback done) override {
    done(Status::OK());
  }

  void ReportErrorToAgentAsync(const ReportErrorToAgentRequest* request,
                               ReportErrorToAgentResponse* response,
                               StatusCallback done) override {
    mutex_lock l(mu_);
    status_ = Status(static_cast<errors::Code>(request->error_code()),
                     request->error_message());
    done(Status::OK());
  }

#define UNIMPLEMENTED(method)                                         \
  void method##Async(const method##Request* request,                  \
                     method##Response* response, StatusCallback done) \
      override {                                                      \
    done(errors::Unimplemented(#method "Async"));                     \
  }

  UNIMPLEMENTED(Heartbeat);
  UNIMPLEMENTED(WaitForAllTasks);
  UNIMPLEMENTED(ReportErrorToService);
  UNIMPLEMENTED(InsertKeyValue);
  UNIMPLEMENTED(GetKeyValue);
  UNIMPLEMENTED(DeleteKeyValue);

#undef UNIMPLEMENTED

 private:
  mutex mu_;
  Status status_ TF_GUARDED_BY(mu_);
};

class TestCoordinationClientCache : public CoordinationClientCache {
 public:
  void AddWorker(const std::string& target, CoordinationClient* client) {
    clients_.emplace(target, client);
  }

  CoordinationClient* GetClient(const string& target) override {
    auto it = clients_.find(target);
    if (it == clients_.end()) return nullptr;
    return it->second;
  }

  std::unique_ptr<CoordinationClient> GetOwnedClient(
      const string& target) override {
    LOG(ERROR) << "GetOwnedClient is not supported.";
    return nullptr;
  }

 private:
  std::unordered_map<std::string, CoordinationClient*> clients_;
};

class CoordinationServiceTest : public ::testing::Test {
 public:
  CoordinationServiceTest() {
    worker_env_.env = Env::Default();
    worker_cache_ = new TestWorkerCache;
    session_mgr_ = std::make_unique<SessionMgr>(
        &worker_env_, "/job:worker/replica:0/task:0/device:CPU:0",
        std::unique_ptr<WorkerCacheInterface>(worker_cache_),
        [](const ServerDef& server_def, WorkerCacheInterface** worker_cache) {
          *worker_cache = new TestWorkerCache;
          return Status::OK();
        });
    worker_env_.session_mgr = session_mgr_.get();
    device_mgr_ = std::make_unique<StaticDeviceMgr>(DeviceFactory::NewDevice(
        "CPU", SessionOptions(), "/job:worker/replica:0/task:0"));
    worker_env_.device_mgr = device_mgr_.get();
  }

 protected:
  WorkerEnv worker_env_;
  TestWorkerCache* worker_cache_;
  std::unique_ptr<SessionMgr> session_mgr_;
  std::unique_ptr<DeviceMgr> device_mgr_;
};

TEST_F(CoordinationServiceTest, TestStandaloneService) {
  const ServerDef& server_def = GetMultiClientServerDef("worker", 2);
  Status status = Status::OK();
  const uint64 w0_incarnation = random::New64();
  const uint64 w1_incarnation = random::New64();

  auto client_cache = std::make_unique<TestCoordinationClientCache>();
  TestCoordinationClient wi0;
  client_cache->AddWorker("/job:worker/replica:0/task:0", &wi0);
  TestCoordinationClient wi1;
  client_cache->AddWorker("/job:worker/replica:0/task:1", &wi1);
  std::unique_ptr<CoordinationServiceInterface> coord_service =
      CoordinationServiceInterface::EnableCoordinationService(
          kCoordinationServiceType, worker_env_.env, server_def,
          std::move(client_cache));

  absl::Notification register0;
  coord_service->RegisterWorker("worker", 0, w0_incarnation, [&](Status s) {
    TF_ASSERT_OK(s);
    register0.Notify();
  });
  register0.WaitForNotification();
  absl::Notification wait_for_all;
  coord_service->WaitForAllTasks("worker", 0, {}, [&](Status s) {
    TF_ASSERT_OK(s);
    wait_for_all.Notify();
  });
  // Not all workers are registered, so must not be notified here.
  ASSERT_FALSE(wait_for_all.HasBeenNotified());
  absl::Notification register1;
  coord_service->RegisterWorker("worker", 1, w1_incarnation, [&](Status s) {
    TF_ASSERT_OK(s);
    register1.Notify();
  });
  register1.WaitForNotification();
  coord_service->WaitForAllTasks("worker", 1, {},
                                 [&](Status s) { TF_ASSERT_OK(s); });
  // All tasks have registered.
  wait_for_all.WaitForNotification();

  TF_ASSERT_OK(coord_service->RecordHeartbeat("worker", 0, w0_incarnation));
  TF_ASSERT_OK(coord_service->RecordHeartbeat("worker", 1, w1_incarnation));
  EXPECT_TRUE(errors::IsInvalidArgument(
      coord_service->RecordHeartbeat("worker", 2, 0)));

  // Sending heartbeat with incarnation mismatch leads to Aborted error
  EXPECT_TRUE(
      errors::IsAborted(coord_service->RecordHeartbeat("worker", 1, 0)));
  EXPECT_TRUE(
      errors::IsAborted(coord_service->RecordHeartbeat("worker", 1, 0)));
  // Error is propagated to other workers
  EXPECT_TRUE(errors::IsAborted(wi0.GetStatus()));
}

TEST_F(CoordinationServiceTest, TestCoordinatedJobs) {
  ServerDef server_def = GetMultiClientServerDef("chief", 1);

  // Add a worker job with 2 tasks
  ClusterDef* cluster_def = server_def.mutable_cluster();
  JobDef* job_def = cluster_def->add_job();
  job_def->set_name("worker");
  job_def->mutable_tasks()->insert({0, "dummy address"});
  job_def->mutable_tasks()->insert({1, "dummy address"});

  // Add an evaluator job with 1 task
  job_def = cluster_def->add_job();
  job_def->set_name("evaluator");
  job_def->mutable_tasks()->insert({0, "dummy address"});

  CoordinationServiceConfig* configs =
      server_def.mutable_default_session_config()
          ->mutable_experimental()
          ->mutable_coordination_config();
  configs->mutable_coordinated_jobs()->Add("chief");
  configs->mutable_coordinated_jobs()->Add("worker");

  auto client_cache = std::make_unique<TestCoordinationClientCache>();
  TestCoordinationClient ci;
  client_cache->AddWorker("/job:chief/replica:0/task:0", &ci);
  TestCoordinationClient wi0;
  client_cache->AddWorker("/job:worker/replica:0/task:0", &wi0);
  TestCoordinationClient wi1;
  client_cache->AddWorker("/job:worker/replica:0/task:1", &wi1);
  TestCoordinationClient ei;
  client_cache->AddWorker("/job:evaluator/replica:0/task:0", &ei);
  std::unique_ptr<CoordinationServiceInterface> coord_service =
      CoordinationServiceInterface::EnableCoordinationService(
          kCoordinationServiceType, worker_env_.env, server_def,
          std::move(client_cache));

  absl::Notification register_chief;
  coord_service->RegisterWorker("chief", 0, 0, [&](Status s) {
    TF_ASSERT_OK(s);
    coord_service->WaitForAllTasks("chief", 0, {}, [&](Status s) {
      TF_ASSERT_OK(s);
      register_chief.Notify();
    });
  });
  absl::Notification register_worker0;
  coord_service->RegisterWorker("worker", 0, 0, [&](Status s) {
    TF_ASSERT_OK(s);
    coord_service->WaitForAllTasks("worker", 0, {}, [&](Status s) {
      TF_ASSERT_OK(s);
      register_worker0.Notify();
    });
  });
  absl::Notification register_worker1;
  coord_service->RegisterWorker("worker", 1, 0, [&](Status s) {
    TF_ASSERT_OK(s);
    coord_service->WaitForAllTasks("worker", 1, {}, [&](Status s) {
      TF_ASSERT_OK(s);
      register_worker1.Notify();
    });
  });
  // All tasks in the coordinated jobs have registered.
  register_chief.WaitForNotification();
  register_worker0.WaitForNotification();
  register_worker1.WaitForNotification();

  Status status = Status::OK();
  // Registering the evaluator task is unexpected
  absl::Notification register_evaluator;
  coord_service->RegisterWorker("evaluator", 0, 0, [&](Status s) {
    status = s;
    register_evaluator.Notify();
  });
  register_evaluator.WaitForNotification();
  EXPECT_TRUE(errors::IsInvalidArgument(status)) << status;
}

TEST_F(CoordinationServiceTest, TestWorkerHeartbeatTimeout) {
  ServerDef server_def = GetMultiClientServerDef("worker", 2);
  const uint64 w0_incarnation = random::New64();
  const uint64 w1_incarnation = random::New64();

  auto client_cache = std::make_unique<TestCoordinationClientCache>();
  TestCoordinationClient wi0;
  client_cache->AddWorker("/job:worker/replica:0/task:0", &wi0);
  TestCoordinationClient wi1;
  client_cache->AddWorker("/job:worker/replica:0/task:1", &wi1);

  auto coord_config = server_def.mutable_default_session_config()
                          ->mutable_experimental()
                          ->mutable_coordination_config();
  coord_config->set_service_type(kCoordinationServiceType);
  coord_config->set_heartbeat_timeout_in_ms(kHeartbeatTimeoutMs);
  std::unique_ptr<CoordinationServiceInterface> coord_service =
      CoordinationServiceInterface::EnableCoordinationService(
          kCoordinationServiceType, worker_env_.env, server_def,
          std::move(client_cache));

  absl::Notification register0;
  coord_service->RegisterWorker("worker", 0, w0_incarnation, [&](Status s) {
    TF_ASSERT_OK(s);
    register0.Notify();
  });
  register0.WaitForNotification();
  absl::Notification register1;
  coord_service->RegisterWorker("worker", 1, w1_incarnation, [&](Status s) {
    TF_ASSERT_OK(s);
    register1.Notify();
  });
  register1.WaitForNotification();

  // No heartbeat for a while, leader consider the worker as stale
  Env::Default()->SleepForMicroseconds(2 * kHeartbeatTimeoutMs * 1000);
  EXPECT_TRUE(errors::IsUnavailable(
      coord_service->RecordHeartbeat("worker", 0, w0_incarnation)));
  EXPECT_TRUE(errors::IsUnavailable(
      coord_service->RecordHeartbeat("worker", 1, w1_incarnation)));
}

TEST_F(CoordinationServiceTest, TestWorkerRestart) {
  const ServerDef& server_def = GetMultiClientServerDef("worker", 2);
  const uint64 w0_incarnation = random::New64();
  const uint64 w1_incarnation = random::New64();

  auto client_cache = std::make_unique<TestCoordinationClientCache>();
  TestCoordinationClient wi0;
  client_cache->AddWorker("/job:worker/replica:0/task:0", &wi0);
  TestCoordinationClient wi1;
  client_cache->AddWorker("/job:worker/replica:0/task:1", &wi1);
  std::unique_ptr<CoordinationServiceInterface> coord_service;
  coord_service = CoordinationServiceInterface::EnableCoordinationService(
      kCoordinationServiceType, worker_env_.env, server_def,
      std::move(client_cache));

  absl::Notification register0;
  coord_service->RegisterWorker("worker", 0, w0_incarnation, [&](Status s) {
    TF_ASSERT_OK(s);
    register0.Notify();
  });
  register0.WaitForNotification();
  absl::Notification register1;
  coord_service->RegisterWorker("worker", 1, w1_incarnation, [&](Status s) {
    TF_ASSERT_OK(s);
    register1.Notify();
  });
  register1.WaitForNotification();

  // Simulate worker restart scenario: trying to register to cluster again.
  absl::Notification n_repeated_register;
  coord_service->RegisterWorker("worker", 1, random::New64(), [&](Status s) {
    EXPECT_TRUE(errors::IsAborted(s));
    n_repeated_register.Notify();
  });
  n_repeated_register.WaitForNotification();
  // Aborted error is also propagated to other tasks in cluster.
  EXPECT_TRUE(errors::IsAborted(wi0.GetStatus())) << wi0.GetStatus();
}

TEST_F(CoordinationServiceTest, TestSetGetValues) {
  const ServerDef& server_def = GetMultiClientServerDef("worker", 1);
  Status status = Status::OK();

  auto client_cache = std::make_unique<TestCoordinationClientCache>();
  std::unique_ptr<CoordinationServiceInterface> coord_service;
  coord_service = CoordinationServiceInterface::EnableCoordinationService(
      kCoordinationServiceType, worker_env_.env, server_def,
      std::move(client_cache));

  // Simple key
  TF_ASSERT_OK(coord_service->InsertKeyValue("key0", "value0"));
  // Unix file like key path
  TF_ASSERT_OK(coord_service->InsertKeyValue("/path", "value"));
  TF_ASSERT_OK(coord_service->InsertKeyValue("/path/to/key1", "value1"));
  // Key with redundant slashes
  TF_ASSERT_OK(coord_service->InsertKeyValue("path/to//key2/", "value2"));
  // Error when repeatedly inserting the same key
  EXPECT_TRUE(errors::IsInvalidArgument(
      coord_service->InsertKeyValue("/path/to/key1/", "value2")));

  // Get simple key
  auto ret = coord_service->GetKeyValue("key0");
  TF_ASSERT_OK(ret.status());
  EXPECT_EQ(ret.ValueOrDie(), "value0");
  // Get key with redundant slashes
  ret = coord_service->GetKeyValue("path//to///key1////");
  EXPECT_EQ(ret.ValueOrDie(), "value1");

  // Delete single key-value
  TF_ASSERT_OK(coord_service->DeleteKeyValue("key0"));
  // Get key that is not available
  absl::Notification n;
  coord_service->GetKeyValueAsync(
      "key0", [&](const StatusOr<std::string>& status_or_value) {
        ret = status_or_value;
        n.Notify();
      });
  EXPECT_FALSE(n.HasBeenNotified());
  // Insert the previously deleted key again
  TF_ASSERT_OK(coord_service->InsertKeyValue("key0", "value0_new"));
  n.WaitForNotification();
  EXPECT_EQ(ret.ValueOrDie(), "value0_new");

  // Delete key-values recursively
  TF_ASSERT_OK(coord_service->DeleteKeyValue("/path"));
  // Get key that is not available
  absl::Notification n2;
  coord_service->GetKeyValueAsync(
      "/path/to/key1",
      [&](const StatusOr<std::string>& status_or_value) { n2.Notify(); });
  EXPECT_FALSE(n2.HasBeenNotified());
}

}  // namespace
}  // namespace tensorflow
