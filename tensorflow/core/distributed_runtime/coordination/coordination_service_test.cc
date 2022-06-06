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

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/compiler/xla/pjrt/distributed/protocol.pb.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/distributed_runtime/test_utils.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"

namespace tensorflow {
namespace {
using ::testing::EqualsProto;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;
using ::testing::proto::IgnoringRepeatedFieldOrdering;

constexpr absl::Duration kHeartbeatTimeout = absl::Seconds(2);
constexpr absl::Duration kShutdownBarrierTimeout = absl::Seconds(1);
constexpr char kCoordinationServiceType[] = "standalone";

KeyValueEntry CreateKv(const std::string& key, const std::string& value) {
  KeyValueEntry kv;
  kv.set_key(key);
  kv.set_value(value);
  return kv;
}

class TestCoordinationClient : public CoordinationClient {
 public:
  TestCoordinationClient() = default;

  Status GetStatus() {
    mutex_lock l(mu_);
    return status_;
  }

  void RegisterTaskAsync(CallOptions* opts, const RegisterTaskRequest* request,
                         RegisterTaskResponse* response,
                         StatusCallback done) override {
    done(OkStatus());
  }

  void ReportErrorToTaskAsync(CallOptions* call_opts,
                              const ReportErrorToTaskRequest* request,
                              ReportErrorToTaskResponse* response,
                              StatusCallback done) override {
    mutex_lock l(mu_);
    status_ = Status(static_cast<errors::Code>(request->error_code()),
                     request->error_message());
    done(OkStatus());
  }

#define UNIMPLEMENTED(method)                                         \
  void method##Async(const method##Request* request,                  \
                     method##Response* response, StatusCallback done) \
      override {                                                      \
    done(errors::Unimplemented(#method "Async"));                     \
  }

  UNIMPLEMENTED(WaitForAllTasks);
  UNIMPLEMENTED(ResetTask);
  UNIMPLEMENTED(ReportErrorToService);
  UNIMPLEMENTED(InsertKeyValue);
  UNIMPLEMENTED(TryGetKeyValue);
  UNIMPLEMENTED(GetKeyValueDir);
  UNIMPLEMENTED(DeleteKeyValue);
  UNIMPLEMENTED(Barrier);
  UNIMPLEMENTED(CancelBarrier);
#undef UNIMPLEMENTED

#define UNIMPLEMENTED_WITH_CALL_OPTS(method)                                 \
  void method##Async(CallOptions* call_opts, const method##Request* request, \
                     method##Response* response, StatusCallback done)        \
      override {                                                             \
    done(errors::Unimplemented(#method "Async"));                            \
  }

  UNIMPLEMENTED_WITH_CALL_OPTS(GetKeyValue);
  UNIMPLEMENTED_WITH_CALL_OPTS(Heartbeat);
  UNIMPLEMENTED_WITH_CALL_OPTS(ShutdownTask);
#undef UNIMPLEMENTED_WITH_CALL_OPTS

 private:
  mutex mu_;
  Status status_ TF_GUARDED_BY(mu_);
};

class TestCoordinationClientCache : public CoordinationClientCache {
 public:
  void AddTask(const std::string& target, CoordinationClient* client) {
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

class CoordinationBarrierTest : public ::testing::Test {
 protected:
  CoordinationBarrierTest() {
    // Set up fake cluster with 3 tasks.
    const int num_tasks = 3;
    const ServerDef& server_def = GetMultiClientServerDef("worker", num_tasks);
    auto client_cache = std::make_unique<TestCoordinationClientCache>();
    for (int i = 0; i < num_tasks; ++i) {
      CoordinatedTask task;
      task.set_job_name("worker");
      task.set_task_id(i);

      auto client = std::make_unique<TestCoordinationClient>();
      client_cache->AddTask(absl::StrCat("/job:worker/replica:0/task:", i),
                            client.get());

      tasks_.push_back(task);
      clients_.push_back(std::move(client));
    }

    coord_service_ = CoordinationServiceInterface::EnableCoordinationService(
        kCoordinationServiceType, Env::Default(), server_def,
        std::move(client_cache));
    // Register the tasks.
    for (int i = 0; i < num_tasks; ++i) {
      Status s = coord_service_->RegisterTask(tasks_[i], /*incarnation=*/0);
      if (!s.ok()) {
        LOG(FATAL) << "RegisterTask() failed in CoordinationBarrierTest(): "
                   << s;
      }
    }
  }

  CoordinationServiceInterface* GetCoordinationService() {
    return coord_service_.get();
  }
  CoordinatedTask GetTask(int i) { return tasks_[i]; }

 private:
  std::unique_ptr<CoordinationServiceInterface> coord_service_;
  std::vector<CoordinatedTask> tasks_;
  std::vector<std::unique_ptr<TestCoordinationClient>> clients_;
};

// Sets up coordination service that expects 2 worker tasks.
class CoordinateTwoTasksTest : public ::testing::Test {
 protected:
  CoordinateTwoTasksTest() {
    task_0_.set_job_name("worker");
    task_0_.set_task_id(0);
    task_1_.set_job_name("worker");
    task_1_.set_task_id(1);
  }

  // Set up coordination service.
  void EnableCoordinationService(bool has_service_to_client_connection = true,
                                 bool enable_shutdown_barrier = false) {
    ServerDef server_def = GetMultiClientServerDef("worker", /*num_tasks=*/2);
    auto client_cache = std::make_unique<TestCoordinationClientCache>();
    if (has_service_to_client_connection) {
      client_cache->AddTask("/job:worker/replica:0/task:0", &client_0_);
      client_cache->AddTask("/job:worker/replica:0/task:1", &client_1_);
    } else {
      client_cache = nullptr;
    }
    auto coord_config = server_def.mutable_default_session_config()
                            ->mutable_experimental()
                            ->mutable_coordination_config();
    coord_config->set_service_type(kCoordinationServiceType);
    coord_config->set_heartbeat_timeout_in_ms(kHeartbeatTimeout /
                                              absl::Milliseconds(1));
    if (enable_shutdown_barrier) {
      coord_config->set_shutdown_barrier_timeout_in_ms(kShutdownBarrierTimeout /
                                                       absl::Milliseconds(1));
    }
    // Init service.
    coord_service_ = CoordinationServiceInterface::EnableCoordinationService(
        kCoordinationServiceType, Env::Default(), server_def,
        std::move(client_cache));
  }

  CoordinatedTask task_0_;
  const uint64_t incarnation_0_ = random::New64();
  TestCoordinationClient client_0_;
  CoordinatedTask task_1_;
  const uint64_t incarnation_1_ = random::New64();
  TestCoordinationClient client_1_;
  std::unique_ptr<CoordinationServiceInterface> coord_service_;
};

// Construct fake device protos.
DeviceAttributes CreateTestTfDevice(absl::string_view name) {
  DeviceAttributes device;
  device.set_name(name);
  device.set_device_type("CPU");
  return device;
}

xla::DeviceProto CreateTestXlaDevice(absl::string_view name,
                                     const int local_id) {
  xla::DeviceProto device;
  device.set_name(name);
  device.set_local_device_ordinal(local_id);
  return device;
}

TEST_F(CoordinateTwoTasksTest, TestStandaloneService) {
  EnableCoordinationService();
  // Not specified in server def.
  CoordinatedTask task_2;
  task_2.set_job_name("worker");
  task_2.set_task_id(2);

  TF_ASSERT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  absl::Notification wait_for_all;
  coord_service_->WaitForAllTasks(task_0_, {}, [&](Status s) {
    TF_ASSERT_OK(s);
    wait_for_all.Notify();
  });
  // Not all tasks have registered, so must not be notified here.
  ASSERT_FALSE(wait_for_all.HasBeenNotified());
  TF_ASSERT_OK(coord_service_->RegisterTask(task_1_, incarnation_1_));
  coord_service_->WaitForAllTasks(task_1_, {},
                                  [&](Status s) { TF_ASSERT_OK(s); });
  // All tasks have registered.
  wait_for_all.WaitForNotification();

  TF_ASSERT_OK(coord_service_->RecordHeartbeat(task_0_, incarnation_0_));
  TF_ASSERT_OK(coord_service_->RecordHeartbeat(task_1_, incarnation_1_));
  EXPECT_TRUE(
      errors::IsInvalidArgument(coord_service_->RecordHeartbeat(task_2, 0)));

  // Sending heartbeat with incarnation mismatch leads to Aborted error.
  EXPECT_TRUE(errors::IsAborted(coord_service_->RecordHeartbeat(task_1_, 0)));
  EXPECT_TRUE(errors::IsAborted(coord_service_->RecordHeartbeat(task_1_, 0)));
  // Error is propagated to other tasks.
  EXPECT_TRUE(errors::IsAborted(client_0_.GetStatus()));
}

TEST(CoordinationServiceTest, TestCoordinatedJobs) {
  ServerDef server_def = GetMultiClientServerDef("chief", 1);
  CoordinatedTask chief;
  chief.set_job_name("chief");
  chief.set_task_id(0);
  CoordinatedTask task_0;
  task_0.set_job_name("worker");
  task_0.set_task_id(0);
  CoordinatedTask task_1;
  task_1.set_job_name("worker");
  task_1.set_task_id(1);
  CoordinatedTask evaluator;
  evaluator.set_job_name("evaluator");
  evaluator.set_task_id(0);

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
  client_cache->AddTask("/job:chief/replica:0/task:0", &ci);
  TestCoordinationClient wi0;
  client_cache->AddTask("/job:worker/replica:0/task:0", &wi0);
  TestCoordinationClient wi1;
  client_cache->AddTask("/job:worker/replica:0/task:1", &wi1);
  TestCoordinationClient ei;
  client_cache->AddTask("/job:evaluator/replica:0/task:0", &ei);
  std::unique_ptr<CoordinationServiceInterface> coord_service =
      CoordinationServiceInterface::EnableCoordinationService(
          kCoordinationServiceType, Env::Default(), server_def,
          std::move(client_cache));

  // Each coordinated task registers and waits for other tasks.
  absl::Notification register_chief;
  TF_ASSERT_OK(coord_service->RegisterTask(chief, /*incarnation=*/0));
  coord_service->WaitForAllTasks(chief, {}, [&](Status s) {
    TF_ASSERT_OK(s);
    register_chief.Notify();
  });
  absl::Notification register_task0;
  TF_ASSERT_OK(coord_service->RegisterTask(task_0, /*incarnation=*/0));
  coord_service->WaitForAllTasks(task_0, {}, [&](Status s) {
    TF_ASSERT_OK(s);
    register_task0.Notify();
  });
  absl::Notification register_task1;
  TF_ASSERT_OK(coord_service->RegisterTask(task_1, /*incarnation=*/0));
  coord_service->WaitForAllTasks(task_1, {}, [&](Status s) {
    TF_ASSERT_OK(s);
    register_task1.Notify();
  });
  // All tasks in the coordinated jobs have registered.
  register_chief.WaitForNotification();
  register_task0.WaitForNotification();
  register_task1.WaitForNotification();

  // Registering the evaluator task is unexpected
  Status status = coord_service->RegisterTask(evaluator, /*incarnation=*/0);
  EXPECT_TRUE(errors::IsInvalidArgument(status)) << status;
}

TEST(CoordinationServiceTest, RegisterTask_AlreadyConnected_Fails) {
  ServerDef server_def = GetMultiClientServerDef("worker", 1);
  JobDef* job_def = server_def.mutable_cluster()->add_job();
  job_def->set_name("worker");
  job_def->mutable_tasks()->insert({0, "dummy address"});
  CoordinatedTask task_0;
  task_0.set_job_name("worker");
  task_0.set_task_id(0);
  std::unique_ptr<CoordinationServiceInterface> coord_service =
      CoordinationServiceInterface::EnableCoordinationService(
          kCoordinationServiceType, Env::Default(), server_def,
          /*cache=*/nullptr);
  // Task connects to coordination service.
  TF_ASSERT_OK(coord_service->RegisterTask(task_0, /*incarnation=*/0));

  // Registration should fail since task already registered previously.
  const Status status = coord_service->RegisterTask(task_0, /*incarnation=*/0);

  EXPECT_TRUE(errors::IsAborted(status)) << status;
}

TEST(CoordinationServiceTest, RegisterTask_AlreadyInError_Fails) {
  ServerDef server_def = GetMultiClientServerDef("worker", 1);
  JobDef* job_def = server_def.mutable_cluster()->add_job();
  job_def->set_name("worker");
  job_def->mutable_tasks()->insert({0, "dummy address"});
  CoordinatedTask task_0;
  task_0.set_job_name("worker");
  task_0.set_task_id(0);
  std::unique_ptr<CoordinationServiceInterface> coord_service =
      CoordinationServiceInterface::EnableCoordinationService(
          kCoordinationServiceType, Env::Default(), server_def,
          /*cache=*/nullptr);
  // Task connects to coordination service.
  TF_ASSERT_OK(coord_service->RegisterTask(task_0, /*incarnation=*/0));
  // Arbitrarily set task to be in error.
  TF_ASSERT_OK(
      coord_service->ReportTaskError(task_0, errors::Internal("test_error")));

  // Registration should fail since task already registered previously.
  const Status status = coord_service->RegisterTask(task_0, /*incarnation=*/0);

  EXPECT_TRUE(errors::IsAborted(status)) << status;
}

TEST_F(CoordinateTwoTasksTest, TestTaskHeartbeatTimeout) {
  EnableCoordinationService();
  TF_ASSERT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  TF_ASSERT_OK(coord_service_->RegisterTask(task_1_, incarnation_1_));

  // No heartbeat for a while, leader considers the task as stale.
  Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));
  EXPECT_TRUE(errors::IsUnavailable(
      coord_service_->RecordHeartbeat(task_0_, incarnation_0_)));
  EXPECT_TRUE(errors::IsUnavailable(
      coord_service_->RecordHeartbeat(task_1_, incarnation_1_)));
}

TEST_F(CoordinateTwoTasksTest,
       HeartbeatTimeoutWithoutServerToClientConnection) {
  EnableCoordinationService(/*has_service_to_client_connection=*/false);
  TF_ASSERT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  TF_ASSERT_OK(coord_service_->RegisterTask(task_1_, incarnation_1_));

  // No heartbeat for a while, leader consider the task as stale.
  // Service stops and disconnects both tasks.
  Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));
  // Unexpected heartbeat from unregistered tasks since service state has been
  // reset.
  EXPECT_TRUE(errors::IsInvalidArgument(
      coord_service_->RecordHeartbeat(task_0_, incarnation_0_)));
  EXPECT_TRUE(errors::IsInvalidArgument(
      coord_service_->RecordHeartbeat(task_1_, incarnation_1_)));
}

TEST_F(CoordinateTwoTasksTest, TestTaskRestart) {
  EnableCoordinationService();
  TF_ASSERT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  TF_ASSERT_OK(coord_service_->RegisterTask(task_1_, incarnation_1_));

  // Simulate task restart scenario: trying to register to cluster again.
  Status s =
      coord_service_->RegisterTask(task_1_, /*incarnation=*/random::New64());
  EXPECT_TRUE(errors::IsAborted(s)) << s;
  // Aborted error is also propagated to other tasks in cluster.
  EXPECT_TRUE(errors::IsAborted(client_0_.GetStatus()))
      << client_0_.GetStatus();
}

TEST_F(CoordinateTwoTasksTest, TestSetGetValues) {
  EnableCoordinationService();

  // Simple key
  TF_ASSERT_OK(coord_service_->InsertKeyValue("key0", "value0"));
  // Unix file like key path
  TF_ASSERT_OK(coord_service_->InsertKeyValue("/path", "value"));
  TF_ASSERT_OK(coord_service_->InsertKeyValue("/path/to/key1", "value1"));
  // Key with redundant slashes
  TF_ASSERT_OK(coord_service_->InsertKeyValue("path/to//key2/", "value2"));
  // Error when repeatedly inserting the same key
  EXPECT_TRUE(errors::IsAlreadyExists(
      coord_service_->InsertKeyValue("/path/to/key1/", "value2")));

  // Get simple key
  absl::Notification n1;
  StatusOr<std::string> ret;
  coord_service_->GetKeyValueAsync(
      "key0", [&](const StatusOr<std::string>& status_or_value) {
        ret = status_or_value;
        n1.Notify();
      });
  n1.WaitForNotification();
  TF_ASSERT_OK(ret.status());
  EXPECT_EQ(ret.ValueOrDie(), "value0");
  // Get key with redundant slashes
  absl::Notification n2;
  coord_service_->GetKeyValueAsync(
      "path//to///key1////", [&](const StatusOr<std::string>& status_or_value) {
        ret = status_or_value;
        n2.Notify();
      });
  n2.WaitForNotification();
  EXPECT_EQ(ret.ValueOrDie(), "value1");

  // Delete single key-value
  TF_ASSERT_OK(coord_service_->DeleteKeyValue("key0"));
  // Get key that is not available
  absl::Notification n3;
  coord_service_->GetKeyValueAsync(
      "key0", [&](const StatusOr<std::string>& status_or_value) {
        ret = status_or_value;
        n3.Notify();
      });
  EXPECT_FALSE(n3.HasBeenNotified());
  // Insert the previously deleted key again
  TF_ASSERT_OK(coord_service_->InsertKeyValue("key0", "value0_new"));
  n3.WaitForNotification();
  EXPECT_EQ(ret.ValueOrDie(), "value0_new");

  // Delete key-values recursively
  TF_ASSERT_OK(coord_service_->DeleteKeyValue("/path"));
  // Get key that is not available
  auto n4 = std::make_shared<absl::Notification>();
  coord_service_->GetKeyValueAsync(
      "/path/to/key1",
      // Note: this callback will remain pending until it is cleaned up during
      // service shutdown. Hence, we use a shared pointer for notification so
      // that the it will not be deallocated before the pending callback is
      // cleaned up.
      [n4](const StatusOr<std::string>& status_or_value) { n4->Notify(); });
  EXPECT_FALSE(n4->HasBeenNotified());
}

TEST(CoordinationServiceTest, TryGetKeyValue) {
  const ServerDef& server_def = GetMultiClientServerDef("worker", 1);
  auto client_cache = std::make_unique<TestCoordinationClientCache>();
  std::unique_ptr<CoordinationServiceInterface> coord_service =
      CoordinationServiceInterface::EnableCoordinationService(
          kCoordinationServiceType, Env::Default(), server_def,
          std::move(client_cache));

  // Try to get nonexistent key.
  StatusOr<std::string> result = coord_service->TryGetKeyValue("test_key");
  EXPECT_TRUE(errors::IsNotFound(result.status()));

  // Insert key value.
  TF_ASSERT_OK(coord_service->InsertKeyValue("test_key", "test_value"));
  result = coord_service->TryGetKeyValue("test_key");
  EXPECT_EQ(result.ValueOrDie(), "test_value");

  // Delete Key, and try to get the key again.
  TF_ASSERT_OK(coord_service->DeleteKeyValue("test_key"));
  result = coord_service->TryGetKeyValue("test_key");
  EXPECT_TRUE(errors::IsNotFound(result.status()));
}

TEST_F(CoordinateTwoTasksTest, GetKeyValueDir_SingleValueInDirectory) {
  EnableCoordinationService();
  KeyValueEntry kv = CreateKv("dir/path", "value0");
  TF_ASSERT_OK(coord_service_->InsertKeyValue(kv.key(), kv.value()));

  std::vector<KeyValueEntry> result = coord_service_->GetKeyValueDir("dir");

  EXPECT_THAT(result, UnorderedElementsAre(EqualsProto(kv)));
}

TEST_F(CoordinateTwoTasksTest, GetKeyValueDir_MultipleValuesInDirectory) {
  EnableCoordinationService();
  KeyValueEntry kv = CreateKv("dir/path", "value0");
  KeyValueEntry kv2 = CreateKv("dir/path2", "value1");
  // Placed in nested subdirectory.
  KeyValueEntry kv_sub = CreateKv("dir/sub_dir/path", "value_sub");
  TF_ASSERT_OK(coord_service_->InsertKeyValue(kv.key(), kv.value()));
  TF_ASSERT_OK(coord_service_->InsertKeyValue(kv2.key(), kv2.value()));
  TF_ASSERT_OK(coord_service_->InsertKeyValue(kv_sub.key(), kv_sub.value()));

  std::vector<KeyValueEntry> result = coord_service_->GetKeyValueDir("dir");

  EXPECT_THAT(result, UnorderedElementsAre(EqualsProto(kv), EqualsProto(kv2),
                                           EqualsProto(kv_sub)));
}

TEST_F(CoordinateTwoTasksTest, GetKeyValueDir_Empty_ReturnsEmptyList) {
  EnableCoordinationService();

  std::vector<KeyValueEntry> result = coord_service_->GetKeyValueDir("dir");

  EXPECT_THAT(result, IsEmpty());
}

TEST_F(CoordinateTwoTasksTest, GetKeyValueDir_WrongDir_ReturnsEmptyList) {
  EnableCoordinationService();
  // Wrong directory.
  TF_ASSERT_OK(coord_service_->InsertKeyValue("dir0/path", "value0"));

  std::vector<KeyValueEntry> result = coord_service_->GetKeyValueDir("dir");

  EXPECT_THAT(result, IsEmpty());
}

TEST_F(CoordinateTwoTasksTest, GetKeyValueDir_WrongDirPrefix_ReturnsEmptyList) {
  EnableCoordinationService();
  // Check that we don't match with nested subdirectories with the wrong prefix.
  TF_ASSERT_OK(coord_service_->InsertKeyValue("wrong_dir/dir/path", "value0"));

  std::vector<KeyValueEntry> result = coord_service_->GetKeyValueDir("dir");

  EXPECT_THAT(result, IsEmpty());
}

TEST_F(CoordinateTwoTasksTest,
       GetKeyValueDir_NonDirectoryPrefix_ReturnsEmptyList) {
  EnableCoordinationService();
  // Wrong directory.
  TF_ASSERT_OK(coord_service_->InsertKeyValue("dir_key", "value0"));

  std::vector<KeyValueEntry> result = coord_service_->GetKeyValueDir("dir");

  EXPECT_THAT(result, IsEmpty());
}

TEST_F(CoordinateTwoTasksTest,
       GetKeyValueDir_NonDirectoryKey_ReturnsEmptyList) {
  EnableCoordinationService();
  // Insert same key that is not a directory.
  TF_ASSERT_OK(coord_service_->InsertKeyValue("dir", "value0"));

  std::vector<KeyValueEntry> result = coord_service_->GetKeyValueDir("dir");

  EXPECT_THAT(result, IsEmpty());
}

}  // namespace

// Verify that coordination service can gather each task's device info and
// propagate the aggregated cluster device info correctly.
TEST(CoordinationServiceTest, ListClusterDevices_TfDevice) {
  const ServerDef& server_def = GetMultiClientServerDef("worker", 3);
  CoordinatedTask task_0;
  task_0.set_job_name("worker");
  task_0.set_task_id(0);
  CoordinatedTask task_1;
  task_1.set_job_name("worker");
  task_1.set_task_id(1);
  CoordinatedTask task_2;
  task_2.set_job_name("worker");
  task_2.set_task_id(2);
  Status status = OkStatus();
  auto client_cache = std::make_unique<TestCoordinationClientCache>();
  std::unique_ptr<CoordinationServiceInterface> coord_service =
      CoordinationServiceInterface::EnableCoordinationService(
          kCoordinationServiceType, Env::Default(), server_def,
          std::move(client_cache));
  absl::Notification n;
  // Map fake devices to each task.
  CoordinationServiceDeviceInfo local_devices_0;
  CoordinationServiceDeviceInfo local_devices_1;
  CoordinationServiceDeviceInfo local_devices_2;
  *local_devices_0.mutable_tf()->mutable_devices()->Add() =
      CreateTestTfDevice("task0_device0");
  *local_devices_0.mutable_tf()->mutable_devices()->Add() =
      CreateTestTfDevice("task0_device1");
  *local_devices_1.mutable_tf()->mutable_devices()->Add() =
      CreateTestTfDevice("task1_device0");
  *local_devices_2.mutable_tf()->mutable_devices()->Add() =
      CreateTestTfDevice("task2_device0");

  // Each task sends its device info.
  CoordinationServiceDeviceInfo cluster_devices;
  coord_service->WaitForAllTasks(task_0, local_devices_0,
                                 [&](Status s) { TF_ASSERT_OK(s); });
  coord_service->WaitForAllTasks(task_1, local_devices_1,
                                 [&](Status s) { TF_ASSERT_OK(s); });
  coord_service->WaitForAllTasks(task_2, local_devices_2, [&](Status s) {
    TF_ASSERT_OK(s);
    // Gather the cluster device info.
    cluster_devices = coord_service->ListClusterDevices();
    n.Notify();
  });
  n.WaitForNotification();

  CoordinationServiceDeviceInfo expected_cluster_devices;
  auto expected_devices =
      expected_cluster_devices.mutable_tf()->mutable_devices();
  expected_devices->Add(local_devices_0.mutable_tf()->devices().begin(),
                        local_devices_0.mutable_tf()->devices().end());
  expected_devices->Add(local_devices_1.mutable_tf()->devices().begin(),
                        local_devices_1.mutable_tf()->devices().end());
  expected_devices->Add(local_devices_2.mutable_tf()->devices().begin(),
                        local_devices_2.mutable_tf()->devices().end());
  EXPECT_THAT(cluster_devices, IgnoringRepeatedFieldOrdering(
                                   EqualsProto(expected_cluster_devices)));
}

TEST(CoordinationServiceTest, ListClusterDevices_XlaDevice) {
  const ServerDef& server_def = GetMultiClientServerDef("worker", 3);
  CoordinatedTask task_0;
  task_0.set_job_name("worker");
  task_0.set_task_id(0);
  CoordinatedTask task_1;
  task_1.set_job_name("worker");
  task_1.set_task_id(1);
  CoordinatedTask task_2;
  task_2.set_job_name("worker");
  task_2.set_task_id(2);
  Status status = OkStatus();
  auto client_cache = std::make_unique<TestCoordinationClientCache>();
  std::unique_ptr<CoordinationServiceInterface> coord_service =
      CoordinationServiceInterface::EnableCoordinationService(
          kCoordinationServiceType, Env::Default(), server_def,
          std::move(client_cache));
  absl::Notification n;
  // Map fake devices to each task.
  CoordinationServiceDeviceInfo local_devices_0;
  CoordinationServiceDeviceInfo local_devices_1;
  CoordinationServiceDeviceInfo local_devices_2;
  xla::LocalTopologyProto local_0;
  xla::LocalTopologyProto local_1;
  xla::LocalTopologyProto local_2;
  local_0.set_node_id(0);
  local_1.set_node_id(1);
  local_2.set_node_id(2);
  *local_0.add_devices() = CreateTestXlaDevice("task0_device0", 0);
  *local_0.add_devices() = CreateTestXlaDevice("task0_device1", 1);
  *local_1.add_devices() = CreateTestXlaDevice("task1_device0", 0);
  *local_2.add_devices() = CreateTestXlaDevice("task2_device0", 0);
  *local_devices_0.mutable_xla()->mutable_devices()->add_nodes() = local_0;
  *local_devices_1.mutable_xla()->mutable_devices()->add_nodes() = local_1;
  *local_devices_2.mutable_xla()->mutable_devices()->add_nodes() = local_2;

  // Each task sends its device info.
  CoordinationServiceDeviceInfo cluster_devices;
  coord_service->WaitForAllTasks(task_0, local_devices_0,
                                 [&](Status s) { TF_ASSERT_OK(s); });
  coord_service->WaitForAllTasks(task_1, local_devices_1,
                                 [&](Status s) { TF_ASSERT_OK(s); });
  coord_service->WaitForAllTasks(task_2, local_devices_2, [&](Status s) {
    TF_ASSERT_OK(s);
    // Gather the cluster device info.
    cluster_devices = coord_service->ListClusterDevices();
    n.Notify();
  });
  n.WaitForNotification();

  CoordinationServiceDeviceInfo expected_cluster_devices;
  local_0.mutable_devices(0)->set_global_device_id(0);
  local_0.mutable_devices(1)->set_global_device_id(1);
  local_1.mutable_devices(0)->set_global_device_id(2);
  local_2.mutable_devices(0)->set_global_device_id(3);
  *expected_cluster_devices.mutable_xla()->mutable_devices()->add_nodes() =
      local_0;
  *expected_cluster_devices.mutable_xla()->mutable_devices()->add_nodes() =
      local_1;
  *expected_cluster_devices.mutable_xla()->mutable_devices()->add_nodes() =
      local_2;
  EXPECT_THAT(cluster_devices, IgnoringRepeatedFieldOrdering(
                                   EqualsProto(expected_cluster_devices)));
}

// Task devices should not be added twice if same task calls WaitForAllDevices()
// twice.
TEST(CoordinationServiceTest, ListClusterDevices_DevicesAreNotAddedTwice) {
  const ServerDef& server_def = GetMultiClientServerDef("worker", 2);
  CoordinatedTask task_0;
  task_0.set_job_name("worker");
  task_0.set_task_id(0);
  CoordinatedTask task_1;
  task_1.set_job_name("worker");
  task_1.set_task_id(1);
  Status status = OkStatus();
  auto client_cache = std::make_unique<TestCoordinationClientCache>();
  std::unique_ptr<CoordinationServiceInterface> coord_service =
      CoordinationServiceInterface::EnableCoordinationService(
          kCoordinationServiceType, Env::Default(), server_def,
          std::move(client_cache));
  absl::Notification n;
  // Map fake devices to each task.
  CoordinationServiceDeviceInfo local_devices_0;
  CoordinationServiceDeviceInfo local_devices_1;
  *local_devices_0.mutable_tf()->mutable_devices()->Add() =
      CreateTestTfDevice("task0_device0");
  *local_devices_0.mutable_tf()->mutable_devices()->Add() =
      CreateTestTfDevice("task0_device1");
  *local_devices_1.mutable_tf()->mutable_devices()->Add() =
      CreateTestTfDevice("task1_device0");
  // Task0 sends device info.
  CoordinationServiceDeviceInfo cluster_devices;
  coord_service->WaitForAllTasks(task_0, local_devices_0,
                                 [](Status s) { TF_ASSERT_OK(s); });

  // Task0 sends device info sgain.
  coord_service->WaitForAllTasks(task_0, local_devices_0,
                                 [](Status s) { TF_ASSERT_OK(s); });
  coord_service->WaitForAllTasks(
      task_1, local_devices_1,
      [coord_service = coord_service.get(), &cluster_devices, &n](Status s) {
        TF_ASSERT_OK(s);
        // Gather the cluster device info.
        cluster_devices = coord_service->ListClusterDevices();
        n.Notify();
      });
  n.WaitForNotification();

  // No duplicates found.
  CoordinationServiceDeviceInfo expected_cluster_devices;
  auto expected_devices =
      expected_cluster_devices.mutable_tf()->mutable_devices();
  expected_devices->Add(local_devices_0.mutable_tf()->devices().begin(),
                        local_devices_0.mutable_tf()->devices().end());
  expected_devices->Add(local_devices_1.mutable_tf()->devices().begin(),
                        local_devices_1.mutable_tf()->devices().end());
  EXPECT_THAT(cluster_devices, IgnoringRepeatedFieldOrdering(
                                   EqualsProto(expected_cluster_devices)));
}

TEST_F(CoordinationBarrierTest, Barrier) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  Status barrier_status_1;
  Status barrier_status_2;
  absl::Notification n_0;
  absl::Notification n_1;
  absl::Notification n_2;

  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(0),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_0, &n_0](Status s) {
                                           barrier_status_0 = s;
                                           n_0.Notify();
                                         });
  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(1),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_1, &n_1](Status s) {
                                           barrier_status_1 = s;
                                           n_1.Notify();
                                         });
  // Make sure barrier has not been exited prematurely.
  EXPECT_FALSE(n_0.HasBeenNotified());
  EXPECT_FALSE(n_1.HasBeenNotified());
  EXPECT_FALSE(n_2.HasBeenNotified());

  // Last task calls the barrier.
  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(2),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_2, &n_2](Status s) {
                                           barrier_status_2 = s;
                                           n_2.Notify();
                                         });

  EXPECT_TRUE(n_0.HasBeenNotified());
  EXPECT_TRUE(n_1.HasBeenNotified());
  EXPECT_TRUE(n_2.HasBeenNotified());
  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);
  TF_EXPECT_OK(barrier_status_2);
}

TEST_F(CoordinationBarrierTest, BarrierWithSubsetOfTasks) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  Status barrier_status_1;
  absl::Notification n_0;
  absl::Notification n_1;

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_0, &n_0](Status s) {
        barrier_status_0 = s;
        n_0.Notify();
      });
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(1),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_1, &n_1](Status s) {
        barrier_status_1 = s;
        n_1.Notify();
      });

  // All listed tasks passed the barrier.
  EXPECT_TRUE(n_0.HasBeenNotified());
  EXPECT_TRUE(n_1.HasBeenNotified());
  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);
}

TEST_F(CoordinationBarrierTest, BarrierWithMismatchedTasks) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  Status barrier_status_1;

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_0](Status s) { barrier_status_0 = s; });
  // task_1's barrier call specified a conflicting set of tasks (task_2 instead
  // of task_0).
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(1),
      /*participating_tasks=*/{GetTask(1), GetTask(2)},
      [&barrier_status_1](Status s) { barrier_status_1 = s; });

  EXPECT_TRUE(errors::IsInvalidArgument(barrier_status_0));
  EXPECT_TRUE(errors::IsInvalidArgument(barrier_status_1));
}

TEST_F(CoordinationBarrierTest, BarrierByNonParticipatingTask) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  Status barrier_status_1;
  absl::Notification n_0;
  absl::Notification n_1;

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_0](Status s) { barrier_status_0 = s; });
  // Task 2 unexpectedly calls a barrier that it is not participating in.
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(2),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_1](Status s) { barrier_status_1 = s; });

  // Barrier should fail for all tasks with the unexpected call.
  EXPECT_TRUE(errors::IsInvalidArgument(barrier_status_0));
  EXPECT_TRUE(errors::IsInvalidArgument(barrier_status_1));
}

TEST_F(CoordinationBarrierTest, BarrierByNonClusterTask) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  absl::Notification n_0;
  CoordinatedTask unspecified_task;
  unspecified_task.set_job_name("task_from_another_cluster");
  unspecified_task.set_task_id(2);

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{GetTask(0), unspecified_task},
      [&barrier_status_0, &n_0](Status s) {
        barrier_status_0 = s;
        n_0.Notify();
      });
  n_0.WaitForNotification();

  // Barrier should fail with the unexpected participating task argument.
  EXPECT_TRUE(errors::IsInvalidArgument(barrier_status_0));
}

TEST_F(CoordinationBarrierTest, BarrierTimeout) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(1);
  Status barrier_status_0;
  absl::Notification n_0;

  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(0),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_0, &n_0](Status s) {
                                           barrier_status_0 = s;
                                           n_0.Notify();
                                         });

  // Block until user-specified timeout.
  n_0.WaitForNotification();
  EXPECT_TRUE(errors::IsDeadlineExceeded(barrier_status_0));
}

TEST_F(CoordinationBarrierTest, BarrierReturnsPreviousError) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(1);
  Status barrier_status_0;
  Status barrier_status_1;
  absl::Notification n_0;

  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(0),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_0, &n_0](Status s) {
                                           barrier_status_0 = s;
                                           n_0.Notify();
                                         });
  TF_ASSERT_OK(GetCoordinationService()->ReportTaskError(
      GetTask(0), errors::Internal("test_error")));
  // Block until barrier has failed due to task error.
  n_0.WaitForNotification();
  // Same response should be returned immediately.
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(1),
      /*participating_tasks=*/{},
      [&barrier_status_1](Status s) { barrier_status_1 = s; });

  EXPECT_TRUE(errors::IsInternal(barrier_status_0));
  EXPECT_TRUE(errors::IsInternal(barrier_status_1));
}

TEST_F(CoordinationBarrierTest, BarrierCancelled) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status;

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{},
      [&barrier_status](Status s) { barrier_status = s; });
  Status cancelled_status =
      GetCoordinationService()->CancelBarrier(barrier_id, GetTask(0));

  EXPECT_TRUE(errors::IsCancelled(barrier_status));
  TF_EXPECT_OK(cancelled_status);
}

TEST_F(CoordinationBarrierTest, CancelNonExistentBarrier_FutureBarrierFails) {
  const std::string barrier_id = "cancelled_barrier_id";
  absl::Duration timeout = absl::Seconds(1);
  Status barrier_status;

  // Cancel barrier should still succeed.
  TF_ASSERT_OK(GetCoordinationService()->CancelBarrier(barrier_id, GetTask(0)));
  // Calling a cancelled barrier should fail instantly.
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{},
      [&barrier_status](Status s) { barrier_status = s; });

  EXPECT_TRUE(errors::IsCancelled(barrier_status)) << barrier_status;
}

TEST_F(CoordinationBarrierTest, CancelAfterBarrierHasPassed) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  Status barrier_status_1;
  Status barrier_status_2;

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{},
      [&barrier_status_0](Status s) { barrier_status_0 = s; });
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(1),
      /*participating_tasks=*/{},
      [&barrier_status_1](Status s) { barrier_status_1 = s; });
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(2),
      /*participating_tasks=*/{},
      [&barrier_status_2](Status s) { barrier_status_2 = s; });
  // Cancel barrier should fail if barrier has already been passed.
  Status cancelled_status =
      GetCoordinationService()->CancelBarrier(barrier_id, GetTask(0));

  EXPECT_TRUE(errors::IsFailedPrecondition(cancelled_status));
  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);
  TF_EXPECT_OK(barrier_status_2);
}

TEST_F(CoordinationBarrierTest, PassedBarrierReturnsImmediately) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  Status barrier_status_1;
  Status barrier_status_2;
  Status barrier_status_repeat;
  absl::Notification n0;
  absl::Notification n1;
  absl::Notification n2;
  absl::Notification n_repeat;

  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(0),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_0, &n0](Status s) {
                                           barrier_status_0 = s;
                                           n0.Notify();
                                         });
  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(1),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_1, &n1](Status s) {
                                           barrier_status_1 = s;
                                           n1.Notify();
                                         });
  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(2),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_2, &n2](Status s) {
                                           barrier_status_2 = s;
                                           n2.Notify();
                                         });
  // Repeated call should return the same result.
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(1),
      /*participating_tasks=*/{},
      [&barrier_status_repeat, &n_repeat](Status s) {
        barrier_status_repeat = s;
        n_repeat.Notify();
      });

  EXPECT_TRUE(n0.HasBeenNotified());
  EXPECT_TRUE(n1.HasBeenNotified());
  EXPECT_TRUE(n2.HasBeenNotified());
  EXPECT_TRUE(n_repeat.HasBeenNotified());
  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);
  TF_EXPECT_OK(barrier_status_2);
  TF_EXPECT_OK(barrier_status_repeat);
}

TEST_F(CoordinationBarrierTest, BarrierFailsIfTaskIsAlreadyInError) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  // Set task 0 to error state.
  TF_ASSERT_OK(GetCoordinationService()->ReportTaskError(
      GetTask(0), errors::Internal("test_error")));
  Status barrier_status;

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(1),
      /*participating_tasks=*/{},
      [&barrier_status](Status s) { barrier_status = s; });

  EXPECT_TRUE(errors::IsInternal(barrier_status));
}

TEST_F(CoordinationBarrierTest, BarrierFailsUponTaskError) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Notification n0;
  Status barrier_status;

  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(0),
                                         /*participating_tasks=*/{},
                                         [&barrier_status, &n0](Status s) {
                                           barrier_status = s;
                                           n0.Notify();
                                         });
  TF_ASSERT_OK(GetCoordinationService()->ReportTaskError(
      GetTask(0), errors::Internal("test_error")));
  n0.WaitForNotification();

  EXPECT_TRUE(errors::IsInternal(barrier_status));
}

TEST_F(CoordinationBarrierTest,
       BarrierStillBlocksIfSameTaskCallsOngoingBarrierRepeatedly) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  Status barrier_status_1;
  Status barrier_status_2;
  absl::Notification n_0;
  absl::Notification n_1;
  absl::Notification n_2;

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_0, &n_0](Status s) {
        barrier_status_0 = s;
        n_0.Notify();
      });
  // Duplicate call.
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_1, &n_1](Status s) {
        barrier_status_1 = s;
        n_1.Notify();
      });
  // All listed tasks passed the barrier.
  EXPECT_FALSE(n_0.HasBeenNotified());
  EXPECT_FALSE(n_1.HasBeenNotified());

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(1),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_2, &n_2](Status s) {
        barrier_status_2 = s;
        n_2.Notify();
      });
  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);
  TF_EXPECT_OK(barrier_status_2);
}

TEST_F(CoordinateTwoTasksTest, ResetAndRegisterAgain) {
  EnableCoordinationService();
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));

  TF_EXPECT_OK(coord_service_->ResetTask(task_0_));

  // Task should be allowed to register again after being reset.
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
}

TEST_F(CoordinateTwoTasksTest, Reset_HeartbeatsAreAcceptedForAGracePeriod) {
  EnableCoordinationService();
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));

  TF_EXPECT_OK(coord_service_->ResetTask(task_0_));
  // Heartbeat should be allowed for a short grace period after reset.
  TF_EXPECT_OK(coord_service_->RecordHeartbeat(task_0_, incarnation_0_));

  // Heartbeat failure should be triggered for disconnected task after grace
  // period.
  Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(3 * kHeartbeatTimeout));
  EXPECT_TRUE(errors::IsInvalidArgument(
      coord_service_->RecordHeartbeat(task_0_, incarnation_0_)));
}

TEST_F(CoordinateTwoTasksTest, Reset_FailsOngoingBarrier) {
  EnableCoordinationService(/*has_service_to_client_connection=*/true,
                            /*enable_shutdown_barrier=*/false);
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  Status barrier_status;
  absl::Notification barrier_n;
  coord_service_->BarrierAsync(
      "ongoing_barrier", absl::InfiniteDuration(), task_0_,
      /*participating_tasks=*/{}, [&barrier_status, &barrier_n](Status s) {
        barrier_status = s;
        barrier_n.Notify();
      });

  TF_EXPECT_OK(coord_service_->ResetTask(task_0_));

  // Ongoing barrier should fail with error after shutdown.
  EXPECT_TRUE(barrier_n.HasBeenNotified());
  EXPECT_TRUE(errors::IsInternal(barrier_status)) << barrier_status;
}

TEST_F(CoordinateTwoTasksTest, Shutdown_HeartbeatsAreAcceptedForAGracePeriod) {
  EnableCoordinationService(/*has_service_to_client_connection=*/true,
                            /*enable_shutdown_barrier=*/false);
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));

  absl::Notification n;
  coord_service_->ShutdownTaskAsync(task_0_, [&n](Status s) {
    TF_EXPECT_OK(s);
    n.Notify();
  });
  n.WaitForNotification();

  // Heartbeat should be allowed for a short grace period after shutdown.
  TF_EXPECT_OK(coord_service_->RecordHeartbeat(task_0_, incarnation_0_));

  // Heartbeat failure should be triggered for disconnected task after grace
  // period.
  Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(3 * kHeartbeatTimeout));
  EXPECT_TRUE(errors::IsInvalidArgument(
      coord_service_->RecordHeartbeat(task_0_, incarnation_0_)));
}

TEST_F(CoordinateTwoTasksTest, Shutdown_FailsOngoingBarrier) {
  EnableCoordinationService(/*has_service_to_client_connection=*/true,
                            /*enable_shutdown_barrier=*/false);
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  Status barrier_status;
  absl::Notification barrier_n;
  coord_service_->BarrierAsync(
      "ongoing_barrier", absl::InfiniteDuration(), task_0_,
      /*participating_tasks=*/{}, [&barrier_status, &barrier_n](Status s) {
        barrier_status = s;
        barrier_n.Notify();
      });

  absl::Notification shutdown_n;
  coord_service_->ShutdownTaskAsync(task_0_, [&shutdown_n](Status s) {
    TF_EXPECT_OK(s);
    shutdown_n.Notify();
  });
  shutdown_n.WaitForNotification();

  // Ongoing barrier should fail with error after shutdown.
  EXPECT_TRUE(barrier_n.HasBeenNotified());
  EXPECT_TRUE(errors::IsInternal(barrier_status)) << barrier_status;
}

TEST_F(CoordinateTwoTasksTest, ShutdownWithBarrier_BarrierSucceeds) {
  EnableCoordinationService(/*has_service_to_client_connection=*/true,
                            /*enable_shutdown_barrier=*/true);
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  TF_EXPECT_OK(coord_service_->RegisterTask(task_1_, incarnation_1_));
  Status barrier_status;
  Status barrier_status_2;

  coord_service_->ShutdownTaskAsync(
      task_0_, [&barrier_status](Status s) { barrier_status = s; });
  coord_service_->ShutdownTaskAsync(
      task_1_, [&barrier_status_2](Status s) { barrier_status_2 = s; });

  TF_EXPECT_OK(barrier_status);
  TF_EXPECT_OK(barrier_status_2);

  // Confirm that both tasks have disconnected.
  // Note: this should not happen in prod where RegisterTask() is called after
  // Shutdown(), which is prevented by agent-side logic.
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  TF_EXPECT_OK(coord_service_->RegisterTask(task_1_, incarnation_1_));
}

TEST_F(CoordinateTwoTasksTest,
       ShutdownWithBarrier_BarrierFails_TaskDisconnectsOtherTaskIsAlerted) {
  EnableCoordinationService(/*has_service_to_client_connection=*/true,
                            /*enable_shutdown_barrier=*/true);
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  TF_EXPECT_OK(coord_service_->RegisterTask(task_1_, incarnation_1_));
  Status barrier_status;

  absl::Notification n;
  coord_service_->ShutdownTaskAsync(task_0_, [&n, &barrier_status](Status s) {
    barrier_status = s;
    n.Notify();
  });
  // Block until barrier times out.
  n.WaitForNotification();

  EXPECT_TRUE(errors::IsDeadlineExceeded(barrier_status)) << barrier_status;
  // Confirm that task_0_ has disconnected.
  // Note: this should not happen in prod where RegisterTask() is called after
  // Shutdown(), which is prevented by agent-side logic.
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));

  // Other task is alerted that shutdown has been initiated without it.
  Status other_task_status = client_1_.GetStatus();
  EXPECT_TRUE(errors::IsInternal(other_task_status)) << other_task_status;
}

TEST_F(CoordinateTwoTasksTest,
       ShutdownWithBarrier_BarrierFailsWithoutClientConnection_ServiceStops) {
  EnableCoordinationService(/*has_service_to_client_connection=*/false,
                            /*enable_shutdown_barrier=*/true);
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  TF_EXPECT_OK(coord_service_->RegisterTask(task_1_, incarnation_1_));
  Status barrier_status;

  absl::Notification n;
  coord_service_->ShutdownTaskAsync(task_0_, [&n, &barrier_status](Status s) {
    barrier_status = s;
    n.Notify();
  });
  // Block until barrier times out.
  n.WaitForNotification();
  // Provide time for coordination service to shut down after barrier timeout.
  Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(absl::Seconds(1)));

  EXPECT_TRUE(errors::IsDeadlineExceeded(barrier_status)) << barrier_status;

  // Service stops because no service-to-client connection is available for
  // error propagation.
  // Task 1 still sends unexpected heartbeat because it doesn't know that
  // service has stopped yet, which should fail.
  Status s = coord_service_->RecordHeartbeat(task_1_, incarnation_1_);

  EXPECT_TRUE(errors::IsInvalidArgument(s)) << s;
}
}  // namespace tensorflow
