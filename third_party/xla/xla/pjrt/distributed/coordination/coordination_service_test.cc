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

#include "xla/pjrt/distributed/coordination/coordination_service.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "xla/pjrt/distributed/coordination/coordination_service.pb.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/protobuf/coordination_config.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/random.h"

namespace xla {
namespace {

using ::testing::Each;
using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;
using ::testing::status::IsOk;
using ::testing::status::StatusIs;
using ::tsl::proto_testing::EqualsProto;
using xla::coordination::KeyValueEntry;

constexpr absl::Duration kHeartbeatTimeout = absl::Seconds(2);
constexpr absl::Duration kShutdownBarrierTimeout = absl::Milliseconds(500);

// Returns a human readable string version of the provided config.
std::string DebugString(const CoordinationService::Config& config) {
  return absl::StrFormat(
      "CoordinationService::Config {\n"
      "  cluster_register_timeout: %s\n"
      "  cluster_register_with_barrier: %v\n"
      "  heartbeat_timeout: %s\n"
      "  num_tasks: %d\n"
      "  shutdown_barrier_timeout: %s\n"
      "  recoverable: %v\n"
      "}",
      absl::FormatDuration(config.cluster_register_timeout),
      config.cluster_register_with_barrier,
      absl::FormatDuration(config.heartbeat_timeout), config.num_tasks,
      absl::FormatDuration(config.shutdown_barrier_timeout),
      config.recoverable);
}

// Creates a KeyValueEntry proto from the provided key and value.
KeyValueEntry CreateKv(const std::string& key, const std::string& value) {
  KeyValueEntry kv;
  kv.set_key(key);
  kv.set_value(value);
  return kv;
}

// Constructs a TaskInfo proto from the provided arguments.
xla::coordination::TaskInfo info(CoordinationService::TaskId task,
                                 IncarnationId incarnation_id,
                                 xla::coordination::TaskState state) {
  xla::coordination::TaskInfo info;
  info.set_task_id(task);
  info.set_incarnation(incarnation_id.value());
  info.set_state(state);
  return info;
}

// Returns a default CoordinationService::Config with the given number of tasks.
CoordinationService::Config DefaultConfig(int num_tasks) {
  CoordinationService::Config config;
  config.num_tasks = num_tasks;
  config.recoverable = false;
  config.heartbeat_timeout = kHeartbeatTimeout;
  return config;
}

// Starts and returns a coordination service.
std::unique_ptr<CoordinationService> Start(
    const CoordinationService::Config& config) {
  VLOG(1) << "Starting service with config:\n" << DebugString(config);
  return std::make_unique<CoordinationService>(tsl::Env::Default(), config);
}

struct Tasks {
  std::vector<CoordinationService::TaskId> tasks;
  std::vector<IncarnationId> incarnations;
};

// Registers all tasks with the coordination service.
Tasks RegisterTasks(const CoordinationService::Config& config,
                    CoordinationService& service) {
  // If config.cluster_register_with_barrier is true, then registration is a
  // barrier, but we register tasks serially.
  CHECK(!config.cluster_register_with_barrier);

  Tasks tasks;
  for (int i = 0; i < config.num_tasks; ++i) {
    EXPECT_THAT(service.RegisterTask(i, IncarnationId(i)), IsOk());
    tasks.tasks.push_back(i);
    tasks.incarnations.push_back(IncarnationId(i));
  }
  return tasks;
}

TEST(CoordinationService, TestStandaloneService) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  ASSERT_OK(service->RecordHeartbeat(0, tasks.incarnations[0]));
  ASSERT_OK(service->RecordHeartbeat(1, tasks.incarnations[1]));
  EXPECT_THAT(service->RecordHeartbeat(2, IncarnationId(0)),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Sending heartbeat with incarnation mismatch leads to Aborted error.
  EXPECT_THAT(service->RecordHeartbeat(1, IncarnationId(67)),
              StatusIs(absl::StatusCode::kAborted));
}

// RegisterTask() may succeed in the service, but the agent response times out.
// In this case, the agent would retry Connect() and should succeed if it has
// the same incarnation.
TEST(CoordinationService, RegisterTask_AlreadyConnected_Succeeds) {
  CoordinationService::Config config = DefaultConfig(1);
  std::unique_ptr<CoordinationService> service = Start(config);
  // Task connects to coordination service.
  ASSERT_OK(service->RegisterTask(0, IncarnationId(0)));

  // Registration should succeed since it is the same task.
  const absl::Status status = service->RegisterTask(0, IncarnationId(0));

  TF_EXPECT_OK(status) << status;
}

TEST(CoordinationService,
     RegisterTask_AlreadyConnectedDifferentIncarnation_Fails) {
  CoordinationService::Config config = DefaultConfig(1);
  std::unique_ptr<CoordinationService> service = Start(config);
  // Task connects to coordination service.
  ASSERT_OK(service->RegisterTask(0, IncarnationId(0)));

  // Registration should fail since task already registered previously with a
  // different incarnation. Note that incarnation usually changes if an agent
  // restarts.
  const absl::Status status = service->RegisterTask(0, IncarnationId(1));

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kAborted));
}

TEST(CoordinationService, RegisterTask_AlreadyInError_Fails) {
  CoordinationService::Config config = DefaultConfig(1);
  std::unique_ptr<CoordinationService> service = Start(config);
  // Task connects to coordination service.
  ASSERT_OK(service->RegisterTask(0, IncarnationId(0)));
  // Arbitrarily set task to be in error.
  ASSERT_OK(service->ReportTaskError(0, absl::InternalError("test_error")));

  // Registration should fail.
  const absl::Status status = service->RegisterTask(0, IncarnationId(0));

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kAborted));
}

TEST(CoordinationService, TestTaskHeartbeatTimeout) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);

  // No heartbeat for a while, leader considers the task as stale.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));
  EXPECT_THAT(service->RecordHeartbeat(0, tasks.incarnations[0]),
              StatusIs(absl::StatusCode::kAborted));
  EXPECT_THAT(service->RecordHeartbeat(1, tasks.incarnations[1]),
              StatusIs(absl::StatusCode::kAborted));
}

TEST(CoordinationService,
     ErrorPollingRequestsGotCancelledErrorUponServiceShutdown) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  std::vector<absl::Status> statuses;
  statuses.reserve(2);

  for (CoordinationService::TaskId task : {0, 1}) {
    service->PollForErrorAsync(
        task, [&](const absl::Status& status) { statuses.push_back(status); });
  }

  // No error polling requests are received before service shutdown.
  EXPECT_EQ(statuses.size(), 0);
  service.reset();

  // The service shutdowns successfully and send the cancellation response to
  // the error polling requests.
  EXPECT_EQ(statuses.size(), 2);
  EXPECT_THAT(statuses, Each(StatusIs(absl::StatusCode::kCancelled)));
}

TEST(CoordinationService, HeartbeatTimeoutWithoutServerToClientConnection) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);

  // No heartbeat for a while, leader consider the task as stale.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));
  // Unexpected heartbeat from errored tasks.
  EXPECT_THAT(service->RecordHeartbeat(0, tasks.incarnations[0]),
              StatusIs(absl::StatusCode::kAborted));
  EXPECT_THAT(service->RecordHeartbeat(1, tasks.incarnations[1]),
              StatusIs(absl::StatusCode::kAborted));
}

TEST(CoordinationService,
     HeartbeatTimeoutErrorCanPropagateThroughErrorPolling) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  // Use notifications to guarantee the ordering of operations across threads.
  absl::Notification n0, n1;
  absl::Status s0, s1;

  service->PollForErrorAsync(0, [&](const absl::Status& status) {
    s0 = status;
    n0.Notify();
  });
  service->PollForErrorAsync(1, [&](const absl::Status& status) {
    s1 = status;
    n1.Notify();
  });

  // No heartbeat for a while, leader consider the task as stale and propagate
  // the error to the tasks.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));
  // Make sure the StatusCallbacks are called.
  n0.WaitForNotification();
  n1.WaitForNotification();

  // Heartbeat errors are propagated to everyone.
  EXPECT_THAT(s0, StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_THAT(s1, StatusIs(absl::StatusCode::kUnavailable));
}

TEST(CoordinationService,
     HeartbeatTimeoutErrorFromOneTaskCanPropagateThroughErrorPolling) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  // Use notifications to guarantee the ordering of operations across threads.
  absl::Status s0, s1;
  absl::Notification n0, n1;

  service->PollForErrorAsync(0, [&](const absl::Status& status) {
    s0 = status;
    n0.Notify();
  });
  service->PollForErrorAsync(1, [&](const absl::Status& status) {
    s1 = status;
    n1.Notify();
  });

  // Use a factor of 0.9 to avoid accidental timeout.
  const int64_t sleeping_time =
      absl::ToInt64Microseconds(0.9 * kHeartbeatTimeout);
  // No heartbeat from task 1 for a while, so leader consider the task as stale
  // and propagate the error to all tasks.
  tsl::Env::Default()->SleepForMicroseconds(sleeping_time);
  TF_EXPECT_OK(service->RecordHeartbeat(0, tasks.incarnations[0]));
  tsl::Env::Default()->SleepForMicroseconds(sleeping_time);
  TF_EXPECT_OK(service->RecordHeartbeat(0, tasks.incarnations[0]));
  tsl::Env::Default()->SleepForMicroseconds(sleeping_time);
  // Make sure the StatusCallbacks are called.
  n0.WaitForNotification();
  n1.WaitForNotification();

  // The heartbeat error from `1` below should be propagated to all tasks.
  EXPECT_THAT(s0, StatusIs(absl::StatusCode::kUnavailable, HasSubstr("1")));
  EXPECT_THAT(s1, StatusIs(absl::StatusCode::kUnavailable, HasSubstr("1")));
}

TEST(CoordinationService, ReportedErrorCanPropagateThroughErrorPolling) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  std::vector<absl::Status> statuses;
  statuses.reserve(2);
  for (CoordinationService::TaskId task : {0, 1}) {
    service->PollForErrorAsync(
        task, [&](const absl::Status& status) { statuses.push_back(status); });
  }

  ASSERT_OK(service->ReportTaskError(1, absl::InternalError("test_error")));
  // The reported error is propagated through error polling.
  EXPECT_EQ(statuses.size(), 2);
  EXPECT_THAT(statuses, Each(StatusIs(absl::StatusCode::kInternal)));
}

TEST(CoordinationService, TestTaskRestart) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);

  // Simulate task restart scenario: trying to register to cluster again.
  absl::Status s =
      service->RegisterTask(1, IncarnationId(tsl::random::New64()));

  EXPECT_THAT(s, StatusIs(absl::StatusCode::kAborted));
}

TEST(CoordinationService, WatchTasksSucceeds) {
  // This test calls WatchTasks on two successfully connected tasks.

  // Connect the tasks.
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);

  // Watch the job state, which should return immediately.
  absl::Notification done;
  service->WatchTasks(
      std::nullopt, [&](std::vector<xla::coordination::TaskInfo> got,
                        int64_t version_number) {
        using State = xla::coordination::TaskState;
        std::vector<xla::coordination::TaskInfo> want(2);
        want[0] = info(0, tasks.incarnations[0], State::CONNECTED);
        want[1] = info(1, tasks.incarnations[1], State::CONNECTED);
        EXPECT_THAT(got, UnorderedElementsAre(EqualsProto(want[0]),
                                              EqualsProto(want[1])));
        done.Notify();
      });
  done.WaitForNotification();
}

TEST(CoordinationService, WatchTasksReturnsDisconnected) {
  // This test calls WatchTasks on one successfully connected task and one
  // disconnected task.

  // Connect the tasks. Disconnect task 1.
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  ASSERT_OK(service->ResetTask(1));

  // Watch the job state, which should return immediately.
  absl::Notification done;
  service->WatchTasks(
      std::nullopt, [&](std::vector<xla::coordination::TaskInfo> got,
                        int64_t version_number) {
        using State = xla::coordination::TaskState;
        std::vector<xla::coordination::TaskInfo> want(2);
        want[0] = info(0, tasks.incarnations[0], State::CONNECTED);
        want[1] = info(1, tasks.incarnations[1], State::DISCONNECTED);
        EXPECT_THAT(got, UnorderedElementsAre(EqualsProto(want[0]),
                                              EqualsProto(want[1])));
        EXPECT_THAT(version_number, Ge(0));
        done.Notify();
      });
  done.WaitForNotification();
}

TEST(CoordinationService, WatchTasksReturnsNewIncarnation) {
  // This test calls WatchTasks after one task has restarted with a new
  // incarnation.
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  ASSERT_OK(service->ResetTask(1));
  ASSERT_OK(service->RegisterTask(1, tasks.incarnations[1] + 1));

  // Watch the job state, which should return immediately.
  absl::Notification done;
  service->WatchTasks(
      std::nullopt, [&](std::vector<xla::coordination::TaskInfo> got,
                        int64_t version_number) {
        using State = xla::coordination::TaskState;
        std::vector<xla::coordination::TaskInfo> want(2);
        want[0] = info(0, tasks.incarnations[0], State::CONNECTED);
        want[1] = info(1, tasks.incarnations[1] + 1, State::CONNECTED);
        EXPECT_THAT(got, UnorderedElementsAre(EqualsProto(want[0]),
                                              EqualsProto(want[1])));
        EXPECT_THAT(version_number, Ge(0));
        done.Notify();
      });
  done.WaitForNotification();
}

TEST(CoordinationService, WatchTasksBlocksUntilChange) {
  // This test calls checks that WatchTasks blocks until the job state
  // changes.

  // Connect the tasks. Disconnect task 1.
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);

  // Watch the job state, which should return immediately.
  absl::Notification done_1;
  int64_t version_number = -1;
  service->WatchTasks(
      std::nullopt,
      [&](std::vector<xla::coordination::TaskInfo> got, int64_t v) {
        EXPECT_THAT(v, Ge(0));
        version_number = v;
        done_1.Notify();
      });
  done_1.WaitForNotification();

  // Watch the job state again, which should block.
  absl::Notification done_2;
  service->WatchTasks(
      version_number,
      [&](std::vector<xla::coordination::TaskInfo> got, int64_t v) {
        using State = xla::coordination::TaskState;
        std::vector<xla::coordination::TaskInfo> want(2);
        want[0] = info(0, tasks.incarnations[0], State::CONNECTED);
        want[1] = info(1, tasks.incarnations[1], State::DISCONNECTED);
        EXPECT_THAT(got, UnorderedElementsAre(EqualsProto(want[0]),
                                              EqualsProto(want[1])));
        EXPECT_THAT(v, Ge(version_number));
        done_2.Notify();
      });
  bool notified = done_2.WaitForNotificationWithTimeout(absl::Seconds(1));
  ASSERT_FALSE(notified);

  // Disconnect task 1.
  ASSERT_OK(service->ResetTask(1));

  done_2.WaitForNotification();
}

TEST(CoordinationService, WatchTasksAfterTwoStateChanges) {
  // This test calls WatchTasks after two state changes.
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);

  // Watch the job state, which should return immediately.
  absl::Notification done_1;
  int64_t version_number = -1;
  service->WatchTasks(
      std::nullopt,
      [&](std::vector<xla::coordination::TaskInfo> got, int64_t v) {
        using State = xla::coordination::TaskState;
        std::vector<xla::coordination::TaskInfo> want(2);
        want[0] = info(0, tasks.incarnations[0], State::CONNECTED);
        want[1] = info(1, tasks.incarnations[1], State::CONNECTED);
        EXPECT_THAT(got, UnorderedElementsAre(EqualsProto(want[0]),
                                              EqualsProto(want[1])));
        EXPECT_THAT(v, Ge(0));
        version_number = v;
        done_1.Notify();
      });
  done_1.WaitForNotification();

  // Restart task 1. This leads to two state changes: the task is disconnected
  // and then reconnected.
  ASSERT_OK(service->ResetTask(1));
  ASSERT_OK(service->RegisterTask(1, tasks.incarnations[1] + 1));

  // Watch the job state, which should return immediately because the state has
  // already changed.
  absl::Notification done_2;
  service->WatchTasks(
      version_number,
      [&](std::vector<xla::coordination::TaskInfo> got, int64_t v) {
        using State = xla::coordination::TaskState;
        std::vector<xla::coordination::TaskInfo> want(2);
        want[0] = info(0, tasks.incarnations[0], State::CONNECTED);
        want[1] = info(1, tasks.incarnations[1] + 1, State::CONNECTED);
        EXPECT_THAT(got, UnorderedElementsAre(EqualsProto(want[0]),
                                              EqualsProto(want[1])));
        EXPECT_THAT(v, Ge(version_number));
        done_2.Notify();
      });
  done_2.WaitForNotification();
}

TEST(CoordinationService, InsertKeyValue_Duplicate_Fail) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  ASSERT_OK(service->InsertKeyValue("key0", "original_value"));

  // Inserting the same key again should fail.
  EXPECT_THAT(service->InsertKeyValue("key0", "never_added"),
              StatusIs(absl::StatusCode::kAlreadyExists));

  // The original value should still be set.
  auto result = service->TryGetKeyValue("key0");
  TF_EXPECT_OK(result.status());
  EXPECT_EQ(result.value(), "original_value");
}

TEST(CoordinationService, InsertKeyValue_Duplicate_Overwrite) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  ASSERT_OK(service->InsertKeyValue("key0", "original_value"));
  TF_EXPECT_OK(service->InsertKeyValue("key0", "overwritten_value",
                                       /*allow_overwrite=*/true));
  auto result = service->TryGetKeyValue("key0");
  TF_EXPECT_OK(result.status());
  EXPECT_EQ(result.value(), "overwritten_value");
}

TEST(CoordinationService, TestSetGetValues) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);

  // Simple key
  ASSERT_OK(service->InsertKeyValue("key0", "value0"));
  // Unix file like key path
  ASSERT_OK(service->InsertKeyValue("/path", "value"));
  ASSERT_OK(service->InsertKeyValue("/path/to/key1", "value1"));
  // Key with redundant slashes
  ASSERT_OK(service->InsertKeyValue("path/to//key2/", "value2"));

  // Get simple key
  absl::Notification n1;
  absl::StatusOr<absl::string_view> ret;
  service->GetKeyValueAsync(
      "key0", [&](const absl::StatusOr<absl::string_view>& status_or_value) {
        ret = status_or_value;
        n1.Notify();
      });
  n1.WaitForNotification();
  ASSERT_OK(ret.status());
  EXPECT_EQ(ret.value(), "value0");
  // Get key with redundant slashes
  absl::Notification n2;
  service->GetKeyValueAsync(
      "path//to///key1////",
      [&](const absl::StatusOr<absl::string_view>& status_or_value) {
        ret = status_or_value;
        n2.Notify();
      });
  n2.WaitForNotification();
  EXPECT_EQ(ret.value(), "value1");

  // Delete single key-value
  ASSERT_OK(service->DeleteKeyValue("key0"));
  // Get key that is not available
  absl::Notification n3;
  service->GetKeyValueAsync(
      "key0", [&](const absl::StatusOr<absl::string_view>& status_or_value) {
        ret = status_or_value;
        n3.Notify();
      });
  EXPECT_FALSE(n3.HasBeenNotified());
  // Insert the previously deleted key again
  ASSERT_OK(service->InsertKeyValue("key0", "value0_new"));
  n3.WaitForNotification();
  EXPECT_EQ(ret.value(), "value0_new");

  // Delete key-values recursively
  ASSERT_OK(service->DeleteKeyValue("/path"));
  // Get key that is not available
  auto n4 = std::make_shared<absl::Notification>();
  service->GetKeyValueAsync(
      "/path/to/key1",
      // Note: this callback will remain pending until it is cleaned up during
      // service shutdown. Hence, we use a shared pointer for notification so
      // that the it will not be deallocated before the pending callback is
      // cleaned up.
      [n4](const absl::StatusOr<absl::string_view>& status_or_value) {
        n4->Notify();
      });
  EXPECT_FALSE(n4->HasBeenNotified());
}

TEST(CoordinationService, TryGetKeyValue) {
  CoordinationService::Config config = DefaultConfig(1);
  std::unique_ptr<CoordinationService> service = Start(config);

  // Try to get nonexistent key.
  absl::StatusOr<std::string> result = service->TryGetKeyValue("test_key");
  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kNotFound));

  // Insert key value.
  ASSERT_OK(service->InsertKeyValue("test_key", "test_value"));
  result = service->TryGetKeyValue("test_key");
  EXPECT_EQ(result.value(), "test_value");

  // Delete Key, and try to get the key again.
  ASSERT_OK(service->DeleteKeyValue("test_key"));
  result = service->TryGetKeyValue("test_key");
  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kNotFound));
}

TEST(CoordinationService, IncrementKeyValue) {
  CoordinationService::Config config = DefaultConfig(1);
  std::unique_ptr<CoordinationService> service = Start(config);
  ASSERT_OK(service->InsertKeyValue("test_key", "1"));
  ASSERT_OK(service->IncrementKeyValue("test_key", 3));
  ASSERT_OK_AND_ASSIGN(std::string result_0,
                       service->TryGetKeyValue("test_key"));
  EXPECT_EQ(result_0, "4");
  ASSERT_OK(service->IncrementKeyValue("test_key_2", 10));
  ASSERT_OK_AND_ASSIGN(std::string result_1,
                       service->TryGetKeyValue("test_key_2"));
  EXPECT_EQ(result_1, "10");
  ASSERT_OK(service->InsertKeyValue("test_key_3", "bad_value"));
  EXPECT_THAT(service->IncrementKeyValue("test_key_3", 10),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(CoordinationService, GetKeyValueDir_SingleValueInDirectory) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  KeyValueEntry kv = CreateKv("dir/path", "value0");
  ASSERT_OK(service->InsertKeyValue(kv.key(), kv.value()));

  std::vector<KeyValueEntry> result = service->GetKeyValueDir("dir");

  EXPECT_THAT(result, UnorderedElementsAre(EqualsProto(kv)));
}

TEST(CoordinationService, GetKeyValueDir_MultipleValuesInDirectory) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  KeyValueEntry kv = CreateKv("dir/path", "value0");
  KeyValueEntry kv2 = CreateKv("dir/path2", "value1");
  // Placed in nested subdirectory.
  KeyValueEntry kv_sub = CreateKv("dir/sub_dir/path", "value_sub");
  ASSERT_OK(service->InsertKeyValue(kv.key(), kv.value()));
  ASSERT_OK(service->InsertKeyValue(kv2.key(), kv2.value()));
  ASSERT_OK(service->InsertKeyValue(kv_sub.key(), kv_sub.value()));

  std::vector<KeyValueEntry> result = service->GetKeyValueDir("dir");

  EXPECT_THAT(result, UnorderedElementsAre(EqualsProto(kv), EqualsProto(kv2),
                                           EqualsProto(kv_sub)));
}

TEST(CoordinationService, GetKeyValueDir_Empty_ReturnsEmptyList) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);

  std::vector<KeyValueEntry> result = service->GetKeyValueDir("dir");

  EXPECT_THAT(result, IsEmpty());
}

TEST(CoordinationService, GetKeyValueDir_WrongDir_ReturnsEmptyList) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  // Wrong directory.
  ASSERT_OK(service->InsertKeyValue("dir0/path", "value0"));

  std::vector<KeyValueEntry> result = service->GetKeyValueDir("dir");

  EXPECT_THAT(result, IsEmpty());
}

TEST(CoordinationService, GetKeyValueDir_WrongDirPrefix_ReturnsEmptyList) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  // Check that we don't match with nested subdirectories with the wrong prefix.
  ASSERT_OK(service->InsertKeyValue("wrong_dir/dir/path", "value0"));

  std::vector<KeyValueEntry> result = service->GetKeyValueDir("dir");

  EXPECT_THAT(result, IsEmpty());
}

TEST(CoordinationService, GetKeyValueDir_NonDirectoryPrefix_ReturnsEmptyList) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  // Wrong directory.
  ASSERT_OK(service->InsertKeyValue("dir_key", "value0"));

  std::vector<KeyValueEntry> result = service->GetKeyValueDir("dir");

  EXPECT_THAT(result, IsEmpty());
}

TEST(CoordinationService, GetKeyValueDir_NonDirectoryKey_ReturnsEmptyList) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  // Insert same key that is not a directory.
  ASSERT_OK(service->InsertKeyValue("dir", "value0"));

  std::vector<KeyValueEntry> result = service->GetKeyValueDir("dir");

  EXPECT_THAT(result, IsEmpty());
}

TEST(CoordinationService, Barrier) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0, barrier_status_1, barrier_status_2;
  int64_t counter_0, counter_1, counter_2;
  absl::Notification n_0, n_1, n_2;

  service->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0, &counter_0, &n_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
        counter_0 = counter;
        n_0.Notify();
      });
  service->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{},
      [&barrier_status_1, &counter_1, &n_1](absl::Status s, int64_t counter) {
        barrier_status_1 = s;
        counter_1 = counter;
        n_1.Notify();
      });
  // Make sure barrier has not been exited prematurely.
  EXPECT_FALSE(n_0.HasBeenNotified());
  EXPECT_FALSE(n_1.HasBeenNotified());
  EXPECT_FALSE(n_2.HasBeenNotified());

  // Last task calls the barrier.
  service->BarrierAsync(
      barrier_id, 0, timeout, 2,
      /*participating_tasks=*/{},
      [&barrier_status_2, &counter_2, &n_2](absl::Status s, int64_t counter) {
        barrier_status_2 = s;
        counter_2 = counter;
        n_2.Notify();
      });

  EXPECT_TRUE(n_0.HasBeenNotified());
  EXPECT_TRUE(n_1.HasBeenNotified());
  EXPECT_TRUE(n_2.HasBeenNotified());
  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);
  TF_EXPECT_OK(barrier_status_2);
  EXPECT_EQ(counter_0, 0);
  EXPECT_EQ(counter_1, 0);
  EXPECT_EQ(counter_2, 0);
}

TEST(CoordinationService, BarrierWithSubsetOfTasks) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0, barrier_status_1;
  int64_t counter_0, counter_1;
  absl::Notification n_0, n_1;

  service->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{0, 1},
      [&barrier_status_0, &counter_0, &n_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
        counter_0 = counter;
        n_0.Notify();
      });
  service->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{0, 1},
      [&barrier_status_1, &counter_1, &n_1](absl::Status s, int64_t counter) {
        barrier_status_1 = s;
        counter_1 = counter;
        n_1.Notify();
      });

  // All listed tasks passed the barrier.
  EXPECT_TRUE(n_0.HasBeenNotified());
  EXPECT_TRUE(n_1.HasBeenNotified());
  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);
  EXPECT_EQ(counter_0, 0);
  EXPECT_EQ(counter_1, 0);
}

TEST(CoordinationService, BarrierWithMismatchedTasks) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0, barrier_status_1;

  service->BarrierAsync(barrier_id, 0, timeout, 0,
                        /*participating_tasks=*/{0, 1},
                        [&barrier_status_0](absl::Status s, int64_t counter) {
                          barrier_status_0 = s;
                        });
  // task_1's barrier call specified a conflicting set of tasks (task_2 instead
  // of task_0).
  service->BarrierAsync(barrier_id, 0, timeout, 1,
                        /*participating_tasks=*/{1, 2},
                        [&barrier_status_1](absl::Status s, int64_t counter) {
                          barrier_status_1 = s;
                        });

  EXPECT_THAT(barrier_status_0, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(barrier_status_1, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CoordinationService, BarrierByNonParticipatingTask) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0, barrier_status_1;
  absl::Notification n_0, n_1;

  service->BarrierAsync(barrier_id, 0, timeout, 0,
                        /*participating_tasks=*/{0, 1},
                        [&barrier_status_0](absl::Status s, int64_t counter) {
                          barrier_status_0 = s;
                        });
  // Task 2 unexpectedly calls a barrier that it is not participating in.
  service->BarrierAsync(barrier_id, 0, timeout, 2,
                        /*participating_tasks=*/{0, 1},
                        [&barrier_status_1](absl::Status s, int64_t counter) {
                          barrier_status_1 = s;
                        });

  // Barrier should fail for all tasks with the unexpected call.
  EXPECT_THAT(barrier_status_0, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(barrier_status_1, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CoordinationService, BarrierByNonParticipatingTaskThreeTasks) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0, barrier_status_1, barrier_status_2;
  absl::Notification n_0, n_1;

  service->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{0, 1},
      [&barrier_status_0, &n_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
        n_0.Notify();
      });
  service->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{0, 1},
      [&barrier_status_1, &n_1](absl::Status s, int64_t counter) {
        barrier_status_1 = s;
        n_1.Notify();
      });

  n_0.WaitForNotification();
  n_1.WaitForNotification();

  // Barrier should pass because only participating tasks have called it.
  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);

  // Task 2 unexpectedly calls a barrier that it is not participating in.
  service->BarrierAsync(barrier_id, 0, timeout, 2,
                        /*participating_tasks=*/{0, 1},
                        [&barrier_status_2](absl::Status s, int64_t counter) {
                          barrier_status_2 = s;
                        });

  // Barrier should fail for task 2 which is not participating in the barrier.
  EXPECT_THAT(barrier_status_2, StatusIs(absl::StatusCode::kInvalidArgument));

  // Other clients would need to check the barrier key to detect the error.
}

TEST(CoordinationService, BarrierByNonClusterTask) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0;
  absl::Notification n_0;

  service->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{0, 67},
      [&barrier_status_0, &n_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
        n_0.Notify();
      });
  n_0.WaitForNotification();

  // Barrier should fail with the unexpected participating task argument.
  EXPECT_THAT(barrier_status_0, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CoordinationService, BarrierTimeout) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(1);
  absl::Status barrier_status_0, barrier_status_1;
  absl::Notification n_0, n_1;

  service->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{},
      [&barrier_status_1, &n_1](absl::Status s, int64_t counter) {
        barrier_status_1 = s;
        n_1.Notify();
      });
  service->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0, &n_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
        n_0.Notify();
      });

  // Block until user-specified timeout.
  n_0.WaitForNotification();
  n_1.WaitForNotification();

  // All barrier calls should fail with the same error.
  EXPECT_EQ(barrier_status_0, barrier_status_1);
  EXPECT_THAT(barrier_status_0, StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_FALSE(absl::StrContains(barrier_status_0.message(), 0));
  EXPECT_TRUE(absl::StrContains(barrier_status_0.message(),
                                "1"));  // First task at barrier.
  EXPECT_TRUE(
      absl::StrContains(barrier_status_0.message(), "2"));  // Timed-out task.
  EXPECT_TRUE(absl::StrContains(
      barrier_status_0.message(),
      "2/3"));  // Number of tasks at barrier / total number of tasks.
}

TEST(CoordinationService, BarrierReturnsPreviousError) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(1);
  absl::Status barrier_status_0;
  absl::Status barrier_status_1;
  absl::Notification n_0;

  service->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0, &n_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
        n_0.Notify();
      });
  ASSERT_OK(service->ReportTaskError(0, absl::InternalError("test_error")));
  // Block until barrier has failed due to task error.
  n_0.WaitForNotification();
  // Same response should be returned immediately.
  service->BarrierAsync(barrier_id, 0, timeout, 1,
                        /*participating_tasks=*/{},
                        [&barrier_status_1](absl::Status s, int64_t counter) {
                          barrier_status_1 = s;
                        });

  EXPECT_THAT(barrier_status_0, StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(barrier_status_1, StatusIs(absl::StatusCode::kInternal));
}

TEST(CoordinationService, TwoConsecutiveBarriers_Succeed) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  // Corresponds to first barrier for each node.
  absl::Status barrier_status_0, barrier_status_1, barrier_status_2,
      // Corresponds to second barrier for each node.
      barrier_status_0_2, barrier_status_1_2,
      barrier_status_2_2 = absl::UnknownError("Unknown");
  int64_t counter_0, counter_1, counter_2, counter_0_2, counter_1_2,
      counter_2_2 = -1;

  // First barrier.
  service->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0, &counter_0](const absl::Status& s, int64_t counter) {
        barrier_status_0 = s;
        counter_0 = counter;
      });
  service->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{},
      [&barrier_status_1, &counter_1](const absl::Status& s, int64_t counter) {
        barrier_status_1 = s;
        counter_1 = counter;
      });
  service->BarrierAsync(
      barrier_id, 0, timeout, 2,
      /*participating_tasks=*/{},
      [&barrier_status_2, &counter_2](const absl::Status& s, int64_t counter) {
        barrier_status_2 = s;
        counter_2 = counter;
      });

  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);
  TF_EXPECT_OK(barrier_status_2);
  EXPECT_EQ(counter_0, 0);
  EXPECT_EQ(counter_1, 0);
  EXPECT_EQ(counter_2, 0);

  // Second barrier.
  service->BarrierAsync(barrier_id, 1, timeout, 0,
                        /*participating_tasks=*/{},
                        [&barrier_status_0_2, &counter_0_2](
                            const absl::Status& s, int64_t counter) {
                          barrier_status_0_2 = s;
                          counter_0_2 = counter;
                        });
  service->BarrierAsync(barrier_id, 1, timeout, 1,
                        /*participating_tasks=*/{},
                        [&barrier_status_1_2, &counter_1_2](
                            const absl::Status& s, int64_t counter) {
                          barrier_status_1_2 = s;
                          counter_1_2 = counter;
                        });
  service->BarrierAsync(barrier_id, 1, timeout, 2,
                        /*participating_tasks=*/{},
                        [&barrier_status_2_2, &counter_2_2](
                            const absl::Status& s, int64_t counter) {
                          barrier_status_2_2 = s;
                          counter_2_2 = counter;
                        });

  TF_EXPECT_OK(barrier_status_0_2);
  TF_EXPECT_OK(barrier_status_1_2);
  TF_EXPECT_OK(barrier_status_2_2);
  EXPECT_EQ(counter_0_2, 1);
  EXPECT_EQ(counter_1_2, 1);
  EXPECT_EQ(counter_2_2, 1);
}

TEST(CoordinationService, Barrier_OngoingButMismatchedCounter_InternalError) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0, barrier_status_1,
      barrier_status_2 = absl::UnknownError("Unknown");
  int64_t counter_0, counter_1, counter_2 = -1;

  service->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0, &counter_0](const absl::Status& s, int64_t counter) {
        barrier_status_0 = s;
        counter_0 = counter;
      });
  // Task 1 specifies different counter!
  service->BarrierAsync(
      barrier_id, 1, timeout, 1,
      /*participating_tasks=*/{},
      [&barrier_status_1, &counter_1](const absl::Status& s, int64_t counter) {
        barrier_status_1 = s;
        counter_1 = counter;
      });

  EXPECT_THAT(barrier_status_0, StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(barrier_status_1, StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(counter_0, 0);
  EXPECT_EQ(counter_1, 0);  // Specifies the service-side barrier counter.

  // Try failed barrier with correct counter, return same internal error
  // immediately.
  service->BarrierAsync(
      barrier_id, 0, timeout, 2,
      /*participating_tasks=*/{},
      [&barrier_status_2, &counter_2](const absl::Status& s, int64_t counter) {
        barrier_status_2 = s;
        counter_2 = counter;
      });

  EXPECT_THAT(barrier_status_2, StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(counter_2, 0);
}

TEST(CoordinationService, SecondBarrier_UseWrongCounter_InternalError) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0, barrier_status_1,
      barrier_status_2 = absl::UnknownError("Unknown");
  int64_t counter_0, counter_1, counter_2 = -1;
  absl::Notification timeout_n;

  // First barrier.
  service->BarrierAsync(barrier_id, 0, timeout, 0,
                        /*participating_tasks=*/{},
                        [](const absl::Status& s, int64_t counter) {});
  service->BarrierAsync(barrier_id, 0, timeout, 1,
                        /*participating_tasks=*/{},
                        [](const absl::Status& s, int64_t counter) {});
  service->BarrierAsync(barrier_id, 0, timeout, 2,
                        /*participating_tasks=*/{},
                        [](const absl::Status& s, int64_t counter) {});

  // Second barrier.
  // Specify same as previous barrier: return same result.
  service->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0, &counter_0](const absl::Status& s, int64_t counter) {
        barrier_status_0 = s;
        counter_0 = counter;
      });
  EXPECT_EQ(counter_0, 0);
  TF_EXPECT_OK(barrier_status_0);

  // Specify a counter that is too low: fail!
  service->BarrierAsync(
      barrier_id, -1, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_1, &counter_1](const absl::Status& s, int64_t counter) {
        barrier_status_1 = s;
        counter_1 = counter;
      });
  EXPECT_EQ(counter_1, 0);  // Specifies the service-side barrier counter.
  EXPECT_THAT(barrier_status_1, StatusIs(absl::StatusCode::kInternal));

  // Specify a counter that is too high: fail!
  service->BarrierAsync(
      barrier_id, 2, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_2, &counter_2](const absl::Status& s, int64_t counter) {
        barrier_status_2 = s;
        counter_2 = counter;
      });
  EXPECT_EQ(counter_2, 0);  // Specifies the service-side barrier counter.
  EXPECT_THAT(barrier_status_2, StatusIs(absl::StatusCode::kInternal));
}

TEST(CoordinationService, BarrierCancelled) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status;

  service->BarrierAsync(barrier_id, 0, timeout, 0,
                        /*participating_tasks=*/{},
                        [&barrier_status](absl::Status s, int64_t counter) {
                          barrier_status = s;
                        });
  absl::Status cancelled_status = service->CancelBarrier(barrier_id, 0, 0);

  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kCancelled));
  TF_EXPECT_OK(cancelled_status);
}

TEST(CoordinationService, CancelAfterBarrierHasPassed) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0 = absl::UnknownError("Uninitialized error.");
  absl::Status barrier_status_1 = absl::UnknownError("Uninitialized error.");
  absl::Status barrier_status_2 = absl::UnknownError("Uninitialized error.");

  service->BarrierAsync(barrier_id, 0, timeout, 0,
                        /*participating_tasks=*/{},
                        [&barrier_status_0](absl::Status s, int64_t counter) {
                          barrier_status_0 = s;
                        });
  service->BarrierAsync(barrier_id, 0, timeout, 1,
                        /*participating_tasks=*/{},
                        [&barrier_status_1](absl::Status s, int64_t counter) {
                          barrier_status_1 = s;
                        });
  service->BarrierAsync(barrier_id, 0, timeout, 2,
                        /*participating_tasks=*/{},
                        [&barrier_status_2](absl::Status s, int64_t counter) {
                          barrier_status_2 = s;
                        });
  // Cancel barrier should fail if barrier has already been passed.
  absl::Status cancelled_status = service->CancelBarrier(barrier_id, 0, 0);

  EXPECT_THAT(cancelled_status, StatusIs(absl::StatusCode::kFailedPrecondition,
                                         HasSubstr("already been passed")));
  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);
  TF_EXPECT_OK(barrier_status_2);
}

TEST(CoordinationService, CancelBarrier_WrongCounter_FailedPrecondition) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0 = absl::UnknownError("Uninitialized error.");
  absl::Status barrier_status_1 = absl::UnknownError("Uninitialized error.");
  absl::Status barrier_status_2 = absl::UnknownError("Uninitialized error.");
  absl::Status barrier_status_cancelled =
      absl::UnknownError("Uninitialized error.");

  // First barrier passes (counter: 0).
  service->BarrierAsync(barrier_id, 0, timeout, 0,
                        /*participating_tasks=*/{},
                        [&barrier_status_0](absl::Status s, int64_t counter) {
                          barrier_status_0 = s;
                        });
  service->BarrierAsync(barrier_id, 0, timeout, 1,
                        /*participating_tasks=*/{},
                        [&barrier_status_1](absl::Status s, int64_t counter) {
                          barrier_status_1 = s;
                        });
  service->BarrierAsync(barrier_id, 0, timeout, 2,
                        /*participating_tasks=*/{},
                        [&barrier_status_2](absl::Status s, int64_t counter) {
                          barrier_status_2 = s;
                        });
  TF_ASSERT_OK(barrier_status_0);
  TF_ASSERT_OK(barrier_status_1);
  TF_ASSERT_OK(barrier_status_2);
  // Second barrier (counter: 1)
  service->BarrierAsync(
      barrier_id, 1, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_cancelled](absl::Status s, int64_t counter) {
        barrier_status_cancelled = s;
      });
  // Specify low counter (e.g. due to restart on client-side).
  absl::Status cancelled_status_low_counter =
      service->CancelBarrier(barrier_id, 0, 0);
  EXPECT_THAT(barrier_status_cancelled,
              StatusIs(absl::StatusCode::kUnknown, HasSubstr("Uninitialized")));
  // Specify high counter (e.g. due to restart on service-side).
  absl::Status cancelled_status_high_counter =
      service->CancelBarrier(barrier_id, 2, 0);
  EXPECT_THAT(barrier_status_cancelled,
              StatusIs(absl::StatusCode::kUnknown, HasSubstr("Uninitialized")));
  // Specify correct counter.
  absl::Status cancelled_status_correct_counter =
      service->CancelBarrier(barrier_id, 1, 1);

  EXPECT_THAT(barrier_status_cancelled, StatusIs(absl::StatusCode::kCancelled));
  EXPECT_THAT(cancelled_status_low_counter,
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("likely due to a restart")));
  EXPECT_THAT(cancelled_status_high_counter,
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("likely due to a restart")));
  TF_EXPECT_OK(cancelled_status_correct_counter);
}

TEST(CoordinationService, PassedBarrierReturnsImmediately) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0;
  absl::Status barrier_status_1;
  absl::Status barrier_status_2;
  absl::Status barrier_status_repeat;
  absl::Notification n0;
  absl::Notification n1;
  absl::Notification n2;
  absl::Notification n_repeat;

  service->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0, &n0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
        n0.Notify();
      });
  service->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{},
      [&barrier_status_1, &n1](absl::Status s, int64_t counter) {
        barrier_status_1 = s;
        n1.Notify();
      });
  service->BarrierAsync(
      barrier_id, 0, timeout, 2,
      /*participating_tasks=*/{},
      [&barrier_status_2, &n2](absl::Status s, int64_t counter) {
        barrier_status_2 = s;
        n2.Notify();
      });
  // Repeated call should return the same result.
  service->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{},
      [&barrier_status_repeat, &n_repeat](absl::Status s, int64_t counter) {
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

TEST(CoordinationService, BarrierFailsIfTaskIsAlreadyInError) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  // Set task 0 to error state.
  ASSERT_OK(service->ReportTaskError(0, absl::InternalError("test_error")));
  absl::Status barrier_status;

  service->BarrierAsync(barrier_id, 0, timeout, 1,
                        /*participating_tasks=*/{},
                        [&barrier_status](absl::Status s, int64_t counter) {
                          barrier_status = s;
                        });

  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST(CoordinationService, BarrierFailsUponTaskError) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Notification n0;
  absl::Status barrier_status;

  service->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status, &n0](absl::Status s, int64_t counter) {
        barrier_status = s;
        n0.Notify();
      });
  ASSERT_OK(service->ReportTaskError(0, absl::InternalError("test_error")));
  n0.WaitForNotification();

  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST(CoordinationService,
     BarrierStillBlocksIfSameTaskCallsOngoingBarrierRepeatedly) {
  CoordinationService::Config config = DefaultConfig(3);
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0;
  absl::Status barrier_status_1;
  absl::Status barrier_status_2;
  absl::Notification n_0;
  absl::Notification n_1;
  absl::Notification n_2;

  service->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{0, 1},
      [&barrier_status_0, &n_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
        n_0.Notify();
      });
  // Duplicate call.
  service->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{0, 1},
      [&barrier_status_1, &n_1](absl::Status s, int64_t counter) {
        barrier_status_1 = s;
        n_1.Notify();
      });
  // All listed tasks passed the barrier.
  // Second call should cancel the first.
  EXPECT_TRUE(n_0.HasBeenNotified());
  EXPECT_THAT(barrier_status_0, StatusIs(absl::StatusCode::kCancelled));
  EXPECT_FALSE(n_1.HasBeenNotified());

  service->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{0, 1},
      [&barrier_status_2, &n_2](absl::Status s, int64_t counter) {
        barrier_status_2 = s;
        n_2.Notify();
      });
  TF_EXPECT_OK(barrier_status_1);
  TF_EXPECT_OK(barrier_status_2);
}

TEST(CoordinationService, ResetAndRegisterAgain) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  TF_EXPECT_OK(service->RegisterTask(0, incarnation_0));

  TF_EXPECT_OK(service->ResetTask(0));

  // Task should be allowed to register again after being reset.
  TF_EXPECT_OK(service->RegisterTask(0, incarnation_0));
}

TEST(CoordinationService, Reset_HeartbeatsAreAcceptedForAGracePeriod) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  TF_EXPECT_OK(service->RegisterTask(0, incarnation_0));

  TF_EXPECT_OK(service->ResetTask(0));
  // Heartbeat should be allowed for a short grace period after reset.
  TF_EXPECT_OK(service->RecordHeartbeat(0, incarnation_0));

  // Heartbeat failure should be triggered for disconnected task after grace
  // period.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(3 * kHeartbeatTimeout));
  EXPECT_THAT(service->RecordHeartbeat(0, incarnation_0),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CoordinationService, Reset_FailsOngoingBarrier) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  TF_EXPECT_OK(service->RegisterTask(0, incarnation_0));
  absl::Status barrier_status;
  absl::Notification barrier_n;
  service->BarrierAsync(
      "ongoing_barrier", 0, absl::InfiniteDuration(), 0,
      /*participating_tasks=*/{},
      [&barrier_status, &barrier_n](absl::Status s, int64_t counter) {
        barrier_status = s;
        barrier_n.Notify();
      });

  TF_EXPECT_OK(service->ResetTask(0));

  // Ongoing barrier should fail with error after shutdown.
  EXPECT_TRUE(barrier_n.HasBeenNotified());
  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST(CoordinationService, Shutdown_HeartbeatsAreAcceptedForAGracePeriod) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  TF_EXPECT_OK(service->RegisterTask(0, incarnation_0));

  absl::Notification n;
  service->ShutdownTaskAsync(0, [&n](absl::Status s) {
    TF_EXPECT_OK(s);
    n.Notify();
  });
  n.WaitForNotification();

  // Heartbeat should be allowed for a short grace period after shutdown.
  TF_EXPECT_OK(service->RecordHeartbeat(0, incarnation_0));

  // Heartbeat failure should be triggered for disconnected task after grace
  // period.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(3 * kHeartbeatTimeout));
  EXPECT_THAT(service->RecordHeartbeat(0, incarnation_0),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CoordinationService, Shutdown_FailsOngoingBarrier) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  TF_EXPECT_OK(service->RegisterTask(0, incarnation_0));
  absl::Status barrier_status;
  absl::Notification barrier_n;
  service->BarrierAsync(
      "ongoing_barrier", 0, absl::InfiniteDuration(), 0,
      /*participating_tasks=*/{},
      [&barrier_status, &barrier_n](absl::Status s, int64_t counter) {
        barrier_status = s;
        barrier_n.Notify();
      });

  absl::Notification shutdown_n;
  service->ShutdownTaskAsync(0, [&shutdown_n](absl::Status s) {
    TF_EXPECT_OK(s);
    shutdown_n.Notify();
  });
  shutdown_n.WaitForNotification();

  // Ongoing barrier should fail with error after shutdown.
  EXPECT_TRUE(barrier_n.HasBeenNotified());
  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST(CoordinationService, ShutdownWithBarrier_BarrierSucceeds) {
  CoordinationService::Config config = DefaultConfig(2);
  config.shutdown_barrier_timeout = kShutdownBarrierTimeout;
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  TF_EXPECT_OK(service->RegisterTask(0, incarnation_0));
  TF_EXPECT_OK(service->RegisterTask(1, incarnation_1));
  absl::Status barrier_status;
  absl::Status barrier_status_2;

  service->ShutdownTaskAsync(
      0, [&barrier_status](absl::Status s) { barrier_status = s; });
  service->ShutdownTaskAsync(
      1, [&barrier_status_2](absl::Status s) { barrier_status_2 = s; });

  TF_EXPECT_OK(barrier_status);
  TF_EXPECT_OK(barrier_status_2);

  // Confirm that both tasks have disconnected.
  // Note: this should not happen in prod where RegisterTask() is called after
  // Shutdown(), which is prevented by agent-side logic.
  TF_EXPECT_OK(service->RegisterTask(0, incarnation_0));
  TF_EXPECT_OK(service->RegisterTask(1, incarnation_1));
}

TEST(CoordinationService,
     ShutdownWithBarrier_BarrierFails_TaskDisconnectsOtherTaskIsAlerted) {
  CoordinationService::Config config = DefaultConfig(2);
  config.shutdown_barrier_timeout = kShutdownBarrierTimeout;
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  TF_EXPECT_OK(service->RegisterTask(0, incarnation_0));
  TF_EXPECT_OK(service->RegisterTask(1, incarnation_1));
  absl::Status barrier_status;

  absl::Notification n;
  service->ShutdownTaskAsync(0, [&n, &barrier_status](absl::Status s) {
    barrier_status = s;
    n.Notify();
  });
  // Block until barrier times out.
  n.WaitForNotification();

  EXPECT_THAT(barrier_status,
              StatusIs(absl::StatusCode::kInternal, HasSubstr("timed out")));
  // Task 0 should not be allowed to silently register again to the
  // same service instance, regardless of incarnation (same process or
  // restarted).
  EXPECT_THAT(service->RegisterTask(0, incarnation_0),
              StatusIs(absl::StatusCode::kAborted));
  EXPECT_THAT(service->RegisterTask(0, incarnation_1),
              StatusIs(absl::StatusCode::kAborted));
}

TEST(CoordinationService,
     ShutdownWithBarrier_BarrierFailsWithoutClientConnection_SetTaskToError) {
  CoordinationService::Config config = DefaultConfig(2);
  config.shutdown_barrier_timeout = kShutdownBarrierTimeout;
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  TF_EXPECT_OK(service->RegisterTask(0, incarnation_0));
  TF_EXPECT_OK(service->RegisterTask(1, incarnation_1));
  absl::Status barrier_status;

  absl::Notification n;
  service->ShutdownTaskAsync(0, [&n, &barrier_status](absl::Status s) {
    barrier_status = s;
    n.Notify();
  });
  // Block until barrier times out.
  n.WaitForNotification();
  // Provide time for coordination service to shut down after barrier timeout.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(absl::Seconds(1)));

  EXPECT_THAT(barrier_status,
              StatusIs(absl::StatusCode::kInternal, HasSubstr("timed out")));

  // Task 1 sends unexpected heartbeat that is aborted because it is in error.
  absl::Status s = service->RecordHeartbeat(1, incarnation_1);

  EXPECT_THAT(s, StatusIs(absl::StatusCode::kAborted));
}

TEST(CoordinationService, BarrierFailsIfTaskIsInError) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  ASSERT_OK(service->RegisterTask(0, incarnation_0));
  ASSERT_OK(service->RegisterTask(1, incarnation_1));
  absl::Notification n0;
  absl::Status barrier_status;
  // No heartbeat for a while, leader consider the task as stale.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));

  // Barrier should fail when called after stale task is set to error.
  service->BarrierAsync("barrier_id", 0, absl::Seconds(5), 0,
                        /*participating_tasks=*/{},
                        [&](absl::Status s, int64_t counter) {
                          barrier_status = s;
                          n0.Notify();
                        });

  n0.WaitForNotification();
  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST(CoordinationService, BarrierWithParticipatingTasksFailsIfTaskIsStale) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  ASSERT_OK(service->RegisterTask(0, incarnation_0));
  ASSERT_OK(service->RegisterTask(1, incarnation_1));
  absl::Notification n0;
  absl::Status barrier_status;
  // No heartbeat for a while, leader consider the task as stale.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));

  service->BarrierAsync("barrier_id", 0, absl::Seconds(5), 0,
                        /*participating_tasks=*/{0},
                        [&](absl::Status s, int64_t counter) {
                          barrier_status = s;
                          n0.Notify();
                        });

  n0.WaitForNotification();
  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST(CoordinationService, BarrierFailsAfterErrorPollingResponse) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  ASSERT_OK(service->RegisterTask(0, incarnation_0));
  ASSERT_OK(service->RegisterTask(1, incarnation_1));
  // Use notifications to guarantee the ordering of operations across threads.
  absl::Notification n0, n1;
  absl::Status s0, s1;

  service->PollForErrorAsync(0, [&](const absl::Status& status) {
    s0 = status;
    n0.Notify();
  });
  service->PollForErrorAsync(1, [&](const absl::Status& status) {
    s1 = status;
    n1.Notify();
  });

  // No heartbeat for a while, leader consider the task as stale. The error will
  // be propagated through error polling.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));

  // Make sure the StatusCallbacks are called before the barrier is called.
  n0.WaitForNotification();
  n1.WaitForNotification();
  // The heartbeat error should be propagated to all tasks.
  EXPECT_THAT(s0, StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_THAT(s1, StatusIs(absl::StatusCode::kUnavailable));

  absl::Notification n_barrier;
  absl::Status barrier_status;
  // Barrier should fail when called after the error is propagated.
  service->BarrierAsync("barrier_id", 0, absl::Seconds(5), 0,
                        /*participating_tasks=*/{},
                        [&](absl::Status s, int64_t counter) {
                          barrier_status = s;
                          n_barrier.Notify();
                        });

  n_barrier.WaitForNotification();
  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST(CoordinationService, BarrierWithSubsetFailsIfTaskIsStale) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  ASSERT_OK(service->RegisterTask(0, incarnation_0));
  ASSERT_OK(service->RegisterTask(1, incarnation_1));
  absl::Notification n0;
  absl::Status barrier_status;
  // No heartbeat for a while, leader consider the task as stale.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));

  // Barrier should fail if task is in error.
  // Note that this is same as above test, but the barrier only blocks for task
  // 0.
  service->BarrierAsync("barrier_id", 0, absl::Seconds(5), 0,
                        /*participating_tasks=*/{0},
                        [&](absl::Status s, int64_t counter) {
                          barrier_status = s;
                          n0.Notify();
                        });

  n0.WaitForNotification();
  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST(CoordinationService, RecoverableTaskWillNotPropagateError) {
  CoordinationService::Config config = DefaultConfig(2);
  config.recoverable = true;
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};

  TF_EXPECT_OK(service->RegisterTask(0, incarnation_0));
  TF_EXPECT_OK(service->RegisterTask(1, incarnation_1));

  ASSERT_OK(service->ReportTaskError(0, absl::InternalError("test_error")));
}

TEST(CoordinationService,
     RecoverableTaskWithErrorPollingWillNotPropagateError) {
  CoordinationService::Config config = DefaultConfig(2);
  config.recoverable = true;
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  // These callbacks may be invoked after this test (e.g. cancellations during
  // coord service dtor), so we use shared pointers to extend their lifetimes
  // beyond the test to avoid use-after-free errors.
  auto s0 = std::make_shared<absl::Status>();
  auto s1 = std::make_shared<absl::Status>();
  TF_EXPECT_OK(service->RegisterTask(0, incarnation_0));
  service->PollForErrorAsync(
      0, [s0](const absl::Status& status) { *s0 = status; });
  TF_EXPECT_OK(service->RegisterTask(1, incarnation_1));
  service->PollForErrorAsync(
      1, [s1](const absl::Status& status) { *s1 = status; });

  ASSERT_OK(service->ReportTaskError(0, absl::InternalError("test_error")));

  // Since no error propagation for recoverable tasks, other tasks should work
  // as normal.
  EXPECT_THAT(*s0, StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(*s1, StatusIs(absl::StatusCode::kOk));
}

TEST(CoordinationService, RecoverableTaskReportErrorResetAndRegisterAgain) {
  CoordinationService::Config config = DefaultConfig(2);
  config.recoverable = true;
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  const IncarnationId incarnation_0_new{tsl::random::New64()};

  TF_EXPECT_OK(service->RegisterTask(0, incarnation_0));
  TF_EXPECT_OK(service->RegisterTask(1, incarnation_1));

  ASSERT_OK(service->ReportTaskError(0, absl::InternalError("test_error")));

  EXPECT_THAT(service->RecordHeartbeat(0, incarnation_0),
              StatusIs(absl::StatusCode::kAborted));

  // Reset and register the error task again, both tasks should be healthy.
  TF_EXPECT_OK(service->ResetTask(0));
  TF_EXPECT_OK(service->RegisterTask(0, incarnation_0_new));
  TF_EXPECT_OK(service->RecordHeartbeat(0, incarnation_0_new));
}

TEST(CoordinationService, DoNotAllowPollForErrorIfNotInCluster) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  absl::Status s;

  service->PollForErrorAsync(-1,
                             [&](const absl::Status& status) { s = status; });

  EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument,
                          HasSubstr("not in the cluster")));
}

TEST(CoordinationService, DoNotAllowPollForErrorIfTaskNotRegistered) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  absl::Status s;

  service->PollForErrorAsync(0,
                             [&](const absl::Status& status) { s = status; });

  EXPECT_THAT(s, StatusIs(absl::StatusCode::kFailedPrecondition,
                          HasSubstr("has not been registered")));
}

TEST(CoordinationService, AllowPollForErrorWithinGracePeriodIfTaskHasShutDown) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  absl::Status s;
  ASSERT_OK(service->RegisterTask(0, incarnation_0));
  ASSERT_OK(service->RegisterTask(1, incarnation_1));
  service->ShutdownTaskAsync(0, [&](const absl::Status& status) {});
  service->ShutdownTaskAsync(1, [&](const absl::Status& status) {});

  service->PollForErrorAsync(0,
                             [&](const absl::Status& status) { s = status; });
  // Stop the service.
  service.reset();
  // The error polling request will still proceed because of grace period. It
  // will be cancelled.
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kCancelled));
}

TEST(CoordinationService, DoNotAllowPollForErrorIfTaskHasShutDown) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  absl::Status s;
  ASSERT_OK(service->RegisterTask(0, incarnation_0));
  ASSERT_OK(service->RegisterTask(1, incarnation_1));
  service->ShutdownTaskAsync(0, [&](const absl::Status& status) {});
  service->ShutdownTaskAsync(1, [&](const absl::Status& status) {});

  // Sleep past the grace period.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));
  service->PollForErrorAsync(0,
                             [&](const absl::Status& status) { s = status; });
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kFailedPrecondition,
                          HasSubstr("has disconnected")));
}

TEST(CoordinationService, DoNotAllowPollForErrorAfterReset) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  absl::Status s;
  ASSERT_OK(service->RegisterTask(0, incarnation_0));
  ASSERT_OK(service->ResetTask(0));

  // Sleep past the grace period.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));
  service->PollForErrorAsync(0,
                             [&](const absl::Status& status) { s = status; });
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kFailedPrecondition,
                          HasSubstr("has disconnected")));
}

TEST(CoordinationService, DoNotAllowPollForErrorWhenInErrorState) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  absl::Status s;
  ASSERT_OK(service->RegisterTask(0, incarnation_0));
  ASSERT_OK(service->ReportTaskError(0, absl::InternalError("test_error")));

  service->PollForErrorAsync(0,
                             [&](const absl::Status& status) { s = status; });
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(CoordinationService, DoNotAllowPollForErrorIfTaskIsStale) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  ASSERT_OK(service->RegisterTask(0, incarnation_0));
  ASSERT_OK(service->RegisterTask(1, incarnation_1));
  // No heartbeat for a while, leader consider the task as stale.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));

  absl::Status s;
  service->PollForErrorAsync(0,
                             [&](const absl::Status& status) { s = status; });

  EXPECT_THAT(s, StatusIs(absl::StatusCode::kFailedPrecondition,
                          HasSubstr("already in error")));
}

TEST(CoordinationService,
     CanPropagateTaskRegistrationErrorThroughErrorPolling) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  ASSERT_OK(service->RegisterTask(0, incarnation_0));
  ASSERT_OK(service->RegisterTask(1, incarnation_1));
  absl::Status s0;
  // Start polling for error from `0`.
  service->PollForErrorAsync(0,
                             [&](const absl::Status& status) { s0 = status; });

  // Let registration of `1` fail due to incarnation mismatch.
  ASSERT_THAT(service->RegisterTask(1, incarnation_0),
              StatusIs(absl::StatusCode::kAborted));

  // The first error polling request will get the error propagated from the
  // registration failure.
  EXPECT_THAT(s0, StatusIs(absl::StatusCode::kAborted));
}

TEST(CoordinationService, LatePollingTaskCanGetError) {
  CoordinationService::Config config = DefaultConfig(2);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  ASSERT_OK(service->RegisterTask(0, incarnation_0));
  ASSERT_OK(service->RegisterTask(1, incarnation_1));
  std::vector<absl::Status> statuses;
  statuses.reserve(2);
  service->PollForErrorAsync(
      0, [&](const absl::Status& status) { statuses.push_back(status); });

  // Fail `0` with an error because `1` polls for error.
  ASSERT_OK(service->ReportTaskError(
      0, absl::FailedPreconditionError("test_error_from_task_0")));

  // Poll for error from `1` after the error has been propagated to other
  // tasks.
  service->PollForErrorAsync(
      1, [&](const absl::Status& status) { statuses.push_back(status); });

  // Make sure the error is propagated to both tasks.
  EXPECT_EQ(statuses.size(), 2);
  EXPECT_THAT(statuses, Each(StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("test_error_from_task_0"))));
}

TEST(CoordinationService,
     RegisterWithBarrier_OldHeartbeat_RestartedTasksCanReconnect) {
  CoordinationService::Config config = DefaultConfig(2);
  config.cluster_register_with_barrier = true;
  config.cluster_register_timeout = absl::Seconds(1);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  // Service restarted.
  // Old task 0 sends an unexpected heartbeat, which should fail.
  ASSERT_THAT(service->RecordHeartbeat(0, incarnation_0 - 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  absl::Status task0_status = absl::InternalError("uninitialized_status");
  // Task 0 registers first.
  service->RegisterTaskAsync(0, incarnation_0, [](const absl::Status& s) {});
  // Task 0 restarts with a new incarnation, and registers again.
  // This should be allowed since all tasks have not joined the cluster yet.
  service->RegisterTaskAsync(0, incarnation_0 + 1,
                             [&](const absl::Status& s) { task0_status = s; });
  // Now all tasks will register in a synchronized fashion due to the barrier.
  EXPECT_OK(service->RegisterTask(1, incarnation_1));
  EXPECT_OK(task0_status);
}

TEST(CoordinationService, RegisterWithBarrier_RestartBeforeBarrier_Succeeds) {
  CoordinationService::Config config = DefaultConfig(2);
  config.cluster_register_with_barrier = true;
  config.cluster_register_timeout = absl::Seconds(1);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  absl::Status task0_status = absl::InternalError("uninitialized_status");
  absl::Status restarted_task0_status =
      absl::InternalError("uninitialized_status");
  // Task 0 registers first.
  service->RegisterTaskAsync(0, incarnation_0,
                             [&](const absl::Status& s) { task0_status = s; });
  // Task 0 restarts with a new incarnation, and registers again.
  // This should be allowed since all tasks have not joined the cluster yet.
  service->RegisterTaskAsync(0, incarnation_0 + 1, [&](const absl::Status& s) {
    restarted_task0_status = s;
  });
  // Now all tasks will register in a synchronized fashion due to the barrier.
  ASSERT_OK(service->RegisterTask(1, incarnation_1));
  ASSERT_THAT(task0_status, StatusIs(absl::StatusCode::kAlreadyExists));
  ASSERT_OK(restarted_task0_status);
  // Task 0 joins again with the same incarnation.
  // This is okay, it didn't restart, probably sent RPC twice due to network
  // retries.
  EXPECT_OK(service->RegisterTask(0, incarnation_0 + 1));
}

TEST(CoordinationService, RegisterWithBarrier_RestartAfterBarrier_Fails) {
  CoordinationService::Config config = DefaultConfig(2);
  config.cluster_register_with_barrier = true;
  config.cluster_register_timeout = absl::Seconds(1);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  const IncarnationId incarnation_1{tsl::random::New64()};
  absl::Status task0_status = absl::InternalError("uninitialized_status");
  // Task 0 registers first.
  service->RegisterTaskAsync(0, incarnation_0,
                             [&](const absl::Status& s) { task0_status = s; });
  // Now all tasks will register in a synchronized fashion due to the barrier.
  ASSERT_OK(service->RegisterTask(1, incarnation_1));
  ASSERT_OK(task0_status);

  // Task 0 restarts again with a new incarnation.
  // This should fail since this happens after the initial register barrier
  // (i.e. all tasks already acked once).
  ASSERT_THAT(service->RegisterTask(0, incarnation_0 + 2),
              StatusIs(absl::StatusCode::kAborted));
  // All tasks should be set to error and unable to start any barriers.
  absl::Notification n;
  absl::Status barrier_status;
  service->BarrierAsync("barrier_id", 0, absl::Seconds(10), 0, {},
                        [&](const absl::Status& s, int64_t counter) {
                          n.Notify();
                          barrier_status = s;
                        });
  n.WaitForNotification();
  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST(CoordinationService, RegisterWithBarrier_Timeout) {
  CoordinationService::Config config = DefaultConfig(2);
  config.cluster_register_with_barrier = true;
  config.cluster_register_timeout = absl::Seconds(1);
  std::unique_ptr<CoordinationService> service = Start(config);
  const IncarnationId incarnation_0{tsl::random::New64()};
  // Task 0 joins without task 1. Times out eventually as this function is
  // blocking.
  EXPECT_THAT(service->RegisterTask(0, incarnation_0),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST(CoordinationService, SuccessfulGetAliveTasks) {
  CoordinationService::Config config = DefaultConfig(3);
  config.recoverable = true;
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  // This test has three tasks successfully call GetAliveTasks.
  absl::BlockingCounter finished(3);
  auto done = [&](const absl::Status& status,
                  const std::vector<CoordinationService::TaskId>& alive_tasks,
                  const std::vector<IncarnationId>& incarnations) {
    EXPECT_OK(status);
    EXPECT_THAT(alive_tasks, UnorderedElementsAreArray(tasks.tasks));
    EXPECT_THAT(incarnations,
                UnorderedElementsAre(IncarnationId(0), IncarnationId(1),
                                     IncarnationId(2)));
    finished.DecrementCount();
  };
  service->GetAliveTasksAsync(0, tasks.tasks, done);
  service->GetAliveTasksAsync(1, tasks.tasks, done);
  service->GetAliveTasksAsync(2, tasks.tasks, done);
  finished.Wait();
}

TEST(CoordinationService, FailedTaskBeforeCallingGetAliveTasks) {
  CoordinationService::Config config = DefaultConfig(3);
  config.recoverable = true;
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  // This test involves three tasks: 0, 1, and 2. Task 2 is failed. Then, tasks
  // 0 and 1 call GetAliveTasks on tasks [0, 1, 2], which should return [0, 1].
  absl::BlockingCounter finished(2);
  auto done = [&](const absl::Status& status,
                  const std::vector<CoordinationService::TaskId>& alive_tasks,
                  const std::vector<IncarnationId>& incarnations) {
    EXPECT_OK(status);
    EXPECT_THAT(alive_tasks, UnorderedElementsAre(0, 1));
    EXPECT_THAT(incarnations,
                UnorderedElementsAre(IncarnationId(0), IncarnationId(1)));
    finished.DecrementCount();
  };
  ASSERT_OK(service->ReportTaskError(2, absl::InternalError("failed")));
  service->GetAliveTasksAsync(0, tasks.tasks, done);
  service->GetAliveTasksAsync(1, tasks.tasks, done);
  finished.Wait();
}

TEST(CoordinationService, FailedTaskAfterCallingGetAliveTasks) {
  CoordinationService::Config config = DefaultConfig(3);
  config.recoverable = true;
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  // This test involves three tasks: 0, 1, and 2. Tasks 0 and 1 call
  // GetAliveTasks on tasks [0, 1, 2]. Then, task 2 is failed, which should
  // cause GetAliveTasks to return [0, 1].
  absl::BlockingCounter finished(2);
  auto done = [&](const absl::Status& status,
                  const std::vector<CoordinationService::TaskId>& alive_tasks,
                  const std::vector<IncarnationId>& incarnations) {
    EXPECT_OK(status);
    EXPECT_THAT(alive_tasks, UnorderedElementsAre(0, 1));
    EXPECT_THAT(incarnations,
                UnorderedElementsAre(IncarnationId(0), IncarnationId(1)));
    finished.DecrementCount();
  };
  service->GetAliveTasksAsync(0, tasks.tasks, done);
  service->GetAliveTasksAsync(1, tasks.tasks, done);
  ASSERT_OK(service->ReportTaskError(2, absl::InternalError("failed")));
  finished.Wait();
}

TEST(CoordinationService, ConcurrentGetAliveTasks) {
  CoordinationService::Config config = DefaultConfig(3);
  config.recoverable = true;
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  // This test involves three tasks: 0, 1, and 2. Tasks 0 and 1 call
  // GetAliveTasks on tasks [0, 1], and concurrently tasks 1 and 2 call
  // GetAliveTasks on tasks [1, 2].

  // GetAliveTasks on tasks 0 and 1.
  std::vector<CoordinationService::TaskId> tasks_01{0, 1};
  absl::BlockingCounter finished_01(2);
  auto done_01 =
      [&](const absl::Status& status,
          const std::vector<CoordinationService::TaskId>& alive_tasks,
          const std::vector<IncarnationId>& incarnations) {
        EXPECT_OK(status);
        EXPECT_THAT(alive_tasks, UnorderedElementsAre(0, 1));
        EXPECT_THAT(incarnations,
                    UnorderedElementsAre(IncarnationId(0), IncarnationId(1)));
        finished_01.DecrementCount();
      };

  // GetAliveTasks on tasks 1 and 2.
  std::vector<CoordinationService::TaskId> tasks_12{1, 2};
  absl::BlockingCounter finished_12(2);
  auto done_12 =
      [&](const absl::Status& status,
          const std::vector<CoordinationService::TaskId>& alive_tasks,
          const std::vector<IncarnationId>& incarnations) {
        EXPECT_OK(status);
        EXPECT_THAT(alive_tasks, UnorderedElementsAre(1, 2));
        EXPECT_THAT(incarnations,
                    UnorderedElementsAre(IncarnationId(1), IncarnationId(2)));
        finished_12.DecrementCount();
      };

  // Run both GetAliveTasks concurrently.
  service->GetAliveTasksAsync(0, tasks_01, done_01);
  service->GetAliveTasksAsync(1, tasks_12, done_12);
  service->GetAliveTasksAsync(1, tasks_01, done_01);
  service->GetAliveTasksAsync(2, tasks_12, done_12);
  finished_01.Wait();
  finished_12.Wait();
}

TEST(CoordinationService, CallingGetAliveTasksWithoutBeingAMember) {
  CoordinationService::Config config = DefaultConfig(3);
  config.recoverable = true;
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  // This test includes calls to GetAliveTasks where the requesting task is not
  // included in the specified set of tasks. This should return an error.
  absl::BlockingCounter finished(3);
  auto done = [&](const absl::Status& status,
                  const std::vector<CoordinationService::TaskId>&,
                  const std::vector<IncarnationId>&) {
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
    finished.DecrementCount();
  };

  service->GetAliveTasksAsync(0, {1, 2}, done);
  service->GetAliveTasksAsync(1, {0, 2}, done);
  service->GetAliveTasksAsync(2, {0, 1}, done);
  finished.Wait();
}

TEST(CoordinationService, RedundantGetAliveTasks) {
  CoordinationService::Config config = DefaultConfig(3);
  config.recoverable = true;
  std::unique_ptr<CoordinationService> service = Start(config);
  Tasks tasks = RegisterTasks(config, *service);
  // This test has three tasks call GetAliveTasks, with the twist that some
  // tasks call GetAliveTasks multiple times.
  absl::BlockingCounter finished(6);
  auto done = [&](const absl::Status& status,
                  const std::vector<CoordinationService::TaskId>& alive_tasks,
                  const std::vector<IncarnationId>&) {
    EXPECT_OK(status);
    EXPECT_THAT(alive_tasks, UnorderedElementsAreArray(tasks.tasks));
    finished.DecrementCount();
  };
  service->GetAliveTasksAsync(0, tasks.tasks, done);
  service->GetAliveTasksAsync(0, tasks.tasks, done);
  service->GetAliveTasksAsync(0, tasks.tasks, done);
  service->GetAliveTasksAsync(1, tasks.tasks, done);
  service->GetAliveTasksAsync(1, tasks.tasks, done);
  service->GetAliveTasksAsync(2, tasks.tasks, done);
  finished.Wait();
}

}  // namespace
}  // namespace xla
