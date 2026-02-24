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
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "xla/pjrt/distributed/coordination/coordination_client.h"
#include "xla/pjrt/distributed/coordination/coordination_service_error_util.h"
#include "xla/pjrt/distributed/coordination/test_device.pb.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/protobuf/coordination_config.pb.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/random.h"

namespace xla {
namespace {

using ::testing::Each;
using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Matcher;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;
using ::testing::status::StatusIs;
using ::tsl::proto_testing::EqualsProto;

using tensorflow::CoordinatedJob;
using tensorflow::CoordinatedTask;
using tensorflow::CoordinationServiceConfig;
using tensorflow::DeviceInfo;
using tensorflow::KeyValueEntry;
using xla::TestDevice;

constexpr absl::Duration kHeartbeatTimeout = absl::Seconds(2);
constexpr absl::Duration kShutdownBarrierTimeout = absl::Milliseconds(500);
constexpr char kCoordinationServiceType[] = "standalone";

KeyValueEntry CreateKv(const std::string& key, const std::string& value) {
  KeyValueEntry kv;
  kv.set_key(key);
  kv.set_value(value);
  return kv;
}

CoordinationService::Config GetCoordinationServiceConfig(int num_tasks,
                                                         bool recoverable) {
  CoordinationService::Config config;
  config.num_tasks = num_tasks;
  config.recoverable = recoverable;
  return config;
}

class TestCoordinationClient : public CoordinationClient {
 public:
  TestCoordinationClient() = default;

  absl::Status GetStatus() {
    absl::MutexLock l(mu_);
    return status_;
  }

  void RegisterTaskAsync(tsl::CallOptions* opts,
                         const RegisterTaskRequest* request,
                         RegisterTaskResponse* response,
                         tsl::StatusCallback done) override {
    done(absl::OkStatus());
  }

#define UNIMPLEMENTED(method)                                              \
  void method##Async(const method##Request* request,                       \
                     method##Response* response, tsl::StatusCallback done) \
      override {                                                           \
    done(absl::UnimplementedError(#method "Async"));                       \
  }

  UNIMPLEMENTED(ResetTask);
  UNIMPLEMENTED(InsertKeyValue);
  UNIMPLEMENTED(TryGetKeyValue);
  UNIMPLEMENTED(IncrementKeyValue);
  UNIMPLEMENTED(GetKeyValueDir);
  UNIMPLEMENTED(DeleteKeyValue);
  UNIMPLEMENTED(CancelBarrier);
  UNIMPLEMENTED(GetAliveTasks);
#undef UNIMPLEMENTED

#define UNIMPLEMENTED_WITH_CALL_OPTS(method)                           \
  void method##Async(                                                  \
      tsl::CallOptions* call_opts, const method##Request* request,     \
      method##Response* response, tsl::StatusCallback done) override { \
    done(absl::UnimplementedError(#method "Async"));                   \
  }

  UNIMPLEMENTED_WITH_CALL_OPTS(GetKeyValue);
  UNIMPLEMENTED_WITH_CALL_OPTS(Barrier);
  UNIMPLEMENTED_WITH_CALL_OPTS(Heartbeat);
  UNIMPLEMENTED_WITH_CALL_OPTS(ShutdownTask);
  UNIMPLEMENTED_WITH_CALL_OPTS(PollForError);
  UNIMPLEMENTED_WITH_CALL_OPTS(WatchJobState);
#undef UNIMPLEMENTED_WITH_CALL_OPTS

 private:
  absl::Mutex mu_;
  absl::Status status_ ABSL_GUARDED_BY(mu_);
};

class CoordinationBarrierTest : public ::testing::Test {
 protected:
  explicit CoordinationBarrierTest(bool recoverable = false) {
    // Set up fake cluster with 3 tasks.
    const int num_tasks = 3;
    for (int i = 0; i < num_tasks; ++i) {
      auto client = std::make_unique<TestCoordinationClient>();
      tasks_.push_back(i);
      clients_.push_back(std::move(client));
    }
    CoordinationService::Config config =
        GetCoordinationServiceConfig(num_tasks, recoverable);

    coord_service_ =
        std::make_unique<CoordinationService>(tsl::Env::Default(), config);
    // Register the tasks.
    for (int i = 0; i < num_tasks; ++i) {
      absl::Status s =
          coord_service_->RegisterTask(tasks_[i], IncarnationId(i));
      if (!s.ok()) {
        LOG(FATAL) << "RegisterTask() failed in CoordinationBarrierTest(): "
                   << s;
      }
    }
  }

  const std::vector<CoordinationService::TaskId>& tasks() { return tasks_; }

  CoordinationService* GetCoordinationService() { return coord_service_.get(); }

  std::vector<TestCoordinationClient*> GetClients() {
    std::vector<TestCoordinationClient*> clients;
    for (const auto& client : clients_) {
      clients.push_back(client.get());
    }
    return clients;
  }

 private:
  std::unique_ptr<CoordinationService> coord_service_;
  std::vector<CoordinationService::TaskId> tasks_;
  std::vector<std::unique_ptr<TestCoordinationClient>> clients_;
};

// Sets up coordination service that expects 2 worker tasks.
class CoordinateTwoTasksTest : public ::testing::Test {
 protected:
  // Set up coordination service.
  void EnableCoordinationService(
      bool enable_shutdown_barrier = false,
      bool enable_register_barrier = false,
      bool set_worker_job_recoverable = false,
      bool allow_new_incarnation_to_reconnect = false) {
    CoordinationService::Config config = GetCoordinationServiceConfig(
        /*num_tasks=*/2, /*recoverable=*/set_worker_job_recoverable);
    config.heartbeat_timeout = kHeartbeatTimeout;
    if (enable_shutdown_barrier) {
      config.shutdown_barrier_timeout = kShutdownBarrierTimeout;
    }
    if (enable_register_barrier) {
      config.cluster_register_with_barrier = true;
      config.cluster_register_timeout = absl::Seconds(1);
    }
    if (allow_new_incarnation_to_reconnect) {
      config.allow_new_incarnation_to_reconnect = true;
    }
    // Init service.
    coord_service_ =
        std::make_unique<CoordinationService>(tsl::Env::Default(), config);
  }

  const IncarnationId incarnation_0_{tsl::random::New64()};
  const IncarnationId incarnation_0_new_{tsl::random::New64()};
  TestCoordinationClient client_0_;
  const IncarnationId incarnation_1_{tsl::random::New64()};
  const IncarnationId incarnation_1_new_{tsl::random::New64()};
  TestCoordinationClient client_1_;
  std::unique_ptr<CoordinationService> coord_service_;
};

// Construct fake device protos.
TestDevice CreateTestDevice(absl::string_view name, int local_id = 0) {
  TestDevice device;
  device.set_name(name);
  device.set_local_id(local_id);
  return device;
}

TEST_F(CoordinateTwoTasksTest, TestStandaloneService) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  // Not all tasks have registered, so must not be notified here.
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  ASSERT_OK(coord_service_->RecordHeartbeat(0, incarnation_0_));
  ASSERT_OK(coord_service_->RecordHeartbeat(1, incarnation_1_));
  EXPECT_THAT(coord_service_->RecordHeartbeat(2, IncarnationId(0)),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Sending heartbeat with incarnation mismatch leads to Aborted error.
  EXPECT_THAT(coord_service_->RecordHeartbeat(1, IncarnationId(0)),
              StatusIs(absl::StatusCode::kAborted));
  EXPECT_THAT(coord_service_->RecordHeartbeat(1, IncarnationId(0)),
              StatusIs(absl::StatusCode::kAborted));
}

// RegisterTask() may succeed in the service, but the agent response times out.
// In this case, the agent would retry Connect() and should succeed if it has
// the same incarnation.
TEST(CoordinationServiceTest, RegisterTask_AlreadyConnected_Succeeds) {
  const CoordinationService::Config config =
      GetCoordinationServiceConfig(/*num_tasks=*/1, /*recoverable=*/false);
  std::unique_ptr<CoordinationService> coord_service =
      std::make_unique<CoordinationService>(tsl::Env::Default(), config);
  // Task connects to coordination service.
  ASSERT_OK(coord_service->RegisterTask(0, IncarnationId(0)));

  // Registration should succeed since it is the same task.
  const absl::Status status = coord_service->RegisterTask(0, IncarnationId(0));

  TF_EXPECT_OK(status) << status;
}

TEST(CoordinationServiceTest,
     RegisterTask_AlreadyConnectedDifferentIncarnation_Fails) {
  const CoordinationService::Config config =
      GetCoordinationServiceConfig(/*num_tasks=*/1, /*recoverable=*/false);
  CoordinatedTask task_0;
  task_0.set_job_name("worker");
  task_0.set_task_id(0);
  std::unique_ptr<CoordinationService> coord_service =
      std::make_unique<CoordinationService>(tsl::Env::Default(), config);
  // Task connects to coordination service.
  ASSERT_OK(coord_service->RegisterTask(0, IncarnationId(0)));

  // Registration should fail since task already registered previously with a
  // different incarnation. Note that incarnation usually changes if an agent
  // restarts.
  const absl::Status status = coord_service->RegisterTask(0, IncarnationId(1));

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kAborted));
}

TEST(CoordinationServiceTest, RegisterTask_AlreadyInError_Fails) {
  CoordinationService::Config config =
      GetCoordinationServiceConfig(/*num_tasks=*/1, /*recoverable=*/false);
  std::unique_ptr<CoordinationService> coord_service =
      std::make_unique<CoordinationService>(tsl::Env::Default(), config);
  // Task connects to coordination service.
  ASSERT_OK(coord_service->RegisterTask(0, IncarnationId(0)));
  // Arbitrarily set task to be in error.
  ASSERT_OK(
      coord_service->ReportTaskError(0, absl::InternalError("test_error")));

  // Registration should fail.
  const absl::Status status = coord_service->RegisterTask(0, IncarnationId(0));

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kAborted));
}

TEST_F(CoordinateTwoTasksTest, TestTaskHeartbeatTimeout) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));

  // No heartbeat for a while, leader considers the task as stale.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));
  EXPECT_THAT(coord_service_->RecordHeartbeat(0, incarnation_0_),
              StatusIs(absl::StatusCode::kAborted));
  EXPECT_THAT(coord_service_->RecordHeartbeat(1, incarnation_1_),
              StatusIs(absl::StatusCode::kAborted));
}

TEST_F(CoordinateTwoTasksTest,
       ErrorPollingRequestsGotCancelledErrorUponServiceShutdown) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  std::vector<absl::Status> statuses;
  statuses.reserve(2);

  for (CoordinationService::TaskId task : {0, 1}) {
    coord_service_->PollForErrorAsync(
        task, [&](const absl::Status& status) { statuses.push_back(status); });
  }

  // No error polling requests are received before service shutdown.
  EXPECT_EQ(statuses.size(), 0);
  coord_service_.reset();

  // The service shutdowns successfully and send the cancellation response to
  // the error polling requests.
  EXPECT_EQ(statuses.size(), 2);
  EXPECT_THAT(statuses, Each(StatusIs(absl::StatusCode::kCancelled)));
}

TEST_F(CoordinateTwoTasksTest,
       HeartbeatTimeoutWithoutServerToClientConnection) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));

  // No heartbeat for a while, leader consider the task as stale.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));
  // Unexpected heartbeat from errored tasks.
  EXPECT_THAT(coord_service_->RecordHeartbeat(0, incarnation_0_),
              StatusIs(absl::StatusCode::kAborted));
  EXPECT_THAT(coord_service_->RecordHeartbeat(1, incarnation_1_),
              StatusIs(absl::StatusCode::kAborted));
}

TEST_F(CoordinateTwoTasksTest,
       HeartbeatTimeoutErrorCanPropagateThroughErrorPolling) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  // Use notifications to guarantee the ordering of operations across threads.
  absl::Notification n0, n1;
  absl::Status s0, s1;

  coord_service_->PollForErrorAsync(0, [&](const absl::Status& status) {
    s0 = status;
    n0.Notify();
  });
  coord_service_->PollForErrorAsync(1, [&](const absl::Status& status) {
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

TEST_F(CoordinateTwoTasksTest,
       HeartbeatTimeoutErrorFromOneTaskCanPropagateThroughErrorPolling) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  // Use notifications to guarantee the ordering of operations across threads.
  absl::Status s0, s1;
  absl::Notification n0, n1;

  coord_service_->PollForErrorAsync(0, [&](const absl::Status& status) {
    s0 = status;
    n0.Notify();
  });
  coord_service_->PollForErrorAsync(1, [&](const absl::Status& status) {
    s1 = status;
    n1.Notify();
  });

  // Use a factor of 0.9 to avoid accidental timeout.
  const int64_t sleeping_time =
      absl::ToInt64Microseconds(0.9 * kHeartbeatTimeout);
  // No heartbeat from task 1 for a while, so leader consider the task as stale
  // and propagate the error to all tasks.
  tsl::Env::Default()->SleepForMicroseconds(sleeping_time);
  TF_EXPECT_OK(coord_service_->RecordHeartbeat(0, incarnation_0_));
  tsl::Env::Default()->SleepForMicroseconds(sleeping_time);
  TF_EXPECT_OK(coord_service_->RecordHeartbeat(0, incarnation_0_));
  tsl::Env::Default()->SleepForMicroseconds(sleeping_time);
  // Make sure the StatusCallbacks are called.
  n0.WaitForNotification();
  n1.WaitForNotification();

  // The heartbeat error from `1` below should be propagated to all tasks.
  EXPECT_THAT(s0, StatusIs(absl::StatusCode::kUnavailable, HasSubstr("1")));
  EXPECT_THAT(s1, StatusIs(absl::StatusCode::kUnavailable, HasSubstr("1")));
}

TEST_F(CoordinateTwoTasksTest, ReportedErrorCanPropagateThroughErrorPolling) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  std::vector<absl::Status> statuses;
  statuses.reserve(2);
  for (CoordinationService::TaskId task : {0, 1}) {
    coord_service_->PollForErrorAsync(
        task, [&](const absl::Status& status) { statuses.push_back(status); });
  }

  ASSERT_OK(
      coord_service_->ReportTaskError(1, absl::InternalError("test_error")));
  // The reported error is propagated through error polling.
  EXPECT_EQ(statuses.size(), 2);
  EXPECT_THAT(statuses, Each(StatusIs(absl::StatusCode::kInternal)));
}

TEST_F(CoordinateTwoTasksTest, TestTaskRestart) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));

  // Simulate task restart scenario: trying to register to cluster again.
  absl::Status s =
      coord_service_->RegisterTask(1, IncarnationId(tsl::random::New64()));

  EXPECT_THAT(s, StatusIs(absl::StatusCode::kAborted));
}

tensorflow::CoordinatedTaskStateInfo info(
    CoordinationService::TaskId task, IncarnationId incarnation_id,
    tensorflow::CoordinatedTaskState state) {
  tensorflow::CoordinatedTaskStateInfo info;
  info.mutable_task()->set_task_id(task);
  info.set_incarnation(incarnation_id.value());
  info.set_state(state);
  return info;
}

TEST_F(CoordinateTwoTasksTest, WatchJobStateSucceeds) {
  // This test calls WatchJobState on two successfully connected tasks.

  // Connect the tasks.
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));

  // Watch the job state, which should return immediately.
  absl::Notification done;
  coord_service_->WatchJobState(
      std::nullopt,
      [&, this](std::vector<tensorflow::CoordinatedTaskStateInfo> got,
                int64_t version_number) {
        using State = tensorflow::CoordinatedTaskState;
        std::vector<tensorflow::CoordinatedTaskStateInfo> want(2);
        want[0] = info(0, incarnation_0_, State::TASKSTATE_CONNECTED);
        want[1] = info(1, incarnation_1_, State::TASKSTATE_CONNECTED);
        EXPECT_THAT(got, UnorderedElementsAre(EqualsProto(want[0]),
                                              EqualsProto(want[1])));
        done.Notify();
      });
  done.WaitForNotification();
}

TEST_F(CoordinateTwoTasksTest, WatchJobStateReturnsDisconnected) {
  // This test calls WatchJobState on one successfully connected task and one
  // disconnected task.

  // Connect the tasks. Disconnect task 1.
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  ASSERT_OK(coord_service_->ResetTask(1));

  // Watch the job state, which should return immediately.
  absl::Notification done;
  coord_service_->WatchJobState(
      std::nullopt,
      [&, this](std::vector<tensorflow::CoordinatedTaskStateInfo> got,
                int64_t version_number) {
        using State = tensorflow::CoordinatedTaskState;
        std::vector<tensorflow::CoordinatedTaskStateInfo> want(2);
        want[0] = info(0, incarnation_0_, State::TASKSTATE_CONNECTED);
        want[1] = info(1, incarnation_1_, State::TASKSTATE_DISCONNECTED);
        EXPECT_THAT(got, UnorderedElementsAre(EqualsProto(want[0]),
                                              EqualsProto(want[1])));
        EXPECT_THAT(version_number, Ge(0));
        done.Notify();
      });
  done.WaitForNotification();
}

TEST_F(CoordinateTwoTasksTest, WatchJobStateReturnsNewIncarnation) {
  // This test calls WatchJobState after one task has restarted with a new
  // incarnation.
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  ASSERT_OK(coord_service_->ResetTask(1));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_ + 1));

  // Watch the job state, which should return immediately.
  absl::Notification done;
  coord_service_->WatchJobState(
      std::nullopt,
      [&, this](std::vector<tensorflow::CoordinatedTaskStateInfo> got,
                int64_t version_number) {
        using State = tensorflow::CoordinatedTaskState;
        std::vector<tensorflow::CoordinatedTaskStateInfo> want(2);
        want[0] = info(0, incarnation_0_, State::TASKSTATE_CONNECTED);
        want[1] = info(1, incarnation_1_ + 1, State::TASKSTATE_CONNECTED);
        EXPECT_THAT(got, UnorderedElementsAre(EqualsProto(want[0]),
                                              EqualsProto(want[1])));
        EXPECT_THAT(version_number, Ge(0));
        done.Notify();
      });
  done.WaitForNotification();
}

TEST_F(CoordinateTwoTasksTest, WatchJobStateBlocksUntilChange) {
  // This test calls checks that WatchJobState blocks until the job state
  // changes.

  // Connect the tasks. Disconnect task 1.
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));

  // Watch the job state, which should return immediately.
  absl::Notification done_1;
  int64_t version_number = -1;
  coord_service_->WatchJobState(
      std::nullopt,
      [&](std::vector<tensorflow::CoordinatedTaskStateInfo> got, int64_t v) {
        EXPECT_THAT(v, Ge(0));
        version_number = v;
        done_1.Notify();
      });
  done_1.WaitForNotification();

  // Watch the job state again, which should block.
  absl::Notification done_2;
  coord_service_->WatchJobState(
      version_number,
      [&, this](std::vector<tensorflow::CoordinatedTaskStateInfo> got,
                int64_t v) {
        using State = tensorflow::CoordinatedTaskState;
        std::vector<tensorflow::CoordinatedTaskStateInfo> want(2);
        want[0] = info(0, incarnation_0_, State::TASKSTATE_CONNECTED);
        want[1] = info(1, incarnation_1_, State::TASKSTATE_DISCONNECTED);
        EXPECT_THAT(got, UnorderedElementsAre(EqualsProto(want[0]),
                                              EqualsProto(want[1])));
        EXPECT_THAT(v, Ge(version_number));
        done_2.Notify();
      });
  bool notified = done_2.WaitForNotificationWithTimeout(absl::Seconds(1));
  ASSERT_FALSE(notified);

  // Disconnect task 1.
  ASSERT_OK(coord_service_->ResetTask(1));

  done_2.WaitForNotification();
}

TEST_F(CoordinateTwoTasksTest, WatchJobStateAfterTwoStateChanges) {
  // This test calls WatchJobState after two state changes.
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));

  // Watch the job state, which should return immediately.
  absl::Notification done_1;
  int64_t version_number = -1;
  coord_service_->WatchJobState(
      std::nullopt,
      [&, this](std::vector<tensorflow::CoordinatedTaskStateInfo> got,
                int64_t v) {
        using State = tensorflow::CoordinatedTaskState;
        std::vector<tensorflow::CoordinatedTaskStateInfo> want(2);
        want[0] = info(0, incarnation_0_, State::TASKSTATE_CONNECTED);
        want[1] = info(1, incarnation_1_, State::TASKSTATE_CONNECTED);
        EXPECT_THAT(got, UnorderedElementsAre(EqualsProto(want[0]),
                                              EqualsProto(want[1])));
        EXPECT_THAT(v, Ge(0));
        version_number = v;
        done_1.Notify();
      });
  done_1.WaitForNotification();

  // Restart task 1. This leads to two state changes: the task is disconnected
  // and then reconnected.
  ASSERT_OK(coord_service_->ResetTask(1));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_ + 1));

  // Watch the job state, which should return immediately because the state has
  // already changed.
  absl::Notification done_2;
  coord_service_->WatchJobState(
      version_number,
      [&, this](std::vector<tensorflow::CoordinatedTaskStateInfo> got,
                int64_t v) {
        using State = tensorflow::CoordinatedTaskState;
        std::vector<tensorflow::CoordinatedTaskStateInfo> want(2);
        want[0] = info(0, incarnation_0_, State::TASKSTATE_CONNECTED);
        want[1] = info(1, incarnation_1_ + 1, State::TASKSTATE_CONNECTED);
        EXPECT_THAT(got, UnorderedElementsAre(EqualsProto(want[0]),
                                              EqualsProto(want[1])));
        EXPECT_THAT(v, Ge(version_number));
        done_2.Notify();
      });
  done_2.WaitForNotification();
}

TEST_F(CoordinateTwoTasksTest, InsertKeyValue_Duplicate_Fail) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->InsertKeyValue("key0", "original_value"));

  // Inserting the same key again should fail.
  EXPECT_THAT(coord_service_->InsertKeyValue("key0", "never_added"),
              StatusIs(absl::StatusCode::kAlreadyExists));

  // The original value should still be set.
  auto result = coord_service_->TryGetKeyValue("key0");
  TF_EXPECT_OK(result.status());
  EXPECT_EQ(result.value(), "original_value");
}

TEST_F(CoordinateTwoTasksTest, InsertKeyValue_Duplicate_Overwrite) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->InsertKeyValue("key0", "original_value"));
  TF_EXPECT_OK(coord_service_->InsertKeyValue("key0", "overwritten_value",
                                              /*allow_overwrite=*/true));
  auto result = coord_service_->TryGetKeyValue("key0");
  TF_EXPECT_OK(result.status());
  EXPECT_EQ(result.value(), "overwritten_value");
}

TEST_F(CoordinateTwoTasksTest, TestSetGetValues) {
  EnableCoordinationService();

  // Simple key
  ASSERT_OK(coord_service_->InsertKeyValue("key0", "value0"));
  // Unix file like key path
  ASSERT_OK(coord_service_->InsertKeyValue("/path", "value"));
  ASSERT_OK(coord_service_->InsertKeyValue("/path/to/key1", "value1"));
  // Key with redundant slashes
  ASSERT_OK(coord_service_->InsertKeyValue("path/to//key2/", "value2"));

  // Get simple key
  absl::Notification n1;
  absl::StatusOr<absl::string_view> ret;
  coord_service_->GetKeyValueAsync(
      "key0", [&](const absl::StatusOr<absl::string_view>& status_or_value) {
        ret = status_or_value;
        n1.Notify();
      });
  n1.WaitForNotification();
  ASSERT_OK(ret.status());
  EXPECT_EQ(ret.value(), "value0");
  // Get key with redundant slashes
  absl::Notification n2;
  coord_service_->GetKeyValueAsync(
      "path//to///key1////",
      [&](const absl::StatusOr<absl::string_view>& status_or_value) {
        ret = status_or_value;
        n2.Notify();
      });
  n2.WaitForNotification();
  EXPECT_EQ(ret.value(), "value1");

  // Delete single key-value
  ASSERT_OK(coord_service_->DeleteKeyValue("key0"));
  // Get key that is not available
  absl::Notification n3;
  coord_service_->GetKeyValueAsync(
      "key0", [&](const absl::StatusOr<absl::string_view>& status_or_value) {
        ret = status_or_value;
        n3.Notify();
      });
  EXPECT_FALSE(n3.HasBeenNotified());
  // Insert the previously deleted key again
  ASSERT_OK(coord_service_->InsertKeyValue("key0", "value0_new"));
  n3.WaitForNotification();
  EXPECT_EQ(ret.value(), "value0_new");

  // Delete key-values recursively
  ASSERT_OK(coord_service_->DeleteKeyValue("/path"));
  // Get key that is not available
  auto n4 = std::make_shared<absl::Notification>();
  coord_service_->GetKeyValueAsync(
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

TEST(CoordinationServiceTest, TryGetKeyValue) {
  const CoordinationService::Config config =
      GetCoordinationServiceConfig(/*num_tasks=*/1, /*recoverable=*/false);
  std::unique_ptr<CoordinationService> coord_service =
      std::make_unique<CoordinationService>(tsl::Env::Default(), config);

  // Try to get nonexistent key.
  absl::StatusOr<std::string> result =
      coord_service->TryGetKeyValue("test_key");
  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kNotFound));

  // Insert key value.
  ASSERT_OK(coord_service->InsertKeyValue("test_key", "test_value"));
  result = coord_service->TryGetKeyValue("test_key");
  EXPECT_EQ(result.value(), "test_value");

  // Delete Key, and try to get the key again.
  ASSERT_OK(coord_service->DeleteKeyValue("test_key"));
  result = coord_service->TryGetKeyValue("test_key");
  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kNotFound));
}

TEST(CoordinationServiceTest, IncrementKeyValue) {
  const CoordinationService::Config config =
      GetCoordinationServiceConfig(/*num_tasks=*/1, /*recoverable=*/false);
  std::unique_ptr<CoordinationService> coord_service =
      std::make_unique<CoordinationService>(tsl::Env::Default(), config);
  ASSERT_OK(coord_service->InsertKeyValue("test_key", "1"));
  ASSERT_OK(coord_service->IncrementKeyValue("test_key", 3));
  ASSERT_OK_AND_ASSIGN(std::string result_0,
                       coord_service->TryGetKeyValue("test_key"));
  EXPECT_EQ(result_0, "4");
  ASSERT_OK(coord_service->IncrementKeyValue("test_key_2", 10));
  ASSERT_OK_AND_ASSIGN(std::string result_1,
                       coord_service->TryGetKeyValue("test_key_2"));
  EXPECT_EQ(result_1, "10");
  ASSERT_OK(coord_service->InsertKeyValue("test_key_3", "bad_value"));
  EXPECT_THAT(coord_service->IncrementKeyValue("test_key_3", 10),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(CoordinateTwoTasksTest, GetKeyValueDir_SingleValueInDirectory) {
  EnableCoordinationService();
  KeyValueEntry kv = CreateKv("dir/path", "value0");
  ASSERT_OK(coord_service_->InsertKeyValue(kv.key(), kv.value()));

  std::vector<KeyValueEntry> result = coord_service_->GetKeyValueDir("dir");

  EXPECT_THAT(result, UnorderedElementsAre(EqualsProto(kv)));
}

TEST_F(CoordinateTwoTasksTest, GetKeyValueDir_MultipleValuesInDirectory) {
  EnableCoordinationService();
  KeyValueEntry kv = CreateKv("dir/path", "value0");
  KeyValueEntry kv2 = CreateKv("dir/path2", "value1");
  // Placed in nested subdirectory.
  KeyValueEntry kv_sub = CreateKv("dir/sub_dir/path", "value_sub");
  ASSERT_OK(coord_service_->InsertKeyValue(kv.key(), kv.value()));
  ASSERT_OK(coord_service_->InsertKeyValue(kv2.key(), kv2.value()));
  ASSERT_OK(coord_service_->InsertKeyValue(kv_sub.key(), kv_sub.value()));

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
  ASSERT_OK(coord_service_->InsertKeyValue("dir0/path", "value0"));

  std::vector<KeyValueEntry> result = coord_service_->GetKeyValueDir("dir");

  EXPECT_THAT(result, IsEmpty());
}

TEST_F(CoordinateTwoTasksTest, GetKeyValueDir_WrongDirPrefix_ReturnsEmptyList) {
  EnableCoordinationService();
  // Check that we don't match with nested subdirectories with the wrong prefix.
  ASSERT_OK(coord_service_->InsertKeyValue("wrong_dir/dir/path", "value0"));

  std::vector<KeyValueEntry> result = coord_service_->GetKeyValueDir("dir");

  EXPECT_THAT(result, IsEmpty());
}

TEST_F(CoordinateTwoTasksTest,
       GetKeyValueDir_NonDirectoryPrefix_ReturnsEmptyList) {
  EnableCoordinationService();
  // Wrong directory.
  ASSERT_OK(coord_service_->InsertKeyValue("dir_key", "value0"));

  std::vector<KeyValueEntry> result = coord_service_->GetKeyValueDir("dir");

  EXPECT_THAT(result, IsEmpty());
}

TEST_F(CoordinateTwoTasksTest,
       GetKeyValueDir_NonDirectoryKey_ReturnsEmptyList) {
  EnableCoordinationService();
  // Insert same key that is not a directory.
  ASSERT_OK(coord_service_->InsertKeyValue("dir", "value0"));

  std::vector<KeyValueEntry> result = coord_service_->GetKeyValueDir("dir");

  EXPECT_THAT(result, IsEmpty());
}

}  // namespace

TEST_F(CoordinationBarrierTest, Barrier) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0, barrier_status_1, barrier_status_2;
  int64_t counter_0, counter_1, counter_2;
  absl::Notification n_0, n_1, n_2;

  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0, &counter_0, &n_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
        counter_0 = counter;
        n_0.Notify();
      });
  GetCoordinationService()->BarrierAsync(
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
  GetCoordinationService()->BarrierAsync(
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

TEST_F(CoordinationBarrierTest, BarrierWithSubsetOfTasks) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0, barrier_status_1;
  int64_t counter_0, counter_1;
  absl::Notification n_0, n_1;

  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{0, 1},
      [&barrier_status_0, &counter_0, &n_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
        counter_0 = counter;
        n_0.Notify();
      });
  GetCoordinationService()->BarrierAsync(
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

TEST_F(CoordinationBarrierTest, BarrierWithMismatchedTasks) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0, barrier_status_1;

  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{0, 1},
      [&barrier_status_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
      });
  // task_1's barrier call specified a conflicting set of tasks (task_2 instead
  // of task_0).
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{1, 2},
      [&barrier_status_1](absl::Status s, int64_t counter) {
        barrier_status_1 = s;
      });

  EXPECT_THAT(barrier_status_0, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(barrier_status_1, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(CoordinationBarrierTest, BarrierByNonParticipatingTask) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0, barrier_status_1;
  absl::Notification n_0, n_1;

  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{0, 1},
      [&barrier_status_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
      });
  // Task 2 unexpectedly calls a barrier that it is not participating in.
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 2,
      /*participating_tasks=*/{0, 1},
      [&barrier_status_1](absl::Status s, int64_t counter) {
        barrier_status_1 = s;
      });

  // Barrier should fail for all tasks with the unexpected call.
  EXPECT_THAT(barrier_status_0, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(barrier_status_1, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(CoordinationBarrierTest, BarrierByNonParticipatingTaskThreeTasks) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0, barrier_status_1, barrier_status_2;
  absl::Notification n_0, n_1;

  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{0, 1},
      [&barrier_status_0, &n_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
        n_0.Notify();
      });
  GetCoordinationService()->BarrierAsync(
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
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 2,
      /*participating_tasks=*/{0, 1},
      [&barrier_status_2](absl::Status s, int64_t counter) {
        barrier_status_2 = s;
      });

  // Barrier should fail for task 2 which is not participating in the barrier.
  EXPECT_THAT(barrier_status_2, StatusIs(absl::StatusCode::kInvalidArgument));

  // Other clients would need to check the barrier key to detect the error.
}

TEST_F(CoordinationBarrierTest, BarrierByNonClusterTask) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0;
  absl::Notification n_0;

  GetCoordinationService()->BarrierAsync(
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

TEST_F(CoordinationBarrierTest, BarrierTimeout) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(1);
  absl::Status barrier_status_0, barrier_status_1;
  absl::Notification n_0, n_1;

  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{},
      [&barrier_status_1, &n_1](absl::Status s, int64_t counter) {
        barrier_status_1 = s;
        n_1.Notify();
      });
  GetCoordinationService()->BarrierAsync(
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

TEST_F(CoordinationBarrierTest, BarrierReturnsPreviousError) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(1);
  absl::Status barrier_status_0;
  absl::Status barrier_status_1;
  absl::Notification n_0;

  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0, &n_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
        n_0.Notify();
      });
  ASSERT_OK(GetCoordinationService()->ReportTaskError(
      0, absl::InternalError("test_error")));
  // Block until barrier has failed due to task error.
  n_0.WaitForNotification();
  // Same response should be returned immediately.
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{},
      [&barrier_status_1](absl::Status s, int64_t counter) {
        barrier_status_1 = s;
      });

  EXPECT_THAT(barrier_status_0, StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(barrier_status_1, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(CoordinationBarrierTest, TwoConsecutiveBarriers_Succeed) {
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
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0, &counter_0](const absl::Status& s, int64_t counter) {
        barrier_status_0 = s;
        counter_0 = counter;
      });
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{},
      [&barrier_status_1, &counter_1](const absl::Status& s, int64_t counter) {
        barrier_status_1 = s;
        counter_1 = counter;
      });
  GetCoordinationService()->BarrierAsync(
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
  GetCoordinationService()->BarrierAsync(
      barrier_id, 1, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0_2, &counter_0_2](const absl::Status& s,
                                          int64_t counter) {
        barrier_status_0_2 = s;
        counter_0_2 = counter;
      });
  GetCoordinationService()->BarrierAsync(
      barrier_id, 1, timeout, 1,
      /*participating_tasks=*/{},
      [&barrier_status_1_2, &counter_1_2](const absl::Status& s,
                                          int64_t counter) {
        barrier_status_1_2 = s;
        counter_1_2 = counter;
      });
  GetCoordinationService()->BarrierAsync(
      barrier_id, 1, timeout, 2,
      /*participating_tasks=*/{},
      [&barrier_status_2_2, &counter_2_2](const absl::Status& s,
                                          int64_t counter) {
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

TEST_F(CoordinationBarrierTest,
       Barrier_OngoingButMismatchedCounter_InternalError) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0, barrier_status_1,
      barrier_status_2 = absl::UnknownError("Unknown");
  int64_t counter_0, counter_1, counter_2 = -1;

  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0, &counter_0](const absl::Status& s, int64_t counter) {
        barrier_status_0 = s;
        counter_0 = counter;
      });
  // Task 1 specifies different counter!
  GetCoordinationService()->BarrierAsync(
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
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 2,
      /*participating_tasks=*/{},
      [&barrier_status_2, &counter_2](const absl::Status& s, int64_t counter) {
        barrier_status_2 = s;
        counter_2 = counter;
      });

  EXPECT_THAT(barrier_status_2, StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(counter_2, 0);
}

TEST_F(CoordinationBarrierTest, SecondBarrier_UseWrongCounter_InternalError) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0, barrier_status_1,
      barrier_status_2 = absl::UnknownError("Unknown");
  int64_t counter_0, counter_1, counter_2 = -1;
  absl::Notification timeout_n;

  // First barrier.
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [](const absl::Status& s, int64_t counter) {});
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{},
      [](const absl::Status& s, int64_t counter) {});
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 2,
      /*participating_tasks=*/{},
      [](const absl::Status& s, int64_t counter) {});

  // Second barrier.
  // Specify same as previous barrier: return same result.
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0, &counter_0](const absl::Status& s, int64_t counter) {
        barrier_status_0 = s;
        counter_0 = counter;
      });
  EXPECT_EQ(counter_0, 0);
  TF_EXPECT_OK(barrier_status_0);

  // Specify a counter that is too low: fail!
  GetCoordinationService()->BarrierAsync(
      barrier_id, -1, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_1, &counter_1](const absl::Status& s, int64_t counter) {
        barrier_status_1 = s;
        counter_1 = counter;
      });
  EXPECT_EQ(counter_1, 0);  // Specifies the service-side barrier counter.
  EXPECT_THAT(barrier_status_1, StatusIs(absl::StatusCode::kInternal));

  // Specify a counter that is too high: fail!
  GetCoordinationService()->BarrierAsync(
      barrier_id, 2, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_2, &counter_2](const absl::Status& s, int64_t counter) {
        barrier_status_2 = s;
        counter_2 = counter;
      });
  EXPECT_EQ(counter_2, 0);  // Specifies the service-side barrier counter.
  EXPECT_THAT(barrier_status_2, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(CoordinationBarrierTest, BarrierCancelled) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status;

  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status](absl::Status s, int64_t counter) {
        barrier_status = s;
      });
  absl::Status cancelled_status =
      GetCoordinationService()->CancelBarrier(barrier_id, 0, 0);

  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kCancelled));
  TF_EXPECT_OK(cancelled_status);
}

TEST_F(CoordinationBarrierTest, CancelAfterBarrierHasPassed) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0 = absl::UnknownError("Uninitialized error.");
  absl::Status barrier_status_1 = absl::UnknownError("Uninitialized error.");
  absl::Status barrier_status_2 = absl::UnknownError("Uninitialized error.");

  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
      });
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{},
      [&barrier_status_1](absl::Status s, int64_t counter) {
        barrier_status_1 = s;
      });
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 2,
      /*participating_tasks=*/{},
      [&barrier_status_2](absl::Status s, int64_t counter) {
        barrier_status_2 = s;
      });
  // Cancel barrier should fail if barrier has already been passed.
  absl::Status cancelled_status =
      GetCoordinationService()->CancelBarrier(barrier_id, 0, 0);

  EXPECT_THAT(cancelled_status, StatusIs(absl::StatusCode::kFailedPrecondition,
                                         HasSubstr("already been passed")));
  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);
  TF_EXPECT_OK(barrier_status_2);
}

TEST_F(CoordinationBarrierTest, CancelBarrier_WrongCounter_FailedPrecondition) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0 = absl::UnknownError("Uninitialized error.");
  absl::Status barrier_status_1 = absl::UnknownError("Uninitialized error.");
  absl::Status barrier_status_2 = absl::UnknownError("Uninitialized error.");
  absl::Status barrier_status_cancelled =
      absl::UnknownError("Uninitialized error.");

  // First barrier passes (counter: 0).
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
      });
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{},
      [&barrier_status_1](absl::Status s, int64_t counter) {
        barrier_status_1 = s;
      });
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 2,
      /*participating_tasks=*/{},
      [&barrier_status_2](absl::Status s, int64_t counter) {
        barrier_status_2 = s;
      });
  TF_ASSERT_OK(barrier_status_0);
  TF_ASSERT_OK(barrier_status_1);
  TF_ASSERT_OK(barrier_status_2);
  // Second barrier (counter: 1)
  GetCoordinationService()->BarrierAsync(
      barrier_id, 1, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_cancelled](absl::Status s, int64_t counter) {
        barrier_status_cancelled = s;
      });
  // Specify low counter (e.g. due to restart on client-side).
  absl::Status cancelled_status_low_counter =
      GetCoordinationService()->CancelBarrier(barrier_id, 0, 0);
  EXPECT_THAT(barrier_status_cancelled,
              StatusIs(absl::StatusCode::kUnknown, HasSubstr("Uninitialized")));
  // Specify high counter (e.g. due to restart on service-side).
  absl::Status cancelled_status_high_counter =
      GetCoordinationService()->CancelBarrier(barrier_id, 2, 0);
  EXPECT_THAT(barrier_status_cancelled,
              StatusIs(absl::StatusCode::kUnknown, HasSubstr("Uninitialized")));
  // Specify correct counter.
  absl::Status cancelled_status_correct_counter =
      GetCoordinationService()->CancelBarrier(barrier_id, 1, 1);

  EXPECT_THAT(barrier_status_cancelled, StatusIs(absl::StatusCode::kCancelled));
  EXPECT_THAT(cancelled_status_low_counter,
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("likely due to a restart")));
  EXPECT_THAT(cancelled_status_high_counter,
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("likely due to a restart")));
  TF_EXPECT_OK(cancelled_status_correct_counter);
}

TEST_F(CoordinationBarrierTest, PassedBarrierReturnsImmediately) {
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

  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status_0, &n0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
        n0.Notify();
      });
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{},
      [&barrier_status_1, &n1](absl::Status s, int64_t counter) {
        barrier_status_1 = s;
        n1.Notify();
      });
  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 2,
      /*participating_tasks=*/{},
      [&barrier_status_2, &n2](absl::Status s, int64_t counter) {
        barrier_status_2 = s;
        n2.Notify();
      });
  // Repeated call should return the same result.
  GetCoordinationService()->BarrierAsync(
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

TEST_F(CoordinationBarrierTest, BarrierFailsIfTaskIsAlreadyInError) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  // Set task 0 to error state.
  ASSERT_OK(GetCoordinationService()->ReportTaskError(
      0, absl::InternalError("test_error")));
  absl::Status barrier_status;

  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{},
      [&barrier_status](absl::Status s, int64_t counter) {
        barrier_status = s;
      });

  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(CoordinationBarrierTest, BarrierFailsUponTaskError) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Notification n0;
  absl::Status barrier_status;

  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{},
      [&barrier_status, &n0](absl::Status s, int64_t counter) {
        barrier_status = s;
        n0.Notify();
      });
  ASSERT_OK(GetCoordinationService()->ReportTaskError(
      0, absl::InternalError("test_error")));
  n0.WaitForNotification();

  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(CoordinationBarrierTest,
       BarrierStillBlocksIfSameTaskCallsOngoingBarrierRepeatedly) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Status barrier_status_0;
  absl::Status barrier_status_1;
  absl::Status barrier_status_2;
  absl::Notification n_0;
  absl::Notification n_1;
  absl::Notification n_2;

  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 0,
      /*participating_tasks=*/{0, 1},
      [&barrier_status_0, &n_0](absl::Status s, int64_t counter) {
        barrier_status_0 = s;
        n_0.Notify();
      });
  // Duplicate call.
  GetCoordinationService()->BarrierAsync(
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

  GetCoordinationService()->BarrierAsync(
      barrier_id, 0, timeout, 1,
      /*participating_tasks=*/{0, 1},
      [&barrier_status_2, &n_2](absl::Status s, int64_t counter) {
        barrier_status_2 = s;
        n_2.Notify();
      });
  TF_EXPECT_OK(barrier_status_1);
  TF_EXPECT_OK(barrier_status_2);
}

TEST_F(CoordinateTwoTasksTest, ResetAndRegisterAgain) {
  EnableCoordinationService();
  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_));

  TF_EXPECT_OK(coord_service_->ResetTask(0));

  // Task should be allowed to register again after being reset.
  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_));
}

TEST_F(CoordinateTwoTasksTest, Reset_HeartbeatsAreAcceptedForAGracePeriod) {
  EnableCoordinationService();
  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_));

  TF_EXPECT_OK(coord_service_->ResetTask(0));
  // Heartbeat should be allowed for a short grace period after reset.
  TF_EXPECT_OK(coord_service_->RecordHeartbeat(0, incarnation_0_));

  // Heartbeat failure should be triggered for disconnected task after grace
  // period.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(3 * kHeartbeatTimeout));
  EXPECT_THAT(coord_service_->RecordHeartbeat(0, incarnation_0_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(CoordinateTwoTasksTest, Reset_FailsOngoingBarrier) {
  EnableCoordinationService(
      /*enable_shutdown_barrier=*/false);
  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  absl::Status barrier_status;
  absl::Notification barrier_n;
  coord_service_->BarrierAsync(
      "ongoing_barrier", 0, absl::InfiniteDuration(), 0,
      /*participating_tasks=*/{},
      [&barrier_status, &barrier_n](absl::Status s, int64_t counter) {
        barrier_status = s;
        barrier_n.Notify();
      });

  TF_EXPECT_OK(coord_service_->ResetTask(0));

  // Ongoing barrier should fail with error after shutdown.
  EXPECT_TRUE(barrier_n.HasBeenNotified());
  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(CoordinateTwoTasksTest, Shutdown_HeartbeatsAreAcceptedForAGracePeriod) {
  EnableCoordinationService(
      /*enable_shutdown_barrier=*/false);
  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_));

  absl::Notification n;
  coord_service_->ShutdownTaskAsync(0, [&n](absl::Status s) {
    TF_EXPECT_OK(s);
    n.Notify();
  });
  n.WaitForNotification();

  // Heartbeat should be allowed for a short grace period after shutdown.
  TF_EXPECT_OK(coord_service_->RecordHeartbeat(0, incarnation_0_));

  // Heartbeat failure should be triggered for disconnected task after grace
  // period.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(3 * kHeartbeatTimeout));
  EXPECT_THAT(coord_service_->RecordHeartbeat(0, incarnation_0_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(CoordinateTwoTasksTest, Shutdown_FailsOngoingBarrier) {
  EnableCoordinationService(
      /*enable_shutdown_barrier=*/false);
  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  absl::Status barrier_status;
  absl::Notification barrier_n;
  coord_service_->BarrierAsync(
      "ongoing_barrier", 0, absl::InfiniteDuration(), 0,
      /*participating_tasks=*/{},
      [&barrier_status, &barrier_n](absl::Status s, int64_t counter) {
        barrier_status = s;
        barrier_n.Notify();
      });

  absl::Notification shutdown_n;
  coord_service_->ShutdownTaskAsync(0, [&shutdown_n](absl::Status s) {
    TF_EXPECT_OK(s);
    shutdown_n.Notify();
  });
  shutdown_n.WaitForNotification();

  // Ongoing barrier should fail with error after shutdown.
  EXPECT_TRUE(barrier_n.HasBeenNotified());
  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(CoordinateTwoTasksTest, ShutdownWithBarrier_BarrierSucceeds) {
  EnableCoordinationService(
      /*enable_shutdown_barrier=*/true);
  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  TF_EXPECT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  absl::Status barrier_status;
  absl::Status barrier_status_2;

  coord_service_->ShutdownTaskAsync(
      0, [&barrier_status](absl::Status s) { barrier_status = s; });
  coord_service_->ShutdownTaskAsync(
      1, [&barrier_status_2](absl::Status s) { barrier_status_2 = s; });

  TF_EXPECT_OK(barrier_status);
  TF_EXPECT_OK(barrier_status_2);

  // Confirm that both tasks have disconnected.
  // Note: this should not happen in prod where RegisterTask() is called after
  // Shutdown(), which is prevented by agent-side logic.
  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  TF_EXPECT_OK(coord_service_->RegisterTask(1, incarnation_1_));
}

TEST_F(CoordinateTwoTasksTest,
       ShutdownWithBarrier_BarrierFails_TaskDisconnectsOtherTaskIsAlerted) {
  EnableCoordinationService(
      /*enable_shutdown_barrier=*/true);
  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  TF_EXPECT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  absl::Status barrier_status;

  absl::Notification n;
  coord_service_->ShutdownTaskAsync(0, [&n, &barrier_status](absl::Status s) {
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
  EXPECT_THAT(coord_service_->RegisterTask(0, incarnation_0_),
              StatusIs(absl::StatusCode::kAborted));
  EXPECT_THAT(coord_service_->RegisterTask(0, incarnation_1_),
              StatusIs(absl::StatusCode::kAborted));
}

TEST_F(CoordinateTwoTasksTest,
       ShutdownWithBarrier_BarrierFailsWithoutClientConnection_SetTaskToError) {
  EnableCoordinationService(
      /*enable_shutdown_barrier=*/true);
  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  TF_EXPECT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  absl::Status barrier_status;

  absl::Notification n;
  coord_service_->ShutdownTaskAsync(0, [&n, &barrier_status](absl::Status s) {
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
  absl::Status s = coord_service_->RecordHeartbeat(1, incarnation_1_);

  EXPECT_THAT(s, StatusIs(absl::StatusCode::kAborted));
}

TEST_F(CoordinateTwoTasksTest, BarrierFailsIfTaskIsInError) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  absl::Notification n0;
  absl::Status barrier_status;
  // No heartbeat for a while, leader consider the task as stale.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));

  // Barrier should fail when called after stale task is set to error.
  coord_service_->BarrierAsync("barrier_id", 0, absl::Seconds(5), 0,
                               /*participating_tasks=*/{},
                               [&](absl::Status s, int64_t counter) {
                                 barrier_status = s;
                                 n0.Notify();
                               });

  n0.WaitForNotification();
  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(CoordinateTwoTasksTest,
       BarrierWithParticipatingTasksFailsIfTaskIsStale) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  absl::Notification n0;
  absl::Status barrier_status;
  // No heartbeat for a while, leader consider the task as stale.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));

  coord_service_->BarrierAsync("barrier_id", 0, absl::Seconds(5), 0,
                               /*participating_tasks=*/{0},
                               [&](absl::Status s, int64_t counter) {
                                 barrier_status = s;
                                 n0.Notify();
                               });

  n0.WaitForNotification();
  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(CoordinateTwoTasksTest, BarrierFailsAfterErrorPollingResponse) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  // Use notifications to guarantee the ordering of operations across threads.
  absl::Notification n0, n1;
  absl::Status s0, s1;

  coord_service_->PollForErrorAsync(0, [&](const absl::Status& status) {
    s0 = status;
    n0.Notify();
  });
  coord_service_->PollForErrorAsync(1, [&](const absl::Status& status) {
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
  coord_service_->BarrierAsync("barrier_id", 0, absl::Seconds(5), 0,
                               /*participating_tasks=*/{},
                               [&](absl::Status s, int64_t counter) {
                                 barrier_status = s;
                                 n_barrier.Notify();
                               });

  n_barrier.WaitForNotification();
  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(CoordinateTwoTasksTest, BarrierWithSubsetFailsIfTaskIsStale) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  absl::Notification n0;
  absl::Status barrier_status;
  // No heartbeat for a while, leader consider the task as stale.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));

  // Barrier should fail if task is in error.
  // Note that this is same as above test, but the barrier only blocks for task
  // 0.
  coord_service_->BarrierAsync("barrier_id", 0, absl::Seconds(5), 0,
                               /*participating_tasks=*/{0},
                               [&](absl::Status s, int64_t counter) {
                                 barrier_status = s;
                                 n0.Notify();
                               });

  n0.WaitForNotification();
  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(CoordinateTwoTasksTest, RecoverableTaskWillNotPropagateError) {
  EnableCoordinationService(
      /*enable_shutdown_barrier=*/false,
      /*enable_register_barrier=*/false,
      /*set_worker_job_recoverable=*/true);

  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  TF_EXPECT_OK(coord_service_->RegisterTask(1, incarnation_1_));

  ASSERT_OK(
      coord_service_->ReportTaskError(0, absl::InternalError("test_error")));

  // Since no error propagation for recoverable tasks, other tasks should work
  // as normal.
  TF_EXPECT_OK(client_1_.GetStatus());
}

TEST_F(CoordinateTwoTasksTest,
       RecoverableTaskWithErrorPollingWillNotPropagateError) {
  EnableCoordinationService(
      /*enable_shutdown_barrier=*/false,
      /*enable_register_barrier=*/false,
      /*set_worker_job_recoverable=*/true);
  // These callbacks may be invoked after this test (e.g. cancellations during
  // coord service dtor), so we use shared pointers to extend their lifetimes
  // beyond the test to avoid use-after-free errors.
  auto s0 = std::make_shared<absl::Status>();
  auto s1 = std::make_shared<absl::Status>();
  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  coord_service_->PollForErrorAsync(
      0, [s0](const absl::Status& status) { *s0 = status; });
  TF_EXPECT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  coord_service_->PollForErrorAsync(
      1, [s1](const absl::Status& status) { *s1 = status; });

  ASSERT_OK(
      coord_service_->ReportTaskError(0, absl::InternalError("test_error")));

  // Since no error propagation for recoverable tasks, other tasks should work
  // as normal.
  EXPECT_THAT(*s0, StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(*s1, StatusIs(absl::StatusCode::kOk));
}

TEST_F(CoordinateTwoTasksTest,
       RecoverableTaskReportErrorResetAndRegisterAgain) {
  EnableCoordinationService(
      /*enable_shutdown_barrier=*/false,
      /*enable_register_barrier=*/false,
      /*set_worker_job_recoverable=*/true);

  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  TF_EXPECT_OK(coord_service_->RegisterTask(1, incarnation_1_));

  ASSERT_OK(
      coord_service_->ReportTaskError(0, absl::InternalError("test_error")));

  EXPECT_THAT(coord_service_->RecordHeartbeat(0, incarnation_0_),
              StatusIs(absl::StatusCode::kAborted));
  // Since no error propagation for recoverable tasks, other tasks should work
  // as normal.
  TF_EXPECT_OK(client_1_.GetStatus());

  // Reset and register the error task again, both tasks should be healthy.
  TF_EXPECT_OK(coord_service_->ResetTask(0));
  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_new_));
  TF_EXPECT_OK(coord_service_->RecordHeartbeat(0, incarnation_0_new_));
  TF_EXPECT_OK(client_1_.GetStatus());
}

TEST_F(CoordinateTwoTasksTest, UnavailableTaskCanReconnect) {
  EnableCoordinationService(
      /*enable_shutdown_barrier=*/false,
      /*enable_register_barrier=*/false,
      /*set_worker_job_recoverable=*/false,
      /*allow_new_incarnation_to_reconnect=*/true);

  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_));

  ASSERT_OK(coord_service_->ReportTaskError(
      0, MakeCoordinationError(absl::UnavailableError("test_error"))));

  TF_EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_new_));
}

TEST_F(CoordinateTwoTasksTest, DoNotAllowPollForErrorIfNotInCluster) {
  EnableCoordinationService();
  absl::Status s;

  coord_service_->PollForErrorAsync(
      -1, [&](const absl::Status& status) { s = status; });

  EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument,
                          HasSubstr("not in the cluster")));
}

TEST_F(CoordinateTwoTasksTest, DoNotAllowPollForErrorIfTaskNotRegistered) {
  EnableCoordinationService();
  absl::Status s;

  coord_service_->PollForErrorAsync(
      0, [&](const absl::Status& status) { s = status; });

  EXPECT_THAT(s, StatusIs(absl::StatusCode::kFailedPrecondition,
                          HasSubstr("has not been registered")));
}

TEST_F(CoordinateTwoTasksTest,
       AllowPollForErrorWithinGracePeriodIfTaskHasShutDown) {
  EnableCoordinationService();
  absl::Status s;
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  coord_service_->ShutdownTaskAsync(0, [&](const absl::Status& status) {});
  coord_service_->ShutdownTaskAsync(1, [&](const absl::Status& status) {});

  coord_service_->PollForErrorAsync(
      0, [&](const absl::Status& status) { s = status; });
  // Stop the service.
  coord_service_.reset();
  // The error polling request will still proceed because of grace period. It
  // will be cancelled.
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kCancelled));
}

TEST_F(CoordinateTwoTasksTest, DoNotAllowPollForErrorIfTaskHasShutDown) {
  EnableCoordinationService();
  absl::Status s;
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  coord_service_->ShutdownTaskAsync(0, [&](const absl::Status& status) {});
  coord_service_->ShutdownTaskAsync(1, [&](const absl::Status& status) {});

  // Sleep past the grace period.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));
  coord_service_->PollForErrorAsync(
      0, [&](const absl::Status& status) { s = status; });
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kFailedPrecondition,
                          HasSubstr("has disconnected")));
}

TEST_F(CoordinateTwoTasksTest, DoNotAllowPollForErrorAfterReset) {
  EnableCoordinationService();
  absl::Status s;
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->ResetTask(0));

  // Sleep past the grace period.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));
  coord_service_->PollForErrorAsync(
      0, [&](const absl::Status& status) { s = status; });
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kFailedPrecondition,
                          HasSubstr("has disconnected")));
}

TEST_F(CoordinateTwoTasksTest, DoNotAllowPollForErrorWhenInErrorState) {
  EnableCoordinationService();
  absl::Status s;
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(
      coord_service_->ReportTaskError(0, absl::InternalError("test_error")));

  coord_service_->PollForErrorAsync(
      0, [&](const absl::Status& status) { s = status; });
  EXPECT_THAT(s, StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(CoordinateTwoTasksTest, DoNotAllowPollForErrorIfTaskIsStale) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  // No heartbeat for a while, leader consider the task as stale.
  tsl::Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));

  absl::Status s;
  coord_service_->PollForErrorAsync(
      0, [&](const absl::Status& status) { s = status; });

  EXPECT_THAT(s, StatusIs(absl::StatusCode::kFailedPrecondition,
                          HasSubstr("already in error")));
}

TEST_F(CoordinateTwoTasksTest,
       CanPropagateTaskRegistrationErrorThroughErrorPolling) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  absl::Status s0;
  // Start polling for error from `0`.
  coord_service_->PollForErrorAsync(
      0, [&](const absl::Status& status) { s0 = status; });

  // Let registration of `1` fail due to incarnation mismatch.
  ASSERT_THAT(coord_service_->RegisterTask(1, incarnation_0_),
              StatusIs(absl::StatusCode::kAborted));

  // The first error polling request will get the error propagated from the
  // registration failure.
  EXPECT_THAT(s0, StatusIs(absl::StatusCode::kAborted));
}

TEST_F(CoordinateTwoTasksTest, LatePollingTaskCanGetError) {
  EnableCoordinationService();
  ASSERT_OK(coord_service_->RegisterTask(0, incarnation_0_));
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  std::vector<absl::Status> statuses;
  statuses.reserve(2);
  coord_service_->PollForErrorAsync(
      0, [&](const absl::Status& status) { statuses.push_back(status); });

  // Fail `0` with an error because `1` polls for error.
  ASSERT_OK(coord_service_->ReportTaskError(
      0, absl::FailedPreconditionError("test_error_from_task_0")));

  // Poll for error from `1` after the error has been propagated to other
  // tasks.
  coord_service_->PollForErrorAsync(
      1, [&](const absl::Status& status) { statuses.push_back(status); });

  // Make sure the error is propagated to both tasks.
  EXPECT_EQ(statuses.size(), 2);
  EXPECT_THAT(statuses, Each(StatusIs(absl::StatusCode::kFailedPrecondition,
                                      HasSubstr("test_error_from_task_0"))));
}

TEST_F(CoordinateTwoTasksTest,
       RegisterWithBarrier_OldHeartbeat_RestartedTasksCanReconnect) {
  EnableCoordinationService(
      /*enable_shutdown_barrier=*/false,
      /*enable_register_barrier=*/true);
  // Service restarted.
  // Old task 0 sends an unexpected heartbeat, which should fail.
  ASSERT_THAT(coord_service_->RecordHeartbeat(0, incarnation_0_ - 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  absl::Status task0_status = absl::InternalError("uninitialized_status");
  // Task 0 registers first.
  coord_service_->RegisterTaskAsync(0, incarnation_0_,
                                    [](const absl::Status& s) {});
  // Task 0 restarts with a new incarnation, and registers again.
  // This should be allowed since all tasks have not joined the cluster yet.
  coord_service_->RegisterTaskAsync(
      0, incarnation_0_ + 1, [&](const absl::Status& s) { task0_status = s; });
  // Now all tasks will register in a synchronized fashion due to the barrier.
  EXPECT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  EXPECT_OK(task0_status);
}

TEST_F(CoordinateTwoTasksTest,
       RegisterWithBarrier_RestartBeforeBarrier_Succeeds) {
  EnableCoordinationService(
      /*enable_shutdown_barrier=*/false,
      /*enable_register_barrier=*/true);
  absl::Status task0_status = absl::InternalError("uninitialized_status");
  absl::Status restarted_task0_status =
      absl::InternalError("uninitialized_status");
  // Task 0 registers first.
  coord_service_->RegisterTaskAsync(
      0, incarnation_0_, [&](const absl::Status& s) { task0_status = s; });
  // Task 0 restarts with a new incarnation, and registers again.
  // This should be allowed since all tasks have not joined the cluster yet.
  coord_service_->RegisterTaskAsync(
      0, incarnation_0_ + 1,
      [&](const absl::Status& s) { restarted_task0_status = s; });
  // Now all tasks will register in a synchronized fashion due to the barrier.
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  ASSERT_THAT(task0_status, StatusIs(absl::StatusCode::kAlreadyExists));
  ASSERT_OK(restarted_task0_status);
  // Task 0 joins again with the same incarnation.
  // This is okay, it didn't restart, probably sent RPC twice due to network
  // retries.
  EXPECT_OK(coord_service_->RegisterTask(0, incarnation_0_ + 1));
}

TEST_F(CoordinateTwoTasksTest, RegisterWithBarrier_RestartAfterBarrier_Fails) {
  EnableCoordinationService(
      /*enable_shutdown_barrier=*/false,
      /*enable_register_barrier=*/true);
  absl::Status task0_status = absl::InternalError("uninitialized_status");
  // Task 0 registers first.
  coord_service_->RegisterTaskAsync(
      0, incarnation_0_, [&](const absl::Status& s) { task0_status = s; });
  // Now all tasks will register in a synchronized fashion due to the barrier.
  ASSERT_OK(coord_service_->RegisterTask(1, incarnation_1_));
  ASSERT_OK(task0_status);

  // Task 0 restarts again with a new incarnation.
  // This should fail since this happens after the initial register barrier
  // (i.e. all tasks already acked once).
  ASSERT_THAT(coord_service_->RegisterTask(0, incarnation_0_ + 2),
              StatusIs(absl::StatusCode::kAborted));
  // All tasks should be set to error and unable to start any barriers.
  absl::Notification n;
  absl::Status barrier_status;
  coord_service_->BarrierAsync("barrier_id", 0, absl::Seconds(10), 0, {},
                               [&](const absl::Status& s, int64_t counter) {
                                 n.Notify();
                                 barrier_status = s;
                               });
  n.WaitForNotification();
  EXPECT_THAT(barrier_status, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(CoordinateTwoTasksTest, RegisterWithBarrier_Timeout) {
  EnableCoordinationService(
      /*enable_shutdown_barrier=*/false,
      /*enable_register_barrier=*/true);
  // Task 0 joins without task 1. Times out eventually as this function is
  // blocking.
  EXPECT_THAT(coord_service_->RegisterTask(0, incarnation_0_),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
}

class GetAliveTasksTest : public CoordinationBarrierTest {
 public:
  GetAliveTasksTest() : CoordinationBarrierTest(true) {}
};

TEST_F(GetAliveTasksTest, SuccessfulGetAliveTasks) {
  // This test has three tasks successfully call GetAliveTasks.
  absl::BlockingCounter finished(3);
  auto done = [&](const absl::Status& status,
                  const std::vector<CoordinationService::TaskId>& alive_tasks,
                  const std::vector<IncarnationId>& incarnations) {
    EXPECT_OK(status);
    EXPECT_THAT(alive_tasks, UnorderedElementsAreArray(tasks()));
    EXPECT_THAT(incarnations,
                UnorderedElementsAre(IncarnationId(0), IncarnationId(1),
                                     IncarnationId(2)));
    finished.DecrementCount();
  };
  GetCoordinationService()->GetAliveTasksAsync(0, tasks(), done);
  GetCoordinationService()->GetAliveTasksAsync(1, tasks(), done);
  GetCoordinationService()->GetAliveTasksAsync(2, tasks(), done);
  finished.Wait();
}

TEST_F(GetAliveTasksTest, FailedTaskBeforeCallingGetAliveTasks) {
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
  ASSERT_OK(GetCoordinationService()->ReportTaskError(
      2, absl::InternalError("failed")));
  GetCoordinationService()->GetAliveTasksAsync(0, tasks(), done);
  GetCoordinationService()->GetAliveTasksAsync(1, tasks(), done);
  finished.Wait();
}

TEST_F(GetAliveTasksTest, FailedTaskAfterCallingGetAliveTasks) {
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
  GetCoordinationService()->GetAliveTasksAsync(0, tasks(), done);
  GetCoordinationService()->GetAliveTasksAsync(1, tasks(), done);
  ASSERT_OK(GetCoordinationService()->ReportTaskError(
      2, absl::InternalError("failed")));
  finished.Wait();
}

TEST_F(GetAliveTasksTest, ConcurrentGetAliveTasks) {
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
  GetCoordinationService()->GetAliveTasksAsync(0, tasks_01, done_01);
  GetCoordinationService()->GetAliveTasksAsync(1, tasks_12, done_12);
  GetCoordinationService()->GetAliveTasksAsync(1, tasks_01, done_01);
  GetCoordinationService()->GetAliveTasksAsync(2, tasks_12, done_12);
  finished_01.Wait();
  finished_12.Wait();
}

TEST_F(GetAliveTasksTest, CallingGetAliveTasksWithoutBeingAMember) {
  // This test includes calls to GetAliveTasks where the requesting task is not
  // included in the specified set of tasks. This should return an error.
  absl::BlockingCounter finished(3);
  auto done = [&](const absl::Status& status,
                  const std::vector<CoordinationService::TaskId>&,
                  const std::vector<IncarnationId>&) {
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
    finished.DecrementCount();
  };

  CoordinationService* s = GetCoordinationService();
  s->GetAliveTasksAsync(0, {1, 2}, done);
  s->GetAliveTasksAsync(1, {0, 2}, done);
  s->GetAliveTasksAsync(2, {0, 1}, done);
  finished.Wait();
}

TEST_F(GetAliveTasksTest, RedundantGetAliveTasks) {
  // This test has three tasks call GetAliveTasks, with the twist that some
  // tasks call GetAliveTasks multiple times.
  absl::BlockingCounter finished(6);
  auto done = [&](const absl::Status& status,
                  const std::vector<CoordinationService::TaskId>& alive_tasks,
                  const std::vector<IncarnationId>&) {
    EXPECT_OK(status);
    EXPECT_THAT(alive_tasks, UnorderedElementsAreArray(tasks()));
    finished.DecrementCount();
  };
  GetCoordinationService()->GetAliveTasksAsync(0, tasks(), done);
  GetCoordinationService()->GetAliveTasksAsync(0, tasks(), done);
  GetCoordinationService()->GetAliveTasksAsync(0, tasks(), done);
  GetCoordinationService()->GetAliveTasksAsync(1, tasks(), done);
  GetCoordinationService()->GetAliveTasksAsync(1, tasks(), done);
  GetCoordinationService()->GetAliveTasksAsync(2, tasks(), done);
  finished.Wait();
}

}  // namespace xla
