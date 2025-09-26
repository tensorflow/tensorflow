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

#include "tensorflow/core/distributed_runtime/partial_run_mgr.h"

#include "absl/synchronization/notification.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(PartialRunMgrFindOrCreate, Create) {
  // Basic test of PartialRunMgr CancellationManager creation.
  PartialRunMgr partial_run_mgr;
  int step_id = 1;
  CancellationManager* cancellation_manager;
  partial_run_mgr.FindOrCreate(step_id, &cancellation_manager);
  EXPECT_TRUE(cancellation_manager != nullptr);
}

TEST(PartialRunMgrFindOrCreate, Find) {
  // Basic test of PartialRunMgr CancellationManager find.
  PartialRunMgr partial_run_mgr;
  int step_id = 1;
  CancellationManager* cancellation_manager;
  partial_run_mgr.FindOrCreate(step_id, &cancellation_manager);
  // Looking for the same step should return the same cancellation_manager.
  CancellationManager* found_cancellation_manager;
  partial_run_mgr.FindOrCreate(step_id, &found_cancellation_manager);
  EXPECT_EQ(cancellation_manager, found_cancellation_manager);
}

TEST(PartialRunMgrFindOrCreate, NewCreate) {
  // Test that PartialRunMgr creates a new CancellationManager for new steps.
  PartialRunMgr partial_run_mgr;
  int step_id = 1;
  CancellationManager* cancellation_manager;
  partial_run_mgr.FindOrCreate(step_id, &cancellation_manager);
  // FindOrCreate on a new step should return a new cancellation_manager.
  int new_step_id = 2;
  CancellationManager* new_cancellation_manager;
  partial_run_mgr.FindOrCreate(new_step_id, &new_cancellation_manager);
  EXPECT_NE(cancellation_manager, new_cancellation_manager);
}

TEST(PartialRunMgr, PartialRunRemoved) {
  // Test that PartialRunMgr ensures that the PartialRun is deleted after
  // ExecutorDone and PartialRunDone are called.
  PartialRunMgr partial_run_mgr;
  int step_id = 1;
  CancellationManager* cancellation_manager;
  partial_run_mgr.FindOrCreate(step_id, &cancellation_manager);

  int called = 0;
  partial_run_mgr.PartialRunDone(
      step_id, [&called](absl::Status status) { called++; }, absl::OkStatus());
  partial_run_mgr.ExecutorDone(step_id, absl::OkStatus());

  // Calling ExecutorDone and PartialRunDone on the step_id should still only
  // result in the callback being called once.
  // This proves that the original PartialRun has been removed.
  partial_run_mgr.PartialRunDone(
      step_id, [&called](absl::Status status) { called++; }, absl::OkStatus());
  partial_run_mgr.ExecutorDone(step_id, absl::OkStatus());
  EXPECT_EQ(1, called);
}

struct StatusTestParam {
  absl::Status executor_status;
  absl::Status partial_run_status;
  absl::Status expected_status;
};

class StatusPropagationTest : public ::testing::TestWithParam<StatusTestParam> {
 protected:
  PartialRunMgr partial_run_mgr_;

  // State to help keep track of when the callback is called.
  absl::Notification invoked_;
  absl::Status status_;

  void set_status(const absl::Status& status) {
    status_ = status;
    invoked_.Notify();
  }

  // Blocks until status is set.
  absl::Status status() {
    invoked_.WaitForNotification();
    return status_;
  }
};

TEST_P(StatusPropagationTest, ExecutorDoneFirst) {
  // Tests error propagation when ExecutorDone is called first.
  StatusTestParam param = GetParam();
  int step_id = 1;

  CancellationManager* cancellation_manager;
  partial_run_mgr_.FindOrCreate(step_id, &cancellation_manager);

  partial_run_mgr_.ExecutorDone(step_id, param.executor_status);
  partial_run_mgr_.PartialRunDone(
      step_id, [this](absl::Status status) { set_status(status); },
      param.partial_run_status);

  EXPECT_EQ(status(), param.expected_status);
}

TEST_P(StatusPropagationTest, PartialRunDoneFirst) {
  // Tests error propagation when PartialRunDone is called first.
  StatusTestParam param = GetParam();
  int step_id = 1;

  CancellationManager* cancellation_manager;
  partial_run_mgr_.FindOrCreate(step_id, &cancellation_manager);

  partial_run_mgr_.PartialRunDone(
      step_id, [this](absl::Status status) { set_status(status); },
      param.partial_run_status);
  partial_run_mgr_.ExecutorDone(step_id, param.executor_status);

  EXPECT_EQ(status(), param.expected_status);
}

// Instantiate tests for all error orderings, for both call orders of
// ExecutorDone and PartialRunDone.
absl::Status ExecutorError() { return errors::Internal("executor error"); }
absl::Status PartialRunError() { return errors::Internal("partial run error"); }
INSTANTIATE_TEST_SUITE_P(
    PartialRunMgr, StatusPropagationTest,
    ::testing::Values(
        StatusTestParam{absl::OkStatus(), absl::OkStatus(), absl::OkStatus()},
        StatusTestParam{ExecutorError(), absl::OkStatus(), ExecutorError()},
        StatusTestParam{absl::OkStatus(), PartialRunError(), PartialRunError()},
        StatusTestParam{ExecutorError(), PartialRunError(), ExecutorError()}));

}  // namespace
}  // namespace tensorflow
