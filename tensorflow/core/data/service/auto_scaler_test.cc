/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/service/auto_scaler.h"

#include "absl/time/time.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace tensorflow {
namespace data {
namespace {

using ::tsl::testing::StatusIs;

TEST(AutoScalerTest, ReportProcessingTimeNewWorker) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(10)));
}

TEST(AutoScalerTest, ReportProcessingTimeExistingWorker) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(20)));
}

TEST(AutoScalerTest, ReportProcessingTimeNewAndExisting) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/1:20000",
                                                absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/2:20000",
                                                absl::Microseconds(30)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(30)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/1:20000",
                                                absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/2:20000",
                                                absl::Microseconds(10)));
}

TEST(AutoScalerTest, ReportProcessingTimeZeroDuration) {
  AutoScaler auto_scaler;
  auto result = auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                 absl::ZeroDuration());
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AutoScalerTest, ReportProcessingTimeNegativeDuration) {
  AutoScaler auto_scaler;
  auto result = auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                 absl::Microseconds(-10));
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AutoScalerTest, ReportTargetProcessingTimeNewConsumer) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(10)));
}

TEST(AutoScalerTest, ReportTargetProcessingTimeExistingConsumer) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(10)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(20)));
}

TEST(AutoScalerTest, ReportTargetProcessingTimeNewAndExisting) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(10)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, absl::Microseconds(20)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(2, absl::Microseconds(30)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(30)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, absl::Microseconds(20)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(2, absl::Microseconds(10)));
}

TEST(AutoScalerTest, ReportTargetProcessingTimeZeroDuration) {
  AutoScaler auto_scaler;
  auto result = auto_scaler.ReportTargetProcessingTime(0, absl::ZeroDuration());
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AutoScalerTest, ReportTargetProcessingTimeNegativeDuration) {
  AutoScaler auto_scaler;
  auto result =
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(-10));
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AutoScalerTest, RemoveWorkerSuccessful) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/1:20000",
                                                absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.RemoveWorker("/worker/task/0:20000"));
  TF_ASSERT_OK(auto_scaler.RemoveWorker("/worker/task/1:20000"));
}

TEST(AutoScalerTest, RemoveNonexistentWorker) {
  AutoScaler auto_scaler;
  EXPECT_THAT(auto_scaler.RemoveWorker("/worker/task/0:20000"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(AutoScalerTest, RemoveWorkerAfterNewPTReported) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(10)));
  TF_ASSERT_OK(auto_scaler.ReportProcessingTime("/worker/task/0:20000",
                                                absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.RemoveWorker("/worker/task/0:20000"));
}

TEST(AutoScalerTest, RemoveConsumerSuccessful) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(30)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(1, absl::Microseconds(30)));
  TF_ASSERT_OK(auto_scaler.RemoveConsumer(0));
  TF_ASSERT_OK(auto_scaler.RemoveConsumer(1));
}

TEST(AutoScalerTest, RemoveNonexistentConsumer) {
  AutoScaler auto_scaler;
  EXPECT_THAT(auto_scaler.RemoveConsumer(0),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(AutoScalerTest, RemoveConsumerAfterNewTPTReported) {
  AutoScaler auto_scaler;
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(30)));
  TF_ASSERT_OK(
      auto_scaler.ReportTargetProcessingTime(0, absl::Microseconds(20)));
  TF_ASSERT_OK(auto_scaler.RemoveConsumer(0));
}

}  // namespace

}  // namespace data
}  // namespace tensorflow
