/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/utils/concurrency/concurrency_utils.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/utils/concurrency/tsl_task_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::concurrency {
namespace {

using ::testing::ElementsAreArray;

TEST(ForEachTest, IterVariantConcurrentlyIncrementsIntegers) {
  TslTaskExecutor task_executor(5);

  constexpr int kx0 = 0;
  constexpr int kx1 = 1;
  constexpr int kx2 = 2;

  int v0 = kx0;
  int v1 = kx1;
  int v2 = kx2;

  std::vector<int*> v = {&v0, &v1, &v2};

  ASSERT_EQ(ForEach(
                v.begin(), v.end(),
                [](int* element) {
                  ++(*element);
                  return absl::OkStatus();
                },
                task_executor),
            absl::OkStatus());

  EXPECT_EQ(v0, kx0 + 1);
  EXPECT_EQ(v1, kx1 + 1);
  EXPECT_EQ(v2, kx2 + 1);
}

TEST(ForEachTest, NonOkStatusPropagatesAsTheFinalResult) {
  const absl::Status status = absl::CancelledError("Test Error");

  TslTaskExecutor task_executor{3};

  constexpr int kx0 = 0;
  constexpr int kx1 = 1;
  constexpr int kx2 = 2;

  int v0 = kx0;
  int v1 = kx1;
  int v2 = kx2;

  std::vector<int*> v = {&v0, &v1, &v2};

  EXPECT_THAT(ForEach(
                  v.begin(), v.end(),
                  [&status](int* element) { return status; }, task_executor)
                  .code(),
              absl::StatusCode::kCancelled);
}

TEST(ForEachTest, ActionReturnedValuesCollected) {
  TslTaskExecutor task_executor{3};

  constexpr int kx0 = 0;
  constexpr int kx1 = 1;
  constexpr int kx2 = 2;

  int v0 = kx0;
  int v1 = kx1;
  int v2 = kx2;

  std::vector<int*> v = {&v0, &v1, &v2};

  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      (ForEach<int>(
          v.begin(), v.end(),
          [](int* element) -> absl::StatusOr<int> { return ++(*element); },
          task_executor)));

  EXPECT_EQ(v0, kx0 + 1);
  EXPECT_EQ(v1, kx1 + 1);
  EXPECT_EQ(v2, kx2 + 1);

  EXPECT_THAT(result, ElementsAreArray({1, 2, 3}));
}

TEST(ForEachTest, FailureOfTheFirstActionPropagates) {
  TslTaskExecutor task_executor{3};

  constexpr int kx0 = 0;
  constexpr int kx1 = 1;
  constexpr int kx2 = 2;

  int v0 = kx0;
  int v1 = kx1;
  int v2 = kx2;

  std::vector<int*> v = {&v0, &v1, &v2};

  EXPECT_EQ(ForEach<int>(
                v.begin(), v.end(),
                [](int* element) -> absl::StatusOr<int> {
                  if (*element % 2 == 1)
                    return absl::CancelledError("Force a failure.");
                  return ++(*element);
                },
                task_executor)
                .status()
                .code(),
            absl::StatusCode::kCancelled);
}

}  // namespace
}  // namespace xla::concurrency
