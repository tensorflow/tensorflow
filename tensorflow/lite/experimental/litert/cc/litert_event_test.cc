// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/cc/litert_event.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/test/matchers.h"

namespace litert {
namespace {

using ::testing::Eq;

constexpr int kFakeSyncFenceFd = 1;

TEST(Event, NoEvent) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event event, Event::CreateFromSyncFenceFd(kFakeSyncFenceFd, true));
  LITERT_ASSERT_OK_AND_ASSIGN(int fd, event.GetSyncFenceFd());
  EXPECT_THAT(fd, Eq(kFakeSyncFenceFd));
}

}  // namespace
}  // namespace litert
