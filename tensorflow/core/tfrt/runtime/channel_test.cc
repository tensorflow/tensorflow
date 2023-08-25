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
#include "tensorflow/core/tfrt/runtime/channel.h"

#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/synchronization/blocking_counter.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

using ::testing::ElementsAreArray;
using ::testing::UnorderedElementsAreArray;
using ::tsl::testing::StatusIs;

TEST(ChannelTest, Basic) {
  UnboundedChannel<int> channel;

  std::vector<int> expected(100);
  std::iota(expected.begin(), expected.end(), 0);

  tsl::Env::Default()->SchedClosure([&]() {
    for (int v : expected) {
      CHECK_OK(channel.Write(v));
    }
    channel.Close();
  });

  std::vector<int> outputs;
  int v = -1;
  while (channel.Read(v)) {
    outputs.push_back(v);
  }

  EXPECT_THAT(outputs, ElementsAreArray(expected));

  EXPECT_THAT(channel.Write(100), StatusIs(absl::StatusCode::kInternal));
}

TEST(ChannelTest, MultipleWriters) {
  UnboundedChannel<int> channel;

  std::vector<int> expected(100);
  std::iota(expected.begin(), expected.end(), 0);

  tsl::Env::Default()->SchedClosure([&]() {
    absl::BlockingCounter bcount(expected.size());
    for (int v : expected) {
      tsl::Env::Default()->SchedClosure([&, v]() {
        CHECK_OK(channel.Write(v));
        bcount.DecrementCount();
      });
    }
    bcount.Wait();
    channel.Close();
  });

  std::vector<int> outputs;
  int v = 0;
  while (channel.Read(v)) {
    outputs.push_back(v);
  }

  EXPECT_THAT(outputs, UnorderedElementsAreArray(expected));
}

TEST(ChannelTest, MultipleReaders) {
  UnboundedChannel<int> channel;

  std::vector<int> expected(100);
  std::iota(expected.begin(), expected.end(), 0);

  absl::Mutex mu;
  std::vector<int> outputs;

  int num_readers = 200;
  absl::BlockingCounter bcount(num_readers);
  for (int i = 0; i < num_readers; ++i) {
    tsl::Env::Default()->SchedClosure([&]() {
      int v = 0;
      while (channel.Read(v)) {
        absl::MutexLock lock(&mu);
        outputs.push_back(v);
      }
      bcount.DecrementCount();
    });
  }

  for (int v : expected) {
    CHECK_OK(channel.Write(v));
  }
  channel.Close();

  bcount.Wait();
  EXPECT_THAT(outputs, UnorderedElementsAreArray(expected));
}

TEST(ChannelTest, FullyBuffered) {
  UnboundedChannel<int> channel;

  std::vector<int> expected(100);
  std::iota(expected.begin(), expected.end(), 0);

  for (int v : expected) {
    CHECK_OK(channel.Write(v));
  }
  channel.Close();

  std::vector<int> outputs;
  int v = -1;
  while (channel.Read(v)) {
    outputs.push_back(v);
  }

  EXPECT_THAT(outputs, ElementsAreArray(expected));
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
