/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/reffed_status_callback.h"

#include <atomic>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace {

TEST(TestReffedStatusCallback, CallsBackOK) {
  bool called = false;
  absl::Status status = absl::InvalidArgumentError("");
  auto done = [&called, &status](const absl::Status& s) {
    called = true;
    status = s;
  };
  auto* cb = new ReffedStatusCallback(std::move(done));
  EXPECT_FALSE(called);
  cb->Unref();
  EXPECT_TRUE(called);
  EXPECT_TRUE(status.ok());
}

TEST(TestReffedStatusCallback, CallsBackFail) {
  bool called = false;
  absl::Status status = absl::OkStatus();
  auto done = [&called, &status](const absl::Status& s) {
    called = true;
    status = s;
  };
  auto* cb = new ReffedStatusCallback(std::move(done));
  cb->UpdateStatus(absl::InternalError("1"));
  cb->UpdateStatus(absl::InvalidArgumentError("2"));
  EXPECT_FALSE(called);
  cb->Unref();
  EXPECT_TRUE(called);
  // Should be one of the two given error codes.
  EXPECT_THAT(status.code(),
              ::testing::AnyOf(error::INTERNAL, error::INVALID_ARGUMENT));
  // Both errors are reported.
  EXPECT_TRUE(absl::StrContains(status.message(), "1"));
  EXPECT_TRUE(absl::StrContains(status.message(), "2"));
}

TEST(TestReffedStatusCallback, RefMulti) {
  int called = false;
  absl::Status status = absl::OkStatus();
  auto done = [&called, &status](const absl::Status& s) {
    called = true;
    status = s;
  };
  auto* cb = new ReffedStatusCallback(std::move(done));
  cb->Ref();
  cb->UpdateStatus(absl::InternalError("1"));
  cb->Ref();
  cb->UpdateStatus(absl::InternalError("2"));
  cb->Unref();
  cb->Unref();
  EXPECT_FALSE(called);
  cb->Unref();  // Created by constructor.
  EXPECT_TRUE(called);
  // Both errors are reported.
  EXPECT_TRUE(absl::StrContains(status.message(), "1"));
  EXPECT_TRUE(absl::StrContains(status.message(), "2"));
}

TEST(TestReffedStatusCallback, MultiThreaded) {
  std::atomic<int> num_called(0);
  absl::Status status;
  absl::Notification n;

  auto done = [&num_called, &status, &n](const absl::Status& s) {
    ++num_called;
    status = s;
    n.Notify();
  };

  auto* cb = new ReffedStatusCallback(std::move(done));

  thread::ThreadPool threads(Env::Default(), "test", 3);
  for (int i = 0; i < 5; ++i) {
    cb->Ref();
    threads.Schedule([cb]() {
      cb->UpdateStatus(absl::InvalidArgumentError("err"));
      cb->Unref();
    });
  }

  // Subtract one for the initial (construction) reference.
  cb->Unref();

  n.WaitForNotification();

  EXPECT_EQ(num_called.load(), 1);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_TRUE(absl::StrContains(status.message(), "err"));
}

}  // namespace
}  // namespace tensorflow
