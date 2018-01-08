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

#include <atomic>

#include "tensorflow/core/util/reffed_status_callback.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(TestReffedStatusCallback, CallsBackOK) {
  bool called = false;
  Status status = errors::InvalidArgument("");
  auto done = [&called, &status](const Status& s) {
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
  Status status = Status::OK();
  auto done = [&called, &status](const Status& s) {
    called = true;
    status = s;
  };
  auto* cb = new ReffedStatusCallback(std::move(done));
  cb->UpdateStatus(errors::Internal("1"));
  cb->UpdateStatus(errors::Internal("2"));  // Will be ignored.
  EXPECT_FALSE(called);
  cb->Unref();
  EXPECT_TRUE(called);
  EXPECT_EQ(status.error_message(), "1");
}

TEST(TestReffedStatusCallback, RefMulti) {
  int called = false;
  Status status = Status::OK();
  auto done = [&called, &status](const Status& s) {
    called = true;
    status = s;
  };
  auto* cb = new ReffedStatusCallback(std::move(done));
  cb->Ref();
  cb->UpdateStatus(errors::Internal("1"));
  cb->Ref();
  cb->UpdateStatus(errors::Internal("2"));  // Will be ignored.
  cb->Unref();
  cb->Unref();
  EXPECT_FALSE(called);
  cb->Unref();  // Created by constructor.
  EXPECT_TRUE(called);
  EXPECT_EQ(status.error_message(), "1");
}

TEST(TestReffedStatusCallback, MultiThreaded) {
  std::atomic<int> num_called(0);
  Status status;
  Notification n;

  auto done = [&num_called, &status, &n](const Status& s) {
    ++num_called;
    status = s;
    n.Notify();
  };

  auto* cb = new ReffedStatusCallback(std::move(done));

  thread::ThreadPool threads(Env::Default(), "test", 3);
  for (int i = 0; i < 5; ++i) {
    cb->Ref();
    threads.Schedule([cb]() {
      cb->UpdateStatus(errors::InvalidArgument("err"));
      cb->Unref();
    });
  }

  // Subtract one for the initial (construction) reference.
  cb->Unref();

  n.WaitForNotification();

  EXPECT_EQ(num_called.load(), 1);
  EXPECT_EQ(status.error_message(), "err");
}

}  // namespace
}  // namespace tensorflow
