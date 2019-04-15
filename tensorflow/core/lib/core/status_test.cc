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

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

TEST(Status, OK) {
  EXPECT_EQ(Status::OK().code(), error::OK);
  EXPECT_EQ(Status::OK().error_message(), "");
  TF_EXPECT_OK(Status::OK());
  TF_ASSERT_OK(Status::OK());
  EXPECT_EQ(Status::OK(), Status());
  Status s;
  EXPECT_TRUE(s.ok());
}

TEST(DeathStatus, CheckOK) {
  Status status(errors::InvalidArgument("Invalid"));
  ASSERT_DEATH(TF_CHECK_OK(status), "Invalid");
}

TEST(Status, Set) {
  Status status;
  status = Status(error::CANCELLED, "Error message");
  EXPECT_EQ(status.code(), error::CANCELLED);
  EXPECT_EQ(status.error_message(), "Error message");
}

TEST(Status, Copy) {
  Status a(errors::InvalidArgument("Invalid"));
  Status b(a);
  ASSERT_EQ(a.ToString(), b.ToString());
}

TEST(Status, Assign) {
  Status a(errors::InvalidArgument("Invalid"));
  Status b;
  b = a;
  ASSERT_EQ(a.ToString(), b.ToString());
}

TEST(Status, Update) {
  Status s;
  s.Update(Status::OK());
  ASSERT_TRUE(s.ok());
  Status a(errors::InvalidArgument("Invalid"));
  s.Update(a);
  ASSERT_EQ(s.ToString(), a.ToString());
  Status b(errors::Internal("Internal"));
  s.Update(b);
  ASSERT_EQ(s.ToString(), a.ToString());
  s.Update(Status::OK());
  ASSERT_EQ(s.ToString(), a.ToString());
  ASSERT_FALSE(s.ok());
}

TEST(Status, EqualsOK) { ASSERT_EQ(Status::OK(), Status()); }

TEST(Status, EqualsSame) {
  Status a(errors::InvalidArgument("Invalid"));
  Status b(errors::InvalidArgument("Invalid"));
  ASSERT_EQ(a, b);
}

TEST(Status, EqualsCopy) {
  const Status a(errors::InvalidArgument("Invalid"));
  const Status b = a;
  ASSERT_EQ(a, b);
}

TEST(Status, EqualsDifferentCode) {
  const Status a(errors::InvalidArgument("message"));
  const Status b(errors::Internal("message"));
  ASSERT_NE(a, b);
}

TEST(Status, EqualsDifferentMessage) {
  const Status a(errors::InvalidArgument("message"));
  const Status b(errors::InvalidArgument("another"));
  ASSERT_NE(a, b);
}

TEST(StatusGroup, AcceptsFirstCode) {
  StatusGroup c;
  const Status internal(errors::Internal("Original error."));
  c.Update(internal);
  c.Update(Status::OK());
  c.Update(Status::OK());
  c.Update(Status::OK());
  ASSERT_EQ(c.as_status().code(), internal.code());
  ASSERT_EQ(c.ok(), false);
}

TEST(StatusGroup, ContainsChildMessages) {
  StatusGroup c;
  const Status internal(errors::Internal("Original error."));
  const Status cancelled(errors::Cancelled("Cancelled after 10 steps."));
  const Status aborted(errors::Aborted("Aborted after 10 steps."));
  c.Update(internal);
  for (size_t i = 0; i < 5; ++i) {
    c.Update(cancelled);
  }
  for (size_t i = 0; i < 10; ++i) {
    c.Update(aborted);
  }
  for (size_t i = 0; i < 100; ++i) {
    c.Update(Status::OK());
  }

  ASSERT_EQ(c.as_status().code(), internal.code());
  EXPECT_TRUE(str_util::StrContains(c.as_status().error_message(),
                                    internal.error_message()));
  EXPECT_TRUE(str_util::StrContains(c.as_status().error_message(),
                                    cancelled.error_message()));
  EXPECT_TRUE(str_util::StrContains(c.as_status().error_message(),
                                    aborted.error_message()));
  StatusGroup d;
  d.Update(c.as_status());
  c.Update(errors::FailedPrecondition("Failed!"));
  d.Update(c.as_status());
  c.Update(errors::DataLoss("Data loss!"));
  d.Update(c.as_status());
  LOG(INFO) << d.as_status();
}

TEST(StatusGroup, ContainsIdenticalMessage) {
  StatusGroup sg;
  const Status internal(errors::Internal("Original error"));
  for (size_t i = 0; i < 10; i++) {
    sg.Update(internal);
  }
  EXPECT_EQ(sg.as_status(), internal);
}

TEST(StatusGroup, ContainsCommonPrefix) {
  StatusGroup sg;
  const Status a(errors::Internal("Original error"));
  const Status b(errors::Internal("Original error is"));
  const Status c(errors::Internal("Original error is invalid"));
  sg.Update(a);
  sg.Update(c);
  sg.Update(c);
  sg.Update(b);
  sg.Update(c);
  sg.Update(b);
  sg.Update(a);
  sg.Update(b);
  EXPECT_EQ(sg.as_status(), c);
}

static void BM_TF_CHECK_OK(int iters) {
  tensorflow::Status s =
      (iters < 0) ? errors::InvalidArgument("Invalid") : Status::OK();
  for (int i = 0; i < iters; i++) {
    TF_CHECK_OK(s);
  }
}
BENCHMARK(BM_TF_CHECK_OK);

}  // namespace tensorflow
