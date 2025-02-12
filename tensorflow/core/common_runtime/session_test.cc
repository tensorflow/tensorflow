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

#include "tensorflow/core/public/session.h"

#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

TEST(SessionTest, InvalidTargetReturnsNull) {
  SessionOptions options;
  options.target = "invalid target";

  EXPECT_EQ(nullptr, tensorflow::NewSession(options));

  Session* session;
  absl::Status s = tensorflow::NewSession(options, &session);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
  EXPECT_TRUE(absl::StrContains(
      s.message(),
      "No session factory registered for the given session options"));
}

// Register a fake session factory to test error handling paths in
// NewSession().
class FakeSessionFactory : public SessionFactory {
 public:
  FakeSessionFactory() {}

  bool AcceptsOptions(const SessionOptions& options) override {
    return absl::StartsWith(options.target, "fake");
  }

  absl::Status NewSession(const SessionOptions& options,
                          Session** out_session) override {
    *out_session = nullptr;
    return absl::OkStatus();
  }
};
class FakeSessionRegistrar {
 public:
  FakeSessionRegistrar() {
    SessionFactory::Register("FAKE_SESSION_1", new FakeSessionFactory());
    SessionFactory::Register("FAKE_SESSION_2", new FakeSessionFactory());
  }
};
static FakeSessionRegistrar registrar;

TEST(SessionTest, MultipleFactoriesForTarget) {
  SessionOptions options;
  options.target = "fakesession";

  Session* session;
  absl::Status s = tensorflow::NewSession(options, &session);
  EXPECT_EQ(s.code(), error::INTERNAL);
  EXPECT_TRUE(absl::StrContains(s.message(), "Multiple session factories"));
  EXPECT_TRUE(absl::StrContains(s.message(), "FAKE_SESSION_1"));
  EXPECT_TRUE(absl::StrContains(s.message(), "FAKE_SESSION_2"));
}

}  // namespace
}  // namespace tensorflow
