/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/exceptions.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

class ScopedEnv {
 public:
  explicit ScopedEnv(const char* var, const char* value) : var_(var) {
    tsl::setenv(var, value, 1);
  }

  ~ScopedEnv() { tsl::unsetenv(var_.data()); }

 private:
  absl::string_view var_;
};

TEST(XlaRuntimeErrorTest, BasicMessage) {
  absl::Status status = absl::InternalError("test error");
  XlaRuntimeError e(status);
  EXPECT_THAT(e.what(), HasSubstr("INTERNAL: test error"));
}

TEST(XlaRuntimeErrorTest, WithPayload) {
  absl::Status status = absl::InternalError("test error");
  status.SetPayload("key1", absl::Cord("value1"));
  XlaRuntimeError e(status);
  EXPECT_THAT(e.what(), HasSubstr(" [key1='value1']"));
}

TEST(XlaRuntimeErrorTest, TruncatedPayloadNoEnv) {
  absl::Status status = absl::InternalError("test error");
  std::string huge_value(500, 'a');
  status.SetPayload("key2", absl::Cord(huge_value));
  XlaRuntimeError e(status);
  EXPECT_THAT(e.what(), HasSubstr(" [key2='"));
  EXPECT_THAT(e.what(), HasSubstr("...[truncated]']"));
}

TEST(XlaRuntimeErrorTest, TruncatedPayloadFilterOn) {
  absl::Status status = absl::InternalError("test error");
  ScopedEnv env("JAX_TRACEBACK_FILTERING", "on");
  std::string huge_value(500, 'a');
  status.SetPayload("key2", absl::Cord(huge_value));
  XlaRuntimeError e(status);
  EXPECT_THAT(e.what(), HasSubstr(" [key2='"));
  EXPECT_THAT(e.what(), HasSubstr("...[truncated]']"));
}

TEST(XlaRuntimeErrorTest, MultiplePayloads) {
  absl::Status status = absl::InternalError("test error");
  status.SetPayload("key1", absl::Cord("value1"));
  status.SetPayload("key2", absl::Cord("value2"));
  XlaRuntimeError e(status);
  EXPECT_THAT(e.what(), HasSubstr(" [key1='value1']"));
  EXPECT_THAT(e.what(), HasSubstr(" [key2='value2']"));
}

TEST(XlaRuntimeErrorTest, ShowStackTracesOn) {
  ScopedEnv env("JAX_TRACEBACK_FILTERING", "off");
  absl::Status status = absl::InternalError("test error");
  std::string huge_value(500, 'a');
  status.SetPayload("key1", absl::Cord(huge_value));
  XlaRuntimeError e(status);
  EXPECT_THAT(e.what(), HasSubstr("key1"));
  EXPECT_THAT(e.what(), HasSubstr(huge_value));
}

TEST(XlaRuntimeErrorTest, EscapesMalformedUtf8) {
  absl::Status status = absl::InternalError("test error");
  std::string malformed_utf8 = "hello\xff";
  status.SetPayload("key1", absl::Cord(malformed_utf8));
  XlaRuntimeError e(status);
  EXPECT_THAT(e.what(), HasSubstr(" [key1='hello\\xff']"));
}

}  // namespace
}  // namespace xla
