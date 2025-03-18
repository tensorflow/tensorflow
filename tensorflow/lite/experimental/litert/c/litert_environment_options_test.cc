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

#include "tensorflow/lite/experimental/litert/c/litert_environment_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_any.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/core/environment_options.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"

namespace {

using testing::AnyOf;
using testing::Eq;
using testing::Not;
using testing::StrEq;
using testing::litert::IsError;

class LiteRtEnvironmentOptionsTest : public testing::Test {
 public:
  void SetUp() override {
    constexpr const char* kStrValue = "string_value";
    dispatch_option_.tag = kLiteRtEnvOptionTagDispatchLibraryDir;
    dispatch_option_.value.type = kLiteRtAnyTypeString;
    dispatch_option_.value.str_value = kStrValue;
    options_.SetOption(dispatch_option_);

    constexpr int kIntValue = 3;
    cl_device_id_option_.tag = kLiteRtEnvOptionTagOpenClDeviceId;
    cl_device_id_option_.value.type = kLiteRtAnyTypeInt;
    cl_device_id_option_.value.int_value = kIntValue;
    options_.SetOption(cl_device_id_option_);

    ASSERT_THAT(NotInsertedOptionTag(),
                Not(AnyOf(dispatch_option_.tag, cl_device_id_option_.tag)));
  }

  LiteRtEnvironmentOptions Options() { return &options_; }
  const LiteRtEnvOption& DispatchOption() const { return dispatch_option_; }
  const LiteRtEnvOption& ClDeviceIdOption() const {
    return cl_device_id_option_;
  }

  static constexpr LiteRtEnvOptionTag NotInsertedOptionTag() {
    return kLiteRtEnvOptionTagOpenClPlatformId;
  }

 private:
  LiteRtEnvironmentOptionsT options_;
  LiteRtEnvOption dispatch_option_;
  LiteRtEnvOption cl_device_id_option_;
};

TEST_F(LiteRtEnvironmentOptionsTest,
       LiteRtGetEnvironmentOptionsValueReturnsAnErrorForInvalidArguments) {
  LiteRtAny option_value;
  EXPECT_THAT(
      LiteRtGetEnvironmentOptionsValue(
          /*options=*/nullptr, kLiteRtEnvOptionTagOpenClContext, &option_value),
      IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(
      LiteRtGetEnvironmentOptionsValue(
          Options(), kLiteRtEnvOptionTagOpenClContext, /*value=*/nullptr),
      IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST_F(LiteRtEnvironmentOptionsTest, LiteRtGetEnvironmentOptionsValueWorks) {
  LiteRtAny option_value;
  LITERT_EXPECT_OK(LiteRtGetEnvironmentOptionsValue(
      Options(), ClDeviceIdOption().tag, &option_value));
  EXPECT_THAT(option_value.type, Eq(ClDeviceIdOption().value.type));
  EXPECT_THAT(option_value.int_value, Eq(ClDeviceIdOption().value.int_value));

  LITERT_EXPECT_OK(LiteRtGetEnvironmentOptionsValue(
      Options(), DispatchOption().tag, &option_value));
  EXPECT_THAT(option_value.type, Eq(DispatchOption().value.type));
  EXPECT_THAT(option_value.str_value, StrEq(DispatchOption().value.str_value));

  EXPECT_THAT(LiteRtGetEnvironmentOptionsValue(
                  Options(), NotInsertedOptionTag(), &option_value),
              IsError(kLiteRtStatusErrorNotFound));
}

}  // namespace
