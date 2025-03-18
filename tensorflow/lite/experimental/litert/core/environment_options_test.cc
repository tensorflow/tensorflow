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

#include "tensorflow/lite/experimental/litert/core/environment_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_any.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment_options.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"

namespace {

using testing::Eq;
using testing::Ne;
using testing::litert::IsError;

TEST(EnvironmentOptionsTest, SetGetStringOptionWorks) {
  LiteRtEnvironmentOptionsT options;
  constexpr const char* kStrValue = "string_value";
  LiteRtEnvOption env_option{/*tag=*/kLiteRtEnvOptionTagDispatchLibraryDir,
                             /*value=*/{/*type=*/kLiteRtAnyTypeString}};
  env_option.value.str_value = kStrValue;
  options.SetOption(env_option);

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny stored_option,
      options.GetOption(kLiteRtEnvOptionTagDispatchLibraryDir));

  EXPECT_THAT(stored_option.type, Eq(kLiteRtAnyTypeString));
  EXPECT_THAT(stored_option.str_value, Ne(nullptr));
  EXPECT_THAT(stored_option.str_value, Ne(kStrValue));
}

TEST(EnvironmentOptionsTest, SetGetIntOptionWorks) {
  constexpr int kIntValue = 3;
  LiteRtEnvironmentOptionsT options;
  LiteRtEnvOption env_option{/*tag=*/kLiteRtEnvOptionTagOpenClDeviceId,
                             /*value=*/{/*type=*/kLiteRtAnyTypeInt}};
  env_option.value.int_value = kIntValue;
  options.SetOption(env_option);

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtAny stored_option,
      options.GetOption(kLiteRtEnvOptionTagOpenClDeviceId));

  EXPECT_THAT(stored_option.type, Eq(kLiteRtAnyTypeInt));
  EXPECT_THAT(stored_option.int_value, Eq(kIntValue));
}

TEST(EnvironmentOptionsTest, GetNotSetReturnsNotFound) {
  LiteRtEnvironmentOptionsT options;

  // Add a non related option.
  constexpr const char* kStrValue = "string_value";
  LiteRtEnvOption env_option{/*tag=*/kLiteRtEnvOptionTagDispatchLibraryDir,
                             /*value=*/{/*type=*/kLiteRtAnyTypeString}};
  env_option.value.str_value = kStrValue;
  options.SetOption(env_option);

  // Request an option that wasn't added.
  EXPECT_THAT(options.GetOption(kLiteRtEnvOptionTagOpenClDeviceId),
              IsError(kLiteRtStatusErrorNotFound));
}

}  // namespace
