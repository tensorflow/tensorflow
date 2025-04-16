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

#include "tensorflow/lite/experimental/litert/c/litert_accelerator_compilation_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/core/version.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"

namespace {

using testing::StrEq;
using testing::litert::IsError;

struct DummyAccleratorCompilationOptions {
  static constexpr const LiteRtApiVersion kVersion = {0, 1, 0};
  static constexpr const char* const kIdentifier = "dummy-accelerator";

  int dummy_option = 3;

  // Allocates and sets the basic structure for the accelerator options.
  static litert::Expected<LiteRtAcceleratorCompilationOptions> CreateOptions() {
    auto* payload = new DummyAccleratorCompilationOptions;
    auto payload_destructor = [](void* payload) {
      delete reinterpret_cast<DummyAccleratorCompilationOptions*>(payload);
    };
    return CreateOptions(kVersion, kIdentifier, payload, payload_destructor);
  }

  static litert::Expected<LiteRtAcceleratorCompilationOptions> CreateOptions(
      LiteRtApiVersion version, const char* identifier, void* payload,
      void (*payload_destructor)(void*)) {
    LiteRtAcceleratorCompilationOptions options;
    LITERT_RETURN_IF_ERROR(LiteRtCreateAcceleratorCompilationOptions(
        &version, identifier, payload, payload_destructor, &options));
    return options;
  }
};

class LiteRtAcceleratorOptionsTest : public testing::Test {
 public:
  void SetUp() override {
    auto options = DummyAccleratorCompilationOptions::CreateOptions();
    EXPECT_TRUE(options);
    options_ = *options;
  }

  void TearDown() override {
    LiteRtDestroyAcceleratorCompilationOptions(options_);
    options_ = nullptr;
  }

  LiteRtAcceleratorCompilationOptions options_ = nullptr;
};

TEST_F(LiteRtAcceleratorOptionsTest, CreateAndDestroyDoesntLeak) {}

TEST_F(LiteRtAcceleratorOptionsTest, GetIdentifier) {
  const char* identifier = nullptr;
  LITERT_EXPECT_OK(
      LiteRtGetAcceleratorCompilationOptionsIdentifier(options_, &identifier));
  EXPECT_THAT(identifier,
              StrEq(DummyAccleratorCompilationOptions::kIdentifier));
  EXPECT_THAT(
      LiteRtGetAcceleratorCompilationOptionsIdentifier(nullptr, &identifier),
      IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_EQ(LiteRtGetAcceleratorCompilationOptionsIdentifier(options_, nullptr),
            kLiteRtStatusErrorInvalidArgument);
}

TEST_F(LiteRtAcceleratorOptionsTest, GetVersion) {
  LiteRtApiVersion version;
  EXPECT_EQ(LiteRtGetAcceleratorCompilationOptionsVersion(options_, &version),
            kLiteRtStatusOk);
  EXPECT_TRUE(litert::internal::IsSameVersion(
      version, DummyAccleratorCompilationOptions::kVersion));
  EXPECT_EQ(LiteRtGetAcceleratorCompilationOptionsVersion(nullptr, &version),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtGetAcceleratorCompilationOptionsVersion(options_, nullptr),
            kLiteRtStatusErrorInvalidArgument);
}

TEST_F(LiteRtAcceleratorOptionsTest, CreatingAndDestroyingAListWorks) {
  auto appended_options1 = DummyAccleratorCompilationOptions::CreateOptions();
  ASSERT_TRUE(appended_options1);
  auto appended_options2 = DummyAccleratorCompilationOptions::CreateOptions();
  ASSERT_TRUE(appended_options2);

  EXPECT_EQ(
      LiteRtAppendAcceleratorCompilationOptions(&options_, *appended_options1),
      kLiteRtStatusOk);
  EXPECT_EQ(
      LiteRtAppendAcceleratorCompilationOptions(&options_, *appended_options2),
      kLiteRtStatusOk);

  // Iterate through the list to check that the links have been correctly added.

  LiteRtAcceleratorCompilationOptions options_it = options_;
  ASSERT_EQ(LiteRtGetNextAcceleratorCompilationOptions(&options_it),
            kLiteRtStatusOk);
  EXPECT_EQ(options_it, *appended_options1);

  ASSERT_EQ(LiteRtGetNextAcceleratorCompilationOptions(&options_it),
            kLiteRtStatusOk);
  EXPECT_EQ(options_it, *appended_options2);

  // The list is destroyed in the `TearDown()` function.
}

TEST_F(LiteRtAcceleratorOptionsTest, FindData) {
  constexpr LiteRtApiVersion appended_options_version = {1, 2, 3};
  constexpr auto* appended_options_id = "appended_options_id";
  void* appended_options_data = reinterpret_cast<void*>(12345);
  constexpr auto appended_options_destructor = [](void*) {};

  auto appended_options = DummyAccleratorCompilationOptions::CreateOptions(
      appended_options_version, appended_options_id, appended_options_data,
      appended_options_destructor);

  EXPECT_EQ(
      LiteRtAppendAcceleratorCompilationOptions(&options_, *appended_options),
      kLiteRtStatusOk);

  LiteRtApiVersion payload_version;
  void* payload_data;
  EXPECT_EQ(LiteRtFindAcceleratorCompilationOptionsData(
                options_, appended_options_id, &payload_version, &payload_data),
            kLiteRtStatusOk);

  EXPECT_EQ(payload_version.major, appended_options_version.major);
  EXPECT_EQ(payload_version.minor, appended_options_version.minor);
  EXPECT_EQ(payload_version.patch, appended_options_version.patch);
  EXPECT_EQ(payload_data, appended_options_data);

  // The list is destroyed in the `TearDown()` function.
}

TEST_F(LiteRtAcceleratorOptionsTest, Pop) {
  constexpr LiteRtApiVersion appended_options_version = {1, 2, 3};
  constexpr auto* appended_options_id = "appended_options_id";
  void* appended_options_data = reinterpret_cast<void*>(12345);
  constexpr auto appended_options_destructor = [](void*) {};

  auto appended_options = DummyAccleratorCompilationOptions::CreateOptions(
      appended_options_version, appended_options_id, appended_options_data,
      appended_options_destructor);

  EXPECT_EQ(
      LiteRtAppendAcceleratorCompilationOptions(&options_, *appended_options),
      kLiteRtStatusOk);

  LiteRtApiVersion payload_version;
  void* payload_data;
  EXPECT_EQ(LiteRtFindAcceleratorCompilationOptionsData(
                options_, appended_options_id, &payload_version, &payload_data),
            kLiteRtStatusOk);

  // After poping the last item, we shouldn't be able to find it any longer.
  EXPECT_EQ(LiteRtPopAcceleratorCompilationOptions(&options_), kLiteRtStatusOk);
  EXPECT_NE(LiteRtFindAcceleratorCompilationOptionsData(
                options_, appended_options_id, &payload_version, &payload_data),
            kLiteRtStatusOk);

  // The list is destroyed in the `TearDown()` function.
}

}  // namespace
