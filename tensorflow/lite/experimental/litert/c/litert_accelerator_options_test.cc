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

#include "tensorflow/lite/experimental/litert/c/litert_accelerator_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/core/accelerator.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"

namespace {
using testing::StrEq;
using testing::litert::IsError;

struct DummyAccleratorCompilationOptions {
  static constexpr const char* const kIdentifier = "dummy-accelerator";
  static constexpr const LiteRtApiVersion kVersion = {0, 1, 0};

  // This NEEDS to be the first non-static field of the structure.
  LiteRtAcceleratorCompilationOptionsHeader link;

  int dummy_option = 3;

  // Allocates and sets the basic structure for the accelerator options.
  static LiteRtStatus Create(LiteRtAcceleratorCompilationOptions* options) {
    if (!options) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    *options = reinterpret_cast<LiteRtAcceleratorCompilationOptions>(
        new DummyAccleratorCompilationOptions());
    LiteRtSetAcceleratorCompilationOptionsDestructor(*options, Destroy);
    LiteRtSetAcceleratorCompilationOptionsIdentifier(*options, kIdentifier);
    LiteRtSetAcceleratorCompilationOptionsVersion(*options, kVersion);
    return kLiteRtStatusOk;
  }

 private:
  // Destroys the options.
  static void Destroy(LiteRtAcceleratorCompilationOptions options) {
    delete reinterpret_cast<DummyAccleratorCompilationOptions*>(options);
  }
};

class LiteRtAcceleratorOptionsTest : public testing::Test {
 public:
  void SetUp() override {
    LITERT_ASSERT_OK(DummyAccleratorCompilationOptions::Create(&options_));
  }

  void TearDown() override {
    LITERT_EXPECT_OK(LiteRtDestroyAcceleratorCompilationOptions(options_));
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
  EXPECT_EQ(version.major, DummyAccleratorCompilationOptions::kVersion.major);
  EXPECT_EQ(version.minor, DummyAccleratorCompilationOptions::kVersion.minor);
  EXPECT_EQ(version.patch, DummyAccleratorCompilationOptions::kVersion.patch);
  EXPECT_EQ(LiteRtGetAcceleratorCompilationOptionsVersion(nullptr, &version),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtGetAcceleratorCompilationOptionsVersion(options_, nullptr),
            kLiteRtStatusErrorInvalidArgument);
}

TEST_F(LiteRtAcceleratorOptionsTest, CreatingAndDestroyingAListWorks) {
  LiteRtAcceleratorCompilationOptions appended_options1, appended_options2;
  ASSERT_EQ(DummyAccleratorCompilationOptions::Create(&appended_options1),
            kLiteRtStatusOk);
  ASSERT_EQ(DummyAccleratorCompilationOptions::Create(&appended_options2),
            kLiteRtStatusOk);

  EXPECT_EQ(
      LiteRtAppendAcceleratorCompilationOptions(&options_, appended_options1),
      kLiteRtStatusOk);
  EXPECT_EQ(
      LiteRtAppendAcceleratorCompilationOptions(&options_, appended_options2),
      kLiteRtStatusOk);

  // Iterate through the list to check that the links have been correctly added.

  LiteRtAcceleratorCompilationOptions options_it = options_;
  ASSERT_EQ(LiteRtGetNextAcceleratorCompilationOptions(&options_it),
            kLiteRtStatusOk);
  EXPECT_EQ(options_it, appended_options1);

  ASSERT_EQ(LiteRtGetNextAcceleratorCompilationOptions(&options_it),
            kLiteRtStatusOk);
  EXPECT_EQ(options_it, appended_options2);

  // The list is destroyed in the `TearDown()` function.
}

}  // namespace
