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

#include "tensorflow/lite/experimental/litert/c/litert_compilation_options.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_accelerator_compilation_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"

namespace {

TEST(LiteRtCompiledModelOptionsTest, CreateAndDestroyDontLeak) {
  LiteRtCompilationOptions options;
  ASSERT_EQ(LiteRtCreateCompilationOptions(&options), kLiteRtStatusOk);
  LiteRtDestroyCompilationOptions(options);
}

TEST(LiteRtCompiledModelOptionsTest, CreateWithANullPointerErrors) {
  EXPECT_EQ(LiteRtCreateCompilationOptions(nullptr),
            kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtCompiledModelOptionsTest, SetAndGetHardwareAcceleratorsWorks) {
  LiteRtCompilationOptions options;
  ASSERT_EQ(LiteRtCreateCompilationOptions(&options), kLiteRtStatusOk);

  LiteRtHwAcceleratorSet hardware_accelerators;

  EXPECT_EQ(LiteRtSetCompilationOptionsHardwareAccelerators(
                options, kLiteRtHwAcceleratorNone),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetCompilationOptionsHardwareAccelerators(
                options, &hardware_accelerators),
            kLiteRtStatusOk);
  EXPECT_EQ(hardware_accelerators, kLiteRtHwAcceleratorNone);

  EXPECT_EQ(LiteRtSetCompilationOptionsHardwareAccelerators(
                options, kLiteRtHwAcceleratorCpu),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetCompilationOptionsHardwareAccelerators(
                options, &hardware_accelerators),
            kLiteRtStatusOk);
  EXPECT_EQ(hardware_accelerators, kLiteRtHwAcceleratorCpu);

  EXPECT_EQ(LiteRtSetCompilationOptionsHardwareAccelerators(
                options, kLiteRtHwAcceleratorGpu),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetCompilationOptionsHardwareAccelerators(
                options, &hardware_accelerators),
            kLiteRtStatusOk);
  EXPECT_EQ(hardware_accelerators, kLiteRtHwAcceleratorGpu);

  EXPECT_EQ(LiteRtSetCompilationOptionsHardwareAccelerators(
                options, kLiteRtHwAcceleratorNpu),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetCompilationOptionsHardwareAccelerators(
                options, &hardware_accelerators),
            kLiteRtStatusOk);
  EXPECT_EQ(hardware_accelerators, kLiteRtHwAcceleratorNpu);

  EXPECT_EQ(LiteRtSetCompilationOptionsHardwareAccelerators(
                options, (kLiteRtHwAcceleratorCpu | kLiteRtHwAcceleratorGpu |
                          kLiteRtHwAcceleratorNpu) +
                             1),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtSetCompilationOptionsHardwareAccelerators(
                nullptr, kLiteRtHwAcceleratorNone),
            kLiteRtStatusErrorInvalidArgument);

  LiteRtDestroyCompilationOptions(options);
}

struct DummyAcceleratorCompilationOptions {
  static constexpr const LiteRtApiVersion kVersion = {1, 0, 0};
  static constexpr const char* const kIdentifier = "dummy-accelerator";

  // Allocates and sets the basic structure for the accelerator options.
  static litert::Expected<LiteRtAcceleratorCompilationOptions> CreateOptions() {
    LiteRtAcceleratorCompilationOptions options;
    auto* payload = new DummyAcceleratorCompilationOptions;
    auto payload_destructor = [](void* payload) {
      delete reinterpret_cast<DummyAcceleratorCompilationOptions*>(payload);
    };
    LITERT_RETURN_IF_ERROR(LiteRtCreateAcceleratorCompilationOptions(
        &kVersion, kIdentifier, payload, payload_destructor, &options));
    return options;
  }
};

TEST(LiteRtCompiledModelOptionsTest, AddAcceleratorCompilationOptionsWorks) {
  LiteRtCompilationOptions options;
  ASSERT_EQ(LiteRtCreateCompilationOptions(&options), kLiteRtStatusOk);

  auto accelerator_compilation_options1 =
      DummyAcceleratorCompilationOptions::CreateOptions();
  EXPECT_TRUE(accelerator_compilation_options1);
  auto accelerator_compilation_options2 =
      DummyAcceleratorCompilationOptions::CreateOptions();
  EXPECT_TRUE(accelerator_compilation_options2);

  EXPECT_EQ(LiteRtAddAcceleratorCompilationOptions(
                nullptr, *accelerator_compilation_options1),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtAddAcceleratorCompilationOptions(options, nullptr),
            kLiteRtStatusErrorInvalidArgument);

  EXPECT_EQ(LiteRtAddAcceleratorCompilationOptions(
                options, *accelerator_compilation_options1),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtAddAcceleratorCompilationOptions(
                options, *accelerator_compilation_options2),
            kLiteRtStatusOk);

  LiteRtAcceleratorCompilationOptions options_it = nullptr;
  EXPECT_EQ(LiteRtGetAcceleratorCompilationOptions(options, &options_it),
            kLiteRtStatusOk);
  EXPECT_EQ(options_it, *accelerator_compilation_options1);

  EXPECT_EQ(LiteRtGetNextAcceleratorCompilationOptions(&options_it),
            kLiteRtStatusOk);
  EXPECT_EQ(options_it, *accelerator_compilation_options2);

  LiteRtDestroyCompilationOptions(options);
}

}  // namespace
