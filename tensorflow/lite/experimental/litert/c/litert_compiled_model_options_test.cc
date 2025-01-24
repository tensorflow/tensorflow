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

#include "tensorflow/lite/experimental/litert/c/litert_compiled_model_options.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_common.h"

namespace {

TEST(LiteRtCompiledModelOptionsTest, CreateAndDestroyDontLeak) {
  LiteRtCompilationOptions options;
  ASSERT_EQ(LiteRtCreateCompilationOptions(&options), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtDestroyCompilationOptions(options), kLiteRtStatusOk);
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
                options, kLiteRtHwAccelatorNone),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetCompilationOptionsHardwareAccelerators(
                options, &hardware_accelerators),
            kLiteRtStatusOk);
  EXPECT_EQ(hardware_accelerators, kLiteRtHwAccelatorNone);

  EXPECT_EQ(LiteRtSetCompilationOptionsHardwareAccelerators(
                options, kLiteRtHwAccelatorCpu),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetCompilationOptionsHardwareAccelerators(
                options, &hardware_accelerators),
            kLiteRtStatusOk);
  EXPECT_EQ(hardware_accelerators, kLiteRtHwAccelatorCpu);

  EXPECT_EQ(LiteRtSetCompilationOptionsHardwareAccelerators(
                options, kLiteRtHwAccelatorGpu),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetCompilationOptionsHardwareAccelerators(
                options, &hardware_accelerators),
            kLiteRtStatusOk);
  EXPECT_EQ(hardware_accelerators, kLiteRtHwAccelatorGpu);

  EXPECT_EQ(LiteRtSetCompilationOptionsHardwareAccelerators(
                options, kLiteRtHwAccelatorNpu),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetCompilationOptionsHardwareAccelerators(
                options, &hardware_accelerators),
            kLiteRtStatusOk);
  EXPECT_EQ(hardware_accelerators, kLiteRtHwAccelatorNpu);

  EXPECT_EQ(LiteRtSetCompilationOptionsHardwareAccelerators(
                options, (kLiteRtHwAccelatorCpu | kLiteRtHwAccelatorGpu |
                          kLiteRtHwAccelatorNpu) +
                             1),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtSetCompilationOptionsHardwareAccelerators(
                nullptr, kLiteRtHwAccelatorNone),
            kLiteRtStatusErrorInvalidArgument);

  EXPECT_EQ(LiteRtDestroyCompilationOptions(options), kLiteRtStatusOk);
}

}  // namespace
