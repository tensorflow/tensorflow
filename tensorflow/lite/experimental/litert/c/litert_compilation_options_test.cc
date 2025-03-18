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
#include "tensorflow/lite/experimental/litert/c/litert_accelerator_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/core/accelerator.h"

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

struct DummyAccleratorCompilationOptions {
  // This NEEDS to be the first non-static field of the structure.
  LiteRtAcceleratorCompilationOptionsHeader link;

  // Allocates and sets the basic structure for the accelerator options.
  static LiteRtStatus Create(LiteRtAcceleratorCompilationOptions* options) {
    if (!options) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    *options = reinterpret_cast<LiteRtAcceleratorCompilationOptions>(
        new DummyAccleratorCompilationOptions());
    LiteRtSetAcceleratorCompilationOptionsDestructor(*options, Destroy);
    return kLiteRtStatusOk;
  }

 private:
  // Destroys the options.
  static void Destroy(LiteRtAcceleratorCompilationOptions options) {
    delete reinterpret_cast<DummyAccleratorCompilationOptions*>(options);
  }
};

TEST(LiteRtCompiledModelOptionsTest, AddAcceleratorCompilationOptionsWorks) {
  LiteRtCompilationOptions options;
  ASSERT_EQ(LiteRtCreateCompilationOptions(&options), kLiteRtStatusOk);

  LiteRtAcceleratorCompilationOptions accelerator_options1,
      accelerator_options2;
  ASSERT_EQ(DummyAccleratorCompilationOptions::Create(&accelerator_options1),
            kLiteRtStatusOk);
  ASSERT_EQ(DummyAccleratorCompilationOptions::Create(&accelerator_options2),
            kLiteRtStatusOk);

  EXPECT_EQ(
      LiteRtAddAcceleratorCompilationOptions(nullptr, accelerator_options1),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtAddAcceleratorCompilationOptions(options, nullptr),
            kLiteRtStatusErrorInvalidArgument);

  EXPECT_EQ(
      LiteRtAddAcceleratorCompilationOptions(options, accelerator_options1),
      kLiteRtStatusOk);
  EXPECT_EQ(
      LiteRtAddAcceleratorCompilationOptions(options, accelerator_options2),
      kLiteRtStatusOk);

  LiteRtAcceleratorCompilationOptions options_it = nullptr;
  EXPECT_EQ(LiteRtGetAcceleratorCompilationOptions(options, &options_it),
            kLiteRtStatusOk);
  EXPECT_EQ(options_it, accelerator_options1);

  EXPECT_EQ(LiteRtGetNextAcceleratorCompilationOptions(&options_it),
            kLiteRtStatusOk);
  EXPECT_EQ(options_it, accelerator_options2);

  LiteRtDestroyCompilationOptions(options);
}

}  // namespace
