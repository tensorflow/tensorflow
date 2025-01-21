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

#include "tensorflow/lite/experimental/litert/c/litert_accelerator_registration.h"

#include <memory>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_accelerator.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/core/accelerator.h"

namespace {

class DummyAccelerator {
 public:
  static std::unique_ptr<DummyAccelerator> CpuAccelerator() {
    auto accelerator = std::make_unique<DummyAccelerator>();
    accelerator->hardware_support_ = kLiteRtHwAccelatorCpu;
    return accelerator;
  }

  static void Destroy(void* dummy_accelerator) {
    DummyAccelerator* instance =
        reinterpret_cast<DummyAccelerator*>(dummy_accelerator);
    delete instance;
  }

  static LiteRtStatus GetName(LiteRtAccelerator accelerator,
                              const char** name) {
    return kLiteRtStatusOk;
  }

  static LiteRtStatus GetVersion(LiteRtAccelerator accelerator,
                                 LiteRtApiVersion* version) {
    return kLiteRtStatusOk;
  }

  static LiteRtStatus GetHardwareSupport(
      LiteRtAccelerator accelerator,
      LiteRtHwAcceleratorSet* supported_hardware) {
    return kLiteRtStatusOk;
  }

  static LiteRtStatus ApplyToModel(LiteRtAccelerator accelerator,
                                   LiteRtCompiledModel compiled_model) {
    return kLiteRtStatusOk;
  }

  LiteRtHwAccelerators hardware_support_;
};

TEST(LiteRtAcceleratorRegistrationTest, SetAcceleratorGetNameWorks) {
  LiteRtAcceleratorT accelerator;
  EXPECT_EQ(LiteRtSetAcceleratorGetName(nullptr, DummyAccelerator::GetName),
            kLiteRtStatusErrorInvalidArgument);
  LiteRtSetAcceleratorGetName(&accelerator, DummyAccelerator::GetName);
  EXPECT_EQ(accelerator.GetName, DummyAccelerator::GetName);
}

TEST(LiteRtAcceleratorRegistrationTest, SetAcceleratorGetVersionWorks) {
  LiteRtAcceleratorT accelerator;
  EXPECT_EQ(
      LiteRtSetAcceleratorGetVersion(nullptr, DummyAccelerator::GetVersion),
      kLiteRtStatusErrorInvalidArgument);
  LiteRtSetAcceleratorGetVersion(&accelerator, DummyAccelerator::GetVersion);
  EXPECT_EQ(accelerator.GetVersion, DummyAccelerator::GetVersion);
}

TEST(LiteRtAcceleratorRegistrationTest, SetAcceleratorGetHardwareSupportWorks) {
  LiteRtAcceleratorT accelerator;
  EXPECT_EQ(LiteRtSetAcceleratorGetHardwareSupport(
                nullptr, DummyAccelerator::GetHardwareSupport),
            kLiteRtStatusErrorInvalidArgument);
  LiteRtSetAcceleratorGetHardwareSupport(&accelerator,
                                         DummyAccelerator::GetHardwareSupport);
  EXPECT_EQ(accelerator.GetHardwareSupport,
            DummyAccelerator::GetHardwareSupport);
}

TEST(LiteRtAcceleratorRegistrationTest, SetAcceleratorApplyToModel) {
  LiteRtAcceleratorT accelerator;
  EXPECT_EQ(LiteRtSetApplyToModel(nullptr, DummyAccelerator::ApplyToModel),
            kLiteRtStatusErrorInvalidArgument);
  LiteRtSetApplyToModel(&accelerator, DummyAccelerator::ApplyToModel);
  EXPECT_EQ(accelerator.ApplyToModel, DummyAccelerator::ApplyToModel);
}

TEST(LiteRtAcceleratorRegistrationTest, CreateDestroyAcceleratorDoesntLeak) {
  LiteRtAccelerator accelerator;
  ASSERT_EQ(LiteRtCreateAccelerator(&accelerator), kLiteRtStatusOk);
  ASSERT_EQ(LiteRtDestroyAccelerator(accelerator), kLiteRtStatusOk);
}

TEST(LiteRtAcceleratorRegistrationTest, RegisterAcceleratorWorks) {
  LiteRtEnvironmentCreate(0, nullptr);
  auto dummy_accelerator = DummyAccelerator::CpuAccelerator();
  LiteRtAccelerator accelerator;
  LiteRtCreateAccelerator(&accelerator);
  LiteRtSetAcceleratorGetName(accelerator, DummyAccelerator::GetName);
  LiteRtSetAcceleratorGetVersion(accelerator, DummyAccelerator::GetVersion);
  LiteRtSetAcceleratorGetHardwareSupport(accelerator,
                                         DummyAccelerator::GetHardwareSupport);
  LiteRtRegisterAccelerator(accelerator, dummy_accelerator.release(),
                            DummyAccelerator::Destroy);
  LiteRtEnvironmentDestroy();
}

TEST(LiteRtAcceleratorRegistrationTest,
     RegisterAcceleratorFailsForNullAccelerator) {
  LiteRtEnvironmentCreate(0, nullptr);
  // We check that the memory is correctly deallocated if the registration
  // fails.
  auto dummy_accelerator = DummyAccelerator::CpuAccelerator();
  EXPECT_EQ(LiteRtRegisterAccelerator(nullptr, dummy_accelerator.release(),
                                      DummyAccelerator::Destroy),
            kLiteRtStatusErrorInvalidArgument);
  LiteRtEnvironmentDestroy();
}

}  // namespace
