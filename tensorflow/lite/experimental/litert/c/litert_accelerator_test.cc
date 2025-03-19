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

#include "tensorflow/lite/experimental/litert/c/litert_accelerator.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_accelerator_registration.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/runtime/accelerator.h"

#define LITERT_ENSURE_OK(expr)       \
  do {                               \
    LiteRtStatus status = (expr);    \
    if (status != kLiteRtStatusOk) { \
      return status;                 \
    }                                \
  } while (0)

namespace {
using testing::Eq;
using testing::Ne;
using testing::NotNull;
using testing::StrEq;

class DummyAccelerator {
 public:
  // `hardware_support` is a bitfield of `LiteRtHwAccelerators` values.
  static LiteRtStatus RegisterAccelerator(int hardware_support,
                                          LiteRtEnvironment env) {
    auto dummy_accelerator = std::make_unique<DummyAccelerator>();
    dummy_accelerator->hardware_support_ = hardware_support;
    LiteRtAccelerator accelerator;
    LiteRtCreateAccelerator(&accelerator);
    LITERT_ENSURE_OK(
        LiteRtSetAcceleratorGetName(accelerator, DummyAccelerator::GetName));
    LITERT_ENSURE_OK(LiteRtSetAcceleratorGetVersion(
        accelerator, DummyAccelerator::GetVersion));
    LITERT_ENSURE_OK(LiteRtSetAcceleratorGetHardwareSupport(
        accelerator, DummyAccelerator::GetHardwareSupport));
    LITERT_ENSURE_OK(LiteRtRegisterAccelerator(env, accelerator,
                                               dummy_accelerator.release(),
                                               DummyAccelerator::Destroy));
    return kLiteRtStatusOk;
  }

  static void Destroy(void* dummy_accelerator) {
    DummyAccelerator* instance =
        reinterpret_cast<DummyAccelerator*>(dummy_accelerator);
    delete instance;
  }

  static LiteRtStatus GetName(LiteRtAccelerator accelerator,
                              const char** name) {
    if (!accelerator || !accelerator->data || !name) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    DummyAccelerator& self =
        *reinterpret_cast<DummyAccelerator*>(accelerator->data);
    if (self.name_.empty()) {
      self.name_.append("Dummy");
      if (self.hardware_support_ & kLiteRtHwAcceleratorCpu) {
        self.name_.append("Cpu");
      }
      if (self.hardware_support_ & kLiteRtHwAcceleratorGpu) {
        self.name_.append("Gpu");
      }
      self.name_.append("Accelerator");
    }
    *name = self.name_.c_str();
    return kLiteRtStatusOk;
  }

  static LiteRtStatus GetVersion(LiteRtAccelerator accelerator,
                                 LiteRtApiVersion* version) {
    if (!version) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    version->major = 1;
    version->minor = 2;
    version->patch = 3;
    return kLiteRtStatusOk;
  }

  static LiteRtStatus GetHardwareSupport(
      LiteRtAccelerator accelerator,
      LiteRtHwAcceleratorSet* supported_hardware) {
    if (!accelerator || !accelerator->data || !supported_hardware) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    const DummyAccelerator& self =
        *reinterpret_cast<DummyAccelerator*>(accelerator->data);
    *supported_hardware = self.hardware_support_;
    return kLiteRtStatusOk;
  }

  int hardware_support_;
  std::string name_;
};

class LiteRtAcceleratorTest : public testing::Test {
 public:
  LiteRtEnvironment env_;
  void SetUp() override {
    LiteRtEnvironmentCreate(/*num_options=*/0, nullptr, &env_);
    DummyAccelerator::RegisterAccelerator(kLiteRtHwAcceleratorCpu, env_);
  }

  void TearDown() override { LiteRtDestroyEnvironment(env_); }
};

TEST_F(LiteRtAcceleratorTest, IteratingOverAcceleratorsWorks) {
  // CPU accelerator is registered in the SetUp function.
  DummyAccelerator::RegisterAccelerator(kLiteRtHwAcceleratorGpu, env_);

  LiteRtParamIndex num_accelerators = 0;
  ASSERT_THAT(LiteRtGetNumAccelerators(env_, &num_accelerators),
              kLiteRtStatusOk);
  ASSERT_THAT(num_accelerators, 2);

  EXPECT_THAT(LiteRtGetAccelerator(env_, 0, nullptr),
              kLiteRtStatusErrorInvalidArgument);
  LiteRtAccelerator accelerator0;
  ASSERT_THAT(LiteRtGetAccelerator(env_, 0, &accelerator0), kLiteRtStatusOk);
  EXPECT_THAT(accelerator0, NotNull());

  EXPECT_THAT(LiteRtGetAccelerator(env_, 1, nullptr),
              kLiteRtStatusErrorInvalidArgument);
  LiteRtAccelerator accelerator1;
  ASSERT_THAT(LiteRtGetAccelerator(env_, 1, &accelerator1), kLiteRtStatusOk);
  EXPECT_THAT(accelerator1, NotNull());

  EXPECT_THAT(accelerator0, Ne(accelerator1));

  LiteRtAccelerator accelerator2;
  EXPECT_THAT(LiteRtGetAccelerator(env_, 2, &accelerator2),
              kLiteRtStatusErrorNotFound);
}

TEST_F(LiteRtAcceleratorTest, GetAcceleratorNameWorks) {
  LiteRtParamIndex num_accelerators = 0;
  ASSERT_THAT(LiteRtGetNumAccelerators(env_, &num_accelerators),
              kLiteRtStatusOk);
  ASSERT_THAT(num_accelerators, 1);

  LiteRtAccelerator accelerator;
  ASSERT_THAT(LiteRtGetAccelerator(env_, 0, &accelerator), kLiteRtStatusOk);
  const char* name = nullptr;
  ASSERT_THAT(LiteRtGetAcceleratorName(accelerator, &name), kLiteRtStatusOk);
  EXPECT_THAT(name, StrEq("DummyCpuAccelerator"));

  EXPECT_THAT(LiteRtGetAcceleratorName(nullptr, &name),
              kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(LiteRtGetAcceleratorName(accelerator, nullptr),
              kLiteRtStatusErrorInvalidArgument);
  // Make the accelerator invalid.
  accelerator->GetName = nullptr;
  EXPECT_THAT(LiteRtGetAcceleratorName(accelerator, &name),
              kLiteRtStatusErrorInvalidArgument);
}

TEST_F(LiteRtAcceleratorTest, GetAcceleratorIdWorks) {
  LiteRtParamIndex num_accelerators = 0;
  ASSERT_THAT(LiteRtGetNumAccelerators(env_, &num_accelerators),
              kLiteRtStatusOk);
  ASSERT_THAT(num_accelerators, 1);

  LiteRtAccelerator accelerator;
  ASSERT_THAT(LiteRtGetAccelerator(env_, 0, &accelerator), kLiteRtStatusOk);
  LiteRtAcceleratorId accelerator_id;
  ASSERT_THAT(LiteRtGetAcceleratorId(accelerator, &accelerator_id),
              kLiteRtStatusOk);
  EXPECT_THAT(accelerator_id, Eq(0));

  EXPECT_THAT(LiteRtGetAcceleratorId(nullptr, &accelerator_id),
              kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(LiteRtGetAcceleratorId(accelerator, nullptr),
              kLiteRtStatusErrorInvalidArgument);
  // Make the accelerator invalid.
  accelerator->env = nullptr;
  EXPECT_THAT(LiteRtGetAcceleratorId(accelerator, &accelerator_id),
              kLiteRtStatusErrorInvalidArgument);
}

TEST_F(LiteRtAcceleratorTest, GetAcceleratorVersionWorks) {
  LiteRtParamIndex num_accelerators = 0;
  ASSERT_THAT(LiteRtGetNumAccelerators(env_, &num_accelerators),
              kLiteRtStatusOk);
  ASSERT_THAT(num_accelerators, 1);

  LiteRtAccelerator accelerator;
  ASSERT_THAT(LiteRtGetAccelerator(env_, 0, &accelerator), kLiteRtStatusOk);
  LiteRtApiVersion version;
  ASSERT_THAT(LiteRtGetAcceleratorVersion(accelerator, &version),
              kLiteRtStatusOk);
  EXPECT_THAT(version.major, Eq(1));
  EXPECT_THAT(version.minor, Eq(2));
  EXPECT_THAT(version.patch, Eq(3));

  EXPECT_THAT(LiteRtGetAcceleratorVersion(nullptr, &version),
              kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(LiteRtGetAcceleratorVersion(accelerator, nullptr),
              kLiteRtStatusErrorInvalidArgument);
  // Make the accelerator invalid.
  accelerator->GetVersion = nullptr;
  EXPECT_THAT(LiteRtGetAcceleratorVersion(accelerator, &version),
              kLiteRtStatusErrorInvalidArgument);
}

TEST_F(LiteRtAcceleratorTest, GetAcceleratorHardwareSupportWorks) {
  LiteRtParamIndex num_accelerators = 0;
  ASSERT_THAT(LiteRtGetNumAccelerators(env_, &num_accelerators),
              kLiteRtStatusOk);
  ASSERT_THAT(num_accelerators, 1);

  LiteRtAccelerator accelerator;
  ASSERT_THAT(LiteRtGetAccelerator(env_, 0, &accelerator), kLiteRtStatusOk);
  int hardware_support;
  ASSERT_THAT(
      LiteRtGetAcceleratorHardwareSupport(accelerator, &hardware_support),
      kLiteRtStatusOk);
  EXPECT_THAT(hardware_support & kLiteRtHwAcceleratorCpu, true);
  EXPECT_THAT(hardware_support & kLiteRtHwAcceleratorGpu, false);
  EXPECT_THAT(hardware_support & kLiteRtHwAcceleratorNpu, false);

  EXPECT_THAT(LiteRtGetAcceleratorHardwareSupport(nullptr, &hardware_support),
              kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(LiteRtGetAcceleratorHardwareSupport(accelerator, nullptr),
              kLiteRtStatusErrorInvalidArgument);
  // Make the accelerator invalid.
  accelerator->GetHardwareSupport = nullptr;
  EXPECT_THAT(
      LiteRtGetAcceleratorHardwareSupport(accelerator, &hardware_support),
      kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
