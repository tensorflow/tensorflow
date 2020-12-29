/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/nnapi/nnapi_handler.h"

#include <cstdint>
#include <cstdio>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace nnapi {

using testing::Eq;
using testing::Ne;
using testing::NotNull;

void ExpectEquals(const NnApi& left, const NnApi& right);

class NnApiHandlerTest : public ::testing::Test {
 protected:
  ~NnApiHandlerTest() override { NnApiHandler::Instance()->Reset(); }
};

TEST_F(NnApiHandlerTest, ShouldAlterNnApiInstanceBehaviour) {
  const NnApi* nnapi = NnApiImplementation();

  const auto device_count_stub = [](uint32_t* device_count) -> int {
    *device_count = 999;
    return ANEURALNETWORKS_NO_ERROR;
  };

  NnApiHandler::Instance()->StubGetDeviceCountWith(device_count_stub);

  ASSERT_THAT(nnapi->ANeuralNetworks_getDeviceCount, NotNull());

  uint32_t device_count = 0;
  nnapi->ANeuralNetworks_getDeviceCount(&device_count);
  EXPECT_THAT(device_count, Eq(999));
}

TEST_F(NnApiHandlerTest, ShouldRestoreNnApiToItsOriginalValueWithReset) {
  NnApi nnapi_orig_copy = *NnApiImplementation();

  auto device_count_override = [](uint32_t* device_count) -> int {
    *device_count = 777;
    return ANEURALNETWORKS_NO_ERROR;
  };

  NnApiHandler::Instance()->StubGetDeviceCountWith(device_count_override);

  EXPECT_THAT(nnapi_orig_copy.ANeuralNetworks_getDeviceCount,
              Ne(NnApiImplementation()->ANeuralNetworks_getDeviceCount));

  NnApiHandler::Instance()->Reset();

  ExpectEquals(nnapi_orig_copy, *NnApiImplementation());
}

int (*device_count_ptr)(uint32_t*);
TEST_F(NnApiHandlerTest, ShouldSupportPassthroughCalls) {
  const NnApi* nnapi = NnApiImplementation();
  device_count_ptr = nnapi->ANeuralNetworks_getDeviceCount;

  NnApiHandler::Instance()->StubGetDeviceCountWith(
      [](uint32_t* device_count) -> int {
        return NnApiPassthroughInstance()->ANeuralNetworks_getDeviceCount ==
               device_count_ptr;
      });

  uint32_t device_count = 0;
  EXPECT_THAT(nnapi->ANeuralNetworks_getDeviceCount(&device_count), Eq(1));
}

TEST_F(NnApiHandlerTest, ShouldSetNnApiMembersToNullAsPerSdkVersion_NNAPI11) {
  auto* handler = NnApiHandler::Instance();

  // Setting non null values for nnapi functions
  handler->SetNnapiSupportedDevice("devvice", 1000);
  handler->GetSupportedOperationsForDevicesReturns<1>();
  handler->CompilationCreateForDevicesReturns<1>();
  handler->ExecutionComputeReturns<1>();
  handler->MemoryCreateFromFdReturns<1>();

  handler->SetAndroidSdkVersion(28, /*set_unsupported_ops_to_null=*/true);

  const NnApi* nnapi = NnApiImplementation();

  using ::testing::IsNull;

  EXPECT_THAT(nnapi->ANeuralNetworks_getDeviceCount, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworks_getDevice, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getName, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getVersion, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getFeatureLevel, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getType, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_getSupportedOperationsForDevices,
              IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksCompilation_createForDevices, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksCompilation_setCaching, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_compute, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_getOutputOperandRank, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_getOutputOperandDimensions,
              IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksBurst_create, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksBurst_free, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_burstCompute, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksMemory_createFromAHardwareBuffer, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_setMeasureTiming, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_getDuration, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getExtensionSupport, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_getExtensionOperandType, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_getExtensionOperationType, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_setOperandExtensionData, IsNull());
}

TEST_F(NnApiHandlerTest, ShouldSetNnApiMembersToNullAsPerSdkVersion_NNAPI10) {
  auto* handler = NnApiHandler::Instance();

  // Setting non null values for nnapi functions
  handler->SetNnapiSupportedDevice("devvice", 1000);
  handler->GetSupportedOperationsForDevicesReturns<1>();
  handler->CompilationCreateForDevicesReturns<1>();
  handler->ExecutionComputeReturns<1>();
  handler->MemoryCreateFromFdReturns<1>();

  handler->SetAndroidSdkVersion(27, /*set_unsupported_ops_to_null=*/true);

  const NnApi* nnapi = NnApiImplementation();

  using ::testing::IsNull;

  EXPECT_THAT(nnapi->ANeuralNetworks_getDeviceCount, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworks_getDevice, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getName, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getVersion, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getFeatureLevel, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getType, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_getSupportedOperationsForDevices,
              IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksCompilation_createForDevices, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksCompilation_setCaching, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_compute, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_getOutputOperandRank, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_getOutputOperandDimensions,
              IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksBurst_create, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksBurst_free, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_burstCompute, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksMemory_createFromAHardwareBuffer, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_setMeasureTiming, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_getDuration, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getExtensionSupport, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_getExtensionOperandType, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_getExtensionOperationType, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_setOperandExtensionData, IsNull());

  EXPECT_THAT(nnapi->ANeuralNetworksModel_relaxComputationFloat32toFloat16,
              IsNull());
}

void ExpectEquals(const NnApi& left, const NnApi& right) {
#define EXPECT_NNAPI_MEMBER_EQ(name) EXPECT_EQ(left.name, right.name)

  EXPECT_NNAPI_MEMBER_EQ(nnapi_exists);
  EXPECT_NNAPI_MEMBER_EQ(android_sdk_version);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksMemory_createFromFd);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksMemory_free);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_create);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_free);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_finish);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_addOperand);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_setOperandValue);
  EXPECT_NNAPI_MEMBER_EQ(
      ANeuralNetworksModel_setOperandSymmPerChannelQuantParams);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_setOperandValueFromMemory);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_addOperation);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_identifyInputsAndOutputs);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_relaxComputationFloat32toFloat16);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksCompilation_create);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksCompilation_free);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksCompilation_setPreference);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksCompilation_finish);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_create);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_free);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_setInput);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_setInputFromMemory);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_setOutput);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_setOutputFromMemory);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_startCompute);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksEvent_wait);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksEvent_free);
  EXPECT_NNAPI_MEMBER_EQ(ASharedMemory_create);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworks_getDeviceCount);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworks_getDevice);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksDevice_getName);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksDevice_getVersion);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksDevice_getFeatureLevel);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksDevice_getType);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_getSupportedOperationsForDevices);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksCompilation_createForDevices);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksCompilation_setCaching);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_compute);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_getOutputOperandRank);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_getOutputOperandDimensions);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksBurst_create);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksBurst_free);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_burstCompute);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksMemory_createFromAHardwareBuffer);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_setMeasureTiming);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_getDuration);

#undef EXPECT_NNAPI_MEMBER_EQ
}

}  // namespace nnapi
}  // namespace tflite
