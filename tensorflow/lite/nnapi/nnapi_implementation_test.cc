/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include <gtest/gtest.h>

namespace {

TEST(NnapiLibTest, NnApiImplementation) {
  const NnApi* nnapi = NnApiImplementation();
  EXPECT_NE(nnapi, nullptr);
#ifdef __ANDROID__
  EXPECT_GT(nnapi->android_sdk_version, 0);
  if (nnapi.android_sdk_version < 27) {
    EXPECT_FALSE(nnapi->nnapi_exists);
    EXPECT_EQ(nnapi->ANeuralNetworksMemory_createFromFd, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksMemory_free, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_create, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_free, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_finish, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_addOperand, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_setOperandValue, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_setOperandValueFromMemory, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_addOperation, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_identifyInputsAndOutputs, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_relaxComputationFloat32toFloat16,
              nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksCompilation_create, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksCompilation_free, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksCompilation_setPreference, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksCompilation_finish, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksExecution_create, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksExecution_free, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksExecution_setInput, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksExecution_setInputFromMemory, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksExecution_setOutput, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksExecution_setOutputFromMemory, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksExecution_startCompute, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksEvent_wait, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksEvent_free, nullptr);
    EXPECT_EQ(nnapi->ASharedMemory_create, nullptr);
  } else {
    EXPECT_TRUE(nnapi->nnapi_exists);
    EXPECT_NE(nnapi->ANeuralNetworksMemory_createFromFd, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksMemory_free, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_create, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_free, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_finish, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_addOperand, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_setOperandValue, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_setOperandValueFromMemory, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_addOperation, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_identifyInputsAndOutputs, nullptr);
    if (nnapi->android_sdk_version >= 28) {
      // relaxComputationFloat32toFloat16 only available with Android 9.0 (P).
      EXPECT_NE(nnapi->ANeuralNetworksModel_relaxComputationFloat32toFloat16,
                nullptr);
    } else {
      EXPECT_EQ(nnapi->ANeuralNetworksModel_relaxComputationFloat32toFloat16,
                nullptr);
    }
    EXPECT_NE(nnapi->ANeuralNetworksCompilation_create, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksCompilation_free, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksCompilation_setPreference, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksCompilation_finish, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksExecution_create, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksExecution_free, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksExecution_setInput, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksExecution_setInputFromMemory, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksExecution_setOutput, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksExecution_setOutputFromMemory, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksExecution_startCompute, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksEvent_wait, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksEvent_free, nullptr);
    EXPECT_NE(nnapi->ASharedMemory_create, nullptr);
    // TODO(b/123423795): Test Q-specific APIs after release.
  }
#else
  EXPECT_FALSE(nnapi->nnapi_exists);
  EXPECT_EQ(nnapi->android_sdk_version, 0);
  EXPECT_EQ(nnapi->ANeuralNetworksMemory_createFromFd, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksMemory_free, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_create, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_free, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_finish, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_addOperand, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_setOperandValue, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_setOperandSymmPerChannelQuantParams,
            nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_setOperandValueFromMemory, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_addOperation, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_identifyInputsAndOutputs, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_relaxComputationFloat32toFloat16,
            nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksCompilation_create, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksCompilation_free, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksCompilation_setPreference, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksCompilation_finish, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_create, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_free, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_setInput, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_setInputFromMemory, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_setOutput, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_setOutputFromMemory, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_startCompute, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksEvent_wait, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksEvent_free, nullptr);
  EXPECT_EQ(nnapi->ASharedMemory_create, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworks_getDeviceCount, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworks_getDevice, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksDevice_getName, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksDevice_getVersion, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksDevice_getFeatureLevel, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_getSupportedOperationsForDevices,
            nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksCompilation_createForDevices, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksCompilation_setCaching, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_compute, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_getOutputOperandRank, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_getOutputOperandDimensions,
            nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksBurst_create, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksBurst_free, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_burstCompute, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksMemory_createFromAHardwareBuffer, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_setMeasureTiming, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_getDuration, nullptr);
#endif
}

}  // namespace
