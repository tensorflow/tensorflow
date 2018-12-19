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
#include <gtest/gtest.h>
#include "tensorflow/lite/nnapi/NeuralNetworksShim.h"

namespace {

TEST(NnapiLibTest, NnApiImplementation) {
  const NnApi* nnapi_ = NnApiImplementation();
  EXPECT_NE(nnapi_, nullptr);
#ifdef __ANDROID__
  EXPECT_TRUE(nnapi_->nnapi_exists);
  EXPECT_GT(nnapi_->android_sdk_version, 0);
  EXPECT_NE(nnapi_->ANeuralNetworksMemory_createFromFd, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksMemory_free, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksModel_create, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksModel_free, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksModel_finish, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksModel_addOperand, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksModel_setOperandValue, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksModel_setOperandValueFromMemory, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksModel_addOperation, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksModel_identifyInputsAndOutputs, nullptr);
  if (nnapi_->android_sdk_version >= 28) {
    // relaxComputationFloat32toFloat16 only available with Android 9.0 (P).
    EXPECT_NE(nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16,
              nullptr);
  } else {
    EXPECT_EQ(nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16,
              nullptr);
  }
  EXPECT_NE(nnapi_->ANeuralNetworksCompilation_create, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksCompilation_free, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksCompilation_setPreference, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksCompilation_finish, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksExecution_create, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksExecution_free, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksExecution_setInput, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksExecution_setInputFromMemory, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksExecution_setOutput, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksExecution_setOutputFromMemory, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksExecution_startCompute, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksEvent_wait, nullptr);
  EXPECT_NE(nnapi_->ANeuralNetworksEvent_free, nullptr);
  EXPECT_NE(nnapi_->ASharedMemory_create, nullptr);
#else
  EXPECT_FALSE(nnapi_->nnapi_exists);
  EXPECT_EQ(nnapi_->android_sdk_version, 0);
  EXPECT_EQ(nnapi_->ANeuralNetworksMemory_createFromFd, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksMemory_free, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksModel_create, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksModel_free, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksModel_finish, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksModel_addOperand, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksModel_setOperandValue, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksModel_setOperandValueFromMemory, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksModel_addOperation, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksModel_identifyInputsAndOutputs, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16,
            nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksCompilation_create, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksCompilation_free, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksCompilation_setPreference, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksCompilation_finish, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksExecution_create, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksExecution_free, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksExecution_setInput, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksExecution_setInputFromMemory, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksExecution_setOutput, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksExecution_setOutputFromMemory, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksExecution_startCompute, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksEvent_wait, nullptr);
  EXPECT_EQ(nnapi_->ANeuralNetworksEvent_free, nullptr);
  EXPECT_EQ(nnapi_->ASharedMemory_create, nullptr);
#endif
}

}  // namespace
