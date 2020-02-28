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
#ifndef TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_MOCK_TEST_H_
#define TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_MOCK_TEST_H_

// Cannot mock the delegate when using the disabled version
// (see the condition in the BUILD file).
#ifndef NNAPI_DELEGATE_DISABLED

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <memory>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/nnapi/nnapi_handler.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace delegate {
namespace nnapi {

class NnApiMock : public ::tflite::nnapi::NnApiHandler {
 public:
  explicit NnApiMock(NnApi* nnapi, int android_sdk_version = 29)
      : ::tflite::nnapi::NnApiHandler(nnapi) {
    nnapi_->nnapi_exists = true;
    nnapi_->android_sdk_version = android_sdk_version;

    nnapi_->ANeuralNetworksCompilation_free =
        [](ANeuralNetworksCompilation* compilation) {};
    nnapi_->ANeuralNetworksMemory_free = [](ANeuralNetworksMemory* memory) {};
    nnapi_->ANeuralNetworksModel_free = [](ANeuralNetworksModel* model) {};
    nnapi_->ANeuralNetworksExecution_free =
        [](ANeuralNetworksExecution* execution) {};
    nnapi_->ASharedMemory_create = [](const char* name, size_t size) -> int {
      return open("/dev/zero", O_RDWR);
    };

    ModelCreateReturns<ANEURALNETWORKS_NO_ERROR>();
    AddOperandReturns<ANEURALNETWORKS_NO_ERROR>();
    SetOperandValueReturns<ANEURALNETWORKS_NO_ERROR>();
    AddOperationReturns<ANEURALNETWORKS_NO_ERROR>();
    IdentifyInputAndOutputsReturns<ANEURALNETWORKS_NO_ERROR>();
    RelaxComputationFloatReturns<ANEURALNETWORKS_NO_ERROR>();
    ModelFinishReturns<ANEURALNETWORKS_NO_ERROR>();
    MemoryCreateFromFdReturns<ANEURALNETWORKS_NO_ERROR>();
    CompilationCreateReturns<ANEURALNETWORKS_NO_ERROR>();
    CompilationCreateForDevicesReturns<ANEURALNETWORKS_NO_ERROR>();
    CompilationFinishReturns<ANEURALNETWORKS_NO_ERROR>();
    ExecutionCreateReturns<ANEURALNETWORKS_NO_ERROR>();
    ExecutionSetInputFromMemoryReturns<ANEURALNETWORKS_NO_ERROR>();
    ExecutionSetOutputFromMemoryReturns<ANEURALNETWORKS_NO_ERROR>();
    ExecutionComputeReturns<ANEURALNETWORKS_NO_ERROR>();
    SetNnapiSupportedDevice("test-device", android_sdk_version);
  }

  ~NnApiMock() { Reset(); }
};

class NnApiDelegateMockTest : public ::testing::Test {
 protected:
  void SetUp() override {
    nnapi_ = *NnApiImplementation();
    nnapi_mock_ = absl::make_unique<NnApiMock>(&nnapi_);
  }

  std::unique_ptr<NnApiMock> nnapi_mock_;

 private:
  NnApi nnapi_;
};

}  // namespace nnapi
}  // namespace delegate
}  // namespace tflite

#endif  // #ifndef NNAPI_DELEGATE_DISABLED

#endif  // TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_MOCK_TEST_H_
