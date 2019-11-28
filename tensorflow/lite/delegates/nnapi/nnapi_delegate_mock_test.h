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

    GetDeviceCountReturns<0>();
    ModelCreateReturns<0>();
    AddOperandReturns<0>();
    SetOperandValueReturns<0>();
    AddOperationReturns<0>();
    IdentifyInputAndOutputsReturns<0>();
    RelaxComputationFloatReturns<0>();
    ModelFinishReturns<0>();
    MemoryCreateFromFdReturns<0>();
    CompilationCreateReturns<0>();
    CompilationFinishReturns<0>();
    ExecutionCreateReturns<0>();
    ExecutionSetInputFromMemoryReturns<0>();
    ExecutionSetOutputFromMemoryReturns<0>();
    ExecutionComputeReturns<0>();
  }

  ~NnApiMock() { Reset(); }
};

class NnApiDelegateMockTest : public ::testing::Test {
  void SetUp() override {
    nnapi_ = const_cast<NnApi*>(NnApiImplementation());
    nnapi_mock_ = absl::make_unique<NnApiMock>(nnapi_);
  }

 protected:
  NnApi* nnapi_;
  std::unique_ptr<NnApiMock> nnapi_mock_;
};

}  // namespace nnapi
}  // namespace delegate
}  // namespace tflite

#endif  // #ifndef NNAPI_DELEGATE_DISABLED

#endif  // TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_MOCK_TEST_H_
