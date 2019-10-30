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
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace delegate {
namespace nnapi {

class NnApiMock {
 public:
  template <int Value>
  void GetDeviceCountReturns() {
    nnapi_->ANeuralNetworks_getDeviceCount = [](uint32_t* numDevices) -> int {
      *numDevices = 2;
      return Value;
    };
  }

  template <int Value>
  void ModelCreateReturns() {
    nnapi_->ANeuralNetworksModel_create = [](ANeuralNetworksModel** model) {
      *model = reinterpret_cast<ANeuralNetworksModel*>(1);
      return Value;
    };
  }

  template <int Value>
  void AddOperandReturns() {
    nnapi_->ANeuralNetworksModel_addOperand =
        [](ANeuralNetworksModel* model,
           const ANeuralNetworksOperandType* type) { return Value; };
  }

  template <int Value>
  void SetOperandValueReturns() {
    nnapi_->ANeuralNetworksModel_setOperandValue =
        [](ANeuralNetworksModel* model, int32_t index, const void* buffer,
           size_t length) { return Value; };
  }

  template <int Value>
  void AddOperationReturns() {
    nnapi_->ANeuralNetworksModel_addOperation =
        [](ANeuralNetworksModel* model, ANeuralNetworksOperationType type,
           uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
           const uint32_t* outputs) { return Value; };
  }

  template <int Value>
  void IdentifyInputAndOutputsReturns() {
    nnapi_->ANeuralNetworksModel_identifyInputsAndOutputs =
        [](ANeuralNetworksModel* model, uint32_t inputCount,
           const uint32_t* inputs, uint32_t outputCount,
           const uint32_t* outputs) { return Value; };
  }

  template <int Value>
  void RelaxComputationFloatReturns() {
    nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16 =
        [](ANeuralNetworksModel* model, bool allow) { return Value; };
  }

  template <int Value>
  void ModelFinishReturns() {
    nnapi_->ANeuralNetworksModel_finish = [](ANeuralNetworksModel* model) {
      return Value;
    };
  }

  template <int Value>
  void MemoryCreateFromFdReturns() {
    nnapi_->ANeuralNetworksMemory_createFromFd =
        [](size_t size, int protect, int fd, size_t offset,
           ANeuralNetworksMemory** memory) {
          *memory = reinterpret_cast<ANeuralNetworksMemory*>(2);
          return Value;
        };
  }

  template <int Value>
  void CompilationCreateReturns() {
    nnapi_->ANeuralNetworksCompilation_create =
        [](ANeuralNetworksModel* model,
           ANeuralNetworksCompilation** compilation) {
          *compilation = reinterpret_cast<ANeuralNetworksCompilation*>(3);
          return Value;
        };
  }

  template <int Value>
  void CompilationFinishReturns() {
    nnapi_->ANeuralNetworksCompilation_finish =
        [](ANeuralNetworksCompilation* compilation) { return Value; };
  }

  template <int Value>
  void ExecutionCreateReturns() {
    nnapi_->ANeuralNetworksExecution_create =
        [](ANeuralNetworksCompilation* compilation,
           ANeuralNetworksExecution** execution) {
          if (compilation == nullptr) return 1;
          *execution = reinterpret_cast<ANeuralNetworksExecution*>(4);
          return Value;
        };
  }
  template <int Value>
  void ExecutionSetInputFromMemoryReturns() {
    nnapi_->ANeuralNetworksExecution_setInputFromMemory =
        [](ANeuralNetworksExecution* execution, int32_t index,
           const ANeuralNetworksOperandType* type,
           const ANeuralNetworksMemory* memory, size_t offset,
           size_t length) { return Value; };
  }
  template <int Value>
  void ExecutionSetOutputFromMemoryReturns() {
    nnapi_->ANeuralNetworksExecution_setOutputFromMemory =
        [](ANeuralNetworksExecution* execution, int32_t index,
           const ANeuralNetworksOperandType* type,
           const ANeuralNetworksMemory* memory, size_t offset,
           size_t length) { return Value; };
  }

  template <int Value>
  void ExecutionComputeReturns() {
    nnapi_->ANeuralNetworksExecution_compute =
        [](ANeuralNetworksExecution* execution) { return Value; };
  }

  explicit NnApiMock(NnApi* nnapi, int android_sdk_version = 29)
      : nnapi_(nnapi), prev_nnapi_(*nnapi) {
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

  ~NnApiMock() {
    // Restores global NNAPI to original value for non mocked tests
    *nnapi_ = prev_nnapi_;
  }

 private:
  NnApi* nnapi_;
  NnApi prev_nnapi_;
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
