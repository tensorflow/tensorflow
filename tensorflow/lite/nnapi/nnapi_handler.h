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
#ifndef TENSORFLOW_LITE_NNAPI_NNAPI_HANDLER_H_
#define TENSORFLOW_LITE_NNAPI_NNAPI_HANDLER_H_

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace nnapi {

// Offers an interface to alter the behaviour of the NNAPI instance.
// As for NNAPI, it is designed to be a singleton.
// It allows to change the behaviour of some of the methods with some stub
// implementation and then to reset the behavior to the original one using
// Reset().
//
class NnApiHandler {
 public:
  // No destructor defined to allow this class to be used as singleton.

  // Factory method, only one instance per process/jni library.
  static NnApiHandler* Instance();

  // Makes the current object a transparent proxy again, resetting any
  // applied changes to its methods.
  void Reset();

  // Using templates in the ...Returns methods because the functions need to be
  // stateless and the template generated code is more readable than using a
  // file-local variable in the method implementation to store the configured
  // result.

  template <int Value>
  void GetDeviceCountReturns() {
    nnapi_->ANeuralNetworks_getDeviceCount = [](uint32_t* numDevices) -> int {
      *numDevices = 1;
      return Value;
    };
  }

  template <int DeviceCount>
  void GetDeviceCountReturnsCount() {
    nnapi_->ANeuralNetworks_getDeviceCount = [](uint32_t* numDevices) -> int {
      *numDevices = DeviceCount;
      return ANEURALNETWORKS_NO_ERROR;
    };
  }

  void StubGetDeviceCountWith(int(stub)(uint32_t*)) {
    nnapi_->ANeuralNetworks_getDeviceCount = stub;
  }

  template <int Value>
  void GetDeviceReturns() {
    nnapi_->ANeuralNetworks_getDevice =
        [](uint32_t devIndex, ANeuralNetworksDevice** device) -> int {
      *device =
          reinterpret_cast<ANeuralNetworksDevice*>(NnApiHandler::kNnapiDevice);
      return Value;
    };
  }

  void StubGetDeviceWith(int(stub)(uint32_t, ANeuralNetworksDevice**)) {
    nnapi_->ANeuralNetworks_getDevice = stub;
  }

  template <int Value>
  void GetDeviceNameReturns() {
    nnapi_->ANeuralNetworksDevice_getName =
        [](const ANeuralNetworksDevice* device, const char** name) -> int {
      *name = NnApiHandler::nnapi_device_name_;
      return Value;
    };
  }

  void GetDeviceNameReturnsName(const std::string& name);

  void StubGetDeviceNameWith(int(stub)(const ANeuralNetworksDevice*,
                                       const char**)) {
    nnapi_->ANeuralNetworksDevice_getName = stub;
  }

  // Configure all the functions related to device browsing to support
  // a device with the given name and the cpu fallback nnapi-reference.
  // The extra device will return support the specified feature level
  void SetNnapiSupportedDevice(const std::string& name, int feature_level = 29);

  template <int Value>
  void ModelCreateReturns() {
    nnapi_->ANeuralNetworksModel_create = [](ANeuralNetworksModel** model) {
      *model = reinterpret_cast<ANeuralNetworksModel*>(1);
      return Value;
    };
  }

  void StubModelCreateWith(int(stub)(ANeuralNetworksModel** model)) {
    nnapi_->ANeuralNetworksModel_create = stub;
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

  void StubAddOperationWith(
      int(stub)(ANeuralNetworksModel* model, ANeuralNetworksOperationType type,
                uint32_t inputCount, const uint32_t* inputs,
                uint32_t outputCount, const uint32_t* outputs)) {
    nnapi_->ANeuralNetworksModel_addOperation = stub;
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
  void CompilationCreateForDevicesReturns() {
    nnapi_->ANeuralNetworksCompilation_createForDevices =
        [](ANeuralNetworksModel* model,
           const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
           ANeuralNetworksCompilation** compilation) {
          *compilation = reinterpret_cast<ANeuralNetworksCompilation*>(3);
          return Value;
        };
  }

  void StubCompilationCreateForDevicesWith(int(stub)(
      ANeuralNetworksModel* model, const ANeuralNetworksDevice* const* devices,
      uint32_t numDevices, ANeuralNetworksCompilation** compilation)) {
    nnapi_->ANeuralNetworksCompilation_createForDevices = stub;
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

  template <int Value>
  void GetSupportedOperationsForDevicesReturns() {
    nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices =
        [](const ANeuralNetworksModel* model,
           const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
           bool* supportedOps) { return Value; };
  }

  void StubGetSupportedOperationsForDevicesWith(
      int(stub)(const ANeuralNetworksModel* model,
                const ANeuralNetworksDevice* const* devices,
                uint32_t numDevices, bool* supportedOps)) {
    nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices = stub;
  }

  template <int Value>
  void ExecutionStartComputeReturns() {
    nnapi_->ANeuralNetworksExecution_startCompute =
        [](ANeuralNetworksExecution* execution, ANeuralNetworksEvent** event) {
          *event = reinterpret_cast<ANeuralNetworksEvent*>(1);
          return Value;
        };
  }

  template <int Value>
  void EventWaitReturns() {
    nnapi_->ANeuralNetworksEvent_wait = [](ANeuralNetworksEvent* event) {
      return Value;
    };
  }

  /*
   * Sets the SDK Version in the nnapi structure.
   * If set_unsupported_ops_to_null is set to true, all the functions not
   * available at the given sdk level will be set to null too.
   */
  void SetAndroidSdkVersion(int version,
                            bool set_unsupported_ops_to_null = false);

  const NnApi* GetNnApi() { return nnapi_; }

 protected:
  explicit NnApiHandler(NnApi* nnapi) : nnapi_(nnapi) { DCHECK(nnapi); }

  NnApi* nnapi_;

  static const char kNnapiReferenceDeviceName[];
  static const int kNnapiReferenceDevice;
  static const int kNnapiDevice;

  static void SetDeviceName(const std::string& name);

 private:
  static char* nnapi_device_name_;
  static int nnapi_device_feature_level_;
};

// Returns a pointer to an unaltered instance of NNAPI. Is intended
// to be used by stub methods when wanting to pass-through to original
// implementation for example:
//
// NnApiTestUtility()->StubGetDeviceWith(
//  [](uint32_t devIndex, ANeuralNetworksDevice** device) -> int {
//        static int count = 0;
//        if (count++ < 1) {
//          NnApiPassthroughInstance()->ANeuralNetworks_getDevice(
//                devIndex, device);
//        } else {
//            return ANEURALNETWORKS_BAD_DATA;
//        }
//   });
const NnApi* NnApiPassthroughInstance();

// Returns an instance of NnApiProxy that can be used to alter
// the behaviour of the TFLite wide instance of NnApi.
NnApiHandler* NnApiProxyInstance();

}  // namespace nnapi
}  // namespace tflite

#endif  // TENSORFLOW_LITE_NNAPI_NNAPI_HANDLER_H_
