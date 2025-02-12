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
#include <cstring>
#include <string>

#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace nnapi {

// static
const char NnApiHandler::kNnapiReferenceDeviceName[] = "nnapi-reference";
// static
const int NnApiHandler::kNnapiReferenceDevice = 1;
// static
const int NnApiHandler::kNnapiDevice = 2;

char* NnApiHandler::nnapi_device_name_ = nullptr;
int NnApiHandler::nnapi_device_feature_level_;

const NnApi* NnApiPassthroughInstance() {
  static const NnApi orig_nnapi_copy = *NnApiImplementation();
  return &orig_nnapi_copy;
}

// static
NnApiHandler* NnApiHandler::Instance() {
  // Ensuring that the original copy of nnapi is saved before we return
  // access to NnApiHandler
  NnApiPassthroughInstance();
  static NnApiHandler handler{const_cast<NnApi*>(NnApiImplementation())};
  return &handler;
}

void NnApiHandler::Reset() {
  // Restores global NNAPI to original value
  *nnapi_ = *NnApiPassthroughInstance();
}

void NnApiHandler::SetAndroidSdkVersion(int version,
                                        bool set_unsupported_ops_to_null) {
  nnapi_->android_sdk_version = version;
  nnapi_->nnapi_runtime_feature_level = version;

  if (!set_unsupported_ops_to_null) {
    return;
  }

  if (version < 29) {
    nnapi_->ANeuralNetworks_getDeviceCount = nullptr;
    nnapi_->ANeuralNetworks_getDevice = nullptr;
    nnapi_->ANeuralNetworksDevice_getName = nullptr;
    nnapi_->ANeuralNetworksDevice_getVersion = nullptr;
    nnapi_->ANeuralNetworksDevice_getFeatureLevel = nullptr;
    nnapi_->ANeuralNetworksDevice_getType = nullptr;
    nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices = nullptr;
    nnapi_->ANeuralNetworksCompilation_createForDevices = nullptr;
    nnapi_->ANeuralNetworksCompilation_setCaching = nullptr;
    nnapi_->ANeuralNetworksExecution_compute = nullptr;
    nnapi_->ANeuralNetworksExecution_getOutputOperandRank = nullptr;
    nnapi_->ANeuralNetworksExecution_getOutputOperandDimensions = nullptr;
    nnapi_->ANeuralNetworksBurst_create = nullptr;
    nnapi_->ANeuralNetworksBurst_free = nullptr;
    nnapi_->ANeuralNetworksExecution_burstCompute = nullptr;
    nnapi_->ANeuralNetworksMemory_createFromAHardwareBuffer = nullptr;
    nnapi_->ANeuralNetworksExecution_setMeasureTiming = nullptr;
    nnapi_->ANeuralNetworksExecution_getDuration = nullptr;
    nnapi_->ANeuralNetworksDevice_getExtensionSupport = nullptr;
    nnapi_->ANeuralNetworksModel_getExtensionOperandType = nullptr;
    nnapi_->ANeuralNetworksModel_getExtensionOperationType = nullptr;
    nnapi_->ANeuralNetworksModel_setOperandExtensionData = nullptr;
  }
  if (version < 28) {
    nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16 = nullptr;
  }
}

void NnApiHandler::SetDeviceName(const std::string& name) {
  delete[] nnapi_device_name_;
  nnapi_device_name_ = new char[name.size() + 1];
  std::strcpy(nnapi_device_name_, name.c_str());  // NOLINT
}

void NnApiHandler::GetDeviceNameReturnsName(const std::string& name) {
  NnApiHandler::SetDeviceName(name);
  GetDeviceNameReturns<0>();
}

void NnApiHandler::SetNnapiSupportedDevice(const std::string& name,
                                           int feature_level) {
  NnApiHandler::SetDeviceName(name);
  nnapi_device_feature_level_ = feature_level;

  GetDeviceCountReturnsCount<2>();
  nnapi_->ANeuralNetworks_getDevice =
      [](uint32_t devIndex, ANeuralNetworksDevice** device) -> int {
    if (devIndex > 1) {
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (devIndex == 1) {
      *device =
          reinterpret_cast<ANeuralNetworksDevice*>(NnApiHandler::kNnapiDevice);
    } else {
      *device = reinterpret_cast<ANeuralNetworksDevice*>(
          NnApiHandler::kNnapiReferenceDevice);
    }
    return ANEURALNETWORKS_NO_ERROR;
  };
  nnapi_->ANeuralNetworksDevice_getName =
      [](const ANeuralNetworksDevice* device, const char** name) -> int {
    if (device ==
        reinterpret_cast<ANeuralNetworksDevice*>(NnApiHandler::kNnapiDevice)) {
      *name = NnApiHandler::nnapi_device_name_;
      return ANEURALNETWORKS_NO_ERROR;
    }
    if (device == reinterpret_cast<ANeuralNetworksDevice*>(
                      NnApiHandler::kNnapiReferenceDevice)) {
      *name = NnApiHandler::kNnapiReferenceDeviceName;
      return ANEURALNETWORKS_NO_ERROR;
    }

    return ANEURALNETWORKS_BAD_DATA;
  };
  nnapi_->ANeuralNetworksDevice_getFeatureLevel =
      [](const ANeuralNetworksDevice* device, int64_t* featureLevel) -> int {
    if (device ==
        reinterpret_cast<ANeuralNetworksDevice*>(NnApiHandler::kNnapiDevice)) {
      *featureLevel = NnApiHandler::nnapi_device_feature_level_;
      return ANEURALNETWORKS_NO_ERROR;
    }
    if (device == reinterpret_cast<ANeuralNetworksDevice*>(
                      NnApiHandler::kNnapiReferenceDevice)) {
      *featureLevel = 1000;
      return ANEURALNETWORKS_NO_ERROR;
    }

    return ANEURALNETWORKS_BAD_DATA;
  };
}

}  // namespace nnapi
}  // namespace tflite
