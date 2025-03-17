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

#include "tensorflow/lite/nnapi/nnapi_util.h"

#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace nnapi {

namespace {
std::string SimpleJoin(const std::vector<const char*>& elements,
                       const char* separator) {
  // Note that we avoid use of sstream to avoid binary size bloat.
  std::string joined_elements;
  for (auto it = elements.begin(); it != elements.end(); ++it) {
    if (separator && it != elements.begin()) {
      joined_elements += separator;
    }
    if (*it) {
      joined_elements += *it;
    }
  }
  return joined_elements;
}

}  // namespace

std::vector<const char*> GetDeviceNamesList() {
  return GetDeviceNamesList(NnApiImplementation());
}

std::vector<const char*> GetDeviceNamesList(const NnApi* nnapi) {
  std::vector<const char*> device_names;

  // Only build the list if NnApiImplementation has the methods we need,
  // leaving it empty otherwise.
  if (nnapi->ANeuralNetworks_getDeviceCount != nullptr) {
    uint32_t num_devices = 0;

    nnapi->ANeuralNetworks_getDeviceCount(&num_devices);

    for (uint32_t i = 0; i < num_devices; i++) {
      ANeuralNetworksDevice* device = nullptr;
      const char* buffer = nullptr;
      nnapi->ANeuralNetworks_getDevice(i, &device);
      nnapi->ANeuralNetworksDevice_getName(device, &buffer);
      device_names.push_back(buffer);
    }
  }

  return device_names;
}

std::string GetStringDeviceNamesList() {
  return GetStringDeviceNamesList(NnApiImplementation());
}

std::string GetStringDeviceNamesList(const NnApi* nnapi) {
  std::vector<const char*> device_names = GetDeviceNamesList(nnapi);
  return SimpleJoin(device_names, ",");
}

}  // namespace nnapi
}  // namespace tflite
