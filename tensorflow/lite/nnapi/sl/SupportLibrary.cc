/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// Changed when importing from AOSP
#include "tensorflow/lite/nnapi/sl/include/SupportLibrary.h"

// Changed when importing from AOSP
#include <dlfcn.h>

#include <cinttypes>
#include <cstring>
#include <memory>
#include <string>

#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"

namespace tflite {
namespace nnapi {

using tflite::TFLITE_LOG_ERROR;

NnApiSupportLibrary::NnApiSupportLibrary(const NnApiSLDriverImplFL5& impl,
                                         void* libHandle)
    : NnApiSLDriverImplFL5(impl), libHandle(libHandle) {
  base.implFeatureLevel = ANEURALNETWORKS_FEATURE_LEVEL_5;
}

NnApiSupportLibrary::~NnApiSupportLibrary() {
  if (libHandle != nullptr) {
    dlclose(libHandle);
    libHandle = nullptr;
  }
}

std::unique_ptr<const NnApiSupportLibrary> loadNnApiSupportLibrary(
    const std::string& libName) {
  void* libHandle = dlopen(libName.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (libHandle == nullptr) {
    TFLITE_LOG(TFLITE_LOG_ERROR, "nnapi error: unable to open library %s: %s",
               libName.c_str(), dlerror());
    return nullptr;
  }

  auto result = loadNnApiSupportLibrary(libHandle);
  if (!result) {
    dlclose(libHandle);
  }
  return result;
}

std::unique_ptr<const NnApiSupportLibrary> loadNnApiSupportLibrary(
    void* libHandle) {
  NnApiSLDriverImpl* (*getSlDriverImpl)();
  getSlDriverImpl = reinterpret_cast<decltype(getSlDriverImpl)>(
      dlsym(libHandle, "ANeuralNetworks_getSLDriverImpl"));
  if (getSlDriverImpl == nullptr) {
    TFLITE_LOG(TFLITE_LOG_ERROR,
               "Failed to find ANeuralNetworks_getSLDriverImpl symbol");
    return nullptr;
  }

  NnApiSLDriverImpl* impl = getSlDriverImpl();
  if (impl == nullptr) {
    TFLITE_LOG(TFLITE_LOG_ERROR,
               "ANeuralNetworks_getSLDriverImpl returned nullptr");
    return nullptr;
  }

  if (impl->implFeatureLevel < ANEURALNETWORKS_FEATURE_LEVEL_5) {
    int64_t impl_feature_level = impl->implFeatureLevel;
    TFLITE_LOG(TFLITE_LOG_ERROR,
               "Unsupported NnApiSLDriverImpl->implFeatureLevel: %" PRId64,
               impl_feature_level);
    return nullptr;
  }

  return std::make_unique<NnApiSupportLibrary>(
      *reinterpret_cast<NnApiSLDriverImplFL5*>(impl), libHandle);
}

}  // namespace nnapi
}  // namespace tflite
