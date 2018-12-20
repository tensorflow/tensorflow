/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/nnapi/NeuralNetworksShim.h"

#include <cstdlib>

#ifdef __ANDROID__
#include <sys/mman.h>
#include <sys/system_properties.h>
#include <unistd.h>
#endif

#define NNAPI_LOG(format, ...) fprintf(stderr, format "\n", __VA_ARGS__);

namespace {

#ifdef __ANDROID__
int32_t GetAndroidSdkVersion() {
  const char* sdkProp = "ro.build.version.sdk";
  char sdkVersion[PROP_VALUE_MAX];
  int length = __system_property_get(sdkProp, sdkVersion);
  if (length != 0) {
    int32_t result = 0;
    for (int i = 0; i < length; ++i) {
      int digit = sdkVersion[i] - '0';
      if (digit < 0 || digit > 9) {
        // Non-numeric SDK version, assume it's higher than expected;
        return 0xffff;
      }
      result = result * 10 + digit;
    }
    return result;
  }
  return 0;
}

void* LoadFunction(void* handle, const char* name) {
  if (handle == nullptr) {
    return nullptr;
  }
  void* fn = dlsym(handle, name);
  if (fn == nullptr) {
    NNAPI_LOG("nnapi error: unable to open function %s", name);
  }
  return fn;
}

#define LOAD_FUNCTION(handle, name) \
  nnapi.name = reinterpret_cast<name##_fn>(LoadFunction(handle, #name));

#else

#define LOAD_FUNCTION(handle, name) nnapi.name = nullptr;

#endif

const NnApi LoadNnApi() {
  NnApi nnapi = {};

#ifdef __ANDROID__
  // TODO: change RTLD_LOCAL? Assumes there can be multiple instances of nn
  // api RT
  void* libneuralnetworks =
      dlopen("libneuralnetworks.so", RTLD_LAZY | RTLD_LOCAL);
  if (libneuralnetworks == nullptr) {
    NNAPI_LOG("nnapi error: unable to open library %s", "libneuralnetworks.so");
  }
  void* libandroid = dlopen("libandroid.so", RTLD_LAZY | RTLD_LOCAL);
  if (libneuralnetworks == nullptr) {
    NNAPI_LOG("nnapi error: unable to open library %s", "libandroid.so");
  }
  nnapi.nnapi_exists = libneuralnetworks != nullptr;
  nnapi.android_sdk_version = GetAndroidSdkVersion();
#else
  nnapi.nnapi_exists = false;
  nnapi.android_sdk_version = 0;
#endif

  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksMemory_createFromFd);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksMemory_free);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_create);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_free);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_finish);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_addOperand);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_setOperandValue);
  LOAD_FUNCTION(libneuralnetworks,
                ANeuralNetworksModel_setOperandValueFromMemory);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_addOperation);
  LOAD_FUNCTION(libneuralnetworks,
                ANeuralNetworksModel_identifyInputsAndOutputs);
  LOAD_FUNCTION(libneuralnetworks,
                ANeuralNetworksModel_relaxComputationFloat32toFloat16);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_create);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_free);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_setPreference);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_finish);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_create);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_free);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_setInput);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_setInputFromMemory);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_setOutput);
  LOAD_FUNCTION(libneuralnetworks,
                ANeuralNetworksExecution_setOutputFromMemory);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_startCompute);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksEvent_wait);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksEvent_free);
  LOAD_FUNCTION(libandroid, ASharedMemory_create);

  return nnapi;
}

}  // namespace

const NnApi* NnApiImplementation() {
  static const NnApi nnapi = LoadNnApi();
  return &nnapi;
}
