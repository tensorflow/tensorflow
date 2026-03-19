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
#include <stdlib.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "tensorflow/lite/nnapi/sl/include/SupportLibrary.h"
#include "tensorflow/lite/nnapi/sl/public/NeuralNetworksSupportLibraryImpl.h"

namespace {

std::string GetTempDir() {
  const char* temp_dir = getenv("TEST_TMPDIR");
  if (temp_dir == nullptr || temp_dir[0] == '\0') {
#ifdef __ANDROID__
    return "/data/local/tmp";
#else
    return "/tmp";
#endif
  } else {
    return temp_dir;
  }
}

std::string CallCountFilePath() {
  return GetTempDir() + "/nnapi_sl_fake_impl.out";
}
// We write a . in the trace file to allow a caller to count the number of
// calls to NNAPI SL.
void TraceCall() {
  std::ofstream trace_file(CallCountFilePath().c_str(), std::ofstream::app);
  if (trace_file) {
    std::cerr << "Tracing call\n";
    trace_file << '.';
    if (!trace_file) {
      std::cerr << "Error writing to '" << CallCountFilePath() << "'\n";
    }
  } else {
    std::cerr << "FAKE_NNAPI_SL: UNABLE TO TRACE CALL\n";
  }
}

template <int return_value, typename... Types>
int TraceCallAndReturn(Types... args) {
  TraceCall();
  return return_value;
}

template <typename... Types>
void JustTraceCall(Types... args) {
  TraceCall();
}

const uint32_t kDefaultMemoryPaddingAndAlignment = 64;

NnApiSLDriverImplFL5 GetNnApiSlDriverImpl() {
  NnApiSLDriverImplFL5 sl_driver_impl;

  sl_driver_impl.base = {ANEURALNETWORKS_FEATURE_LEVEL_5};
  sl_driver_impl.ANeuralNetworksBurst_create =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksBurst_free = JustTraceCall;
  sl_driver_impl.ANeuralNetworksCompilation_createForDevices =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksCompilation_finish =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksCompilation_free = JustTraceCall;
  sl_driver_impl
      .ANeuralNetworksCompilation_getPreferredMemoryAlignmentForInput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* alignment) -> int {
    TraceCall();
    *alignment = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl
      .ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* alignment) -> int {
    TraceCall();
    *alignment = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworksCompilation_getPreferredMemoryPaddingForInput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* padding) -> int {
    TraceCall();
    *padding = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* padding) -> int {
    TraceCall();
    *padding = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworksCompilation_setCaching =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksCompilation_setPreference =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksCompilation_setPriority =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksCompilation_setTimeout =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksDevice_getExtensionSupport =
      [](const ANeuralNetworksDevice* device, const char* extensionName,
         bool* isExtensionSupported) -> int {
    *isExtensionSupported = false;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworksDevice_getFeatureLevel =
      [](const ANeuralNetworksDevice* device, int64_t* featureLevel) -> int {
    TraceCall();
    *featureLevel = ANEURALNETWORKS_FEATURE_LEVEL_5;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworksDevice_getName =
      [](const ANeuralNetworksDevice* device, const char** name) -> int {
    TraceCall();
    *name = "mockDevice";
    return ANEURALNETWORKS_BAD_DATA;
  };
  sl_driver_impl.ANeuralNetworksDevice_getType =
      [](const ANeuralNetworksDevice* device, int32_t* type) -> int {
    *type = ANEURALNETWORKS_DEVICE_CPU;
    return ANEURALNETWORKS_BAD_DATA;
  };
  sl_driver_impl.ANeuralNetworksDevice_getVersion =
      [](const ANeuralNetworksDevice* device, const char** version) -> int {
    TraceCall();
    *version = "mock";
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworksDevice_wait =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksEvent_createFromSyncFenceFd =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksEvent_free = JustTraceCall;
  sl_driver_impl.ANeuralNetworksEvent_getSyncFenceFd =
      TraceCallAndReturn<ANEURALNETWORKS_BAD_DATA>;
  sl_driver_impl.ANeuralNetworksEvent_wait =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_burstCompute =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_compute =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_create =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_enableInputAndOutputPadding =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_free = JustTraceCall;
  sl_driver_impl.ANeuralNetworksExecution_getDuration =
      [](const ANeuralNetworksExecution* execution, int32_t durationCode,
         uint64_t* duration) -> int {
    TraceCall();
    *duration = UINT64_MAX;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworksExecution_getOutputOperandDimensions =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_getOutputOperandRank =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setInput =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setInputFromMemory =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setLoopTimeout =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setMeasureTiming =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setOutput =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setOutputFromMemory =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setReusable =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setTimeout =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_startComputeWithDependencies =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemoryDesc_addInputRole =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemoryDesc_addOutputRole =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemoryDesc_create =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemoryDesc_finish =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemoryDesc_free = JustTraceCall;
  sl_driver_impl.ANeuralNetworksMemoryDesc_setDimensions =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemory_copy =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemory_createFromAHardwareBuffer =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemory_createFromDesc =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemory_createFromFd =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemory_free = JustTraceCall;
  sl_driver_impl.ANeuralNetworksModel_addOperand =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_addOperation =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_create =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_finish =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_free = JustTraceCall;
  sl_driver_impl.ANeuralNetworksModel_getExtensionOperandType =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_getExtensionOperationType =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_getSupportedOperationsForDevices =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_identifyInputsAndOutputs =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_relaxComputationFloat32toFloat16 =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_setOperandExtensionData =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_setOperandSymmPerChannelQuantParams =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_setOperandValue =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_setOperandValueFromMemory =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_setOperandValueFromModel =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworks_getDefaultLoopTimeout = []() -> uint64_t {
    TraceCall();
    return UINT64_MAX;
  };
  sl_driver_impl.ANeuralNetworks_getDevice =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworks_getDeviceCount =
      [](uint32_t* num_devices) -> int {
    TraceCall();
    *num_devices = 0;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworks_getMaximumLoopTimeout = []() -> uint64_t {
    TraceCall();
    return UINT64_MAX;
  };
  sl_driver_impl.ANeuralNetworks_getRuntimeFeatureLevel = []() -> int64_t {
    TraceCall();
    return ANEURALNETWORKS_FEATURE_LEVEL_5;
  };

  return sl_driver_impl;
}

}  // namespace

extern "C" NnApiSLDriverImpl* ANeuralNetworks_getSLDriverImpl() {
  static NnApiSLDriverImplFL5 sl_driver_impl = GetNnApiSlDriverImpl();
  return reinterpret_cast<NnApiSLDriverImpl*>(&sl_driver_impl);
}

namespace tflite {
namespace acceleration {

void InitNnApiSlInvocationStatus() { unlink(CallCountFilePath().c_str()); }

bool WasNnApiSlInvoked() {
  std::cerr << "Checking if file '" << CallCountFilePath() << "' exists.\n";
  if (FILE* trace_file = fopen(CallCountFilePath().c_str(), "r")) {
    fclose(trace_file);
    return true;
  } else {
    return false;
  }
}

int CountNnApiSlApiCalls() {
  FILE* trace_file = fopen(CallCountFilePath().c_str(), "r");
  int call_count = 0;
  while (fgetc(trace_file) != EOF) {
    call_count++;
  }
  return call_count;
}

}  // namespace acceleration
}  // namespace tflite
