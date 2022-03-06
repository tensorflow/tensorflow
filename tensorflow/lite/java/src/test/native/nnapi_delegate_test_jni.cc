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

#include <jni.h>

#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/nnapi/sl/public/NeuralNetworksSupportLibraryImpl.h"

namespace {

constexpr const char kFakeDeviceIds[] = "nnapi_test=1.0";

constexpr const uint8_t kModelArchHash[32] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

bool g_sl_has_been_called = false;

template <int return_value, typename... Types>
int DoNothingAndReturn(Types... args) {
  g_sl_has_been_called = true;
  return return_value;
}

template <typename... Types>
void DoNothing(Types... args) {
  g_sl_has_been_called = true;
}

const uint32_t kDefaultMemoryPaddingAndAlignment = 64;

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegateTest_getMockSlHandle(JNIEnv* env,
                                                                 jclass clazz) {
  g_sl_has_been_called = false;

  NnApiSLDriverImplFL5* supportLibraryImplementation =
      new NnApiSLDriverImplFL5();

  supportLibraryImplementation->base.implFeatureLevel =
      ANEURALNETWORKS_FEATURE_LEVEL_5;

  // Most calls do nothing and return NO_ERROR as result. Errors are returned in
  // get* calls that are not trivial to mock.
  supportLibraryImplementation->ANeuralNetworksBurst_create =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksBurst_free = DoNothing;
  supportLibraryImplementation->ANeuralNetworksCompilation_createForDevices =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksCompilation_finish =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksCompilation_free = DoNothing;
  supportLibraryImplementation
      ->ANeuralNetworksCompilation_getPreferredMemoryAlignmentForInput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* alignment) -> int {
    g_sl_has_been_called = true;
    *alignment = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation
      ->ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* alignment) -> int {
    g_sl_has_been_called = true;
    *alignment = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation
      ->ANeuralNetworksCompilation_getPreferredMemoryPaddingForInput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* padding) -> int {
    g_sl_has_been_called = true;
    *padding = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation
      ->ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* padding) -> int {
    g_sl_has_been_called = true;
    *padding = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksCompilation_setCaching =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksCompilation_setPreference =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksCompilation_setPriority =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksCompilation_setTimeout =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksDevice_getExtensionSupport =
      [](const ANeuralNetworksDevice* device, const char* extensionName,
         bool* isExtensionSupported) -> int {
    g_sl_has_been_called = true;
    *isExtensionSupported = false;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_getFeatureLevel =
      [](const ANeuralNetworksDevice* device, int64_t* featureLevel) -> int {
    g_sl_has_been_called = true;
    *featureLevel = ANEURALNETWORKS_FEATURE_LEVEL_5;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_getName =
      [](const ANeuralNetworksDevice* device, const char** name) -> int {
    g_sl_has_been_called = true;
    *name = "mockDevice";
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_getType =
      [](const ANeuralNetworksDevice* device, int32_t* type) -> int {
    g_sl_has_been_called = true;
    *type = ANEURALNETWORKS_DEVICE_CPU;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_getVersion =
      [](const ANeuralNetworksDevice* device, const char** version) -> int {
    g_sl_has_been_called = true;
    *version = "mock";
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_wait =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksEvent_createFromSyncFenceFd =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksEvent_free = DoNothing;
  supportLibraryImplementation->ANeuralNetworksEvent_getSyncFenceFd =
      DoNothingAndReturn<ANEURALNETWORKS_BAD_DATA>;
  supportLibraryImplementation->ANeuralNetworksEvent_wait =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_burstCompute =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_compute =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_create =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation
      ->ANeuralNetworksExecution_enableInputAndOutputPadding =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_free = DoNothing;
  supportLibraryImplementation->ANeuralNetworksExecution_getDuration =
      [](const ANeuralNetworksExecution* execution, int32_t durationCode,
         uint64_t* duration) -> int {
    g_sl_has_been_called = true;
    *duration = UINT64_MAX;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation
      ->ANeuralNetworksExecution_getOutputOperandDimensions =
      DoNothingAndReturn<ANEURALNETWORKS_BAD_DATA>;
  supportLibraryImplementation->ANeuralNetworksExecution_getOutputOperandRank =
      DoNothingAndReturn<ANEURALNETWORKS_BAD_DATA>;
  supportLibraryImplementation->ANeuralNetworksExecution_setInput =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_setInputFromMemory =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_setLoopTimeout =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_setMeasureTiming =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_setOutput =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_setOutputFromMemory =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_setReusable =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_setTimeout =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation
      ->ANeuralNetworksExecution_startComputeWithDependencies =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_addInputRole =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_addOutputRole =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_create =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_finish =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_free = DoNothing;
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_setDimensions =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemory_copy =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation
      ->ANeuralNetworksMemory_createFromAHardwareBuffer =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemory_createFromDesc =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemory_createFromFd =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemory_free = DoNothing;
  supportLibraryImplementation->ANeuralNetworksModel_addOperand =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_addOperation =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_create =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_finish =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_free = DoNothing;
  supportLibraryImplementation->ANeuralNetworksModel_getExtensionOperandType =
      DoNothingAndReturn<ANEURALNETWORKS_BAD_DATA>;
  supportLibraryImplementation->ANeuralNetworksModel_getExtensionOperationType =
      DoNothingAndReturn<ANEURALNETWORKS_BAD_DATA>;
  supportLibraryImplementation
      ->ANeuralNetworksModel_getSupportedOperationsForDevices =
      DoNothingAndReturn<ANEURALNETWORKS_BAD_DATA>;
  supportLibraryImplementation->ANeuralNetworksModel_identifyInputsAndOutputs =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation
      ->ANeuralNetworksModel_relaxComputationFloat32toFloat16 =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_setOperandExtensionData =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation
      ->ANeuralNetworksModel_setOperandSymmPerChannelQuantParams =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_setOperandValue =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_setOperandValueFromMemory =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_setOperandValueFromModel =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworks_getDefaultLoopTimeout =
      []() -> uint64_t {
    g_sl_has_been_called = true;
    return UINT64_MAX;
  };
  supportLibraryImplementation->ANeuralNetworks_getDevice =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworks_getDeviceCount =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworks_getMaximumLoopTimeout =
      []() -> uint64_t {
    g_sl_has_been_called = true;
    return UINT64_MAX;
  };
  supportLibraryImplementation->ANeuralNetworks_getRuntimeFeatureLevel =
      []() -> int64_t {
    g_sl_has_been_called = true;
    return ANEURALNETWORKS_FEATURE_LEVEL_5;
  };
  supportLibraryImplementation->SL_ANeuralNetworksDevice_getPerformanceInfo =
      [](const ANeuralNetworksDevice*, int32_t,
         SL_ANeuralNetworksPerformanceInfo*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDevice_forEachOperandPerformanceInfo =
      [](const ANeuralNetworksDevice*, void*,
         void (*)(SL_ANeuralNetworksOperandPerformanceInfo, void*)) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDevice_getVendorExtensionCount =
      [](const ANeuralNetworksDevice*, uint32_t*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDevice_getVendorExtensionName =
      [](const ANeuralNetworksDevice*, uint32_t, const char**) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDevice_forEachVendorExtensionOperandTypeInformation =
      [](const ANeuralNetworksDevice*, uint32_t, void*,
         void (*)(SL_ANeuralNetworksExtensionOperandTypeInformation,
                  void*)) -> int { return ANEURALNETWORKS_INCOMPLETE; };

  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticCompilationInfo_getSessionId =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };

  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticCompilationInfo_getNnApiVersion =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> int64_t {
    return ANEURALNETWORKS_FEATURE_LEVEL_1;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticCompilationInfo_getModelArchHash =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> const uint8_t* {
    return kModelArchHash;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticCompilationInfo_getDeviceIds =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> const char* {
    return kFakeDeviceIds;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticCompilationInfo_getErrorCode =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> int32_t {
    return ANEURALNETWORKS_BAD_STATE;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticCompilationInfo_getInputDataClass =
      [](const ANeuralNetworksDiagnosticCompilationInfo*)
      -> ANeuralNetworksDiagnosticDataClass {
    return ANNDIAG_DATA_CLASS_UNKNOWN;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticCompilationInfo_getOutputDataClass =
      [](const ANeuralNetworksDiagnosticCompilationInfo*)
      -> ANeuralNetworksDiagnosticDataClass {
    return ANNDIAG_DATA_CLASS_UNKNOWN;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticCompilationInfo_getCompilationTimeNanos =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> uint64_t {
    return 0;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticCompilationInfo_isCachingEnabled =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> bool {
    return false;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticCompilationInfo_isControlFlowUsed =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> bool {
    return false;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticCompilationInfo_areDynamicTensorsUsed =
      [](const ANeuralNetworksDiagnosticCompilationInfo*) -> bool {
    return false;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticExecutionInfo_getSessionId =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> int32_t {
    return 0;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticExecutionInfo_getNnApiVersion =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> int64_t {
    return ANEURALNETWORKS_FEATURE_LEVEL_1;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticExecutionInfo_getModelArchHash =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> const uint8_t* {
    return kModelArchHash;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticExecutionInfo_getDeviceIds =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> const char* {
    return kFakeDeviceIds;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticExecutionInfo_getExecutionMode =
      [](const ANeuralNetworksDiagnosticExecutionInfo*)
      -> ANeuralNetworksDiagnosticExecutionMode {
    return ANNDIAG_EXECUTION_MODE_UNKNOWN;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticExecutionInfo_getInputDataClass =
      [](const ANeuralNetworksDiagnosticExecutionInfo*)
      -> ANeuralNetworksDiagnosticDataClass {
    return ANNDIAG_DATA_CLASS_UNKNOWN;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticExecutionInfo_getOutputDataClass =
      [](const ANeuralNetworksDiagnosticExecutionInfo*)
      -> ANeuralNetworksDiagnosticDataClass {
    return ANNDIAG_DATA_CLASS_UNKNOWN;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticExecutionInfo_getErrorCode =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> uint32_t {
    return 0;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticExecutionInfo_getRuntimeExecutionTimeNanos =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> uint64_t {
    return 0;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticExecutionInfo_getDriverExecutionTimeNanos =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> uint64_t {
    return 0;
  };
  // clang-format off
  supportLibraryImplementation
    ->SL_ANeuralNetworksDiagnosticExecutionInfo_getHardwareExecutionTimeNanos =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> uint64_t {
    return 0;
  };
  // clang-format on
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticExecutionInfo_isCachingEnabled =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> bool {
    return false;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticExecutionInfo_isControlFlowUsed =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> bool {
    return false;
  };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDiagnosticExecutionInfo_areDynamicTensorsUsed =
      [](const ANeuralNetworksDiagnosticExecutionInfo*) -> bool {
    return false;
  };
  supportLibraryImplementation->SL_ANeuralNetworksDiagnostic_registerCallbacks =
      [](ANeuralNetworksDiagnosticCompilationFinishedCallback
             compilationCallback,
         ANeuralNetworksDiagnosticExecutionFinishedCallback,
         void* context) -> void {};

  return reinterpret_cast<jlong>(supportLibraryImplementation);
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegateTest_hasNnApiSlBeenCalled(
    JNIEnv* env, jclass clazz) {
  return g_sl_has_been_called;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegateTest_closeMockSl(JNIEnv* env,
                                                             jclass clazz,
                                                             jlong handle) {
  delete reinterpret_cast<NnApiSLDriverImplFL5*>(handle);
}

}  // extern "C"
