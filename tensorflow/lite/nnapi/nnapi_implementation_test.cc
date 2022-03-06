/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

#include <algorithm>
#include <memory>

#include <gtest/gtest.h>
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/nnapi/sl/public/NeuralNetworksSupportLibraryImpl.h"

struct ANeuralNetworksCompilation {};
struct ANeuralNetworksExecution {};
struct ANeuralNetworksEvent {};
struct ANeuralNetworksModel {
  int operation_count;
  ANeuralNetworksModel() : operation_count(0) {}
};
struct ANeuralNetworksMemory {};
struct ANeuralNetworksDevice {};
struct ANeuralNetworksMemoryDesc {};
struct ANeuralNetworksDiagnosticCompilationInfo {};

constexpr const char kFakeDeviceName[] = "nnapi_test";
constexpr const char kFakeDeviceVersion[] = "1.0";
constexpr const char kFakeDeviceIds[] = "nnapi_test=1.0";

constexpr const uint8_t kModelArchHash[32] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
constexpr ANeuralNetworksDiagnosticCompilationInfo kCompilationInfo{};

// The user can optionally register a compilation callback and
// context object by calling 'SL_ANeuralNetworksDiagnostic_registerCallbacks'.
// We store the user-supplied callback and context at the global scope
// so that we have them available in functions that deal with model compilation.
ANeuralNetworksDiagnosticCompilationFinishedCallback
    g_user_supplied_compilation_callback{nullptr};
void* g_user_supplied_compilation_callback_context{nullptr};

// Create a NnApiSLDriverImplFL5 struct and assign all function pointers to a
// callable lambda.  The returned NnApiSLDriverImplFL5 object does not guarnatee
// any specific behavior (e.g. ability to simulate a model inference).
std::unique_ptr<NnApiSLDriverImplFL5> CreateCompleteSupportLibrary() {
  auto supportLibraryImplementation = std::make_unique<NnApiSLDriverImplFL5>();

  NnApiSLDriverImpl driver_impl;
  driver_impl.implFeatureLevel = ANEURALNETWORKS_FEATURE_LEVEL_5;
  supportLibraryImplementation->base = driver_impl;

  supportLibraryImplementation->ANeuralNetworksBurst_create =
      [](ANeuralNetworksCompilation*, ANeuralNetworksBurst**) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksBurst_free =
      [](ANeuralNetworksBurst*) {};
  supportLibraryImplementation->ANeuralNetworksCompilation_createForDevices =
      [](ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         ANeuralNetworksCompilation** compilation) -> int {
    *compilation = new ANeuralNetworksCompilation();
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksCompilation_finish =
      [](ANeuralNetworksCompilation*) -> int {
    // The user might not have registered compilation callbacks
    if (g_user_supplied_compilation_callback) {
      g_user_supplied_compilation_callback(
          g_user_supplied_compilation_callback_context, &kCompilationInfo);
    }

    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksCompilation_free =
      [](ANeuralNetworksCompilation* compilation) { delete compilation; };

  supportLibraryImplementation
      ->ANeuralNetworksCompilation_getPreferredMemoryAlignmentForInput =
      [](const ANeuralNetworksCompilation*, uint32_t, uint32_t*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation
      ->ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput =
      [](const ANeuralNetworksCompilation*, uint32_t, uint32_t*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation
      ->ANeuralNetworksCompilation_getPreferredMemoryPaddingForInput =
      [](const ANeuralNetworksCompilation*, uint32_t, uint32_t*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation
      ->ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput =
      [](const ANeuralNetworksCompilation*, uint32_t, uint32_t*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksCompilation_setCaching =
      [](ANeuralNetworksCompilation*, const char*, const uint8_t*) -> int {
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksCompilation_setPreference =
      [](ANeuralNetworksCompilation*, int32_t) -> int {
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksCompilation_setPriority =
      [](ANeuralNetworksCompilation*, int) -> int {
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksCompilation_setTimeout =
      [](ANeuralNetworksCompilation*, uint64_t) -> int {
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_getExtensionSupport =
      [](const ANeuralNetworksDevice*, const char*, bool*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_getFeatureLevel =
      [](const ANeuralNetworksDevice*, int64_t* featureLevel) -> int {
    *featureLevel = 5;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_getName =
      [](const ANeuralNetworksDevice* device, const char** name) -> int {
    *name = kFakeDeviceName;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_getType =
      [](const ANeuralNetworksDevice* device, int32_t* type) -> int {
    *type = ANEURALNETWORKS_DEVICE_ACCELERATOR;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_getVersion =
      [](const ANeuralNetworksDevice* device, const char** version) -> int {
    *version = kFakeDeviceVersion;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_wait =
      [](const ANeuralNetworksDevice*) -> int {
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksEvent_createFromSyncFenceFd =
      [](int, ANeuralNetworksEvent**) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksEvent_free =
      [](ANeuralNetworksEvent*) {};
  supportLibraryImplementation->ANeuralNetworksEvent_getSyncFenceFd =
      [](const ANeuralNetworksEvent*, int*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksEvent_wait =
      [](ANeuralNetworksEvent*) -> int { return ANEURALNETWORKS_INCOMPLETE; };
  supportLibraryImplementation->ANeuralNetworksExecution_burstCompute =
      [](ANeuralNetworksExecution*, ANeuralNetworksBurst*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksExecution_compute =
      [](ANeuralNetworksExecution*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksExecution_create =
      [](ANeuralNetworksCompilation*, ANeuralNetworksExecution**) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation
      ->ANeuralNetworksExecution_enableInputAndOutputPadding =
      [](ANeuralNetworksExecution*, bool) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksExecution_free =
      [](ANeuralNetworksExecution*) {};
  supportLibraryImplementation->ANeuralNetworksExecution_getDuration =
      [](const ANeuralNetworksExecution*, int32_t, uint64_t*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation
      ->ANeuralNetworksExecution_getOutputOperandDimensions =
      [](ANeuralNetworksExecution*, int32_t, uint32_t*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksExecution_getOutputOperandRank =
      [](ANeuralNetworksExecution*, int32_t, uint32_t*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksExecution_setInput =
      [](ANeuralNetworksExecution*, int32_t, const ANeuralNetworksOperandType*,
         const void*, size_t) -> int { return ANEURALNETWORKS_INCOMPLETE; };
  supportLibraryImplementation->ANeuralNetworksExecution_setInputFromMemory =
      [](ANeuralNetworksExecution*, int32_t, const ANeuralNetworksOperandType*,
         const ANeuralNetworksMemory*, size_t,
         size_t) -> int { return ANEURALNETWORKS_INCOMPLETE; };
  supportLibraryImplementation->ANeuralNetworksExecution_setLoopTimeout =
      [](ANeuralNetworksExecution*, uint64_t) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksExecution_setMeasureTiming =
      [](ANeuralNetworksExecution*, bool) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksExecution_setOutput =
      [](ANeuralNetworksExecution*, int32_t, const ANeuralNetworksOperandType*,
         void*, size_t) -> int { return ANEURALNETWORKS_INCOMPLETE; };
  supportLibraryImplementation->ANeuralNetworksExecution_setOutputFromMemory =
      [](ANeuralNetworksExecution*, int32_t, const ANeuralNetworksOperandType*,
         const ANeuralNetworksMemory*, size_t,
         size_t) -> int { return ANEURALNETWORKS_INCOMPLETE; };
  supportLibraryImplementation->ANeuralNetworksExecution_setReusable =
      [](ANeuralNetworksExecution*, bool) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksExecution_setTimeout =
      [](ANeuralNetworksExecution*, uint64_t) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation
      ->ANeuralNetworksExecution_startComputeWithDependencies =
      [](ANeuralNetworksExecution*, const ANeuralNetworksEvent* const*,
         uint32_t, uint64_t,
         ANeuralNetworksEvent**) -> int { return ANEURALNETWORKS_INCOMPLETE; };
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_addInputRole =
      [](ANeuralNetworksMemoryDesc*, const ANeuralNetworksCompilation*,
         uint32_t, float) -> int { return ANEURALNETWORKS_NO_ERROR; };
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_addOutputRole =
      [](ANeuralNetworksMemoryDesc*, const ANeuralNetworksCompilation*,
         uint32_t, float) -> int { return ANEURALNETWORKS_NO_ERROR; };
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_create =
      [](ANeuralNetworksMemoryDesc** desc) -> int {
    *desc = new ANeuralNetworksMemoryDesc();
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_finish =
      [](ANeuralNetworksMemoryDesc*) -> int {
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_free =
      [](ANeuralNetworksMemoryDesc* desc) { delete desc; };
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_setDimensions =
      [](ANeuralNetworksMemoryDesc*, uint32_t, const uint32_t*) -> int {
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksMemory_copy =
      [](const ANeuralNetworksMemory*, const ANeuralNetworksMemory*) -> int {
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation
      ->ANeuralNetworksMemory_createFromAHardwareBuffer =
      [](const AHardwareBuffer*, ANeuralNetworksMemory** memory) -> int {
    *memory = new ANeuralNetworksMemory();
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksMemory_createFromDesc =
      [](const ANeuralNetworksMemoryDesc*,
         ANeuralNetworksMemory** memory) -> int {
    *memory = new ANeuralNetworksMemory();
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksMemory_createFromFd =
      [](size_t, int, int, size_t, ANeuralNetworksMemory** memory) -> int {
    *memory = new ANeuralNetworksMemory();
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksMemory_free =
      [](ANeuralNetworksMemory* memory) { delete memory; };
  supportLibraryImplementation->ANeuralNetworksModel_addOperand =
      [](ANeuralNetworksModel* model,
         const ANeuralNetworksOperandType*) -> int {
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksModel_addOperation =
      [](ANeuralNetworksModel* model, ANeuralNetworksOperationType, uint32_t,
         const uint32_t*, uint32_t, const uint32_t*) -> int {
    model->operation_count++;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksModel_create =
      [](ANeuralNetworksModel** model) -> int {
    *model = new ANeuralNetworksModel();
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksModel_finish =
      [](ANeuralNetworksModel*) -> int { return ANEURALNETWORKS_NO_ERROR; };
  supportLibraryImplementation->ANeuralNetworksModel_free =
      [](ANeuralNetworksModel* model) { delete model; };
  supportLibraryImplementation->ANeuralNetworksModel_getExtensionOperandType =
      [](ANeuralNetworksModel*, const char*, uint16_t, int32_t*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksModel_getExtensionOperationType =
      [](ANeuralNetworksModel*, const char*, uint16_t,
         ANeuralNetworksOperationType*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation
      ->ANeuralNetworksModel_getSupportedOperationsForDevices =
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
    std::fill(supportedOps, supportedOps + model->operation_count, true);
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksModel_identifyInputsAndOutputs =
      [](ANeuralNetworksModel*, uint32_t, const uint32_t*, uint32_t,
         const uint32_t*) -> int { return ANEURALNETWORKS_NO_ERROR; };
  supportLibraryImplementation
      ->ANeuralNetworksModel_relaxComputationFloat32toFloat16 =
      [](ANeuralNetworksModel*, bool) -> int {
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksModel_setOperandExtensionData =
      [](ANeuralNetworksModel*, int32_t, const void*, size_t) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation
      ->ANeuralNetworksModel_setOperandSymmPerChannelQuantParams =
      [](ANeuralNetworksModel*, int32_t,
         const ANeuralNetworksSymmPerChannelQuantParams*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
  };
  supportLibraryImplementation->ANeuralNetworksModel_setOperandValue =
      [](ANeuralNetworksModel*, int32_t, const void*, size_t) -> int {
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksModel_setOperandValueFromMemory =
      [](ANeuralNetworksModel*, int32_t, const ANeuralNetworksMemory*, size_t,
         size_t) -> int { return ANEURALNETWORKS_NO_ERROR; };
  supportLibraryImplementation->ANeuralNetworksModel_setOperandValueFromModel =
      [](ANeuralNetworksModel*, int32_t, const ANeuralNetworksModel*) -> int {
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworks_getDefaultLoopTimeout =
      []() -> uint64_t { return 0; };

  supportLibraryImplementation->ANeuralNetworks_getDevice =
      [](uint32_t devIndex, ANeuralNetworksDevice** device) -> int {
    *device = new ANeuralNetworksDevice();
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworks_getDeviceCount =
      [](uint32_t* deviceCnt) -> int {
    *deviceCnt = 1;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworks_getMaximumLoopTimeout =
      []() -> uint64_t { return 0; };
  supportLibraryImplementation->ANeuralNetworks_getRuntimeFeatureLevel =
      []() -> int64_t { return 0; };
  supportLibraryImplementation
      ->SL_ANeuralNetworksCompilation_setCachingFromFds =
      [](ANeuralNetworksCompilation*, const int*, const uint32_t, const int*,
         const uint32_t,
         const uint8_t*) -> int { return ANEURALNETWORKS_INCOMPLETE; };
  supportLibraryImplementation
      ->SL_ANeuralNetworksDevice_getNumberOfCacheFilesNeeded =
      [](const ANeuralNetworksDevice*, uint32_t*, uint32_t*) -> int {
    return ANEURALNETWORKS_INCOMPLETE;
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
         void* context) -> void {
    g_user_supplied_compilation_callback = compilationCallback;
    g_user_supplied_compilation_callback_context = context;
  };
  return supportLibraryImplementation;
}  // NOLINT(readability/fn_size)

namespace {

TEST(NnapiLibTest, NnApiImplementation) {
  const NnApi* nnapi = NnApiImplementation();
  EXPECT_NE(nnapi, nullptr);
#ifdef __ANDROID__
  EXPECT_GT(nnapi->android_sdk_version, 0);
  if (nnapi.android_sdk_version < 27) {
    EXPECT_FALSE(nnapi->nnapi_exists);
    EXPECT_EQ(nnapi->ANeuralNetworksMemory_createFromFd, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksMemory_free, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_create, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_free, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_finish, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_addOperand, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_setOperandValue, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_setOperandValueFromMemory, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_addOperation, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_identifyInputsAndOutputs, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksModel_relaxComputationFloat32toFloat16,
              nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksCompilation_create, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksCompilation_free, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksCompilation_setPreference, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksCompilation_finish, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksExecution_create, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksExecution_free, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksExecution_setInput, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksExecution_setInputFromMemory, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksExecution_setOutput, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksExecution_setOutputFromMemory, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksExecution_startCompute, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksEvent_wait, nullptr);
    EXPECT_EQ(nnapi->ANeuralNetworksEvent_free, nullptr);
    EXPECT_EQ(nnapi->ASharedMemory_create, nullptr);
  } else {
    EXPECT_TRUE(nnapi->nnapi_exists);
    EXPECT_NE(nnapi->ANeuralNetworksMemory_createFromFd, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksMemory_free, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_create, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_free, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_finish, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_addOperand, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_setOperandValue, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_setOperandValueFromMemory, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_addOperation, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksModel_identifyInputsAndOutputs, nullptr);
    if (nnapi->android_sdk_version >= 28) {
      // relaxComputationFloat32toFloat16 only available with Android 9.0 (P).
      EXPECT_NE(nnapi->ANeuralNetworksModel_relaxComputationFloat32toFloat16,
                nullptr);
    } else {
      EXPECT_EQ(nnapi->ANeuralNetworksModel_relaxComputationFloat32toFloat16,
                nullptr);
    }
    EXPECT_NE(nnapi->ANeuralNetworksCompilation_create, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksCompilation_free, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksCompilation_setPreference, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksCompilation_finish, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksExecution_create, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksExecution_free, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksExecution_setInput, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksExecution_setInputFromMemory, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksExecution_setOutput, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksExecution_setOutputFromMemory, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksExecution_startCompute, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksEvent_wait, nullptr);
    EXPECT_NE(nnapi->ANeuralNetworksEvent_free, nullptr);
    EXPECT_NE(nnapi->ASharedMemory_create, nullptr);
    // TODO(b/123423795): Test Q-specific APIs after release.
  }
#else
  EXPECT_FALSE(nnapi->nnapi_exists);
  EXPECT_EQ(nnapi->android_sdk_version, 0);
  EXPECT_EQ(nnapi->ANeuralNetworksMemory_createFromFd, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksMemory_free, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_create, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_free, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_finish, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_addOperand, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_setOperandValue, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_setOperandSymmPerChannelQuantParams,
            nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_setOperandValueFromMemory, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_addOperation, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_identifyInputsAndOutputs, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_relaxComputationFloat32toFloat16,
            nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksCompilation_create, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksCompilation_free, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksCompilation_setPreference, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksCompilation_finish, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_create, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_free, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_setInput, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_setInputFromMemory, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_setOutput, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_setOutputFromMemory, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_startCompute, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksEvent_wait, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksEvent_free, nullptr);
  EXPECT_EQ(nnapi->ASharedMemory_create, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworks_getDeviceCount, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworks_getDevice, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksDevice_getName, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksDevice_getVersion, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksDevice_getFeatureLevel, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksModel_getSupportedOperationsForDevices,
            nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksCompilation_createForDevices, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksCompilation_setCaching, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_compute, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_getOutputOperandRank, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_getOutputOperandDimensions,
            nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksBurst_create, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksBurst_free, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_burstCompute, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksMemory_createFromAHardwareBuffer, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_setMeasureTiming, nullptr);
  EXPECT_EQ(nnapi->ANeuralNetworksExecution_getDuration, nullptr);
#endif
}

TEST(NnapiLibTest, CreateNnApiFromSupportLibrary) {
  auto support_library = CreateCompleteSupportLibrary();
  EXPECT_TRUE(CreateNnApiFromSupportLibrary(support_library.get()));
}

TEST(NnapiLibTest, CreateNnApiFromSupportLibrary_SucceedsWhenSlIsIncomplete) {
  auto support_library = CreateCompleteSupportLibrary();
  support_library->ANeuralNetworksBurst_create = nullptr;

  EXPECT_TRUE(CreateNnApiFromSupportLibrary(support_library.get()));
}

TEST(NnapiLibTest, CreateCompleteNnApiFromSupportLibraryOrFail) {
  auto support_library = CreateCompleteSupportLibrary();
  EXPECT_TRUE(
      CreateCompleteNnApiFromSupportLibraryOrFail(support_library.get()));
}

TEST(NnapiLibTest,
     CreateCompleteNnApiFromSupportLibraryOrFail_FailsWhenSlIsIncomplete) {
  auto support_library = CreateCompleteSupportLibrary();
  support_library->ANeuralNetworksBurst_create = nullptr;

  EXPECT_DEATH(
      CreateCompleteNnApiFromSupportLibraryOrFail(support_library.get()),
      "support_library");
}

}  // namespace
