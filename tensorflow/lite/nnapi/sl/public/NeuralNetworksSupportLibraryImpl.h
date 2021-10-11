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

#ifndef TENSORFLOW_LITE_NNAPI_SL_PUBLIC_NEURAL_NETWORKS_SUPPORT_LIBRARY_IMPL_H_
#define TENSORFLOW_LITE_NNAPI_SL_PUBLIC_NEURAL_NETWORKS_SUPPORT_LIBRARY_IMPL_H_

/******************************************************************
 *
 * IMPORTANT NOTICE:
 *
 *   This file is part of Android's set of stable system headers
 *   exposed by the Android NDK (Native Development Kit).
 *
 *   Third-party source AND binary code relies on the definitions
 *   here to be FROZEN ON ALL UPCOMING PLATFORM RELEASES.
 *
 *   - DO NOT MODIFY ENUMS (EXCEPT IF YOU ADD NEW 32-BIT VALUES)
 *   - DO NOT MODIFY CONSTANTS OR FUNCTIONAL MACROS
 *   - DO NOT CHANGE THE SIGNATURE OF FUNCTIONS IN ANY WAY
 *   - DO NOT CHANGE THE LAYOUT OR SIZE OF STRUCTURES
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Changed when importing from AOSP
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Performance information for the reference workload.
 *
 * Used by a driver to report its performance characteristics.
 */
typedef struct {
  /**
   * Ratio of the time taken by the driver to execute the workload compared to
   * the time the CPU would take for the same workload. A lower number is
   * better.
   */
  float execTime;

  /**
   * Ratio of the energy used by the driver compared to what the CPU would use
   * for doing the same workload. A lower number is better.
   */
  float powerUsage;
} SL_ANeuralNetworksPerformanceInfo;

/**
 * Driver performance when operating on a particular data type. In the case of
 * float32 data, this is used when the calculations are not relaxed.
 */
typedef struct {
  int32_t operandType;
  SL_ANeuralNetworksPerformanceInfo performanceInfo;
} SL_ANeuralNetworksOperandPerformanceInfo;

/**
 * Information about NNAPI Vendor extension operand type.
 */
typedef struct {
  /**
   * The byte size of the operand (if scalar) or of a single element (if
   * tensor).
   */
  uint32_t byteSize;

  /**
   * The extension operand type.
   */
  uint16_t type;

  /**
   * Indicates whether the extension operand type represents a tensor or a
   * scalar.
   */
  bool isTensor;
} SL_ANeuralNetworksExtensionOperandTypeInformation;

/**
 * The different performance info kinds.
 */
typedef enum {
  /**
   * Driver performance when operating on float32 data but performing
   * calculations with range and/or precision as low as that of the IEEE 754
   * 16-bit floating-point format.
   */
  SL_ANEURALNETWORKS_CAPABILITIES_PERFORMANCE_RELAXED_SCALAR = 0,

  /**
   * Driver performance when operating on float32 data but performing
   * calculations with range and/or precision as low as that of the IEEE 754
   * 16-bit floating-point format.
   */
  SL_ANEURALNETWORKS_CAPABILITIES_PERFORMANCE_RELAXED_TENSOR = 1,

  /**
   * Performance of an {@link ANEURALNETWORKS_IF} operation is the sum of {@link
   * ANEURALNETWORKS_IF}'s performance and the mean of performance for the two
   * branch subgraphs, where performance for a subgraph is the sum of the
   * performance of all operations within the subgraph.
   */
  SL_ANEURALNETWORKS_CAPABILITIES_PERFORMANCE_IF = 2,

  /**
   * Performance of a {@link ANEURALNETWORKS_WHILE} operation is the sum of
   * {@link ANEURALNETWORKS_WHILE}'s performance, performance for the condition
   * subgraph and performance for the body subgraph, where performance for a
   * subgraph is the sum of the performance of all operations within the
   * subgraph.
   */
  SL_ANEURALNETWORKS_CAPABILITIES_PERFORMANCE_WHILE = 3,
} SL_ANeuralNetworksPerformanceInfoCode;

typedef struct ANeuralNetworksDiagnosticCompilationInfo
    ANeuralNetworksDiagnosticCompilationInfo;
typedef struct ANeuralNetworksDiagnosticExecutionInfo
    ANeuralNetworksDiagnosticExecutionInfo;
typedef void (*ANeuralNetworksDiagnosticCompilationFinishedCallback)(
    const void* context, const ANeuralNetworksDiagnosticCompilationInfo* info);
typedef void (*ANeuralNetworksDiagnosticExecutionFinishedCallback)(
    const void* context, const ANeuralNetworksDiagnosticExecutionInfo* info);

/**
 * Base version of NnApiSLDriverImpl with version information.
 *
 * NnApiSLDriverImpl is non-opaque, versioning struct to make it possible to
 * pass its instance straight from the SL Driver to the shim registration. The
 * glue code that loads the SL and calls the shim is non-updatable. An opaque
 * struct would require the glue code to be updated if we would like to use
 * newer NNAPI Feature Level.
 *
 * There's expectation that for M>N, NnApiSLDriverImplFL(M) is
 * a strict superset of NnApiSLDriverImplFL(N), and NnApiSLDriverImplFL(M)* can
 * be reinterpret_cast to NnApiSLDriverImplFL(N)* safely.
 */
typedef struct NnApiSLDriverImpl {
  /**
   * Version of the NnApiSLDriverImpl struct. Uses {@link FeatureLevelCode}
   * values for versioning.
   */
  int64_t implFeatureLevel;
} NnApiSLDriverImpl;

/**
 * NnApiSLDriverImpl for an Updatable SL Driver implementing {@link
 * ANEURALNETWORKS_FEATURE_LEVEL_5}.
 *
 * This struct must set its implFeatureLevel to {@link
 * ANEURALNETWORKS_FEATURE_LEVEL_5}.
 */
typedef struct NnApiSLDriverImplFL5 {
  /**
   * Base type with version information. Allows to cast a pointer of this type
   * to NnApiSLDriverImpl* with valid version information.
   * For this type, its .version fields should be always set to {@link
   * ANEURALNETWORKS_FEATURE_LEVEL_5}.
   */
  NnApiSLDriverImpl base;

  /**
   * SL Driver implementation of {@link ANeuralNetworksBurst_create}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksBurst_create},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksBurst_create)(ANeuralNetworksCompilation* compilation,
                                     ANeuralNetworksBurst** burst);

  /**
   * SL Driver implementation of {@link ANeuralNetworksBurst_free}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksBurst_free},
   * at the feature level of this NnApiSLDriver struct.
   */
  void (*ANeuralNetworksBurst_free)(ANeuralNetworksBurst* burst);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksCompilation_createForDevices}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link ANeuralNetworksCompilation_createForDevices},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksCompilation_createForDevices)(
      ANeuralNetworksModel* model, const ANeuralNetworksDevice* const* devices,
      uint32_t numDevices, ANeuralNetworksCompilation** compilation);

  /**
   * SL Driver implementation of {@link ANeuralNetworksCompilation_finish}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksCompilation_finish},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksCompilation_finish)(
      ANeuralNetworksCompilation* compilation);

  /**
   * SL Driver implementation of {@link ANeuralNetworksCompilation_free}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksCompilation_free},
   * at the feature level of this NnApiSLDriver struct.
   */
  void (*ANeuralNetworksCompilation_free)(
      ANeuralNetworksCompilation* compilation);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksCompilation_getPreferredMemoryAlignmentForInput}. Behavior,
   * arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksCompilation_getPreferredMemoryAlignmentForInput},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksCompilation_getPreferredMemoryAlignmentForInput)(
      const ANeuralNetworksCompilation* compilation, uint32_t index,
      uint32_t* alignment);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput}. Behavior,
   * arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput)(
      const ANeuralNetworksCompilation* compilation, uint32_t index,
      uint32_t* alignment);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksCompilation_getPreferredMemoryPaddingForInput}. Behavior,
   * arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksCompilation_getPreferredMemoryPaddingForInput},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksCompilation_getPreferredMemoryPaddingForInput)(
      const ANeuralNetworksCompilation* compilation, uint32_t index,
      uint32_t* padding);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput}. Behavior,
   * arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput)(
      const ANeuralNetworksCompilation* compilation, uint32_t index,
      uint32_t* padding);

  /**
   * SL Driver implementation of {@link ANeuralNetworksCompilation_setCaching}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksCompilation_setCaching},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksCompilation_setCaching)(
      ANeuralNetworksCompilation* compilation, const char* cacheDir,
      const uint8_t* token);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksCompilation_setPreference}. Behavior, arguments, and outputs
   * match NNAPI Runtime function
   * {@link ANeuralNetworksCompilation_setPreference},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksCompilation_setPreference)(
      ANeuralNetworksCompilation* compilation, int32_t preference);

  /**
   * SL Driver implementation of {@link ANeuralNetworksCompilation_setPriority}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksCompilation_setPriority},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksCompilation_setPriority)(
      ANeuralNetworksCompilation* compilation, int priority);

  /**
   * SL Driver implementation of {@link ANeuralNetworksCompilation_setTimeout}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksCompilation_setTimeout},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksCompilation_setTimeout)(
      ANeuralNetworksCompilation* compilation, uint64_t duration);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksDevice_getExtensionSupport}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link ANeuralNetworksDevice_getExtensionSupport},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksDevice_getExtensionSupport)(
      const ANeuralNetworksDevice* device, const char* extensionName,
      bool* isExtensionSupported);

  /**
   * SL Driver implementation of {@link ANeuralNetworksDevice_getFeatureLevel}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksDevice_getFeatureLevel},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksDevice_getFeatureLevel)(
      const ANeuralNetworksDevice* device, int64_t* featureLevel);

  /**
   * SL Driver implementation of {@link ANeuralNetworksDevice_getName}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksDevice_getName},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksDevice_getName)(const ANeuralNetworksDevice* device,
                                       const char** name);

  /**
   * SL Driver implementation of {@link ANeuralNetworksDevice_getType}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksDevice_getType},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksDevice_getType)(const ANeuralNetworksDevice* device,
                                       int32_t* type);

  /**
   * SL Driver implementation of {@link ANeuralNetworksDevice_getVersion}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksDevice_getVersion},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksDevice_getVersion)(const ANeuralNetworksDevice* device,
                                          const char** version);

  /**
   * SL Driver implementation of {@link ANeuralNetworksDevice_wait}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksDevice_wait},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksDevice_wait)(const ANeuralNetworksDevice* device);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksEvent_createFromSyncFenceFd}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link ANeuralNetworksEvent_createFromSyncFenceFd},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksEvent_createFromSyncFenceFd)(
      int sync_fence_fd, ANeuralNetworksEvent** event);

  /**
   * SL Driver implementation of {@link ANeuralNetworksEvent_free}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksEvent_free},
   * at the feature level of this NnApiSLDriver struct.
   */
  void (*ANeuralNetworksEvent_free)(ANeuralNetworksEvent* event);

  /**
   * SL Driver implementation of {@link ANeuralNetworksEvent_getSyncFenceFd}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksEvent_getSyncFenceFd},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksEvent_getSyncFenceFd)(const ANeuralNetworksEvent* event,
                                             int* sync_fence_fd);

  /**
   * SL Driver implementation of {@link ANeuralNetworksEvent_wait}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksEvent_wait},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksEvent_wait)(ANeuralNetworksEvent* event);

  /**
   * SL Driver implementation of {@link ANeuralNetworksExecution_burstCompute}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_burstCompute},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_burstCompute)(
      ANeuralNetworksExecution* execution, ANeuralNetworksBurst* burst);

  /**
   * SL Driver implementation of {@link ANeuralNetworksExecution_compute}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_compute},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_compute)(ANeuralNetworksExecution* execution);

  /**
   * SL Driver implementation of {@link ANeuralNetworksExecution_create}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_create},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_create)(
      ANeuralNetworksCompilation* compilation,
      ANeuralNetworksExecution** execution);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksExecution_enableInputAndOutputPadding}. Behavior, arguments,
   * and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_enableInputAndOutputPadding},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_enableInputAndOutputPadding)(
      ANeuralNetworksExecution* execution, bool enable);

  /**
   * SL Driver implementation of {@link ANeuralNetworksExecution_free}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_free},
   * at the feature level of this NnApiSLDriver struct.
   */
  void (*ANeuralNetworksExecution_free)(ANeuralNetworksExecution* execution);

  /**
   * SL Driver implementation of {@link ANeuralNetworksExecution_getDuration}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_getDuration},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_getDuration)(
      const ANeuralNetworksExecution* execution, int32_t durationCode,
      uint64_t* duration);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksExecution_getOutputOperandDimensions}. Behavior, arguments,
   * and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_getOutputOperandDimensions},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_getOutputOperandDimensions)(
      ANeuralNetworksExecution* execution, int32_t index, uint32_t* dimensions);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksExecution_getOutputOperandRank}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_getOutputOperandRank},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_getOutputOperandRank)(
      ANeuralNetworksExecution* execution, int32_t index, uint32_t* rank);

  /**
   * SL Driver implementation of {@link ANeuralNetworksExecution_setInput}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_setInput},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_setInput)(
      ANeuralNetworksExecution* execution, int32_t index,
      const ANeuralNetworksOperandType* type, const void* buffer,
      size_t length);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksExecution_setInputFromMemory}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_setInputFromMemory},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_setInputFromMemory)(
      ANeuralNetworksExecution* execution, int32_t index,
      const ANeuralNetworksOperandType* type,
      const ANeuralNetworksMemory* memory, size_t offset, size_t length);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksExecution_setLoopTimeout}. Behavior, arguments, and outputs
   * match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_setLoopTimeout},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_setLoopTimeout)(
      ANeuralNetworksExecution* execution, uint64_t duration);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksExecution_setMeasureTiming}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_setMeasureTiming},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_setMeasureTiming)(
      ANeuralNetworksExecution* execution, bool measure);

  /**
   * SL Driver implementation of {@link ANeuralNetworksExecution_setOutput}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_setOutput},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_setOutput)(
      ANeuralNetworksExecution* execution, int32_t index,
      const ANeuralNetworksOperandType* type, void* buffer, size_t length);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksExecution_setOutputFromMemory}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_setOutputFromMemory},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_setOutputFromMemory)(
      ANeuralNetworksExecution* execution, int32_t index,
      const ANeuralNetworksOperandType* type,
      const ANeuralNetworksMemory* memory, size_t offset, size_t length);

  /**
   * SL Driver implementation of {@link ANeuralNetworksExecution_setReusable}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_setReusable},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_setReusable)(
      ANeuralNetworksExecution* execution, bool reusable);

  /**
   * SL Driver implementation of {@link ANeuralNetworksExecution_setTimeout}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_setTimeout},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_setTimeout)(
      ANeuralNetworksExecution* execution, uint64_t duration);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksExecution_startComputeWithDependencies}. Behavior,
   * arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksExecution_startComputeWithDependencies},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksExecution_startComputeWithDependencies)(
      ANeuralNetworksExecution* execution,
      const ANeuralNetworksEvent* const* dependencies,
      uint32_t num_dependencies, uint64_t duration,
      ANeuralNetworksEvent** event);

  /**
   * SL Driver implementation of {@link ANeuralNetworksMemoryDesc_addInputRole}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksMemoryDesc_addInputRole},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksMemoryDesc_addInputRole)(
      ANeuralNetworksMemoryDesc* desc,
      const ANeuralNetworksCompilation* compilation, uint32_t index,
      float frequency);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksMemoryDesc_addOutputRole}. Behavior, arguments, and outputs
   * match NNAPI Runtime function
   * {@link ANeuralNetworksMemoryDesc_addOutputRole},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksMemoryDesc_addOutputRole)(
      ANeuralNetworksMemoryDesc* desc,
      const ANeuralNetworksCompilation* compilation, uint32_t index,
      float frequency);

  /**
   * SL Driver implementation of {@link ANeuralNetworksMemoryDesc_create}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksMemoryDesc_create},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksMemoryDesc_create)(ANeuralNetworksMemoryDesc** desc);

  /**
   * SL Driver implementation of {@link ANeuralNetworksMemoryDesc_finish}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksMemoryDesc_finish},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksMemoryDesc_finish)(ANeuralNetworksMemoryDesc* desc);

  /**
   * SL Driver implementation of {@link ANeuralNetworksMemoryDesc_free}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksMemoryDesc_free},
   * at the feature level of this NnApiSLDriver struct.
   */
  void (*ANeuralNetworksMemoryDesc_free)(ANeuralNetworksMemoryDesc* desc);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksMemoryDesc_setDimensions}. Behavior, arguments, and outputs
   * match NNAPI Runtime function
   * {@link ANeuralNetworksMemoryDesc_setDimensions},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksMemoryDesc_setDimensions)(
      ANeuralNetworksMemoryDesc* desc, uint32_t rank,
      const uint32_t* dimensions);

  /**
   * SL Driver implementation of {@link ANeuralNetworksMemory_copy}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksMemory_copy},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksMemory_copy)(const ANeuralNetworksMemory* src,
                                    const ANeuralNetworksMemory* dst);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksMemory_createFromAHardwareBuffer}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link ANeuralNetworksMemory_createFromAHardwareBuffer},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksMemory_createFromAHardwareBuffer)(
      const AHardwareBuffer* ahwb, ANeuralNetworksMemory** memory);

  /**
   * SL Driver implementation of {@link ANeuralNetworksMemory_createFromDesc}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksMemory_createFromDesc},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksMemory_createFromDesc)(
      const ANeuralNetworksMemoryDesc* desc, ANeuralNetworksMemory** memory);

  /**
   * SL Driver implementation of {@link ANeuralNetworksMemory_createFromFd}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksMemory_createFromFd},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksMemory_createFromFd)(size_t size, int protect, int fd,
                                            size_t offset,
                                            ANeuralNetworksMemory** memory);

  /**
   * SL Driver implementation of {@link ANeuralNetworksMemory_free}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksMemory_free},
   * at the feature level of this NnApiSLDriver struct.
   */
  void (*ANeuralNetworksMemory_free)(ANeuralNetworksMemory* memory);

  /**
   * SL Driver implementation of {@link ANeuralNetworksModel_addOperand}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksModel_addOperand},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksModel_addOperand)(
      ANeuralNetworksModel* model, const ANeuralNetworksOperandType* type);

  /**
   * SL Driver implementation of {@link ANeuralNetworksModel_addOperation}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksModel_addOperation},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksModel_addOperation)(ANeuralNetworksModel* model,
                                           ANeuralNetworksOperationType type,
                                           uint32_t inputCount,
                                           const uint32_t* inputs,
                                           uint32_t outputCount,
                                           const uint32_t* outputs);

  /**
   * SL Driver implementation of {@link ANeuralNetworksModel_create}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksModel_create},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksModel_create)(ANeuralNetworksModel** model);

  /**
   * SL Driver implementation of {@link ANeuralNetworksModel_finish}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksModel_finish},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksModel_finish)(ANeuralNetworksModel* model);

  /**
   * SL Driver implementation of {@link ANeuralNetworksModel_free}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksModel_free},
   * at the feature level of this NnApiSLDriver struct.
   */
  void (*ANeuralNetworksModel_free)(ANeuralNetworksModel* model);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksModel_getExtensionOperandType}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link ANeuralNetworksModel_getExtensionOperandType},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksModel_getExtensionOperandType)(
      ANeuralNetworksModel* model, const char* extensionName,
      uint16_t operandCodeWithinExtension, int32_t* type);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksModel_getExtensionOperationType}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link ANeuralNetworksModel_getExtensionOperationType},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksModel_getExtensionOperationType)(
      ANeuralNetworksModel* model, const char* extensionName,
      uint16_t operationCodeWithinExtension,
      ANeuralNetworksOperationType* type);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksModel_getSupportedOperationsForDevices}. Behavior,
   * arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksModel_getSupportedOperationsForDevices},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksModel_getSupportedOperationsForDevices)(
      const ANeuralNetworksModel* model,
      const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
      bool* supportedOps);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksModel_identifyInputsAndOutputs}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link ANeuralNetworksModel_identifyInputsAndOutputs},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksModel_identifyInputsAndOutputs)(
      ANeuralNetworksModel* model, uint32_t inputCount, const uint32_t* inputs,
      uint32_t outputCount, const uint32_t* outputs);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksModel_relaxComputationFloat32toFloat16}. Behavior,
   * arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksModel_relaxComputationFloat32toFloat16},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksModel_relaxComputationFloat32toFloat16)(
      ANeuralNetworksModel* model, bool allow);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksModel_setOperandExtensionData}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link ANeuralNetworksModel_setOperandExtensionData},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksModel_setOperandExtensionData)(
      ANeuralNetworksModel* model, int32_t index, const void* data,
      size_t length);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksModel_setOperandSymmPerChannelQuantParams}. Behavior,
   * arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksModel_setOperandSymmPerChannelQuantParams},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksModel_setOperandSymmPerChannelQuantParams)(
      ANeuralNetworksModel* model, int32_t index,
      const ANeuralNetworksSymmPerChannelQuantParams* channelQuant);

  /**
   * SL Driver implementation of {@link ANeuralNetworksModel_setOperandValue}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksModel_setOperandValue},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksModel_setOperandValue)(ANeuralNetworksModel* model,
                                              int32_t index, const void* buffer,
                                              size_t length);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksModel_setOperandValueFromMemory}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link ANeuralNetworksModel_setOperandValueFromMemory},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksModel_setOperandValueFromMemory)(
      ANeuralNetworksModel* model, int32_t index,
      const ANeuralNetworksMemory* memory, size_t offset, size_t length);

  /**
   * SL Driver implementation of {@link
   * ANeuralNetworksModel_setOperandValueFromModel}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link ANeuralNetworksModel_setOperandValueFromModel},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworksModel_setOperandValueFromModel)(
      ANeuralNetworksModel* model, int32_t index,
      const ANeuralNetworksModel* value);

  /**
   * SL Driver implementation of {@link ANeuralNetworks_getDefaultLoopTimeout}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworks_getDefaultLoopTimeout},
   * at the feature level of this NnApiSLDriver struct.
   */
  uint64_t (*ANeuralNetworks_getDefaultLoopTimeout)();

  /**
   * SL Driver implementation of {@link ANeuralNetworks_getDevice}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworks_getDevice},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworks_getDevice)(uint32_t devIndex,
                                   ANeuralNetworksDevice** device);

  /**
   * SL Driver implementation of {@link ANeuralNetworks_getDeviceCount}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworks_getDeviceCount},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*ANeuralNetworks_getDeviceCount)(uint32_t* numDevices);

  /**
   * SL Driver implementation of {@link ANeuralNetworks_getMaximumLoopTimeout}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworks_getMaximumLoopTimeout},
   * at the feature level of this NnApiSLDriver struct.
   */
  uint64_t (*ANeuralNetworks_getMaximumLoopTimeout)();

  /**
   * SL Driver implementation of {@link ANeuralNetworks_getRuntimeFeatureLevel}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link ANeuralNetworks_getRuntimeFeatureLevel},
   * at the feature level of this NnApiSLDriver struct.
   */
  int64_t (*ANeuralNetworks_getRuntimeFeatureLevel)();

  /**
   * SL Driver implementation of a function similar to
   * {@link ANeuralNetworksCompilation_setCaching} that takes file descriptors
   * instead of a cache directory.
   * Behavior and outputs match NNAPI Runtime function
   * {@link ANeuralNetworksCompilation_setCaching},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*SL_ANeuralNetworksCompilation_setCachingFromFds)(
      ANeuralNetworksCompilation* compilation, const int* modelCacheFds,
      const uint32_t numModelCacheFiles, const int* dataCacheFds,
      const uint32_t numDataCacheFiles, const uint8_t* token);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDevice_getNumberOfCacheFilesNeeded}. Behavior, arguments,
   * and outputs match NNAPI Runtime function
   * {@link SL_ANeuralNetworksDevice_getNumberOfCacheFilesNeeded},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*SL_ANeuralNetworksDevice_getNumberOfCacheFilesNeeded)(
      const ANeuralNetworksDevice* device, uint32_t* numModelCacheFiles,
      uint32_t* numDataCacheFiles);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDevice_getPerformanceInfo}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link SL_ANeuralNetworksDevice_getPerformanceInfo},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*SL_ANeuralNetworksDevice_getPerformanceInfo)(
      const ANeuralNetworksDevice* device, int32_t performanceInfoKind,
      SL_ANeuralNetworksPerformanceInfo* performanceInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDevice_forEachOperandPerformanceInfo}. Behavior,
   * arguments, and outputs match NNAPI Runtime function
   * {@link SL_ANeuralNetworksDevice_forEachOperandPerformanceInfo},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*SL_ANeuralNetworksDevice_forEachOperandPerformanceInfo)(
      const ANeuralNetworksDevice* device, void* context,
      void (*callback)(SL_ANeuralNetworksOperandPerformanceInfo, void*));

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDevice_getVendorExtensionCount}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link SL_ANeuralNetworksDevice_getVendorExtensionCount},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*SL_ANeuralNetworksDevice_getVendorExtensionCount)(
      const ANeuralNetworksDevice* device, uint32_t* vendorExtensionCount);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDevice_getVendorExtensionName}. Behavior, arguments, and
   * outputs match NNAPI Runtime function
   * {@link SL_ANeuralNetworksDevice_getVendorExtensionName},
   * at the feature level of this NnApiSLDriver struct.
   */
  int (*SL_ANeuralNetworksDevice_getVendorExtensionName)(
      const ANeuralNetworksDevice* device, uint32_t vendorExtensionIndex,
      const char** extensionName);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDevice_forEachVendorExtensionOperandTypeInformation}.
   * Behavior, arguments, and outputs match NNAPI Runtime function
   * {@link
   * SL_ANeuralNetworksDevice_forEachVendorExtensionOperandTypeInformation}, at
   * the feature level of this NnApiSLDriver struct.
   */
  int (*SL_ANeuralNetworksDevice_forEachVendorExtensionOperandTypeInformation)(
      const ANeuralNetworksDevice* device, uint32_t vendorExtensionIndex,
      void* context,
      void (*callback)(SL_ANeuralNetworksExtensionOperandTypeInformation,
                       void*));

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getSessionId}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getSessionId}, at the feature
   * level of this NnApiSLDriver struct.
   */
  int32_t (*SL_ANeuralNetworksDiagnosticCompilationInfo_getSessionId)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getNnApiVersion}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getNnApiVersion}, at the
   * feature level of this NnApiSLDriver struct.
   */
  int64_t (*SL_ANeuralNetworksDiagnosticCompilationInfo_getNnApiVersion)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getModelArchHash}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getModelArchHash}, at the
   * feature level of this NnApiSLDriver struct.
   */
  const uint8_t* (
      *SL_ANeuralNetworksDiagnosticCompilationInfo_getModelArchHash)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getDeviceIds}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getDeviceIds}, at the feature
   * level of this NnApiSLDriver struct.
   */
  const char* (*SL_ANeuralNetworksDiagnosticCompilationInfo_getDeviceIds)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getErrorCode}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getErrorCode}, at the feature
   * level of this NnApiSLDriver struct.
   */
  int32_t (*SL_ANeuralNetworksDiagnosticCompilationInfo_getErrorCode)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getInputDataClass}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getInputDataClass}, at the
   * feature level of this NnApiSLDriver struct.
   */
  ANeuralNetworksDiagnosticDataClass (
      *SL_ANeuralNetworksDiagnosticCompilationInfo_getInputDataClass)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getOutputDataClass}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getOutputDataClass}, at the
   * feature level of this NnApiSLDriver struct.
   */
  ANeuralNetworksDiagnosticDataClass (
      *SL_ANeuralNetworksDiagnosticCompilationInfo_getOutputDataClass)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getCompilationTimeNanos}.
   * Behavior, arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_getCompilationTimeNanos}, at
   * the feature level of this NnApiSLDriver struct.
   */
  uint64_t (
      *SL_ANeuralNetworksDiagnosticCompilationInfo_getCompilationTimeNanos)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_isCachingEnabled}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_isCachingEnabled}, at the
   * feature level of this NnApiSLDriver struct.
   */
  bool (*SL_ANeuralNetworksDiagnosticCompilationInfo_isCachingEnabled)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_isControlFlowUsed}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_isControlFlowUsed}, at the
   * feature level of this NnApiSLDriver struct.
   */
  bool (*SL_ANeuralNetworksDiagnosticCompilationInfo_isControlFlowUsed)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_areDynamicTensorsUsed}.
   * Behavior, arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticCompilationInfo_areDynamicTensorsUsed}, at the
   * feature level of this NnApiSLDriver struct.
   */
  bool (*SL_ANeuralNetworksDiagnosticCompilationInfo_areDynamicTensorsUsed)(
      const ANeuralNetworksDiagnosticCompilationInfo*
          diagnosticCompilationInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getSessionId}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getSessionId}, at the feature
   * level of this NnApiSLDriver struct.
   */
  int32_t (*SL_ANeuralNetworksDiagnosticExecutionInfo_getSessionId)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getNnApiVersion}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getNnApiVersion}, at the feature
   * level of this NnApiSLDriver struct.
   */
  int64_t (*SL_ANeuralNetworksDiagnosticExecutionInfo_getNnApiVersion)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getModelArchHash}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getModelArchHash}, at the feature
   * level of this NnApiSLDriver struct.
   */
  const uint8_t* (*SL_ANeuralNetworksDiagnosticExecutionInfo_getModelArchHash)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getDeviceIds}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getDeviceIds}, at the feature
   * level of this NnApiSLDriver struct.
   */
  const char* (*SL_ANeuralNetworksDiagnosticExecutionInfo_getDeviceIds)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getExecutionMode}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getExecutionMode}, at the feature
   * level of this NnApiSLDriver struct.
   */
  ANeuralNetworksDiagnosticExecutionMode (
      *SL_ANeuralNetworksDiagnosticExecutionInfo_getExecutionMode)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getInputDataClass}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getInputDataClass}, at the
   * feature level of this NnApiSLDriver struct.
   */
  ANeuralNetworksDiagnosticDataClass (
      *SL_ANeuralNetworksDiagnosticExecutionInfo_getInputDataClass)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getOutputDataClass}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getOutputDataClass}, at the
   * feature level of this NnApiSLDriver struct.
   */
  ANeuralNetworksDiagnosticDataClass (
      *SL_ANeuralNetworksDiagnosticExecutionInfo_getOutputDataClass)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getErrorCode}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getErrorCode}, at the feature
   * level of this NnApiSLDriver struct.
   */
  uint32_t (*SL_ANeuralNetworksDiagnosticExecutionInfo_getErrorCode)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getRuntimeExecutionTimeNanos}.
   * Behavior, arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getRuntimeExecutionTimeNanos}, at
   * the feature level of this NnApiSLDriver struct.
   */
  uint64_t (
      *SL_ANeuralNetworksDiagnosticExecutionInfo_getRuntimeExecutionTimeNanos)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getDriverExecutionTimeNanos}.
   * Behavior, arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getDriverExecutionTimeNanos}, at
   * the feature level of this NnApiSLDriver struct.
   */
  uint64_t (
      *SL_ANeuralNetworksDiagnosticExecutionInfo_getDriverExecutionTimeNanos)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getHardwareExecutionTimeNanos}.
   * Behavior, arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_getHardwareExecutionTimeNanos},
   * at the feature level of this NnApiSLDriver struct.
   */
  uint64_t (
      *SL_ANeuralNetworksDiagnosticExecutionInfo_getHardwareExecutionTimeNanos)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_isCachingEnabled}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_isCachingEnabled}, at the feature
   * level of this NnApiSLDriver struct.
   */
  bool (*SL_ANeuralNetworksDiagnosticExecutionInfo_isCachingEnabled)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_isControlFlowUsed}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_isControlFlowUsed}, at the
   * feature level of this NnApiSLDriver struct.
   */
  bool (*SL_ANeuralNetworksDiagnosticExecutionInfo_isControlFlowUsed)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_areDynamicTensorsUsed}. Behavior,
   * arguments, and outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnosticExecutionInfo_areDynamicTensorsUsed}, at the
   * feature level of this NnApiSLDriver struct.
   */
  bool (*SL_ANeuralNetworksDiagnosticExecutionInfo_areDynamicTensorsUsed)(
      const ANeuralNetworksDiagnosticExecutionInfo* diagnosticExecutionInfo);

  /**
   * SL Driver implementation of {@link
   * SL_ANeuralNetworksDiagnostic_registerCallbacks}. Behavior, arguments, and
   * outputs match NNAPI Runtime function {@link
   * SL_ANeuralNetworksDiagnostic_registerCallbacks}, at the feature level of
   * this NnApiSLDriver struct.
   */
  void (*SL_ANeuralNetworksDiagnostic_registerCallbacks)(
      ANeuralNetworksDiagnosticCompilationFinishedCallback compilationCallback,
      ANeuralNetworksDiagnosticExecutionFinishedCallback executionCallback,
      void* callbackContext);
} NnApiSLDriverImplFL5;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_NNAPI_SL_PUBLIC_NEURAL_NETWORKS_SUPPORT_LIBRARY_IMPL_H_
