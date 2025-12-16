/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_INTERFACE_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_INTERFACE_H_

#include <cstddef>
#include <cstdint>

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_profiler_target.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_target.h"
#include "third_party/gpus/cuda/include/cuda.h"

// These types are only available starting from CUPTI 2024.3, therefore we
// forward declare them here, so that we can still compile this header against
// older CUDA versions.
extern "C" {
struct CUpti_Profiler_Host_Initialize_Params;
struct CUpti_Profiler_Host_Deinitialize_Params;
struct CUpti_Profiler_Host_GetSupportedChips_Params;
struct CUpti_Profiler_Host_GetBaseMetrics_Params;
struct CUpti_Profiler_Host_GetSubMetrics_Params;
struct CUpti_Profiler_Host_GetMetricProperties_Params;
struct CUpti_Profiler_Host_GetRangeName_Params;
struct CUpti_Profiler_Host_EvaluateToGpuValues_Params;
struct CUpti_Profiler_Host_ConfigAddMetrics_Params;
struct CUpti_Profiler_Host_GetConfigImageSize_Params;
struct CUpti_Profiler_Host_GetConfigImage_Params;
struct CUpti_Profiler_Host_GetNumOfPasses_Params;
struct CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params;

struct CUpti_PmSampling_SetConfig_Params;
struct CUpti_PmSampling_Enable_Params;
struct CUpti_PmSampling_Disable_Params;
struct CUpti_PmSampling_Start_Params;
struct CUpti_PmSampling_Stop_Params;
struct CUpti_PmSampling_DecodeData_Params;
struct CUpti_PmSampling_GetCounterAvailability_Params;
struct CUpti_PmSampling_GetCounterDataSize_Params;
struct CUpti_PmSampling_CounterDataImage_Initialize_Params;
struct CUpti_PmSampling_GetCounterDataInfo_Params;
struct CUpti_PmSampling_CounterData_GetSampleInfo_Params;
}

namespace xla {
namespace profiler {

// Provides a wrapper interface to every single CUPTI API function. This class
// is needed to create an easy mock object for CUPTI API calls. All member
// functions are defined in the following order: activity related APIs, callback
// related APIs, Event APIs, and metric APIs. Within each category, we follow
// the order in the original CUPTI documentation.
class CuptiInterface {
 public:
  CuptiInterface() {}

  virtual ~CuptiInterface() {}

  // CUPTI activity API
  virtual CUptiResult ActivityDisable(CUpti_ActivityKind kind) = 0;

  virtual CUptiResult ActivityEnable(CUpti_ActivityKind kind) = 0;

  virtual CUptiResult ActivityFlushAll(uint32_t flag) = 0;

  virtual CUptiResult ActivityGetNextRecord(uint8_t* buffer,
                                            size_t valid_buffer_size_bytes,
                                            CUpti_Activity** record) = 0;

  virtual CUptiResult ActivityGetNumDroppedRecords(CUcontext context,
                                                   uint32_t stream_id,
                                                   size_t* dropped) = 0;

  virtual CUptiResult ActivityConfigureUnifiedMemoryCounter(
      CUpti_ActivityUnifiedMemoryCounterConfig* config, uint32_t count) = 0;

  virtual CUptiResult ActivityRegisterCallbacks(
      CUpti_BuffersCallbackRequestFunc func_buffer_requested,
      CUpti_BuffersCallbackCompleteFunc func_buffer_completed) = 0;

  virtual CUptiResult ActivityUsePerThreadBuffer() = 0;

  virtual CUptiResult SetActivityFlushPeriod(uint32_t period_ms) = 0;

  virtual CUptiResult GetDeviceId(CUcontext context, uint32_t* deviceId) = 0;

  virtual CUptiResult GetTimestamp(uint64_t* timestamp) = 0;

  virtual CUptiResult Finalize() = 0;

  // CUPTI callback API
  virtual CUptiResult EnableCallback(uint32_t enable,
                                     CUpti_SubscriberHandle subscriber,
                                     CUpti_CallbackDomain domain,
                                     CUpti_CallbackId cbid) = 0;

  virtual CUptiResult EnableDomain(uint32_t enable,
                                   CUpti_SubscriberHandle subscriber,
                                   CUpti_CallbackDomain domain) = 0;

  virtual CUptiResult Subscribe(CUpti_SubscriberHandle* subscriber,
                                CUpti_CallbackFunc callback,
                                void* userdata) = 0;

  virtual CUptiResult Unsubscribe(CUpti_SubscriberHandle subscriber) = 0;

  virtual CUptiResult GetResultString(CUptiResult result, const char** str) = 0;

  virtual CUptiResult GetContextId(CUcontext context, uint32_t* context_id) = 0;

  virtual CUptiResult GetStreamIdEx(CUcontext context, CUstream stream,
                                    uint8_t per_thread_stream,
                                    uint32_t* stream_id) = 0;

  virtual CUptiResult GetGraphId(CUgraph graph, uint32_t* graph_id) = 0;

  // Gets the graph node id.
  virtual CUptiResult GetGraphNodeId(CUgraphNode node, uint64_t* nodeId) = 0;

  virtual CUptiResult GetGraphExecId(CUgraphExec graph_exec,
                                     uint32_t* graph_id) = 0;

  virtual CUptiResult SetThreadIdType(CUpti_ActivityThreadIdType type) = 0;

  virtual CUptiResult ActivityEnableHWTrace(bool enable) = 0;

  // Functions related to profiling APIs - range profiling, PC sampling, PM
  // sampling Equivalent functions are declared in
  // cuda/extras/CUPTI/include/cupti_profiler_host.h and
  // cuda/extras/CUPTI/include/cupti_profiler_target.h
  virtual CUptiResult ProfilerHostInitialize(
      CUpti_Profiler_Host_Initialize_Params* params) = 0;
  virtual CUptiResult ProfilerHostDeinitialize(
      CUpti_Profiler_Host_Deinitialize_Params* params) = 0;
  virtual CUptiResult ProfilerHostGetSupportedChips(
      CUpti_Profiler_Host_GetSupportedChips_Params* params) = 0;
  virtual CUptiResult ProfilerHostGetBaseMetrics(
      CUpti_Profiler_Host_GetBaseMetrics_Params* params) = 0;
  virtual CUptiResult ProfilerHostGetSubMetrics(
      CUpti_Profiler_Host_GetSubMetrics_Params* params) = 0;
  virtual CUptiResult ProfilerHostGetMetricProperties(
      CUpti_Profiler_Host_GetMetricProperties_Params* params) = 0;
  virtual CUptiResult ProfilerHostGetRangeName(
      CUpti_Profiler_Host_GetRangeName_Params* params) = 0;
  virtual CUptiResult ProfilerHostEvaluateToGpuValues(
      CUpti_Profiler_Host_EvaluateToGpuValues_Params* params) = 0;
  virtual CUptiResult ProfilerHostConfigAddMetrics(
      CUpti_Profiler_Host_ConfigAddMetrics_Params* params) = 0;
  virtual CUptiResult ProfilerHostGetConfigImageSize(
      CUpti_Profiler_Host_GetConfigImageSize_Params* params) = 0;
  virtual CUptiResult ProfilerHostGetConfigImage(
      CUpti_Profiler_Host_GetConfigImage_Params* params) = 0;
  virtual CUptiResult ProfilerHostGetNumOfPasses(
      CUpti_Profiler_Host_GetNumOfPasses_Params* params) = 0;
  virtual CUptiResult ProfilerHostGetMaxNumHardwareMetricsPerPass(
      CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params* params) = 0;

  virtual CUptiResult ProfilerInitialize(
      CUpti_Profiler_Initialize_Params* params) = 0;
  virtual CUptiResult ProfilerDeInitialize(
      CUpti_Profiler_DeInitialize_Params* params) = 0;
  virtual CUptiResult ProfilerCounterDataImageCalculateSize(
      CUpti_Profiler_CounterDataImage_CalculateSize_Params* params) = 0;
  virtual CUptiResult ProfilerCounterDataImageInitialize(
      CUpti_Profiler_CounterDataImage_Initialize_Params* params) = 0;
  virtual CUptiResult ProfilerCounterDataImageCalculateScratchBufferSize(
      CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params*
          params) = 0;
  virtual CUptiResult ProfilerCounterDataImageInitializeScratchBuffer(
      CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params*
          params) = 0;
  virtual CUptiResult ProfilerBeginSession(
      CUpti_Profiler_BeginSession_Params* params) = 0;
  virtual CUptiResult ProfilerEndSession(
      CUpti_Profiler_EndSession_Params* params) = 0;
  virtual CUptiResult ProfilerSetConfig(
      CUpti_Profiler_SetConfig_Params* params) = 0;
  virtual CUptiResult ProfilerUnsetConfig(
      CUpti_Profiler_UnsetConfig_Params* params) = 0;
  virtual CUptiResult ProfilerBeginPass(
      CUpti_Profiler_BeginPass_Params* params) = 0;
  virtual CUptiResult ProfilerEndPass(
      CUpti_Profiler_EndPass_Params* params) = 0;
  virtual CUptiResult ProfilerEnableProfiling(
      CUpti_Profiler_EnableProfiling_Params* params) = 0;
  virtual CUptiResult ProfilerDisableProfiling(
      CUpti_Profiler_DisableProfiling_Params* params) = 0;
  virtual CUptiResult ProfilerIsPassCollected(
      CUpti_Profiler_IsPassCollected_Params* params) = 0;
  virtual CUptiResult ProfilerFlushCounterData(
      CUpti_Profiler_FlushCounterData_Params* params) = 0;
  virtual CUptiResult ProfilerPushRange(
      CUpti_Profiler_PushRange_Params* params) = 0;
  virtual CUptiResult ProfilerPopRange(
      CUpti_Profiler_PopRange_Params* params) = 0;
  virtual CUptiResult ProfilerGetCounterAvailability(
      CUpti_Profiler_GetCounterAvailability_Params* params) = 0;
  virtual CUptiResult ProfilerDeviceSupported(
      CUpti_Profiler_DeviceSupported_Params* params) = 0;

  // PM sampling specific functions from
  // cuda/extras/CUPTI/include/cupti_pmsampling.h
  virtual CUptiResult PmSamplingSetConfig(
      CUpti_PmSampling_SetConfig_Params* params) = 0;
  virtual CUptiResult PmSamplingEnable(
      CUpti_PmSampling_Enable_Params* params) = 0;
  virtual CUptiResult PmSamplingDisable(
      CUpti_PmSampling_Disable_Params* params) = 0;
  virtual CUptiResult PmSamplingStart(
      CUpti_PmSampling_Start_Params* params) = 0;
  virtual CUptiResult PmSamplingStop(CUpti_PmSampling_Stop_Params* params) = 0;
  virtual CUptiResult PmSamplingDecodeData(
      CUpti_PmSampling_DecodeData_Params* params) = 0;
  virtual CUptiResult PmSamplingGetCounterAvailability(
      CUpti_PmSampling_GetCounterAvailability_Params* params) = 0;
  virtual CUptiResult PmSamplingGetCounterDataSize(
      CUpti_PmSampling_GetCounterDataSize_Params* params) = 0;
  virtual CUptiResult PmSamplingCounterDataImageInitialize(
      CUpti_PmSampling_CounterDataImage_Initialize_Params* params) = 0;
  virtual CUptiResult PmSamplingGetCounterDataInfo(
      CUpti_PmSampling_GetCounterDataInfo_Params* params) = 0;
  virtual CUptiResult PmSamplingCounterDataGetSampleInfo(
      CUpti_PmSampling_CounterData_GetSampleInfo_Params* params) = 0;

  virtual CUptiResult DeviceGetChipName(
      CUpti_Device_GetChipName_Params* params) = 0;

  // Interface maintenance functions. Not directly related to CUPTI, but
  // required for implementing an error resilient layer over CUPTI API.

  // Performance any clean up work that is required each time profile session
  // is done. Therefore this can be called multiple times during process life
  // time.
  virtual void CleanUp() = 0;

  // Whether CUPTI API is currently disabled due to unrecoverable errors.
  // All subsequent calls will fail immediately without forwarding calls to
  // CUPTI library.
  virtual bool Disabled() const = 0;

 private:
  CuptiInterface(const CuptiInterface&) = delete;
  void operator=(const CuptiInterface&) = delete;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_INTERFACE_H_
