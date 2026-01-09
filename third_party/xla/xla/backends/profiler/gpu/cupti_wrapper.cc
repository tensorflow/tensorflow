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

#include "xla/backends/profiler/gpu/cupti_wrapper.h"

#include <cstdint>

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_profiler_target.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_result.h"
#include "third_party/gpus/cuda/include/cuda.h"

#if CUPTI_API_VERSION >= 24
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_pmsampling.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_profiler_host.h"
#endif

namespace xla {
namespace profiler {

CUptiResult CuptiWrapper::ActivityDisable(CUpti_ActivityKind kind) {
  return cuptiActivityDisable(kind);
}

CUptiResult CuptiWrapper::ActivityEnable(CUpti_ActivityKind kind) {
  return cuptiActivityEnable(kind);
}

CUptiResult CuptiWrapper::ActivityFlushAll(uint32_t flag) {
  return cuptiActivityFlushAll(flag);
}

CUptiResult CuptiWrapper::ActivityGetNextRecord(uint8_t* buffer,
                                                size_t valid_buffer_size_bytes,
                                                CUpti_Activity** record) {
  return cuptiActivityGetNextRecord(buffer, valid_buffer_size_bytes, record);
}

CUptiResult CuptiWrapper::ActivityGetNumDroppedRecords(CUcontext context,
                                                       uint32_t stream_id,
                                                       size_t* dropped) {
  return cuptiActivityGetNumDroppedRecords(context, stream_id, dropped);
}

CUptiResult CuptiWrapper::ActivityConfigureUnifiedMemoryCounter(
    CUpti_ActivityUnifiedMemoryCounterConfig* config, uint32_t count) {
  return cuptiActivityConfigureUnifiedMemoryCounter(config, count);
}

CUptiResult CuptiWrapper::ActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc func_buffer_requested,
    CUpti_BuffersCallbackCompleteFunc func_buffer_completed) {
  return cuptiActivityRegisterCallbacks(func_buffer_requested,
                                        func_buffer_completed);
}

CUptiResult CuptiWrapper::ActivityUsePerThreadBuffer() {
#if CUDA_VERSION >= 12030
  uint8_t use_per_thread_activity_buffer = 1;
  size_t value_size = sizeof(use_per_thread_activity_buffer);
  return cuptiActivitySetAttribute(
      CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER, &value_size,
      &use_per_thread_activity_buffer);
#else
  // cuptiActivitySetAttribute returns CUPTI_ERROR_INVALID_PARAMETER if invoked
  // with an invalid first parameter.
  return CUPTI_ERROR_INVALID_PARAMETER;
#endif
}

CUptiResult CuptiWrapper::SetActivityFlushPeriod(uint32_t period_ms) {
#if CUDA_VERSION >= 11010
  return cuptiActivityFlushPeriod(period_ms);
#else
  return CUPTI_ERROR_NOT_SUPPORTED;
#endif
}

CUptiResult CuptiWrapper::GetDeviceId(CUcontext context, uint32_t* deviceId) {
  return cuptiGetDeviceId(context, deviceId);
}

CUptiResult CuptiWrapper::GetTimestamp(uint64_t* timestamp) {
  return cuptiGetTimestamp(timestamp);
}

CUptiResult CuptiWrapper::Finalize() { return cuptiFinalize(); }

CUptiResult CuptiWrapper::EnableCallback(uint32_t enable,
                                         CUpti_SubscriberHandle subscriber,
                                         CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid) {
  return cuptiEnableCallback(enable, subscriber, domain, cbid);
}

CUptiResult CuptiWrapper::EnableDomain(uint32_t enable,
                                       CUpti_SubscriberHandle subscriber,
                                       CUpti_CallbackDomain domain) {
  return cuptiEnableDomain(enable, subscriber, domain);
}

CUptiResult CuptiWrapper::Subscribe(CUpti_SubscriberHandle* subscriber,
                                    CUpti_CallbackFunc callback,
                                    void* userdata) {
  return cuptiSubscribe(subscriber, callback, userdata);
}

CUptiResult CuptiWrapper::Unsubscribe(CUpti_SubscriberHandle subscriber) {
  return cuptiUnsubscribe(subscriber);
}

CUptiResult CuptiWrapper::GetResultString(CUptiResult result,
                                          const char** str) {
  return cuptiGetResultString(result, str);
}

CUptiResult CuptiWrapper::GetContextId(CUcontext context,
                                       uint32_t* context_id) {
  return cuptiGetContextId(context, context_id);
}

CUptiResult CuptiWrapper::GetGraphId(CUgraph graph, uint32_t* graph_id) {
#if CUDA_VERSION >= 11010
  return cuptiGetGraphId(graph, graph_id);
#else
  // Do not treat it as error if the interface is not available.
  if (graph_id) *graph_id = 0;
  return CUPTI_SUCCESS;
#endif
}

CUptiResult CuptiWrapper::GetGraphNodeId(CUgraphNode node, uint64_t* nodeId) {
#if CUDA_VERSION >= 11010
  return cuptiGetGraphNodeId(node, nodeId);
#else
  // Do not treat it as error if the interface is not available.
  return CUPTI_SUCCESS;
#endif
}

CUptiResult CuptiWrapper::GetGraphExecId(CUgraphExec graph_exec,
                                         uint32_t* graph_id) {
  // TODO: (b/350105610), Using cuptiGetGraphExecId() for CUDA 12.3 and later
  return GetGraphId(reinterpret_cast<CUgraph>(graph_exec), graph_id);
}

CUptiResult CuptiWrapper::SetThreadIdType(CUpti_ActivityThreadIdType type) {
  return cuptiSetThreadIdType(type);
}

CUptiResult CuptiWrapper::GetStreamIdEx(CUcontext context, CUstream stream,
                                        uint8_t per_thread_stream,
                                        uint32_t* stream_id) {
  return cuptiGetStreamIdEx(context, stream, per_thread_stream, stream_id);
}

extern "C" {
// Prototype for cuptiActivityEnableHWTrace if headers are not present.
[[gnu::weak]] CUptiResult cuptiActivityEnableHWTrace(uint8_t enable);
}  // extern "C"

CUptiResult CuptiWrapper::ActivityEnableHWTrace(bool enable) {
  return (cuptiActivityEnableHWTrace == nullptr)
             ? CUPTI_ERROR_NOT_SUPPORTED
             : cuptiActivityEnableHWTrace(enable ? 1 : 0);
}

// Prototypes for PM sampling and profiler host APIs if headers are not present
extern "C" {
[[gnu::weak]] CUptiResult cuptiProfilerHostInitialize(
    CUpti_Profiler_Host_Initialize_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerHostDeinitialize(
    CUpti_Profiler_Host_Deinitialize_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerHostGetSupportedChips(
    CUpti_Profiler_Host_GetSupportedChips_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerHostGetBaseMetrics(
    CUpti_Profiler_Host_GetBaseMetrics_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerHostGetSubMetrics(
    CUpti_Profiler_Host_GetSubMetrics_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerHostGetMetricProperties(
    CUpti_Profiler_Host_GetMetricProperties_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerHostGetRangeName(
    CUpti_Profiler_Host_GetRangeName_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerHostEvaluateToGpuValues(
    CUpti_Profiler_Host_EvaluateToGpuValues_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerHostConfigAddMetrics(
    CUpti_Profiler_Host_ConfigAddMetrics_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerHostGetConfigImageSize(
    CUpti_Profiler_Host_GetConfigImageSize_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerHostGetConfigImage(
    CUpti_Profiler_Host_GetConfigImage_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerHostGetNumOfPasses(
    CUpti_Profiler_Host_GetNumOfPasses_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerHostGetMaxNumHardwareMetricsPerPass(
    CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerInitialize(
    CUpti_Profiler_Initialize_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerDeInitialize(
    CUpti_Profiler_DeInitialize_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerCounterDataImageCalculateSize(
    CUpti_Profiler_CounterDataImage_CalculateSize_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerCounterDataImageInitialize(
    CUpti_Profiler_CounterDataImage_Initialize_Params* params);
[[gnu::weak]] CUptiResult
cuptiProfilerCounterDataImageCalculateScratchBufferSize(
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerCounterDataImageInitializeScratchBuffer(
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerBeginSession(
    CUpti_Profiler_BeginSession_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerEndSession(
    CUpti_Profiler_EndSession_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerSetConfig(
    CUpti_Profiler_SetConfig_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerBeginPass(
    CUpti_Profiler_BeginPass_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerEndPass(
    CUpti_Profiler_EndPass_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerEnableProfiling(
    CUpti_Profiler_EnableProfiling_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerDisableProfiling(
    CUpti_Profiler_DisableProfiling_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerIsPassCollected(
    CUpti_Profiler_IsPassCollected_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerFlushCounterData(
    CUpti_Profiler_FlushCounterData_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerPushRange(
    CUpti_Profiler_PushRange_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerPopRange(
    CUpti_Profiler_PopRange_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerGetCounterAvailability(
    CUpti_Profiler_GetCounterAvailability_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerDeviceSupported(
    CUpti_Profiler_DeviceSupported_Params* params);
[[gnu::weak]] CUptiResult cuptiProfilerUnsetConfig(
    CUpti_Profiler_UnsetConfig_Params* params);

[[gnu::weak]] CUptiResult cuptiPmSamplingSetConfig(
    CUpti_PmSampling_SetConfig_Params* params);
[[gnu::weak]] CUptiResult cuptiPmSamplingEnable(
    CUpti_PmSampling_Enable_Params* params);
[[gnu::weak]] CUptiResult cuptiPmSamplingDisable(
    CUpti_PmSampling_Disable_Params* params);
[[gnu::weak]] CUptiResult cuptiPmSamplingStart(
    CUpti_PmSampling_Start_Params* params);
[[gnu::weak]] CUptiResult cuptiPmSamplingStop(
    CUpti_PmSampling_Stop_Params* params);
[[gnu::weak]] CUptiResult cuptiPmSamplingDecodeData(
    CUpti_PmSampling_DecodeData_Params* params);
[[gnu::weak]] CUptiResult cuptiPmSamplingGetCounterAvailability(
    CUpti_PmSampling_GetCounterAvailability_Params* params);
[[gnu::weak]] CUptiResult cuptiPmSamplingGetCounterDataSize(
    CUpti_PmSampling_GetCounterDataSize_Params* params);
[[gnu::weak]] CUptiResult cuptiPmSamplingCounterDataImageInitialize(
    CUpti_PmSampling_CounterDataImage_Initialize_Params* params);
[[gnu::weak]] CUptiResult cuptiPmSamplingGetCounterDataInfo(
    CUpti_PmSampling_GetCounterDataInfo_Params* params);
[[gnu::weak]] CUptiResult cuptiPmSamplingCounterDataGetSampleInfo(
    CUpti_PmSampling_CounterData_GetSampleInfo_Params* params);
}

// Profiler Host APIs
CUptiResult CuptiWrapper::ProfilerHostInitialize(
    CUpti_Profiler_Host_Initialize_Params* params) {
  if (cuptiProfilerHostInitialize != nullptr) {
    return cuptiProfilerHostInitialize(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerHostDeinitialize(
    CUpti_Profiler_Host_Deinitialize_Params* params) {
  if (cuptiProfilerHostDeinitialize != nullptr) {
    return cuptiProfilerHostDeinitialize(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerHostGetSupportedChips(
    CUpti_Profiler_Host_GetSupportedChips_Params* params) {
  if (cuptiProfilerHostGetSupportedChips != nullptr) {
    return cuptiProfilerHostGetSupportedChips(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerHostGetBaseMetrics(
    CUpti_Profiler_Host_GetBaseMetrics_Params* params) {
  if (cuptiProfilerHostGetBaseMetrics != nullptr) {
    return cuptiProfilerHostGetBaseMetrics(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerHostGetSubMetrics(
    CUpti_Profiler_Host_GetSubMetrics_Params* params) {
  if (cuptiProfilerHostGetSubMetrics != nullptr) {
    return cuptiProfilerHostGetSubMetrics(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerHostGetMetricProperties(
    CUpti_Profiler_Host_GetMetricProperties_Params* params) {
  if (cuptiProfilerHostGetMetricProperties != nullptr) {
    return cuptiProfilerHostGetMetricProperties(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerHostGetRangeName(
    CUpti_Profiler_Host_GetRangeName_Params* params) {
  if (cuptiProfilerHostGetRangeName != nullptr) {
    return cuptiProfilerHostGetRangeName(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerHostEvaluateToGpuValues(
    CUpti_Profiler_Host_EvaluateToGpuValues_Params* params) {
  if (cuptiProfilerHostEvaluateToGpuValues != nullptr) {
    return cuptiProfilerHostEvaluateToGpuValues(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerHostConfigAddMetrics(
    CUpti_Profiler_Host_ConfigAddMetrics_Params* params) {
  if (cuptiProfilerHostConfigAddMetrics != nullptr) {
    return cuptiProfilerHostConfigAddMetrics(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerHostGetConfigImageSize(
    CUpti_Profiler_Host_GetConfigImageSize_Params* params) {
  if (cuptiProfilerHostGetConfigImageSize != nullptr) {
    return cuptiProfilerHostGetConfigImageSize(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerHostGetConfigImage(
    CUpti_Profiler_Host_GetConfigImage_Params* params) {
  if (cuptiProfilerHostGetConfigImage != nullptr) {
    return cuptiProfilerHostGetConfigImage(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerHostGetNumOfPasses(
    CUpti_Profiler_Host_GetNumOfPasses_Params* params) {
  if (cuptiProfilerHostGetNumOfPasses != nullptr) {
    return cuptiProfilerHostGetNumOfPasses(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerHostGetMaxNumHardwareMetricsPerPass(
    CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params* params) {
  if (cuptiProfilerHostGetMaxNumHardwareMetricsPerPass != nullptr) {
    return cuptiProfilerHostGetMaxNumHardwareMetricsPerPass(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

// Profiler Target APIs
CUptiResult CuptiWrapper::ProfilerInitialize(
    CUpti_Profiler_Initialize_Params* params) {
  if (cuptiProfilerInitialize != nullptr) {
    return cuptiProfilerInitialize(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerDeInitialize(
    CUpti_Profiler_DeInitialize_Params* params) {
  if (cuptiProfilerDeInitialize != nullptr) {
    return cuptiProfilerDeInitialize(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerCounterDataImageCalculateSize(
    CUpti_Profiler_CounterDataImage_CalculateSize_Params* params) {
  if (cuptiProfilerCounterDataImageCalculateSize != nullptr) {
    return cuptiProfilerCounterDataImageCalculateSize(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerCounterDataImageInitialize(
    CUpti_Profiler_CounterDataImage_Initialize_Params* params) {
  if (cuptiProfilerCounterDataImageInitialize != nullptr) {
    return cuptiProfilerCounterDataImageInitialize(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerCounterDataImageCalculateScratchBufferSize(
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params* params) {
  if (cuptiProfilerCounterDataImageCalculateScratchBufferSize != nullptr) {
    return cuptiProfilerCounterDataImageCalculateScratchBufferSize(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerCounterDataImageInitializeScratchBuffer(
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* params) {
  if (cuptiProfilerCounterDataImageInitializeScratchBuffer != nullptr) {
    return cuptiProfilerCounterDataImageInitializeScratchBuffer(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerBeginSession(
    CUpti_Profiler_BeginSession_Params* params) {
  if (cuptiProfilerBeginSession != nullptr) {
    return cuptiProfilerBeginSession(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerEndSession(
    CUpti_Profiler_EndSession_Params* params) {
  if (cuptiProfilerEndSession != nullptr) {
    return cuptiProfilerEndSession(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerSetConfig(
    CUpti_Profiler_SetConfig_Params* params) {
  if (cuptiProfilerSetConfig != nullptr) {
    return cuptiProfilerSetConfig(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerUnsetConfig(
    CUpti_Profiler_UnsetConfig_Params* params) {
  if (cuptiProfilerUnsetConfig != nullptr) {
    return cuptiProfilerUnsetConfig(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerBeginPass(
    CUpti_Profiler_BeginPass_Params* params) {
  if (cuptiProfilerBeginPass != nullptr) {
    return cuptiProfilerBeginPass(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerEndPass(
    CUpti_Profiler_EndPass_Params* params) {
  if (cuptiProfilerEndPass != nullptr) {
    return cuptiProfilerEndPass(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerEnableProfiling(
    CUpti_Profiler_EnableProfiling_Params* params) {
  if (cuptiProfilerEnableProfiling != nullptr) {
    return cuptiProfilerEnableProfiling(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerDisableProfiling(
    CUpti_Profiler_DisableProfiling_Params* params) {
  if (cuptiProfilerDisableProfiling != nullptr) {
    return cuptiProfilerDisableProfiling(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerIsPassCollected(
    CUpti_Profiler_IsPassCollected_Params* params) {
  if (cuptiProfilerIsPassCollected != nullptr) {
    return cuptiProfilerIsPassCollected(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerFlushCounterData(
    CUpti_Profiler_FlushCounterData_Params* params) {
  if (cuptiProfilerFlushCounterData != nullptr) {
    return cuptiProfilerFlushCounterData(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerPushRange(
    CUpti_Profiler_PushRange_Params* params) {
  if (cuptiProfilerPushRange != nullptr) {
    return cuptiProfilerPushRange(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerPopRange(
    CUpti_Profiler_PopRange_Params* params) {
  if (cuptiProfilerPopRange != nullptr) {
    return cuptiProfilerPopRange(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerGetCounterAvailability(
    CUpti_Profiler_GetCounterAvailability_Params* params) {
  if (cuptiProfilerGetCounterAvailability != nullptr) {
    return cuptiProfilerGetCounterAvailability(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::ProfilerDeviceSupported(
    CUpti_Profiler_DeviceSupported_Params* params) {
  if (cuptiProfilerDeviceSupported != nullptr) {
    return cuptiProfilerDeviceSupported(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::PmSamplingSetConfig(
    CUpti_PmSampling_SetConfig_Params* params) {
  if (cuptiPmSamplingSetConfig != nullptr) {
    return cuptiPmSamplingSetConfig(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::PmSamplingEnable(
    CUpti_PmSampling_Enable_Params* params) {
  if (cuptiPmSamplingEnable != nullptr) {
    return cuptiPmSamplingEnable(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::PmSamplingDisable(
    CUpti_PmSampling_Disable_Params* params) {
  if (cuptiPmSamplingDisable != nullptr) {
    return cuptiPmSamplingDisable(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::PmSamplingStart(
    CUpti_PmSampling_Start_Params* params) {
  if (cuptiPmSamplingStart != nullptr) {
    return cuptiPmSamplingStart(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::PmSamplingStop(CUpti_PmSampling_Stop_Params* params) {
  if (cuptiPmSamplingStop != nullptr) {
    return cuptiPmSamplingStop(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::PmSamplingDecodeData(
    CUpti_PmSampling_DecodeData_Params* params) {
  if (cuptiPmSamplingDecodeData != nullptr) {
    return cuptiPmSamplingDecodeData(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::PmSamplingGetCounterAvailability(
    CUpti_PmSampling_GetCounterAvailability_Params* params) {
  if (cuptiPmSamplingGetCounterAvailability != nullptr) {
    return cuptiPmSamplingGetCounterAvailability(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::PmSamplingGetCounterDataSize(
    CUpti_PmSampling_GetCounterDataSize_Params* params) {
  if (cuptiPmSamplingGetCounterDataSize != nullptr) {
    return cuptiPmSamplingGetCounterDataSize(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::PmSamplingCounterDataImageInitialize(
    CUpti_PmSampling_CounterDataImage_Initialize_Params* params) {
  if (cuptiPmSamplingCounterDataImageInitialize != nullptr) {
    return cuptiPmSamplingCounterDataImageInitialize(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::PmSamplingGetCounterDataInfo(
    CUpti_PmSampling_GetCounterDataInfo_Params* params) {
  if (cuptiPmSamplingGetCounterDataInfo != nullptr) {
    return cuptiPmSamplingGetCounterDataInfo(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

CUptiResult CuptiWrapper::PmSamplingCounterDataGetSampleInfo(
    CUpti_PmSampling_CounterData_GetSampleInfo_Params* params) {
  if (cuptiPmSamplingCounterDataGetSampleInfo != nullptr) {
    return cuptiPmSamplingCounterDataGetSampleInfo(params);
  } else {
    return CUPTI_ERROR_NOT_SUPPORTED;
  }
}

// Restore disabled warnings (required because CUPTI declares non-weak functions
// which then cause -Wtautological-pointer-compare to fail in the above code)
#if CUPTI_PM_SAMPLING_SUPPORTED
#pragma clang diagnostic pop
#endif

CUptiResult CuptiWrapper::DeviceGetChipName(
    CUpti_Device_GetChipName_Params* params) {
  return cuptiDeviceGetChipName(params);
}

}  // namespace profiler
}  // namespace xla
