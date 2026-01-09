/* Copyright 2024 The OpenXLA Authors.

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

#include <cstddef>
#include <cstdint>

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_profiler_target.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_target.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "xla/backends/profiler/gpu/cupti_wrapper.h"

namespace xla {
namespace profiler {

CUptiResult CuptiWrapperStub::ActivityDisable(CUpti_ActivityKind kind) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ActivityEnable(CUpti_ActivityKind kind) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ActivityFlushAll(uint32_t flag) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ActivityGetNextRecord(
    uint8_t* buffer, size_t valid_buffer_size_bytes, CUpti_Activity** record) {
  return CUPTI_ERROR_MAX_LIMIT_REACHED;
}

CUptiResult CuptiWrapperStub::ActivityGetNumDroppedRecords(CUcontext context,
                                                           uint32_t stream_id,
                                                           size_t* dropped) {
  *dropped = 0;
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ActivityConfigureUnifiedMemoryCounter(
    CUpti_ActivityUnifiedMemoryCounterConfig* config, uint32_t count) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc func_buffer_requested,
    CUpti_BuffersCallbackCompleteFunc func_buffer_completed) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ActivityUsePerThreadBuffer() {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::SetActivityFlushPeriod(uint32_t period_ms) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::GetDeviceId(CUcontext context,
                                          uint32_t* deviceId) {
  return cuptiGetDeviceId(context, deviceId);
}

CUptiResult CuptiWrapperStub::GetTimestamp(uint64_t* timestamp) {
  return cuptiGetTimestamp(timestamp);
}

CUptiResult CuptiWrapperStub::Finalize() { return CUPTI_SUCCESS; }

CUptiResult CuptiWrapperStub::EnableCallback(uint32_t enable,
                                             CUpti_SubscriberHandle subscriber,
                                             CUpti_CallbackDomain domain,
                                             CUpti_CallbackId cbid) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::EnableDomain(uint32_t enable,
                                           CUpti_SubscriberHandle subscriber,
                                           CUpti_CallbackDomain domain) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::Subscribe(CUpti_SubscriberHandle* subscriber,
                                        CUpti_CallbackFunc callback,
                                        void* userdata) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::Unsubscribe(CUpti_SubscriberHandle subscriber) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::GetResultString(CUptiResult result,
                                              const char** str) {
  return cuptiGetResultString(result, str);
}

CUptiResult CuptiWrapperStub::GetContextId(CUcontext context,
                                           uint32_t* context_id) {
  return cuptiGetContextId(context, context_id);
}

CUptiResult CuptiWrapperStub::GetStreamIdEx(CUcontext context, CUstream stream,
                                            uint8_t per_thread_stream,
                                            uint32_t* stream_id) {
  return cuptiGetStreamIdEx(context, stream, per_thread_stream, stream_id);
}

CUptiResult CuptiWrapperStub::GetGraphId(CUgraph graph, uint32_t* graph_id) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::GetGraphNodeId(CUgraphNode node,
                                             uint64_t* nodeId) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::GetGraphExecId(CUgraphExec graph_exec,
                                             uint32_t* graph_id) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::SetThreadIdType(CUpti_ActivityThreadIdType type) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ActivityEnableHWTrace(bool enable) {
  return CUPTI_SUCCESS;
}

// Profiler Host APIs
CUptiResult CuptiWrapperStub::ProfilerHostInitialize(
    CUpti_Profiler_Host_Initialize_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerHostDeinitialize(
    CUpti_Profiler_Host_Deinitialize_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerHostGetSupportedChips(
    CUpti_Profiler_Host_GetSupportedChips_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerHostGetBaseMetrics(
    CUpti_Profiler_Host_GetBaseMetrics_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerHostGetSubMetrics(
    CUpti_Profiler_Host_GetSubMetrics_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerHostGetMetricProperties(
    CUpti_Profiler_Host_GetMetricProperties_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerHostGetRangeName(
    CUpti_Profiler_Host_GetRangeName_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerHostEvaluateToGpuValues(
    CUpti_Profiler_Host_EvaluateToGpuValues_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerHostConfigAddMetrics(
    CUpti_Profiler_Host_ConfigAddMetrics_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerHostGetConfigImageSize(
    CUpti_Profiler_Host_GetConfigImageSize_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerHostGetConfigImage(
    CUpti_Profiler_Host_GetConfigImage_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerHostGetNumOfPasses(
    CUpti_Profiler_Host_GetNumOfPasses_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerHostGetMaxNumHardwareMetricsPerPass(
    CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params* params) {
  return CUPTI_SUCCESS;
}

// Profiler Target APIs
CUptiResult CuptiWrapperStub::ProfilerInitialize(
    CUpti_Profiler_Initialize_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerDeInitialize(
    CUpti_Profiler_DeInitialize_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerCounterDataImageCalculateSize(
    CUpti_Profiler_CounterDataImage_CalculateSize_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerCounterDataImageInitialize(
    CUpti_Profiler_CounterDataImage_Initialize_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult
CuptiWrapperStub::ProfilerCounterDataImageCalculateScratchBufferSize(
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerCounterDataImageInitializeScratchBuffer(
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerBeginSession(
    CUpti_Profiler_BeginSession_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerEndSession(
    CUpti_Profiler_EndSession_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerSetConfig(
    CUpti_Profiler_SetConfig_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerUnsetConfig(
    CUpti_Profiler_UnsetConfig_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerBeginPass(
    CUpti_Profiler_BeginPass_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerEndPass(
    CUpti_Profiler_EndPass_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerEnableProfiling(
    CUpti_Profiler_EnableProfiling_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerDisableProfiling(
    CUpti_Profiler_DisableProfiling_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerIsPassCollected(
    CUpti_Profiler_IsPassCollected_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerFlushCounterData(
    CUpti_Profiler_FlushCounterData_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerPushRange(
    CUpti_Profiler_PushRange_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerPopRange(
    CUpti_Profiler_PopRange_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerGetCounterAvailability(
    CUpti_Profiler_GetCounterAvailability_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::ProfilerDeviceSupported(
    CUpti_Profiler_DeviceSupported_Params* params) {
  return CUPTI_SUCCESS;
}

// PM sampling specific functions
CUptiResult CuptiWrapperStub::PmSamplingSetConfig(
    CUpti_PmSampling_SetConfig_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::PmSamplingEnable(
    CUpti_PmSampling_Enable_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::PmSamplingDisable(
    CUpti_PmSampling_Disable_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::PmSamplingStart(
    CUpti_PmSampling_Start_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::PmSamplingStop(
    CUpti_PmSampling_Stop_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::PmSamplingDecodeData(
    CUpti_PmSampling_DecodeData_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::PmSamplingGetCounterAvailability(
    CUpti_PmSampling_GetCounterAvailability_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::PmSamplingGetCounterDataSize(
    CUpti_PmSampling_GetCounterDataSize_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::PmSamplingCounterDataImageInitialize(
    CUpti_PmSampling_CounterDataImage_Initialize_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::PmSamplingGetCounterDataInfo(
    CUpti_PmSampling_GetCounterDataInfo_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::PmSamplingCounterDataGetSampleInfo(
    CUpti_PmSampling_CounterData_GetSampleInfo_Params* params) {
  return CUPTI_SUCCESS;
}

CUptiResult CuptiWrapperStub::DeviceGetChipName(
    CUpti_Device_GetChipName_Params* params) {
  return CUPTI_SUCCESS;
}

}  // namespace profiler
}  // namespace xla
