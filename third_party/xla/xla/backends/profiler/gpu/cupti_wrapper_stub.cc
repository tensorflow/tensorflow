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

#include <type_traits>

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

}  // namespace profiler
}  // namespace xla
