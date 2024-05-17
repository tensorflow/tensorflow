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

#include <type_traits>

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

CUptiResult CuptiWrapper::GetStreamIdEx(CUcontext context, CUstream stream,
                                        uint8_t per_thread_stream,
                                        uint32_t* stream_id) {
  return cuptiGetStreamIdEx(context, stream, per_thread_stream, stream_id);
}

}  // namespace profiler
}  // namespace xla
