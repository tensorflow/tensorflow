/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/internal/gpu/cupti_wrapper.h"

#include <type_traits>

namespace tensorflow {
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

CUptiResult CuptiWrapper::GetDeviceId(CUcontext context, uint32* deviceId) {
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

CUptiResult CuptiWrapper::DeviceEnumEventDomains(
    CUdevice device, size_t* array_size_bytes,
    CUpti_EventDomainID* domain_array) {
  return cuptiDeviceEnumEventDomains(device, array_size_bytes, domain_array);
}

CUptiResult CuptiWrapper::DeviceGetEventDomainAttribute(
    CUdevice device, CUpti_EventDomainID event_domain,
    CUpti_EventDomainAttribute attrib, size_t* value_size, void* value) {
  return cuptiDeviceGetEventDomainAttribute(device, event_domain, attrib,
                                            value_size, value);
}

CUptiResult CuptiWrapper::DisableKernelReplayMode(CUcontext context) {
  return cuptiDisableKernelReplayMode(context);
}

CUptiResult CuptiWrapper::EnableKernelReplayMode(CUcontext context) {
  return cuptiEnableKernelReplayMode(context);
}

CUptiResult CuptiWrapper::DeviceGetNumEventDomains(CUdevice device,
                                                   uint32_t* num_domains) {
  return cuptiDeviceGetNumEventDomains(device, num_domains);
}

CUptiResult CuptiWrapper::EventDomainEnumEvents(
    CUpti_EventDomainID event_domain, size_t* array_size_bytes,
    CUpti_EventID* event_array) {
  return cuptiEventDomainEnumEvents(event_domain, array_size_bytes,
                                    event_array);
}

CUptiResult CuptiWrapper::EventDomainGetNumEvents(
    CUpti_EventDomainID event_domain, uint32_t* num_events) {
  return cuptiEventDomainGetNumEvents(event_domain, num_events);
}

CUptiResult CuptiWrapper::EventGetAttribute(CUpti_EventID event,
                                            CUpti_EventAttribute attrib,
                                            size_t* value_size, void* value) {
  return cuptiEventGetAttribute(event, attrib, value_size, value);
}

CUptiResult CuptiWrapper::EventGetIdFromName(CUdevice device,
                                             const char* event_name,
                                             CUpti_EventID* event) {
  return cuptiEventGetIdFromName(device, event_name, event);
}

CUptiResult CuptiWrapper::EventGroupDisable(CUpti_EventGroup event_group) {
  return cuptiEventGroupDisable(event_group);
}

CUptiResult CuptiWrapper::EventGroupEnable(CUpti_EventGroup event_group) {
  return cuptiEventGroupEnable(event_group);
}

CUptiResult CuptiWrapper::EventGroupGetAttribute(
    CUpti_EventGroup event_group, CUpti_EventGroupAttribute attrib,
    size_t* value_size, void* value) {
  return cuptiEventGroupGetAttribute(event_group, attrib, value_size, value);
}

CUptiResult CuptiWrapper::EventGroupReadEvent(
    CUpti_EventGroup event_group, CUpti_ReadEventFlags flags,
    CUpti_EventID event, size_t* event_value_buffer_size_bytes,
    uint64_t* event_value_buffer) {
  return cuptiEventGroupReadEvent(event_group, flags, event,
                                  event_value_buffer_size_bytes,
                                  event_value_buffer);
}

CUptiResult CuptiWrapper::EventGroupSetAttribute(
    CUpti_EventGroup event_group, CUpti_EventGroupAttribute attrib,
    size_t value_size, void* value) {
  return cuptiEventGroupSetAttribute(event_group, attrib, value_size, value);
}

CUptiResult CuptiWrapper::EventGroupSetsCreate(
    CUcontext context, size_t event_id_array_size_bytes,
    CUpti_EventID* event_id_array, CUpti_EventGroupSets** event_group_passes) {
  return cuptiEventGroupSetsCreate(context, event_id_array_size_bytes,
                                   event_id_array, event_group_passes);
}

CUptiResult CuptiWrapper::EventGroupSetsDestroy(
    CUpti_EventGroupSets* event_group_sets) {
  return cuptiEventGroupSetsDestroy(event_group_sets);
}

// CUPTI metric API
CUptiResult CuptiWrapper::DeviceEnumMetrics(CUdevice device,
                                            size_t* arraySizeBytes,
                                            CUpti_MetricID* metricArray) {
  return cuptiDeviceEnumMetrics(device, arraySizeBytes, metricArray);
}

CUptiResult CuptiWrapper::DeviceGetNumMetrics(CUdevice device,
                                              uint32_t* num_metrics) {
  return cuptiDeviceGetNumMetrics(device, num_metrics);
}

CUptiResult CuptiWrapper::MetricGetIdFromName(CUdevice device,
                                              const char* metric_name,
                                              CUpti_MetricID* metric) {
  return cuptiMetricGetIdFromName(device, metric_name, metric);
}

CUptiResult CuptiWrapper::MetricGetNumEvents(CUpti_MetricID metric,
                                             uint32_t* num_events) {
  return cuptiMetricGetNumEvents(metric, num_events);
}

CUptiResult CuptiWrapper::MetricEnumEvents(CUpti_MetricID metric,
                                           size_t* event_id_array_size_bytes,
                                           CUpti_EventID* event_id_array) {
  return cuptiMetricEnumEvents(metric, event_id_array_size_bytes,
                               event_id_array);
}

CUptiResult CuptiWrapper::MetricGetAttribute(CUpti_MetricID metric,
                                             CUpti_MetricAttribute attrib,
                                             size_t* value_size, void* value) {
  return cuptiMetricGetAttribute(metric, attrib, value_size, value);
}

CUptiResult CuptiWrapper::MetricGetValue(CUdevice device, CUpti_MetricID metric,
                                         size_t event_id_array_size_bytes,
                                         CUpti_EventID* event_id_array,
                                         size_t event_value_array_size_bytes,
                                         uint64_t* event_value_array,
                                         uint64_t time_duration,
                                         CUpti_MetricValue* metric_value) {
  return cuptiMetricGetValue(device, metric, event_id_array_size_bytes,
                             event_id_array, event_value_array_size_bytes,
                             event_value_array, time_duration, metric_value);
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
}  // namespace tensorflow
