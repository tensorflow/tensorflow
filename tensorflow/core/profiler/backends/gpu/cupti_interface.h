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

#ifndef TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_CUPTI_INTERFACE_H_
#define TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_CUPTI_INTERFACE_H_

#include <stddef.h>
#include <stdint.h>

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
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

  virtual CUptiResult GetDeviceId(CUcontext context, uint32* deviceId) = 0;

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

  // CUPTI event API
  virtual CUptiResult DeviceEnumEventDomains(
      CUdevice device, size_t* array_size_bytes,
      CUpti_EventDomainID* domain_array) = 0;

  virtual CUptiResult DeviceGetEventDomainAttribute(
      CUdevice device, CUpti_EventDomainID event_domain,
      CUpti_EventDomainAttribute attrib, size_t* value_size, void* value) = 0;

  virtual CUptiResult DisableKernelReplayMode(CUcontext context) = 0;

  virtual CUptiResult EnableKernelReplayMode(CUcontext context) = 0;

  virtual CUptiResult DeviceGetNumEventDomains(CUdevice device,
                                               uint32_t* num_domains) = 0;

  virtual CUptiResult EventDomainEnumEvents(CUpti_EventDomainID event_domain,
                                            size_t* array_size_bytes,
                                            CUpti_EventID* event_array) = 0;

  virtual CUptiResult EventDomainGetNumEvents(CUpti_EventDomainID event_domain,
                                              uint32_t* num_events) = 0;

  virtual CUptiResult EventGetAttribute(CUpti_EventID event,
                                        CUpti_EventAttribute attrib,
                                        size_t* value_size, void* value) = 0;

  virtual CUptiResult EventGetIdFromName(CUdevice device,
                                         const char* event_name,
                                         CUpti_EventID* event) = 0;

  virtual CUptiResult EventGroupDisable(CUpti_EventGroup event_group) = 0;

  virtual CUptiResult EventGroupEnable(CUpti_EventGroup event_group) = 0;

  virtual CUptiResult EventGroupGetAttribute(CUpti_EventGroup event_group,
                                             CUpti_EventGroupAttribute attrib,
                                             size_t* value_size,
                                             void* value) = 0;

  virtual CUptiResult EventGroupReadEvent(CUpti_EventGroup event_group,
                                          CUpti_ReadEventFlags flags,
                                          CUpti_EventID event,
                                          size_t* event_value_buffer_size_bytes,
                                          uint64_t* eventValueBuffer) = 0;

  virtual CUptiResult EventGroupSetAttribute(CUpti_EventGroup event_group,
                                             CUpti_EventGroupAttribute attrib,
                                             size_t value_size,
                                             void* value) = 0;

  virtual CUptiResult EventGroupSetsCreate(
      CUcontext context, size_t event_id_array_size_bytes,
      CUpti_EventID* event_id_array,
      CUpti_EventGroupSets** event_group_passes) = 0;

  virtual CUptiResult EventGroupSetsDestroy(
      CUpti_EventGroupSets* event_group_sets) = 0;

  // CUPTI metric API
  virtual CUptiResult DeviceEnumMetrics(CUdevice device, size_t* arraySizeBytes,
                                        CUpti_MetricID* metricArray) = 0;

  virtual CUptiResult DeviceGetNumMetrics(CUdevice device,
                                          uint32_t* num_metrics) = 0;

  virtual CUptiResult MetricGetIdFromName(CUdevice device,
                                          const char* metric_name,
                                          CUpti_MetricID* metric) = 0;

  virtual CUptiResult MetricGetNumEvents(CUpti_MetricID metric,
                                         uint32_t* num_events) = 0;

  virtual CUptiResult MetricEnumEvents(CUpti_MetricID metric,
                                       size_t* event_id_array_size_bytes,
                                       CUpti_EventID* event_id_array) = 0;

  virtual CUptiResult MetricGetAttribute(CUpti_MetricID metric,
                                         CUpti_MetricAttribute attrib,
                                         size_t* value_size, void* value) = 0;

  virtual CUptiResult MetricGetValue(CUdevice device, CUpti_MetricID metric,
                                     size_t event_id_array_size_bytes,
                                     CUpti_EventID* event_id_array,
                                     size_t event_value_array_size_bytes,
                                     uint64_t* event_value_array,
                                     uint64_t time_duration,
                                     CUpti_MetricValue* metric_value) = 0;

  virtual CUptiResult GetResultString(CUptiResult result, const char** str) = 0;

  virtual CUptiResult GetContextId(CUcontext context, uint32_t* context_id) = 0;

  virtual CUptiResult GetStreamIdEx(CUcontext context, CUstream stream,
                                    uint8_t per_thread_stream,
                                    uint32_t* stream_id) = 0;

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
  TF_DISALLOW_COPY_AND_ASSIGN(CuptiInterface);
};

CuptiInterface* GetCuptiInterface();

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_CUPTI_INTERFACE_H_
