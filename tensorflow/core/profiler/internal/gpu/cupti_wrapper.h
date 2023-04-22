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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_WRAPPER_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_WRAPPER_H_

#include <stddef.h>
#include <stdint.h>

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/core/profiler/internal/gpu/cupti_interface.h"

namespace tensorflow {
namespace profiler {

class CuptiWrapper : public tensorflow::profiler::CuptiInterface {
 public:
  CuptiWrapper() {}

  ~CuptiWrapper() override {}

  // CUPTI activity API
  CUptiResult ActivityDisable(CUpti_ActivityKind kind) override;

  CUptiResult ActivityEnable(CUpti_ActivityKind kind) override;

  CUptiResult ActivityFlushAll(uint32_t flag) override;

  CUptiResult ActivityGetNextRecord(uint8_t* buffer,
                                    size_t valid_buffer_size_bytes,
                                    CUpti_Activity** record) override;

  CUptiResult ActivityGetNumDroppedRecords(CUcontext context,
                                           uint32_t stream_id,
                                           size_t* dropped) override;

  CUptiResult ActivityConfigureUnifiedMemoryCounter(
      CUpti_ActivityUnifiedMemoryCounterConfig* config,
      uint32_t count) override;

  CUptiResult ActivityRegisterCallbacks(
      CUpti_BuffersCallbackRequestFunc func_buffer_requested,
      CUpti_BuffersCallbackCompleteFunc func_buffer_completed) override;

  CUptiResult GetDeviceId(CUcontext context, uint32* deviceId) override;

  CUptiResult GetTimestamp(uint64_t* timestamp) override;

  // cuptiFinalize is only defined in CUDA8 and above.
  // To enable it in CUDA8, the environment variable CUPTI_ENABLE_FINALIZE must
  // be set to 1.
  CUptiResult Finalize() override;

  // CUPTI callback API
  CUptiResult EnableCallback(uint32_t enable, CUpti_SubscriberHandle subscriber,
                             CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbid) override;

  CUptiResult EnableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber,
                           CUpti_CallbackDomain domain) override;

  CUptiResult Subscribe(CUpti_SubscriberHandle* subscriber,
                        CUpti_CallbackFunc callback, void* userdata) override;

  CUptiResult Unsubscribe(CUpti_SubscriberHandle subscriber) override;

  // CUPTI event API
  CUptiResult DeviceEnumEventDomains(
      CUdevice device, size_t* array_size_bytes,
      CUpti_EventDomainID* domain_array) override;

  CUptiResult DeviceGetEventDomainAttribute(CUdevice device,
                                            CUpti_EventDomainID event_domain,
                                            CUpti_EventDomainAttribute attrib,
                                            size_t* value_size,
                                            void* value) override;

  CUptiResult DisableKernelReplayMode(CUcontext context) override;

  CUptiResult EnableKernelReplayMode(CUcontext context) override;

  CUptiResult DeviceGetNumEventDomains(CUdevice device,
                                       uint32_t* num_domains) override;

  CUptiResult EventDomainEnumEvents(CUpti_EventDomainID event_domain,
                                    size_t* array_size_bytes,
                                    CUpti_EventID* event_array) override;

  CUptiResult EventDomainGetNumEvents(CUpti_EventDomainID event_domain,
                                      uint32_t* num_events) override;

  CUptiResult EventGetAttribute(CUpti_EventID event,
                                CUpti_EventAttribute attrib, size_t* value_size,
                                void* value) override;

  CUptiResult EventGetIdFromName(CUdevice device, const char* event_name,
                                 CUpti_EventID* event) override;

  CUptiResult EventGroupDisable(CUpti_EventGroup event_group) override;

  CUptiResult EventGroupEnable(CUpti_EventGroup event_group) override;

  CUptiResult EventGroupGetAttribute(CUpti_EventGroup event_group,
                                     CUpti_EventGroupAttribute attrib,
                                     size_t* value_size, void* value) override;

  CUptiResult EventGroupReadEvent(CUpti_EventGroup event_group,
                                  CUpti_ReadEventFlags flags,
                                  CUpti_EventID event,
                                  size_t* event_value_buffer_size_bytes,
                                  uint64_t* event_value_buffer) override;

  CUptiResult EventGroupSetAttribute(CUpti_EventGroup event_group,
                                     CUpti_EventGroupAttribute attrib,
                                     size_t value_size, void* value) override;

  CUptiResult EventGroupSetsCreate(
      CUcontext context, size_t event_id_array_size_bytes,
      CUpti_EventID* event_id_array,
      CUpti_EventGroupSets** event_group_passes) override;

  CUptiResult EventGroupSetsDestroy(
      CUpti_EventGroupSets* event_group_sets) override;

  // CUPTI metric API
  CUptiResult DeviceEnumMetrics(CUdevice device, size_t* arraySizeBytes,
                                CUpti_MetricID* metricArray) override;

  CUptiResult DeviceGetNumMetrics(CUdevice device,
                                  uint32_t* num_metrics) override;

  CUptiResult MetricGetIdFromName(CUdevice device, const char* metric_name,
                                  CUpti_MetricID* metric) override;

  CUptiResult MetricGetNumEvents(CUpti_MetricID metric,
                                 uint32_t* num_events) override;

  CUptiResult MetricEnumEvents(CUpti_MetricID metric,
                               size_t* event_id_array_size_bytes,
                               CUpti_EventID* event_id_array) override;

  CUptiResult MetricGetAttribute(CUpti_MetricID metric,
                                 CUpti_MetricAttribute attrib,
                                 size_t* value_size, void* value) override;

  CUptiResult MetricGetValue(CUdevice device, CUpti_MetricID metric,
                             size_t event_id_array_size_bytes,
                             CUpti_EventID* event_id_array,
                             size_t event_value_array_size_bytes,
                             uint64_t* event_value_array,
                             uint64_t time_duration,
                             CUpti_MetricValue* metric_value) override;

  CUptiResult GetResultString(CUptiResult result, const char** str) override;

  CUptiResult GetContextId(CUcontext context, uint32_t* context_id) override;

  CUptiResult GetStreamIdEx(CUcontext context, CUstream stream,
                            uint8_t per_thread_stream,
                            uint32_t* stream_id) override;

  void CleanUp() override {}
  bool Disabled() const override { return false; }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CuptiWrapper);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // PERFTOOLS_ACCELERATORS_XPROF_XPROFILEZ_NVIDIA_GPU_CUPTI_WRAPPER_H_
