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

#ifndef TENSORFLOW_COMPILER_XLA_BACKENDS_PROFILER_GPU_CUPTI_ERROR_MANAGER_H_
#define TENSORFLOW_COMPILER_XLA_BACKENDS_PROFILER_GPU_CUPTI_ERROR_MANAGER_H_

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/backends/profiler/gpu/cupti_interface.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/thread_annotations.h"

namespace xla {
namespace profiler {

class CuptiErrorManager : public xla::profiler::CuptiInterface {
 public:
  explicit CuptiErrorManager(std::unique_ptr<CuptiInterface> interface);

  // Returns whether CUPTI is disabled.
  bool Disabled() const override { return disabled_.load(); }

  // CUPTI activity API: all thread-safe
  // Disables activity monitoring.
  CUptiResult ActivityDisable(CUpti_ActivityKind kind) override;

  // Enables activity monitoring. If this is successfully executed, we add
  // ActivityDisable to the undo log.
  CUptiResult ActivityEnable(CUpti_ActivityKind kind) override;

  // Flushes all outstanding activities.
  CUptiResult ActivityFlushAll(uint32_t flag) override;

  // Gets a next activity record from a pool of already collected activity
  // records.
  CUptiResult ActivityGetNextRecord(uint8_t* buffer,
                                    size_t valid_buffer_size_bytes,
                                    CUpti_Activity** record) override;

  // Reports the number of dropped activity records.
  CUptiResult ActivityGetNumDroppedRecords(CUcontext context,
                                           uint32_t stream_id,
                                           size_t* dropped) override;

  CUptiResult ActivityConfigureUnifiedMemoryCounter(
      CUpti_ActivityUnifiedMemoryCounterConfig* config,
      uint32_t count) override;

  // Registers callback functions handling activity.
  CUptiResult ActivityRegisterCallbacks(
      CUpti_BuffersCallbackRequestFunc func_buffer_requested,
      CUpti_BuffersCallbackCompleteFunc func_buffer_completed) override;

  // Returns device ID for a given context.
  CUptiResult GetDeviceId(CUcontext context, uint32_t* device_id) override;

  // Returns CUPTI timestamp.
  CUptiResult GetTimestamp(uint64_t* timestamp) override;

  // Explicitly destroys and cleans up all resources associated with CUPTI in
  // the current process.
  CUptiResult Finalize() override;

  // CUPTI callback API
  // Enables or disables callback. If we successfully enables callback, we add
  // EnableCallback to disable callback to the undo log.
  CUptiResult EnableCallback(uint32_t enable, CUpti_SubscriberHandle subscriber,
                             CUpti_CallbackDomain domain,
                             CUpti_CallbackId callback_id) override;

  // Enables or disables callback domain. If we successfully enables a domain,
  // we add EnableDomain to disable the domain to the undo log.
  CUptiResult EnableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber,
                           CUpti_CallbackDomain domain) override;

  // Subscribes callbacks. If we successfully subscribes the callback, we add
  // Unsubscribe to the undo log.
  CUptiResult Subscribe(CUpti_SubscriberHandle* subscriber,
                        CUpti_CallbackFunc callback, void* userdata) override;

  // Unsubscribes callbacks.
  CUptiResult Unsubscribe(CUpti_SubscriberHandle subscriber) override;

  // CUPTI event API
  // Returns a list of event domains.
  CUptiResult DeviceEnumEventDomains(
      CUdevice device, size_t* array_size_bytes,
      CUpti_EventDomainID* domain_array) override;

  // Returns domain attributes.
  CUptiResult DeviceGetEventDomainAttribute(CUdevice device,
                                            CUpti_EventDomainID event_domain,
                                            CUpti_EventDomainAttribute attrib,
                                            size_t* value_size,
                                            void* value) override;

  // Disables kernel replay mode.
  CUptiResult DisableKernelReplayMode(CUcontext context) override;

  // Enables kernel replay mode. If we successfully enable kernel replay mode,
  // we add DisableKernelReplayMode to the undo log.
  CUptiResult EnableKernelReplayMode(CUcontext context) override;

  // Returns the number of event domains.
  CUptiResult DeviceGetNumEventDomains(CUdevice device,
                                       uint32_t* num_domains) override;

  // Returns a list of events.
  CUptiResult EventDomainEnumEvents(CUpti_EventDomainID event_domain,
                                    size_t* array_size_bytes,
                                    CUpti_EventID* event_array) override;

  // Returns the number of events.
  CUptiResult EventDomainGetNumEvents(CUpti_EventDomainID event_domain,
                                      uint32_t* num_events) override;

  // Returns an event attribute.
  CUptiResult EventGetAttribute(CUpti_EventID event,
                                CUpti_EventAttribute attrib, size_t* value_size,
                                void* value) override;

  // Convverts event ID from event name.
  CUptiResult EventGetIdFromName(CUdevice device, const char* event_name,
                                 CUpti_EventID* event) override;

  // Disables event group.
  CUptiResult EventGroupDisable(CUpti_EventGroup event_group) override;

  // Enables event group. If we successfully enable an event group, we add
  // EventGroupDisable to the undo log.
  CUptiResult EventGroupEnable(CUpti_EventGroup event_group) override;

  // Returns an event group attribute.
  CUptiResult EventGroupGetAttribute(CUpti_EventGroup event_group,
                                     CUpti_EventGroupAttribute attrib,
                                     size_t* value_size, void* value) override;

  // Returns a performance counter value.
  CUptiResult EventGroupReadEvent(CUpti_EventGroup event_group,
                                  CUpti_ReadEventFlags flags,
                                  CUpti_EventID event,
                                  size_t* event_value_buffer_size_bytes,
                                  uint64_t* event_value_buffer) override;

  // Returns an event group set attribute.
  CUptiResult EventGroupSetAttribute(CUpti_EventGroup event_group,
                                     CUpti_EventGroupAttribute attrib,
                                     size_t value_size, void* value) override;

  // Creates an event group set. If we successfully creates an event group set,
  // we add EventGroupSetsDestroy to the undo log.
  CUptiResult EventGroupSetsCreate(
      CUcontext context, size_t event_id_array_size_bytes,
      CUpti_EventID* event_id_array,
      CUpti_EventGroupSets** event_group_passes) override;

  // Destroys an event group set.
  CUptiResult EventGroupSetsDestroy(
      CUpti_EventGroupSets* event_group_sets) override;

  // CUPTI metric API: all thread-safe
  // Enumerates metrics.
  CUptiResult DeviceEnumMetrics(CUdevice device, size_t* arraySizeBytes,
                                CUpti_MetricID* metricArray) override;

  // Returns the number of metrics.
  CUptiResult DeviceGetNumMetrics(CUdevice device,
                                  uint32_t* num_metrics) override;

  // Converts a metric ID to a metric name.
  CUptiResult MetricGetIdFromName(CUdevice device, const char* metric_name,
                                  CUpti_MetricID* metric) override;

  // Returns the number of events required to calculate a particular metric.
  CUptiResult MetricGetNumEvents(CUpti_MetricID metric,
                                 uint32_t* num_events) override;

  // Returns a list of events required to calculate a particular metric.
  CUptiResult MetricEnumEvents(CUpti_MetricID metric,
                               size_t* event_id_array_size_bytes,
                               CUpti_EventID* event_id_array) override;

  // Returns a metric attribute.
  CUptiResult MetricGetAttribute(CUpti_MetricID metric,
                                 CUpti_MetricAttribute attrib,
                                 size_t* value_size, void* value) override;

  // Returns a metric value.
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

  // Clears Undo stack. We are maintaining undo stack for each profiling phase.
  // Once the profiling is done, we need to clear the undo stack.
  void CleanUp() override;

 private:
  typedef std::function<CUptiResult()> UndoFunction;

  // Register undo function.
  void RegisterUndoFunction(const UndoFunction& func);

  // Resets profiling status by calling some undo functions registered,
  // and then disables profiling.
  void UndoAndDisable();

  // Returns a descriptive string for a CUptiResult.
  std::string ResultString(CUptiResult result) const;

  // Contains a pointer to a cupti interface instance. Normally, this will point
  // to a real CUPTI interface that interacts with underlying hardware, but for
  // testing, we often replace this with a CUPTI mock object to mock hardware
  // behavior. This will be set when CuptiBase singleton was created and an
  // object that this variable points to will die when CuptiBase singleton dies,
  // i.e., at the end of program execution.
  std::unique_ptr<CuptiInterface> interface_;

  // A vector of functions that needs to be called by Undo upon an error
  // detected. This vector is managed like a statck through push_back and
  // pop_back. Whenever an API function is successfully executed, its
  // corresponding undo function will be pushed into this stack and Undo will
  // pop and execute the unroll function upon detecting an error.
  std::vector<UndoFunction> undo_stack_ TF_GUARDED_BY(undo_stack_mu_);

  // A mutex to guarantee atomicity for undo_stack_. Given that threads that
  // can update undo_stack_ are a profiling control thread such as a webserver
  // thread or a thread that executes a kernel during performance counter
  // profiling, which is already serialized, the contention for this lock will
  // be extremely low. In other words, it will be contended only when the
  // profiling is being enabled or disabled, and we will have at most two
  // threads that will contend for this mutex.
  tsl::mutex undo_stack_mu_;

  // Once an error is detected, we will ignore any CUPTI API call.
  std::atomic<int> disabled_;

  // Prevent recursive undo if an UndoFunction fails.
  bool undo_disabled_;

  TF_DISALLOW_COPY_AND_ASSIGN(CuptiErrorManager);
};

}  // namespace profiler
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_BACKENDS_PROFILER_GPU_CUPTI_ERROR_MANAGER_H_
