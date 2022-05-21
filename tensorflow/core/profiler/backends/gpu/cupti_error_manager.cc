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

#include "tensorflow/core/profiler/backends/gpu/cupti_error_manager.h"

#include <utility>

#include "absl/debugging/leak_check.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace profiler {

CuptiErrorManager::CuptiErrorManager(std::unique_ptr<CuptiInterface> interface)
    : interface_(std::move(interface)), disabled_(0), undo_disabled_(false) {}

#define IGNORE_CALL_IF_DISABLED                                                \
  if (disabled_) {                                                             \
    LOG(ERROR) << "cupti" << __func__ << ": ignored due to a previous error."; \
    return CUPTI_ERROR_DISABLED;                                               \
  }                                                                            \
  VLOG(1) << "cupti" << __func__;

#define ALLOW_ERROR(e, ERROR)                                           \
  if (e == ERROR) {                                                     \
    VLOG(1) << "cupti" << __func__ << ": error " << static_cast<int>(e) \
            << ": " << ResultString(e) << " (allowed)";                 \
    return e;                                                           \
  }

#define LOG_AND_DISABLE_IF_ERROR(e)                                        \
  if (e != CUPTI_SUCCESS) {                                                \
    LOG(ERROR) << "cupti" << __func__ << ": error " << static_cast<int>(e) \
               << ": " << ResultString(e);                                 \
    UndoAndDisable();                                                      \
  }

void CuptiErrorManager::RegisterUndoFunction(
    const CuptiErrorManager::UndoFunction& func) {
  mutex_lock lock(undo_stack_mu_);
  undo_stack_.push_back(func);
}

CUptiResult CuptiErrorManager::ActivityDisable(CUpti_ActivityKind kind) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->ActivityDisable(kind);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::ActivityEnable(CUpti_ActivityKind kind) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->ActivityEnable(kind);
  if (error == CUPTI_SUCCESS) {
    auto f = std::bind(&CuptiErrorManager::ActivityDisable, this, kind);
    RegisterUndoFunction(f);
  }
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::ActivityFlushAll(uint32_t flag) {
  // There is a synchronization issue that we were assuming this will flush all
  // the activity buffers. Therefore we need to let CUPTI to flush no matter if
  // previous error is encountered or not.
  CUptiResult error = interface_->ActivityFlushAll(flag);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::ActivityGetNextRecord(
    uint8_t* buffer, size_t valid_buffer_size_bytes, CUpti_Activity** record) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->ActivityGetNextRecord(
      buffer, valid_buffer_size_bytes, record);
  ALLOW_ERROR(error, CUPTI_ERROR_MAX_LIMIT_REACHED);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::ActivityGetNumDroppedRecords(CUcontext context,
                                                            uint32_t stream_id,
                                                            size_t* dropped) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->ActivityGetNumDroppedRecords(context, stream_id, dropped);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::ActivityConfigureUnifiedMemoryCounter(
    CUpti_ActivityUnifiedMemoryCounterConfig* config, uint32_t count) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->ActivityConfigureUnifiedMemoryCounter(config, count);
  // Don't disable cupti just because the gpu don't support unified memory.
  return error;
}

CUptiResult CuptiErrorManager::ActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc func_buffer_requested,
    CUpti_BuffersCallbackCompleteFunc func_buffer_completed) {
  IGNORE_CALL_IF_DISABLED;
  // Disable heap checking for the first CUPTI activity API. See b/22091576.
  absl::LeakCheckDisabler disabler;
  CUptiResult error = interface_->ActivityRegisterCallbacks(
      func_buffer_requested, func_buffer_completed);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::GetDeviceId(CUcontext context,
                                           uint32_t* device_id) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->GetDeviceId(context, device_id);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::GetTimestamp(uint64_t* timestamp) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->GetTimestamp(timestamp);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::Finalize() {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->Finalize();
  ALLOW_ERROR(error, CUPTI_ERROR_API_NOT_IMPLEMENTED);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EnableCallback(uint32_t enable,
                                              CUpti_SubscriberHandle subscriber,
                                              CUpti_CallbackDomain domain,
                                              CUpti_CallbackId callback_id) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->EnableCallback(enable, subscriber, domain, callback_id);
  if (error == CUPTI_SUCCESS) {
    if (enable == 1) {
      auto f = std::bind(&CuptiErrorManager::EnableCallback, this,
                         0 /* DISABLE */, subscriber, domain, callback_id);
      RegisterUndoFunction(f);
    }
  }
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EnableDomain(uint32_t enable,
                                            CUpti_SubscriberHandle subscriber,
                                            CUpti_CallbackDomain domain) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EnableDomain(enable, subscriber, domain);
  if (error == CUPTI_SUCCESS) {
    if (enable == 1) {
      auto f = std::bind(&CuptiErrorManager::EnableDomain, this,
                         0 /* DISABLE */, subscriber, domain);
      RegisterUndoFunction(f);
    }
  }
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::Subscribe(CUpti_SubscriberHandle* subscriber,
                                         CUpti_CallbackFunc callback,
                                         void* userdata) {
  IGNORE_CALL_IF_DISABLED;
  // Disable heap checking for the first CUPTI callback API. See b/22091576.
  absl::LeakCheckDisabler disabler;
  CUptiResult error = interface_->Subscribe(subscriber, callback, userdata);
  if (error == CUPTI_SUCCESS) {
    auto f = std::bind(&CuptiErrorManager::Unsubscribe, this, *subscriber);
    RegisterUndoFunction(f);
  }
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::Unsubscribe(CUpti_SubscriberHandle subscriber) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->Unsubscribe(subscriber);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::DeviceEnumEventDomains(
    CUdevice device, size_t* array_size_bytes,
    CUpti_EventDomainID* domain_array) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->DeviceEnumEventDomains(
      device, array_size_bytes, domain_array);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::DeviceGetEventDomainAttribute(
    CUdevice device, CUpti_EventDomainID event_domain,
    CUpti_EventDomainAttribute attrib, size_t* value_size, void* value) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->DeviceGetEventDomainAttribute(
      device, event_domain, attrib, value_size, value);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::DisableKernelReplayMode(CUcontext context) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->DisableKernelReplayMode(context);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EnableKernelReplayMode(CUcontext context) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EnableKernelReplayMode(context);
  if (error == CUPTI_SUCCESS) {
    auto f =
        std::bind(&CuptiErrorManager::DisableKernelReplayMode, this, context);
    RegisterUndoFunction(f);
  }
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::DeviceEnumMetrics(CUdevice device,
                                                 size_t* arraySizeBytes,
                                                 CUpti_MetricID* metricArray) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->DeviceEnumMetrics(device, arraySizeBytes, metricArray);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::DeviceGetNumEventDomains(CUdevice device,
                                                        uint32_t* num_domains) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->DeviceGetNumEventDomains(device, num_domains);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventDomainEnumEvents(
    CUpti_EventDomainID event_domain, size_t* array_size_bytes,
    CUpti_EventID* event_array) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventDomainEnumEvents(
      event_domain, array_size_bytes, event_array);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventDomainGetNumEvents(
    CUpti_EventDomainID event_domain, uint32_t* num_events) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->EventDomainGetNumEvents(event_domain, num_events);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGetAttribute(CUpti_EventID event,
                                                 CUpti_EventAttribute attrib,
                                                 size_t* value_size,
                                                 void* value) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->EventGetAttribute(event, attrib, value_size, value);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGetIdFromName(CUdevice device,
                                                  const char* event_name,
                                                  CUpti_EventID* event) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGetIdFromName(device, event_name, event);
  ALLOW_ERROR(error, CUPTI_ERROR_INVALID_EVENT_NAME);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGroupDisable(CUpti_EventGroup event_group) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGroupDisable(event_group);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGroupEnable(CUpti_EventGroup event_group) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGroupEnable(event_group);
  if (error == CUPTI_SUCCESS) {
    auto f =
        std::bind(&CuptiErrorManager::EventGroupDisable, this, event_group);
    RegisterUndoFunction(f);
  }
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGroupGetAttribute(
    CUpti_EventGroup event_group, CUpti_EventGroupAttribute attrib,
    size_t* value_size, void* value) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGroupGetAttribute(event_group, attrib,
                                                         value_size, value);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGroupReadEvent(
    CUpti_EventGroup event_group, CUpti_ReadEventFlags flags,
    CUpti_EventID event, size_t* event_value_buffer_size_bytes,
    uint64_t* event_value_buffer) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGroupReadEvent(
      event_group, flags, event, event_value_buffer_size_bytes,
      event_value_buffer);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGroupSetAttribute(
    CUpti_EventGroup event_group, CUpti_EventGroupAttribute attrib,
    size_t value_size, void* value) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGroupSetAttribute(event_group, attrib,
                                                         value_size, value);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGroupSetsCreate(
    CUcontext context, size_t event_id_array_size_bytes,
    CUpti_EventID* event_id_array, CUpti_EventGroupSets** event_group_passes) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGroupSetsCreate(
      context, event_id_array_size_bytes, event_id_array, event_group_passes);
  if (error == CUPTI_SUCCESS) {
    auto f = std::bind(&CuptiErrorManager::EventGroupSetsDestroy, this,
                       *event_group_passes);
    RegisterUndoFunction(f);
  }
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGroupSetsDestroy(
    CUpti_EventGroupSets* event_group_sets) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGroupSetsDestroy(event_group_sets);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

// CUPTI metric API
CUptiResult CuptiErrorManager::DeviceGetNumMetrics(CUdevice device,
                                                   uint32_t* num_metrics) {
  IGNORE_CALL_IF_DISABLED;
  // Disable heap checking for the first CUPTI metric API. See b/22091576.
  absl::LeakCheckDisabler disabler;
  CUptiResult error = interface_->DeviceGetNumMetrics(device, num_metrics);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::MetricGetIdFromName(CUdevice device,
                                                   const char* metric_name,
                                                   CUpti_MetricID* metric) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->MetricGetIdFromName(device, metric_name, metric);
  ALLOW_ERROR(error, CUPTI_ERROR_INVALID_METRIC_NAME);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::MetricGetNumEvents(CUpti_MetricID metric,
                                                  uint32_t* num_events) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->MetricGetNumEvents(metric, num_events);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::MetricEnumEvents(
    CUpti_MetricID metric, size_t* event_id_array_size_bytes,
    CUpti_EventID* event_id_array) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->MetricEnumEvents(
      metric, event_id_array_size_bytes, event_id_array);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::MetricGetAttribute(CUpti_MetricID metric,
                                                  CUpti_MetricAttribute attrib,
                                                  size_t* value_size,
                                                  void* value) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->MetricGetAttribute(metric, attrib, value_size, value);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::MetricGetValue(
    CUdevice device, CUpti_MetricID metric, size_t event_id_array_size_bytes,
    CUpti_EventID* event_id_array, size_t event_value_array_size_bytes,
    uint64_t* event_value_array, uint64_t time_duration,
    CUpti_MetricValue* metric_value) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->MetricGetValue(
      device, metric, event_id_array_size_bytes, event_id_array,
      event_value_array_size_bytes, event_value_array, time_duration,
      metric_value);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

void CuptiErrorManager::UndoAndDisable() {
  if (undo_disabled_) {  // prevent deadlock
    return;
  }
  // Iterates undo log and call undo APIs one by one.
  mutex_lock lock(undo_stack_mu_);
  undo_disabled_ = true;
  while (!undo_stack_.empty()) {
    LOG(ERROR) << "CuptiErrorManager is disabling profiling automatically.";
    undo_stack_.back()();
    undo_stack_.pop_back();
  }
  undo_disabled_ = false;
  disabled_ = 1;
}

CUptiResult CuptiErrorManager::GetResultString(CUptiResult result,
                                               const char** str) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->GetResultString(result, str);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::GetContextId(CUcontext context,
                                            uint32_t* context_id) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->GetContextId(context, context_id);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::GetStreamIdEx(CUcontext context, CUstream stream,
                                             uint8_t per_thread_stream,
                                             uint32_t* stream_id) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->GetStreamIdEx(context, stream, per_thread_stream, stream_id);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

void CuptiErrorManager::CleanUp() {
  if (undo_disabled_) {  // prevent deadlock
    return;
  }
  mutex_lock lock(undo_stack_mu_);
  undo_disabled_ = true;
  while (!undo_stack_.empty()) {
    undo_stack_.pop_back();
  }
  undo_disabled_ = false;
}

std::string CuptiErrorManager::ResultString(CUptiResult error) const {
  const char* error_message = nullptr;
  if (interface_->GetResultString(error, &error_message) == CUPTI_SUCCESS &&
      error_message != nullptr) {
    return error_message;
  }
  return "";
}

}  // namespace profiler
}  // namespace tensorflow
