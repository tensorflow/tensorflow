/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/cupti_error_manager.h"

#include <utility>

#include "absl/debugging/leak_check.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace profiler {

using tsl::mutex_lock;

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
  ALLOW_ERROR(error, CUPTI_ERROR_INVALID_KIND);
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

CUptiResult CuptiErrorManager::ActivityUsePerThreadBuffer() {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->ActivityUsePerThreadBuffer();
  // Don't disable cupti just because the gpu driver or cuda don't support
  // per-thread activity buffer.
  return error;
}

CUptiResult CuptiErrorManager::SetActivityFlushPeriod(uint32_t period_ms) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->SetActivityFlushPeriod(period_ms);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
};

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
  } else {
    LOG(ERROR) << "cupti" << __func__
               << ": error with domain:" << static_cast<int>(domain)
               << " and callback_id:" << static_cast<int>(callback_id);
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

CUptiResult CuptiErrorManager::GetGraphId(CUgraph graph, uint32_t* graph_id) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->GetGraphId(graph, graph_id);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::GetGraphExecId(CUgraphExec graph_exec,
                                              uint32_t* graph_id) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->GetGraphExecId(graph_exec, graph_id);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::SetThreadIdType(
    CUpti_ActivityThreadIdType type) {
  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->SetThreadIdType(type);
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
}  // namespace xla
