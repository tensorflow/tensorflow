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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_ERROR_MANAGER_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_ERROR_MANAGER_H_

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"

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

  CUptiResult ActivityUsePerThreadBuffer() override;

  CUptiResult SetActivityFlushPeriod(uint32_t period_ms) override;

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

  CUptiResult GetResultString(CUptiResult result, const char** str) override;

  CUptiResult GetContextId(CUcontext context, uint32_t* context_id) override;

  CUptiResult GetStreamIdEx(CUcontext context, CUstream stream,
                            uint8_t per_thread_stream,
                            uint32_t* stream_id) override;

  CUptiResult GetGraphId(CUgraph graph, uint32_t* graph_id) override;

  CUptiResult GetGraphExecId(CUgraphExec graph_exec,
                             uint32_t* graph_id) override;

  CUptiResult SetThreadIdType(CUpti_ActivityThreadIdType type) override;

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

  CuptiErrorManager(const CuptiErrorManager&) = delete;
  void operator=(const CuptiErrorManager&) = delete;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_ERROR_MANAGER_H_
