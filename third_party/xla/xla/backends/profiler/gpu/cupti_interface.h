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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_INTERFACE_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_INTERFACE_H_

#include <cstddef>
#include <cstdint>

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/types.h"

namespace xla {
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

  virtual CUptiResult ActivityUsePerThreadBuffer() = 0;

  virtual CUptiResult SetActivityFlushPeriod(uint32_t period_ms) = 0;

  virtual CUptiResult GetDeviceId(CUcontext context, uint32_t* deviceId) = 0;

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

  virtual CUptiResult GetResultString(CUptiResult result, const char** str) = 0;

  virtual CUptiResult GetContextId(CUcontext context, uint32_t* context_id) = 0;

  virtual CUptiResult GetStreamIdEx(CUcontext context, CUstream stream,
                                    uint8_t per_thread_stream,
                                    uint32_t* stream_id) = 0;

  virtual CUptiResult GetGraphId(CUgraph graph, uint32_t* graph_id) = 0;

  virtual CUptiResult GetGraphExecId(CUgraphExec graph_exec,
                                     uint32_t* graph_id) = 0;

  virtual CUptiResult SetThreadIdType(CUpti_ActivityThreadIdType type) = 0;

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
  CuptiInterface(const CuptiInterface&) = delete;
  void operator=(const CuptiInterface&) = delete;
};

CuptiInterface* GetCuptiInterface();

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_INTERFACE_H_
