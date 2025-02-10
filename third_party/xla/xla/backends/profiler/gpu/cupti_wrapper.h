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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_WRAPPER_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_WRAPPER_H_

#include <stddef.h>
#include <stdint.h>

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"

namespace xla {
namespace profiler {

class CuptiWrapper : public xla::profiler::CuptiInterface {
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

  CUptiResult ActivityUsePerThreadBuffer() override;

  CUptiResult SetActivityFlushPeriod(uint32_t period_ms) override;

  CUptiResult GetDeviceId(CUcontext context, uint32_t* deviceId) override;

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

  CUptiResult GetResultString(CUptiResult result, const char** str) override;

  CUptiResult GetContextId(CUcontext context, uint32_t* context_id) override;

  CUptiResult GetStreamIdEx(CUcontext context, CUstream stream,
                            uint8_t per_thread_stream,
                            uint32_t* stream_id) override;

  CUptiResult GetGraphId(CUgraph graph, uint32_t* graph_id) override;

  CUptiResult GetGraphExecId(CUgraphExec graph_exec,
                             uint32_t* graph_id) override;

  CUptiResult SetThreadIdType(CUpti_ActivityThreadIdType type) override;

  void CleanUp() override {}
  bool Disabled() const override { return false; }

 private:
  CuptiWrapper(const CuptiWrapper&) = delete;
  void operator=(const CuptiWrapper&) = delete;
};

// This is an implementation of CuptiWrapper that implements all load bearing
// APIs as no-op. This is a stub that keeps XLA profiler functional, but all
// collected profiles will be empty.
class CuptiWrapperStub : public xla::profiler::CuptiInterface {
 public:
  CuptiWrapperStub() {}

  ~CuptiWrapperStub() override {}

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

  CUptiResult ActivityUsePerThreadBuffer() override;

  CUptiResult SetActivityFlushPeriod(uint32_t period_ms) override;

  CUptiResult GetDeviceId(CUcontext context, uint32_t* deviceId) override;

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

  CUptiResult GetResultString(CUptiResult result, const char** str) override;

  CUptiResult GetContextId(CUcontext context, uint32_t* context_id) override;

  CUptiResult GetStreamIdEx(CUcontext context, CUstream stream,
                            uint8_t per_thread_stream,
                            uint32_t* stream_id) override;

  CUptiResult GetGraphId(CUgraph graph, uint32_t* graph_id) override;

  CUptiResult GetGraphExecId(CUgraphExec graph_exec,
                             uint32_t* graph_id) override;

  CUptiResult SetThreadIdType(CUpti_ActivityThreadIdType type) override;

  void CleanUp() override {}
  bool Disabled() const override { return false; }

 private:
  CuptiWrapperStub(const CuptiWrapperStub&) = delete;
  void operator=(const CuptiWrapperStub&) = delete;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_WRAPPER_H_
