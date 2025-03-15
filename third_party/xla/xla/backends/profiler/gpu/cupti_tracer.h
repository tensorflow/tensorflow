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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_TRACER_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_TRACER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/include/nvtx3/nvToolsExt.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "tsl/platform/types.h"

namespace xla {
namespace profiler {

struct CuptiTracerOptions {
  bool required_callback_api_events = true;
  // The callback ids that will be enabled and monitored, if empty, all
  // Callback ids to be enabled using Callback API.
  // We only care CUPTI_CB_DOMAIN_DRIVER_API domain for now. It is kind of
  // redundant to have both CUPTI_CB_DOMAIN_DRIVER_API and
  // CUPTI_CB_DOMAIN_RUNTIME_API.
  std::vector<CUpti_driver_api_trace_cbid_enum> cbids_selected;
  // Activity kinds to be collected using Activity API. If empty, the Activity
  // API is disable.
  std::vector<CUpti_ActivityKind> activities_selected;
  // Whether to call cuptiFinalize.
  bool cupti_finalize = false;
  // Whether to call cuCtxSynchronize for each device before Stop().
  bool sync_devices_before_stop = false;
  // Whether to enable NVTX tracking, we need this for TensorRT tracking.
  bool enable_nvtx_tracking = false;
};

class CuptiTracer;

class CuptiDriverApiHook {
 public:
  virtual ~CuptiDriverApiHook() {}

  virtual absl::Status OnDriverApiEnter(
      int device_id, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
      const CUpti_CallbackData* callback_info) = 0;
  virtual absl::Status OnDriverApiExit(
      int device_id, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
      const CUpti_CallbackData* callback_info) = 0;
  virtual absl::Status SyncAndFlush() = 0;
};

// The class use to enable cupti callback/activity API and forward the collected
// trace events to CuptiTraceCollector. There should be only one CuptiTracer
// per process.
class CuptiTracer {
 public:
  // Not copyable or movable
  CuptiTracer(const CuptiTracer&) = delete;
  CuptiTracer& operator=(const CuptiTracer&) = delete;

  // Returns a pointer to singleton CuptiTracer.
  static CuptiTracer* GetCuptiTracerSingleton();

  // Only one profile session can be live in the same time.
  bool IsAvailable() const;
  bool NeedRootAccess() const { return need_root_access_; }

  absl::Status Enable(const CuptiTracerOptions& option,
                      CuptiTraceCollector* collector);
  void Disable();

  // Control threads could periodically call this function to flush the
  // collected events to the collector. Note that this function will lock the
  // per-thread data mutex and may impact the performance.
  absl::Status FlushEventsToCollector();

  // Sets the activity event buffer flush period. Set to 0 to disable the
  // periodic flush. Before using the FlushEventsToCollector() function, user
  // either need to set the activity flush period or call the
  // FlushActivityBuffers()
  absl::Status SetActivityFlushPeriod(uint32_t period_ms);

  // Force the cupti to flush activity buffers to this tracer.
  absl::Status FlushActivityBuffers();

  absl::Status HandleCallback(CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid,
                              const CUpti_CallbackData* callback_info);

  // Returns a buffer and its size for CUPTI to store activities. This buffer
  // will be reclaimed when CUPTI makes a callback to ProcessActivityBuffer.
  void RequestActivityBuffer(uint8_t** buffer, size_t* size);

  // Parses CUPTI activity events from activity buffer, and emits events for
  // CuptiTraceCollector. This function is public because called from registered
  // callback.
  absl::Status ProcessActivityBuffer(CUcontext context, uint32_t stream_id,
                                     uint8_t* buffer, size_t size);

  static uint64_t GetTimestamp();
  static int NumGpus();
  // Returns the error (if any) when using libcupti.
  static std::string ErrorIfAny();

  // Returns true if the number of annotation strings is too large. The input
  // count is the per-thread count.
  bool TooManyAnnotationStrings(size_t count) const;

  // Returns true if the total number of callback events across all threads
  // is too large.
  bool TooManyCallbackEvents() const;

  void IncCallbackEventCount() {
    num_callback_events_.fetch_add(1, std::memory_order_relaxed);
  }

  bool IsCallbackApiEventsRequired() const {
    return option_.has_value() ? option_->required_callback_api_events : false;
  }

 protected:
  // protected constructor for injecting mock cupti interface for testing.
  explicit CuptiTracer(CuptiInterface* cupti_interface);

 private:
  // Buffer size and alignment, 32K and 8 as in CUPTI samples.
  static constexpr size_t kBufferSizeInBytes = 32 * 1024;

  std::unique_ptr<CuptiActivityBufferManager> activity_buffers_;
  static_assert(std::atomic<size_t>::is_always_lock_free,
                "std::atomic<size_t> is not lock free! This may cause very bad"
                " profiling overhead in some circumstances.");
  std::atomic<size_t> cupti_dropped_activity_event_count_ = 0;
  std::atomic<size_t> num_activity_events_in_dropped_buffer_ = 0;
  std::atomic<size_t> num_activity_events_in_cached_buffer_ = 0;
  std::atomic<size_t> num_callback_events_ = 0;

  // Clear activity_buffers, reset activity event counters.
  void PrepareActivityStart();

  // Empty all per-thread callback annotations, reset callback event counter.
  void PrepareCallbackStart();

  // Gather all per-thread callback events and annotations.
  std::vector<CallbackAnnotationsAndEvents> GatherCallbackAnnotationsAndEvents(
      bool stop_recording);

  absl::Status EnableApiTracing();
  absl::Status EnableActivityTracing();
  absl::Status DisableApiTracing();
  absl::Status DisableActivityTracing();
  absl::Status Finalize();
  void ConfigureActivityUnifiedMemoryCounter(bool enable);
  absl::Status HandleNVTXCallback(CUpti_CallbackId cbid,
                                  const CUpti_CallbackData* cbdata);
  absl::Status HandleDriverApiCallback(CUpti_CallbackId cbid,
                                       const CUpti_CallbackData* cbdata);
  absl::Status HandleResourceCallback(CUpti_CallbackId cbid,
                                      const CUpti_CallbackData* cbdata);
  int num_gpus_;
  std::optional<CuptiTracerOptions> option_;
  CuptiInterface* cupti_interface_ = nullptr;
  CuptiTraceCollector* collector_ = nullptr;

  // CUPTI 10.1 and higher need root access to profile.
  bool need_root_access_ = false;

  bool api_tracing_enabled_ = false;
  // Cupti handle for driver or runtime API callbacks. Cupti permits a single
  // subscriber to be active at any time and can be used to trace Cuda runtime
  // as and driver calls for all contexts and devices.
  CUpti_SubscriberHandle subscriber_;  // valid when api_tracing_enabled_.

  bool activity_tracing_enabled_ = false;

  std::unique_ptr<CuptiDriverApiHook> cupti_driver_api_hook_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_TRACER_H_
