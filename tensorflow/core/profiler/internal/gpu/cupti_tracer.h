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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_TRACER_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_TRACER_H_

#include "absl/types/optional.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/include/nvtx3/nvToolsExt.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/internal/gpu/cupti_collector.h"
#include "tensorflow/core/profiler/internal/gpu/cupti_interface.h"

namespace tensorflow {
namespace profiler {

struct CuptiTracerOptions {
  bool enable_activity_api = true;

  // Use cuda events to enclose the kernel/memcpy to measure device activity.
  // enable_event_based_activity, if true, will override the enable_activity_api
  // setting.
  bool enable_event_based_activity = false;

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

class CuptiDriverApiHook {
 public:
  virtual ~CuptiDriverApiHook() {}

  virtual Status OnDriverApiEnter(int device_id, CUpti_CallbackDomain domain,
                                  CUpti_CallbackId cbid,
                                  const CUpti_CallbackData* callback_info) = 0;
  virtual Status OnDriverApiExit(int device_id, CUpti_CallbackDomain domain,
                                 CUpti_CallbackId cbid,
                                 const CUpti_CallbackData* callback_info) = 0;
  virtual Status SyncAndFlush() = 0;

 protected:
  static Status AddDriverApiCallbackEvent(
      CuptiTraceCollector* collector, CuptiInterface* cupti_interface,
      int device_id, uint64 start_tsc, uint64 end_tsc,
      CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
      const CUpti_CallbackData* callback_info);
};

// The class use to enable cupti callback/activity API and forward the collected
// trace events to CuptiTraceCollector. There should be only one CuptiTracer
// per process.
class CuptiTracer {
 public:
  // Returns a pointer to singleton CuptiTracer.
  static CuptiTracer* GetCuptiTracerSingleton();

  // Only one profile session can be live in the same time.
  bool IsAvailable() const;
  bool NeedRootAccess() const { return need_root_access_; }

  void Enable(const CuptiTracerOptions& option, CuptiTraceCollector* collector);
  void Disable();

  Status HandleCallback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                        const CUpti_CallbackData* callback_info);

  // This function is public because called from registered callback.
  Status ProcessActivityBuffer(CUcontext context, uint32_t stream_id,
                               uint8_t* buffer, size_t size);

  static uint64 GetTimestamp();
  static int NumGpus();
  // Returns the error (if any) when using libcupti.
  static std::string ErrorIfAny();

 protected:
  // protected constructor for injecting mock cupti interface for testing.
  explicit CuptiTracer(CuptiInterface* cupti_interface)
      : num_gpus_(NumGpus()), cupti_interface_(cupti_interface) {}

 private:
  Status EnableApiTracing();
  Status EnableActivityTracing();
  Status DisableApiTracing();
  Status DisableActivityTracing();
  Status Finalize();
  void ConfigureActivityUnifiedMemoryCounter(bool enable);
  Status HandleNVTXCallback(CUpti_CallbackId cbid,
                            const CUpti_CallbackData* cbdata);

  int num_gpus_;
  absl::optional<CuptiTracerOptions> option_;
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

  TF_DISALLOW_COPY_AND_ASSIGN(CuptiTracer);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_TRACER_H_
