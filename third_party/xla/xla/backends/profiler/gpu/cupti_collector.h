/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_COLLECTOR_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_COLLECTOR_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xla/backends/profiler/gpu/cupti_buffer_events.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

struct CuptiTracerCollectorOptions {
  // Maximum number of events to collect from callback API; if -1, no limit.
  // if 0, the callback API is enabled to build a correlation map, but no
  // events are collected.
  uint64_t max_callback_api_events = 2 * 1024 * 1024;
  // Maximum number of events to collect from activity API; if -1, no limit.
  uint64_t max_activity_api_events = 2 * 1024 * 1024;
  // Maximum number of annotation strings that we can accommodate.
  uint64_t max_annotation_strings = 1024 * 1024;
  // Number of GPUs involved.
  uint32_t num_gpus;
};

class CuptiTraceCollector {
 public:
  explicit CuptiTraceCollector(const CuptiTracerCollectorOptions& options)
      : options_(options),
        annotation_map_(options.max_annotation_strings, options.num_gpus) {}
  virtual ~CuptiTraceCollector() {}

  // Producer side functions (i.e. called by CuptiTracer).
  virtual void AddEvent(CuptiTracerEvent&& event) = 0;
  virtual void OnEventsDropped(const std::string& reason,
                               uint32_t num_events) = 0;
  virtual void Flush() = 0;

  // After CuptiTracer stop, collected per-thread callback data from threads
  // will be send here. Default behavior are: a) create merged annotation map
  // (for later activity event usage), and b) direct add all event by calling
  // AddEvent(). Yet collector could just save those callback events without
  // processing now, but merge annotation and AddEvent() later when needed, such
  // as during export(). If need_callback_events is false, only annotation map
  // will be merged, all events will be dropped.
  virtual void OnTracerCollectedCallbackData(
      std::vector<CallbackAnnotationsAndEvents> callback_events,
      bool need_callback_events);

  // CuptiTracer tracer now cache all activity buffers during tracing.
  // After tracing stop, the cached activity buffers will be send here.
  // Default behavior is direct process those cached activity events and
  // add it into this class by calling AddEvent().
  // Yet collector could just save activity buffers without processing here,
  // but process and AddEvent() later when needed, such as during export().
  // This could make the profiling stop() timestamp, if used by upper
  // level wrapper, do not contains time used by exporting events.
  virtual void OnTracerCachedActivityBuffers(
      std::unique_ptr<CuptiActivityBufferManager> activity_buffers);

  // Consumer side functions (i.e. called by GPU tracer);
  virtual bool Export(tensorflow::profiler::XSpace* space,
                      uint64_t end_gpu_ns) {
    return true;
  }
  virtual std::string ReportNumEventsIfDropped() { return ""; }

  AnnotationMap* annotation_map() { return &annotation_map_; }

  const CuptiTracerCollectorOptions& GetOptions() const { return options_; }

 protected:
  CuptiTracerCollectorOptions options_;
  bool need_callback_events_ = false;

 private:
  AnnotationMap annotation_map_;

  CuptiTraceCollector(const CuptiTraceCollector&) = delete;
  void operator=(const CuptiTraceCollector&) = delete;
};

std::unique_ptr<CuptiTraceCollector> CreateCuptiCollector(
    const CuptiTracerCollectorOptions& options, uint64_t start_walltime_ns,
    uint64_t start_gputime_ns);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_COLLECTOR_H_
