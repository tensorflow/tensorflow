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

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xla/backends/profiler/gpu/cupti_buffer_events.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
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
  // Whether to dump the graph nope mapping.
  bool dump_graph_nope_mapping = false;
};
// This struct will be used to store the PM Sampling data.
// Same as CUDA 12.6.2 extras/CUPTI/samples/pm_sampling/pm_sampling.h
struct SamplerRange {
  size_t range_index;
  uint64_t start_timestamp_ns;
  uint64_t end_timestamp_ns;
  // Instead of map<std::string, double> in the above sample code, we use to
  // vector<double> to save memory.
  std::vector<double> metric_values;
};

// This is to hold multiple PM Sampling data with one std::string vector for
// holding the names.
class PmSamples {
 public:
  PmSamples(std::vector<std::string> metrics,
            std::vector<SamplerRange> sampler_ranges)
      : metrics_(std::move(metrics)),
        sampler_ranges_(std::move(sampler_ranges)) {}
  void PopulateCounterLine(tsl::profiler::XPlaneBuilder* plane);

 private:
  std::vector<std::string> metrics_;
  std::vector<SamplerRange> sampler_ranges_;
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
  // AddEvent(). If need_callback_events is false, only annotation map and scope
  // range id tree will be merged, all events will be dropped.
  virtual void OnTracerCollectedCallbackData(
      std::vector<CallbackAnnotationsAndEvents> callback_events,
      bool need_callback_events);

  // CuptiTracer tracer now cache all activity buffers during tracing.
  // After tracing stop, the cached activity buffers will be send here.
  // Default behavior is direct process those cached activity events and
  // add it into this class by calling AddEvent().
  virtual void OnTracerCachedActivityBuffers(
      std::list<CuptiActivityBufferManager::ActivityBufferAndSize>
          activity_buffers);

  // Consumer side functions (i.e. called by GPU tracer);
  virtual bool Export(tensorflow::profiler::XSpace* space,
                      uint64_t end_gpu_ns) {
    return true;
  }
  virtual std::string ReportNumEventsIfDropped() { return ""; }

  // Set by the cupti tracer right after tracing is stopped.
  void SetTracingEndTimeNs(uint64_t end_time_ns) {
    tracing_end_time_ns_ = end_time_ns;
  }

  uint64_t GetTracingEndTimeNs() const { return tracing_end_time_ns_; }

  AnnotationMap* annotation_map() { return &annotation_map_; }

  const CuptiTracerCollectorOptions& GetOptions() const { return options_; }

 protected:
  CuptiTracerCollectorOptions options_;
  // map of child_scope_id -> parent_scope_id
  ScopeRangeIdTree scope_range_id_tree_;

 private:
  AnnotationMap annotation_map_;
  uint64_t tracing_end_time_ns_ = 0;

  CuptiTraceCollector(const CuptiTraceCollector&) = delete;
  void operator=(const CuptiTraceCollector&) = delete;
};

std::unique_ptr<CuptiTraceCollector> CreateCuptiCollector(
    const CuptiTracerCollectorOptions& options, uint64_t start_walltime_ns,
    uint64_t start_gputime_ns);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_COLLECTOR_H_
