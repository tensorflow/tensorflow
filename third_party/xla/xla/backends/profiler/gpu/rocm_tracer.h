/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_H_

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/stream_executor/rocm/roctracer_wrapper.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/status.h"
#include "tsl/platform/types.h"

namespace xla {
namespace profiler {
// forward declare (interface)
class RocmTraceCollector;

struct RocmTracerOptions {
  // maximum number of annotation strings that AnnotationMap in RocmTracer can
  // store. e.g. 1M
  uint64_t max_annotation_strings;
};

// The class use to enable rocprofiler-sdk buffered callback/activity tracing
// and forward the collected trace events to RocmTraceCollector. There should be
// only one RocmTracer per process.
class RocmTracer {
 public:
  // Returns a reference to the singleton instance of RocmTracer.
  // This ensures that only one global instance exists throughout the process
  // lifetime. The first call to this function lazily constructs the instance in
  // a thread-safe manner. Subsequent calls return the same instance, enabling
  // centralized tracer state management.
  static RocmTracer& GetRocmTracerSingleton();

  // Only one profile session can be live in the same time.
  bool IsAvailable() const;

  void Enable(const RocmTracerOptions& options, RocmTraceCollector* collector_);
  void Disable();

  static uint64_t GetTimestamp();
  uint32_t NumGpus() const { return num_gpus_; };
  RocmTraceCollector* collector() { return collector_; }

  int toolInit(rocprofiler_client_finalize_t finalize_func, void* tool_data);
  static void toolFinalize(void* tool_data);

  void TracingCallback(rocprofiler_context_id_t context,
                       rocprofiler_buffer_id_t buffer_id,
                       rocprofiler_record_header_t** headers,
                       size_t num_headers, uint64_t drop_count);

  void CodeObjectCallback(rocprofiler_callback_tracing_record_t record,
                          void* callback_data);

  AnnotationMap* annotation_map() { return &annotation_map_; }

 protected:
  // protected constructor for injecting mock cupti interface for testing.
  RocmTracer() = default;

  void HipApiEvent(const rocprofiler_record_header_t* hdr, RocmTracerEvent* ev);
  void KernelEvent(const rocprofiler_record_header_t* hdr, RocmTracerEvent* ev);
  void MemcpyEvent(const rocprofiler_record_header_t* hdr, RocmTracerEvent* ev);

 private:
  uint32_t num_gpus_{0};
  std::optional<RocmTracerOptions> options_;
  RocmTraceCollector* collector_{nullptr};
  absl::Mutex collector_mutex_;

  bool api_tracing_enabled_{false};
  bool activity_tracing_enabled_{false};

  AnnotationMap annotation_map_{/* default size, e.g. */ 1024 * 1024};

 public:
  using kernel_symbol_data_t =
      rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

  struct ProfilerKernelInfo {
    std::string name;
    kernel_symbol_data_t data;
  };

  using kernel_info_map_t =
      std::unordered_map<rocprofiler_kernel_id_t, ProfilerKernelInfo>;

  using agent_info_map_t = std::unordered_map<uint64_t, rocprofiler_agent_v0_t>;

  using callback_name_info = rocprofiler::sdk::callback_name_info;

  rocprofiler_client_id_t* client_id_{nullptr};
  // Contexts ----------------------------------------------------------
  // for registering kernel names
  rocprofiler_context_id_t utility_context_{};
  // for buffered callback services
  rocprofiler_context_id_t context_{};
  rocprofiler_buffer_id_t buffer_{};

  // Maps & misc -------------------------------------------------------
  kernel_info_map_t kernel_info_{};
  absl::Mutex kernel_lock_;

  callback_name_info name_info_;
  agent_info_map_t agents_;

 public:
  // Disable copy and move.
  RocmTracer(const RocmTracer&) = delete;
  RocmTracer& operator=(const RocmTracer&) = delete;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_H_
