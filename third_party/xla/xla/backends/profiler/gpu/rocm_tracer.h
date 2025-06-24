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

#include <optional>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/stream_executor/rocm/roctracer_wrapper.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/status.h"
#include "tsl/platform/thread_annotations.h"
#include "tsl/platform/types.h"

namespace xla {
namespace profiler {

enum class RocmTracerSyncTypes {
  InvalidSync = 0,
  StreamSynchronize,  // caller thread wait stream to become empty
  EventSynchronize,   // caller thread will block until event happens
  StreamWait          // compute stream will wait for event to happen
};

struct RocmTracerOptions {
  std::set<uint32_t> api_tracking_set;  // actual api set we want to profile

  // map of domain --> ops for which we need to enable the API callbacks
  // If the ops vector is empty, then enable API callbacks for entire domain
  absl::flat_hash_map<activity_domain_t, std::vector<uint32_t> > api_callbacks;

  // map of domain --> ops for which we need to enable the Activity records
  // If the ops vector is empty, then enable Activity records for entire domain
  absl::flat_hash_map<activity_domain_t, std::vector<uint32_t> >
      activity_tracing;
};

class RocmTracer;

class RocmApiCallbackImpl {
 public:
  RocmApiCallbackImpl(const RocmTracerOptions& options, RocmTracer* tracer,
                      RocmTraceCollector* collector)
      : options_(options), tracer_(tracer), collector_(collector) {}

  absl::Status operator()(uint32_t domain, uint32_t cbid, const void* cbdata);

 private:
  void AddKernelEventUponApiExit(uint32_t cbid, const hip_api_data_t* data,
                                 uint64_t enter_time, uint64_t exit_time);
  void AddNormalMemcpyEventUponApiExit(uint32_t cbid,
                                       const hip_api_data_t* data,
                                       uint64_t enter_time, uint64_t exit_time);
  void AddMemcpyPeerEventUponApiExit(uint32_t cbid, const hip_api_data_t* data,
                                     uint64_t enter_time, uint64_t exit_time);
  void AddMemsetEventUponApiExit(uint32_t cbid, const hip_api_data_t* data,
                                 uint64_t enter_time, uint64_t exit_time);
  void AddMallocFreeEventUponApiExit(uint32_t cbid, const hip_api_data_t* data,
                                     uint32_t device_id, uint64_t enter_time,
                                     uint64_t exit_time);
  void AddStreamSynchronizeEventUponApiExit(uint32_t cbid,
                                            const hip_api_data_t* data,
                                            uint64_t enter_time,
                                            uint64_t exit_time);
  void AddSynchronizeEventUponApiExit(uint32_t cbid, const hip_api_data_t* data,
                                      uint64_t enter_time, uint64_t exit_time);

  RocmTracerOptions options_;
  RocmTracer* tracer_ = nullptr;
  RocmTraceCollector* collector_ = nullptr;
  absl::Mutex api_call_start_mutex_;
  // TODO(rocm-profiler): replace this with absl hashmap
  // keep a map from the corr. id to enter time for API callbacks.
  std::map<uint32_t, uint64_t> api_call_start_time_
      TF_GUARDED_BY(api_call_start_mutex_);
};

class RocmActivityCallbackImpl {
 public:
  RocmActivityCallbackImpl(const RocmTracerOptions& options, RocmTracer* tracer,
                           RocmTraceCollector* collector)
      : options_(options), tracer_(tracer), collector_(collector) {}

  absl::Status operator()(const char* begin, const char* end);

 private:
  void AddHipKernelActivityEvent(const roctracer_record_t* record);
  void AddNormalHipMemcpyActivityEvent(const roctracer_record_t* record);
  void AddHipMemsetActivityEvent(const roctracer_record_t* record);
  void AddHipMallocActivityEvent(const roctracer_record_t* record);
  void AddHipStreamSynchronizeActivityEvent(const roctracer_record_t* record);
  void AddHccKernelActivityEvent(const roctracer_record_t* record);
  void AddNormalHipOpsMemcpyActivityEvent(const roctracer_record_t* record);
  void AddHipOpsMemsetActivityEvent(const roctracer_record_t* record);
  RocmTracerOptions options_;
  RocmTracer* tracer_ = nullptr;
  RocmTraceCollector* collector_ = nullptr;
};

// The class uses roctracer callback/activity API and forward the collected
// trace events to RocmTraceCollector. There should be only one RocmTracer
// per process.
class RocmTracer {
 public:
  // Returns a pointer to singleton RocmTracer.
  static RocmTracer* GetRocmTracerSingleton();

  // Only one profile session can be live in the same time.
  bool IsAvailable() const;

  void Enable(const RocmTracerOptions& options, RocmTraceCollector* collector);
  void Disable();

  absl::Status ApiCallbackHandler(uint32_t domain, uint32_t cbid,
                                  const void* cbdata);
  absl::Status ActivityCallbackHandler(const char* begin, const char* end);

  static uint64_t GetTimestamp();
  static int NumGpus();

  void AddToPendingActivityRecords(uint32_t correlation_id) {
    pending_activity_records_.Add(correlation_id);
  }

  void RemoveFromPendingActivityRecords(uint32_t correlation_id) {
    pending_activity_records_.Remove(correlation_id);
  }

  void ClearPendingActivityRecordsCount() { pending_activity_records_.Clear(); }

  size_t GetPendingActivityRecordsCount() {
    return pending_activity_records_.Count();
  }

 protected:
  // protected constructor for injecting mock cupti interface for testing.
  explicit RocmTracer() : num_gpus_(NumGpus()) {}

 private:
  absl::Status EnableApiTracing();
  absl::Status DisableApiTracing();

  absl::Status EnableActivityTracing();
  absl::Status DisableActivityTracing();

  int num_gpus_;
  std::optional<RocmTracerOptions> options_;
  RocmTraceCollector* collector_ = nullptr;

  bool api_tracing_enabled_ = false;
  bool activity_tracing_enabled_ = false;

  RocmApiCallbackImpl* api_cb_impl_;
  RocmActivityCallbackImpl* activity_cb_impl_;

  class PendingActivityRecords {
   public:
    // add a correlation id to the pending set
    void Add(uint32_t correlation_id) {
      absl::MutexLock lock(&mutex);
      pending_set.insert(correlation_id);
    }
    // remove a correlation id from the pending set
    void Remove(uint32_t correlation_id) {
      absl::MutexLock lock(&mutex);
      pending_set.erase(correlation_id);
    }
    // clear the pending set
    void Clear() {
      absl::MutexLock lock(&mutex);
      pending_set.clear();
    }
    // count the number of correlation ids in the pending set
    size_t Count() {
      absl::MutexLock lock(&mutex);
      return pending_set.size();
    }

   private:
    // set of co-relation ids for which the hcc activity record is pending
    absl::flat_hash_set<uint32_t> pending_set;
    // the callback which processes the activity records (and consequently
    // removes items from the pending set) is called in a separate thread
    // from the one that adds item to the list.
    absl::Mutex mutex;
  };
  PendingActivityRecords pending_activity_records_;

 public:
  // Disable copy and move.
  RocmTracer(const RocmTracer&) = delete;
  RocmTracer& operator=(const RocmTracer&) = delete;
};

}  // namespace profiler
}  // namespace xla
#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_H_
