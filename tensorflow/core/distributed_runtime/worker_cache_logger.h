/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_CACHE_LOGGER_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_CACHE_LOGGER_H_

#include <string>
#include <unordered_map>

#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
class StepStatsCollector;

// WorkerCacheLogger is a thread-safe utility for use by a WorkerCache
// to optionally log some selected RPC activity.  A single instance
// should be owned by a WorkerCache, for use by its RemoteWorker
// instances.

class WorkerCacheLogger {
 public:
  // Start/Stop logging activity.  This function increments/decrements
  // a counter so that if two separate steps turn logging on/off,
  // logging should be on for the union of the durations of both,
  // regardless of relative timing.
  void SetLogging(bool v);

  // Discard any saved log data.
  void ClearLogs();

  // Return logs for the identified step in *ss.  Any returned data will no
  // longer be stored.  Returns true iff *ss was modified.
  bool RetrieveLogs(int64 step_id, StepStats* ss);

  // Return true if there is any outstanding request for logging on
  // the RPC channels.
  bool LoggingActive() {
    mutex_lock l(count_mu_);
    return want_logging_count_ > 0;
  }

  // Generates a NodeExecStats record with the given data, and saves for
  // later retrieval by RetrieveLogs().
  void RecordRecvTensor(int64 step_id, int64 start_usecs, int64 end_usecs,
                        const string& tensor_name, const string& src_device,
                        const string& dst_device, int64 bytes);

  // Generates a NodeExecStats record with the given data, and saves for
  // later retrieval by RetrieveLogs().
  void RecordDataTransfer(int64 step_id, int64 start_usecs, int64 end_usecs,
                          const string& tensor_name, const string& src_device,
                          const string& dst_device, int64 bytes,
                          const string& details,
                          const string& transfer_method_name);

 private:
  mutex count_mu_;
  int32 want_logging_count_ TF_GUARDED_BY(count_mu_) = 0;

  struct StepLog {
    StepStats step_stats;
    StepStatsCollector* collector;
  };
  typedef std::unordered_map<int64, StepLog> LogMap;
  mutex mu_;
  LogMap log_map_ TF_GUARDED_BY(mu_);

  // Records "ns" in log_map_ under the given device and step.
  void Save(const string& device, int64 step_id, NodeExecStats* ns);

  void ClearLogsWithLock() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
};
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_CACHE_LOGGER_H_
