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

#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"

#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {
// Maximum number of step_ids for which RPC logs can be maintained.
// TODO(mrry): Make this configurable if necessary.
const int32 kWorkerCacheLoggerLimit = 1 << 10;
}  // namespace

void WorkerCacheLogger::SetLogging(bool v) {
  mutex_lock l(count_mu_);
  if (v) {
    ++want_logging_count_;
  } else {
    --want_logging_count_;
    // If RPCs get cancelled, it may be possible for the count
    // to go negative.  This should not be a fatal error, since
    // logging is non-critical.
    if (want_logging_count_ < 0) want_logging_count_ = 0;
  }
}

void WorkerCacheLogger::ClearLogs() {
  mutex_lock l(mu_);
  ClearLogsWithLock();
}

void WorkerCacheLogger::ClearLogsWithLock() {
  for (auto& iter : log_map_) {
    delete iter.second.collector;
  }
  log_map_.clear();
}

bool WorkerCacheLogger::RetrieveLogs(int64 step_id, StepStats* ss) {
  mutex_lock l(mu_);
  LogMap::iterator iter = log_map_.find(step_id);
  if (iter != log_map_.end()) {
    iter->second.collector->Swap(ss);
    delete iter->second.collector;
    log_map_.erase(iter);
    return true;
  }
  return false;
}

void WorkerCacheLogger::Save(const string& device, int64 step_id,
                             NodeExecStats* ns) {
  mutex_lock l(mu_);
  StepLog* sl = &log_map_[step_id];
  if (!sl->collector) {
    sl->collector = new StepStatsCollector(&sl->step_stats);
  }
  sl->collector->Save(device, ns);
  if (log_map_.size() > kWorkerCacheLoggerLimit) {
    // Something's gone wrong.  Just empty the cache.
    ClearLogsWithLock();
  }
}

void WorkerCacheLogger::RecordRecvTensor(int64 step_id, int64 start_usecs,
                                         int64 end_usecs,
                                         const string& tensor_name,
                                         const string& src_device,
                                         const string& dst_device,
                                         int64 bytes) {
  NodeExecStats* ns = new NodeExecStats;
  ns->set_node_name("RecvTensor");
  string byte_string = strings::StrCat("[", bytes, "B] ");
  if (bytes >= 0.1 * 1048576.0) {
    byte_string = strings::Printf("[%.1fMB] ", bytes / 1048576.0);
  }
  ns->set_timeline_label(strings::StrCat(byte_string, tensor_name, " from ",
                                         src_device, " to ", dst_device));
  ns->set_all_start_micros(start_usecs);
  ns->set_op_start_rel_micros(0);
  ns->set_op_end_rel_micros(end_usecs - start_usecs);
  NodeOutput* no = ns->add_output();
  no->set_slot(0);
  // TODO(tucker): Maybe set the dimensions too, but then they'll
  // need to be passed in.
  no->mutable_tensor_description()
      ->mutable_allocation_description()
      ->set_requested_bytes(bytes);
  Save(dst_device, step_id, ns);
}

}  // namespace tensorflow
