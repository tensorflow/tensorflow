/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/logging.h"

#include <algorithm>
#include <vector>

#include "absl/log/log_sink_registry.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"

namespace {

tsl::mutex logsink_update_mutex(tsl::LINKER_INITIALIZED);

std::vector<tsl::TFLogSink*>* log_sinks TF_GUARDED_BY(logsink_update_mutex) =
    new std::vector<tsl::TFLogSink*>();
}  // namespace

namespace tsl {

void TFAddLogSink(TFLogSink* sink) {
  tsl::mutex_lock lock(logsink_update_mutex);
  absl::AddLogSink(sink);
  log_sinks->emplace_back(sink);
}

void TFRemoveLogSink(TFLogSink* sink) {
  tsl::mutex_lock lock(logsink_update_mutex);
  absl::RemoveLogSink(sink);
  auto it = std::find(log_sinks->begin(), log_sinks->end(), sink);
  if (it != log_sinks->end()) log_sinks->erase(it);
}

std::vector<TFLogSink*> TFGetLogSinks() {
  tsl::mutex_lock lock(logsink_update_mutex);
  return *log_sinks;
}

}  // namespace tsl
