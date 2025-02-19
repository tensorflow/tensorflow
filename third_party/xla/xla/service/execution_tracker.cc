/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/execution_tracker.h"

#include <memory>
#include <utility>

#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"

namespace xla {

AsyncExecution::AsyncExecution(Backend* backend,
                               std::vector<StreamPool::Ptr> streams,
                               const ExecutionProfile& profile,
                               GlobalDataHandle result)
    : backend_(CHECK_NOTNULL(backend)),
      streams_(std::move(streams)),
      profile_(profile),
      result_(std::move(result)) {
  for (const auto& stream : streams_) {
    CHECK(stream != nullptr);
  }
}

ExecutionTracker::ExecutionTracker() : next_handle_(1) {}

}  // namespace xla
