/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/execution_graph_renderer.h"

#include <memory>

#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"

namespace xla {

absl::Mutex renderer_mu(absl::kConstInit);
ExecutionGraphRenderer* graph_renderer ABSL_GUARDED_BY(renderer_mu) = nullptr;

ExecutionGraphRenderer* GetExecutionGraphRenderer() {
  absl::MutexLock lock(&renderer_mu);
  return graph_renderer;
}

void RegisterExecutionGraphRenderer(
    std::unique_ptr<ExecutionGraphRenderer> renderer) {
  absl::MutexLock lock(&renderer_mu);
  if (graph_renderer != nullptr) {
    LOG(WARNING) << "Multiple calls to RegisterExecutionGraphRenderer. Last "
                    "call wins, but because order of initialization in C++ is "
                    "nondeterministic, this may not be what you want.";
    delete graph_renderer;
  }
  graph_renderer = renderer.release();
}

}  // namespace xla
