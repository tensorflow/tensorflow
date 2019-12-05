/* Copyright 2019 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_CONTEXT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_CONTEXT_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "tensorflow/lite/experimental/ruy/allocator.h"
#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/prepacked_cache.h"
#include "tensorflow/lite/experimental/ruy/thread_pool.h"
#include "tensorflow/lite/experimental/ruy/trace.h"
#include "tensorflow/lite/experimental/ruy/tune.h"

namespace ruy {

// The state private to each Ruy thread.
struct PerThreadState {
  // Each thread may be running on a different microarchitecture. For example,
  // some threads may be on big cores, while others are on little cores. Thus,
  // it's best for the tuning to be per-thread.
  TuningResolver tuning_resolver;
  // Each thread has its own local allocator.
  Allocator allocator;
};

// A Context holds runtime information used by Ruy. It holds runtime resources
// such as the workers thread pool and the allocator (which holds buffers for
// temporary data), as well as runtime options controlling which Paths are
// enabled (typically based on which instruction sets are detected) and how
// many threads to use.
struct Context final {
  Path last_taken_path = Path::kNone;
  Tuning explicit_tuning = Tuning::kAuto;
  // TODO(benoitjacob) rename that thread_pool. Current name is gemmlowp legacy.
  ThreadPool workers_pool;
  int max_num_threads = 1;
  // State for each thread in the thread pool. Entry 0 is the main thread.
  std::vector<std::unique_ptr<PerThreadState>> per_thread_states;
  TracingContext tracing;
  CachePolicy cache_policy = CachePolicy::kNoCache;

  Allocator* GetMainAllocator() {
    if (!main_allocator_) {
      main_allocator_.reset(new Allocator);
    }
    return main_allocator_.get();
  }

  PrepackedCache* GetPrepackedCache() {
    if (!prepacked_cache_) {
      prepacked_cache_.reset(new PrepackedCache);
    }
    return prepacked_cache_.get();
  }

  void EnsureNPerThreadStates(int thread_count) {
    while (per_thread_states.size() < static_cast<std::size_t>(thread_count)) {
      per_thread_states.emplace_back(new PerThreadState);
    }
  }

  Tuning GetMainThreadTuning() {
    EnsureNPerThreadStates(1);
    TuningResolver* tuning_resolver = &per_thread_states[0]->tuning_resolver;
    tuning_resolver->SetTuning(explicit_tuning);
    return tuning_resolver->Resolve();
  }

  template <Path CompiledPaths>
  Path GetPathToTake() {
    last_taken_path =
        GetMostSignificantPath(CompiledPaths & GetRuntimeEnabledPaths());
    return last_taken_path;
  }

  void SetRuntimeEnabledPaths(Path paths);
  Path GetRuntimeEnabledPaths();

 private:
  // Allocator for main thread work before invoking the threadpool.
  // Our simple Allocator does not allow reserving/allocating more blocks
  // while it's already in committed state, so the main thread needs both
  // this allocator, and its per-thread allocator.
  std::unique_ptr<Allocator> main_allocator_;
  std::unique_ptr<PrepackedCache> prepacked_cache_;
  Path runtime_enabled_paths_ = Path::kNone;
};

}  // end namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_CONTEXT_H_
