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

#include "tensorflow/core/common_runtime/process_util.h"

#if defined(ENABLE_MKL) && defined(ENABLE_ONEDNN_OPENMP)
#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP
#endif  // defined(ENABLE_MKL) && defined(ENABLE_ONEDNN_OPENMP)
#include <string.h>

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

namespace {

// Use environment setting if specified (init once)
int32 GetEnvNumInterOpThreads() {
  static int32_t env_num_threads = NumInterOpThreadsFromEnvironment();
  return env_num_threads;
}

int32 DefaultNumInterOpThreads() {
#ifndef __ANDROID__
  int32_t env_num_threads = GetEnvNumInterOpThreads();
  if (env_num_threads > 0) {
    return env_num_threads;
  }

  // Default to the maximum parallelism for the current process.
  return port::MaxParallelism();
#else
  // Historically, -D__ANDROID__ resulted in the inter-op threadpool not being
  // used (regardless of what was chosen here); instead, all work was done on
  // the thread(s) calling Session::Run. That's no longer the case, but we'd
  // like to avoid suddenly higher concurrency and peak resource usage (for the
  // same device shape, graph, and options) versus prior versions - as best we
  // can:
  //
  //   - Single Session::Run (none concurrent), and default options:
  //     Behavior is mostly the same as before.
  //
  //   - Concurrent Session::Runs, and default options:
  //     Reduced concurrency versus before.
  //
  //   - Thread-pool size set explicitly (>1):
  //     Increased concurrency versus before.
  //
  // (We assume the first case is the most common)
  return 1;
#endif
}

static thread::ThreadPool* InitComputePool(const SessionOptions& options) {
  int32_t inter_op_parallelism_threads =
      options.config.inter_op_parallelism_threads();
  if (inter_op_parallelism_threads == 0) {
    inter_op_parallelism_threads = DefaultNumInterOpThreads();
  }
  return new thread::ThreadPool(
      Env::Default(), ThreadOptions(), "Compute", inter_op_parallelism_threads,
      !options.config.experimental().disable_thread_spinning(),
      /*allocator=*/nullptr);
}

}  // namespace

thread::ThreadPool* ComputePool(const SessionOptions& options) {
  static thread::ThreadPool* compute_pool = InitComputePool(options);
  return compute_pool;
}

int32 NumInterOpThreadsFromEnvironment() {
  int32_t num;
  const char* val = std::getenv("TF_NUM_INTEROP_THREADS");
  return (val && strings::safe_strto32(val, &num)) ? num : 0;
}

int32 NumIntraOpThreadsFromEnvironment() {
  int32_t num;
  const char* val = std::getenv("TF_NUM_INTRAOP_THREADS");
  return (val && strings::safe_strto32(val, &num)) ? num : 0;
}
#if defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL)
int32 OMPThreadsFromEnvironment() {
  // 1) std::getenv is thread-safe (as long as no other function modifies the
  // host env) from C++11 onward. 2) Most of TF code (except tests and
  // experimental code) doesn't call setenv and unsetenv
  int32 num;
  const char* val = std::getenv("OMP_NUM_THREADS");
  return (val && strings::safe_strto32(val, &num)) ? num : 0;
}

int32 DefaultNumIntraOpThreads() {
  // Use environment setting if specified (init once)
  static int env_num_threads = NumIntraOpThreadsFromEnvironment();
  if (env_num_threads > 0) {
    return env_num_threads;
  }

  // Default to the maximum parallelism for the current process.
  return port::MaxParallelism();
}
#endif  // defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL)
int32 NumInterOpThreadsFromSessionOptions(const SessionOptions& options) {
  const int32_t inter_op = options.config.inter_op_parallelism_threads();
  if (inter_op > 0) return inter_op;
  const int32_t env_inter_op = GetEnvNumInterOpThreads();
  if (env_inter_op > 0) return env_inter_op;

#if defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL)
  if (IsMKLEnabled()) {
    // MKL library executes ops in parallel using OMP threads.
    // Setting inter_op conservatively to avoid thread oversubscription that
    // could lead to severe perf degradations and OMP resource exhaustion.
    // Inter ops are set such that mkl_inter_op * mkl_intra_op <= NumCores.
    const int32 intra_op = options.config.intra_op_parallelism_threads();
    const int32 omp_max_threads = OMPThreadsFromEnvironment();
    const int32 mkl_intra_op =
        (omp_max_threads > 0)
            ? omp_max_threads
            : (intra_op > 0) ? intra_op : DefaultNumIntraOpThreads();
    DCHECK_GE(mkl_intra_op, 1);
    const int32 mkl_inter_op = std::max(
        (DefaultNumInterOpThreads() + mkl_intra_op - 1) / mkl_intra_op, 2);
    VLOG(0)
        << "Creating new thread pool with default inter op setting: "
        << mkl_inter_op
        << ". Tune using inter_op_parallelism_threads for best performance.";
    return mkl_inter_op;
  }
#endif  // defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL)
  return DefaultNumInterOpThreads();
}

thread::ThreadPool* NewThreadPoolFromSessionOptions(
    const SessionOptions& options) {
  const int32_t num_threads = NumInterOpThreadsFromSessionOptions(options);
  VLOG(1) << "Session inter op parallelism threads: " << num_threads;
  return new thread::ThreadPool(
      options.env, ThreadOptions(), "Compute", num_threads,
      !options.config.experimental().disable_thread_spinning(),
      /*allocator=*/nullptr);
}

void SchedClosure(std::function<void()> closure) {
  if (!tracing::EventCollector::IsEnabled()) {
    return Env::Default()->SchedClosure(std::move(closure));
  }
  uint64 id = tracing::GetUniqueArg();
  tracing::RecordEvent(tracing::EventCategory::kScheduleClosure, id);

  Env::Default()->SchedClosure([id, closure = std::move(closure)]() {
    tracing::ScopedRegion region(tracing::EventCategory::kRunClosure, id);
    closure();
  });
}

void SchedNonBlockingClosureAfter(int64_t micros,
                                  std::function<void()> closure) {
  Env::Default()->SchedClosureAfter(micros, std::move(closure));
}

}  // namespace tensorflow
