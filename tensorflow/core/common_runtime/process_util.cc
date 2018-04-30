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

#ifdef INTEL_MKL
#include <omp.h>
#endif
#include <string.h>

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

static thread::ThreadPool* InitComputePool(const SessionOptions& options) {
  int32 inter_op_parallelism_threads =
      options.config.inter_op_parallelism_threads();
  if (inter_op_parallelism_threads == 0) {
    // Default to using the number of cores available in the process.
    inter_op_parallelism_threads = port::NumSchedulableCPUs();
  }

  return new thread::ThreadPool(Env::Default(), "Compute",
                                inter_op_parallelism_threads);
}

}  // namespace

thread::ThreadPool* ComputePool(const SessionOptions& options) {
  static thread::ThreadPool* compute_pool = InitComputePool(options);
  return compute_pool;
}

int32 NumInterOpThreadsFromSessionOptions(const SessionOptions& options) {
  const int32 inter_op = options.config.inter_op_parallelism_threads();
  if (inter_op != 0) return inter_op;
#ifdef INTEL_MKL
  // MKL library executes ops in parallel using OMP threads
  // Set inter_op conservatively to avoid thread oversubscription that could
  // lead to severe perf degradations and OMP resource exhaustion
  const int mkl_intra_op = omp_get_max_threads();
  CHECK_GE(mkl_intra_op, 1);
  const int32 mkl_inter_op = std::max(
      (port::NumSchedulableCPUs() + mkl_intra_op - 1) / mkl_intra_op, 2);
  VLOG(0) << "Creating new thread pool with default inter op setting: "
          << mkl_inter_op
          << ". Tune using inter_op_parallelism_threads for best performance.";
  return mkl_inter_op;
#else
  // Default to using the number of cores available in the process.
  return port::NumSchedulableCPUs();
#endif
}

thread::ThreadPool* NewThreadPoolFromSessionOptions(
    const SessionOptions& options) {
  const int32 num_threads = NumInterOpThreadsFromSessionOptions(options);
  VLOG(1) << "Direct session inter op parallelism threads: " << num_threads;
  return new thread::ThreadPool(options.env, "Compute", num_threads);
}

void SchedClosure(std::function<void()> closure) {
  if (!tracing::EventCollector::IsEnabled()) {
    return Env::Default()->SchedClosure(std::move(closure));
  }
  uint64 id = tracing::GetUniqueArg();
  tracing::RecordEvent(tracing::EventCategory::kScheduleClosure, id);

  Env::Default()->SchedClosure(std::bind(
      [id](std::function<void()> closure) {
        tracing::ScopedRegion region(tracing::EventCategory::kRunClosure, id);
        closure();
      },
      std::move(closure)));
}

void SchedNonBlockingClosureAfter(int64 micros, std::function<void()> closure) {
  Env::Default()->SchedClosureAfter(micros, std::move(closure));
}

}  // namespace tensorflow
