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
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"
#include "tsl/platform/tracing.h"

namespace tensorflow {

namespace {

constexpr int32 kMaxThreadCount = 1024;

int32 ValidateThreadCount(int32 requested_threads, const char* thread_type) {
  // 0 means auto-detect - valid
  if (requested_threads == 0) {
    return 0;
  }

  // Negative case not valid
  if (requested_threads < 0) {
    LOG(WARNING) << thread_type << " thread count " << requested_threads
                 << " is negative. Using 0 (auto-detect).";
    return 0;
  }

  // Get hardware info for validation
  const int hardware_concurrency = std::thread::hardware_concurrency();
  const int max_reasonable = std::max(hardware_concurrency * 2, 1024);

  // CRITICAL: Clamp to prevent segfault
  if (requested_threads > kMaxThreadCount) {
    LOG(ERROR) << thread_type << " thread count " << requested_threads
               << " exceeds hard limit of " << kMaxThreadCount
               << ". Clamping to " << kMaxThreadCount
               << ". Hardware concurrency: " << hardware_concurrency;
    return kMaxThreadCount;
  }

  // Warn if value is very large
  if (requested_threads > max_reasonable) {
    LOG(WARNING) << thread_type << " thread count " << requested_threads
                 << " is very large for a system with " << hardware_concurrency
                 << " CPUs. This may cause performance degradation.";
  }

  return requested_threads;
}

int32 GetEnvNumInterOpThreads() {
  static int32_t env_num_threads = NumInterOpThreadsFromEnvironment();
  return env_num_threads;
}

int32 DefaultNumInterOpThreads() {
#ifndef __ANDROID__
  int32_t env_num_threads = GetEnvNumInterOpThreads();
  if (env_num_threads > 0) {
    return ValidateThreadCount(env_num_threads, "inter_op_parallelism (env)");
  }

  return port::MaxParallelism();
#else
  return 1;
#endif
}

static thread::ThreadPool* InitComputePool(const SessionOptions& options) {
  int32_t inter_op_parallelism_threads =
      options.config.inter_op_parallelism_threads();

  inter_op_parallelism_threads = ValidateThreadCount(
      inter_op_parallelism_threads, "inter_op_parallelism");

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
  return (val && absl::SimpleAtoi(val, &num)) ? num : 0;
}

int32 NumIntraOpThreadsFromEnvironment() {
  int32_t num;
  const char* val = std::getenv("TF_NUM_INTRAOP_THREADS");
  return (val && absl::SimpleAtoi(val, &num)) ? num : 0;
}

#if defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL)
int32 OMPThreadsFromEnvironment() {
  int32 num;
  const char* val = std::getenv("OMP_NUM_THREADS");
  return (val && strings::safe_strto32(val, &num)) ? num : 0;
}

int32 DefaultNumIntraOpThreads() {
  static int env_num_threads = NumIntraOpThreadsFromEnvironment();
  if (env_num_threads > 0) {
    return ValidateThreadCount(env_num_threads, "intra_op_parallelism (env)");
  }

  return port::MaxParallelism();
}
#endif  // defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL)

int32 NumInterOpThreadsFromSessionOptions(const SessionOptions& options) {
  const int32_t inter_op = ValidateThreadCount(
      options.config.inter_op_parallelism_threads(), "inter_op_parallelism");
  if (inter_op > 0) return inter_op;

  const int32_t env_inter_op = GetEnvNumInterOpThreads();
  if (env_inter_op > 0) return env_inter_op;

#if defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL)
  if (IsMKLEnabled()) {
    const int32 intra_op = ValidateThreadCount(
        options.config.intra_op_parallelism_threads(), "intra_op_parallelism");
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
    const SessionOptions& options, int32_t num_threads) {

  const int32_t validated_num_threads = ValidateThreadCount(
      num_threads, "thread_pool");

  const int32_t num_threads_real =
      validated_num_threads > 0 ? validated_num_threads
                                : NumInterOpThreadsFromSessionOptions(options);

  VLOG(1) << "Session inter op parallelism threads: " << num_threads_real;
  return new thread::ThreadPool(
      options.env, ThreadOptions(), "Compute", num_threads_real,
      !options.config.experimental().disable_thread_spinning(),
      /*allocator=*/nullptr);
}

void SchedClosure(absl::AnyInvocable<void()> closure) {
  if (!tsl::tracing::EventCollector::IsEnabled()) {
    return Env::Default()->SchedClosure(std::move(closure));
  }
  uint64 id = tsl::tracing::GetUniqueArg();
  tsl::tracing::RecordEvent(tsl::tracing::EventCategory::kScheduleClosure, id);

  Env::Default()->SchedClosure([id, closure = std::move(closure)]() mutable {
    tsl::tracing::ScopedRegion region(tsl::tracing::EventCategory::kRunClosure,
                                      id);
    closure();
  });
}

void SchedNonBlockingClosureAfter(int64_t micros,
                                  absl::AnyInvocable<void()> closure) {
  Env::Default()->SchedClosureAfter(micros, std::move(closure));
}

}  // namespace tensorflow
