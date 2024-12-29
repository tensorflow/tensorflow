/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/xnnpack/xnn_threadpool.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/base/call_once.h"
#include "absl/base/optimization.h"
#include "pthreadpool.h"
#include "xla/backends/cpu/runtime/xnnpack/parallel_loop_runner.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/cpu_info.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

// `pthreadpool` API implementation on top of ParallelLoopRunner.
//
// When building with `pthreadpool_header_only` config, `pthreadpool` becomes a
// header-only library, and we implement the API on top of ParallelLoopRunner.
//
// At link time `pthreadpool` symbols resolved to our own implementation. This
// is a temporary hack around the fact that it's impossible to customize
// `pthreadpool` implementation at run time. The downsize is that it's
// impossible to have two `pthreadpool` implementations linked into the same
// binary.
//
// WARNING: This is under construction and implements only the subset of the API
// surface which is needed by XNNPACK uses inside XLA.

namespace xla::cpu {

bool IsCustomPthreadpoolEnabled() {
#if defined(XLA_CPU_USE_CUSTOM_PTHREADPOOL)
  return true;
#else
  return false;
#endif  // XLA_CPU_USE_CUSTOM_PTHREADPOOL
}

// Default XLA:CPU pthreadpool initialized once per process.
static absl::once_flag pthreadpool_init;
static pthreadpool_t default_pthreadpool;

pthreadpool_t DefaultPthreadpool() {
  if (IsCustomPthreadpoolEnabled()) {
    LOG(WARNING) << "Default pthreadpool is not supported when build with "
                    "`--define pthreadpool_header_only=true`";
    return nullptr;
  }

  absl::call_once(pthreadpool_init, []() {
    default_pthreadpool = pthreadpool_create(tsl::port::MaxParallelism());
  });

  return default_pthreadpool;
}

namespace {

class Pthreadpool {
 public:
  virtual ~Pthreadpool() = default;
  virtual ParallelLoopRunner* runner() = 0;
};

// Wraps user-provided parallel loop runner into the custom pthreadpool.
class WrappedParallelLoopRunner : public Pthreadpool {
 public:
  explicit WrappedParallelLoopRunner(ParallelLoopRunner* runner)
      : runner_(runner) {}
  ParallelLoopRunner* runner() final { return runner_; }

 private:
  ParallelLoopRunner* runner_;
};

// Wraps newly created thread pool into the custom pthreadpool.
class OwnedParallelLoopRunner : public Pthreadpool {
 public:
  explicit OwnedParallelLoopRunner(size_t threads_count)
      : thread_pool_(tsl::Env::Default(), "xnn_threadpool", threads_count),
        device_(thread_pool_.AsEigenThreadPool(), threads_count),
        runner_(&device_) {}

  ParallelLoopRunner* runner() final { return &runner_; }

 private:
  tsl::thread::ThreadPool thread_pool_;
  Eigen::ThreadPoolDevice device_;
  ParallelLoopRunner runner_;
};

}  // namespace

pthreadpool_t CreateCustomPthreadpool(ParallelLoopRunner* runner) {
  if (IsCustomPthreadpoolEnabled()) {
    return reinterpret_cast<pthreadpool_t>(
        std::make_unique<WrappedParallelLoopRunner>(runner).release());
  }
  LOG(FATAL) << "To use custom pthreadpool, build with "
                "`--define pthreadpool_header_only=true`";
}

static pthreadpool_t CreateCustomPthreadpool(size_t threads_count) {  // NOLINT
  if (IsCustomPthreadpoolEnabled()) {
    return reinterpret_cast<pthreadpool_t>(
        std::make_unique<OwnedParallelLoopRunner>(threads_count).release());
  }
  LOG(FATAL) << "To use custom pthreadpool, build with "
                "`--define pthreadpool_header_only=true`";
}

static Pthreadpool* Cast(pthreadpool_t threadpool) {
  return reinterpret_cast<Pthreadpool*>(threadpool);
}

xla::cpu::ParallelLoopRunner* GetParallelLoopRunner(pthreadpool_t threadpool) {
  return IsCustomPthreadpoolEnabled() ? Cast(threadpool)->runner() : nullptr;
}

//===----------------------------------------------------------------------===//
// C++ implementation of the subset of `pthreadpool` C API.
//===----------------------------------------------------------------------===//

static void DestroyCustomPthreadpool(pthreadpool_t threadpool) {  // NOLINT
  if (ABSL_PREDICT_FALSE(threadpool == nullptr)) {
    return;
  }

  tsl::BlockUntilReady(Cast(threadpool)->runner()->done_event());
  delete Cast(threadpool);
}

static size_t GetThreadsCount(pthreadpool_t threadpool) {  // NOLINT
  if (ABSL_PREDICT_FALSE(threadpool == nullptr)) {
    return 0;
  }

  return Cast(threadpool)->runner()->num_threads();
}

static void Parallelize1D(  // NOLINT
    pthreadpool_t threadpool, pthreadpool_task_1d_t function, void* context,
    size_t range, uint32_t flags) {
  if (ABSL_PREDICT_FALSE(threadpool == nullptr)) {
    for (size_t i = 0; i < range; ++i) {
      function(context, i);
    }
    return;
  }

  ParallelLoopRunner::Task1D task = [function, context](size_t offset) {
    (*function)(context, offset);
  };
  Cast(threadpool)->runner()->Parallelize(range, task);
}

static void Parallelize1DTile1D(  // NOLINT
    pthreadpool_t threadpool, pthreadpool_task_1d_tile_1d_t function,
    void* context, size_t range, size_t tile, uint32_t flags) {
  if (ABSL_PREDICT_FALSE(threadpool == nullptr)) {
    for (size_t i = 0; i < range; i += tile) {
      function(context, i, std::min(range - i, tile));
    }
    return;
  }

  ParallelLoopRunner::Task1DTile1D task = [function, context](size_t offset,
                                                              size_t extent) {
    (*function)(context, offset, extent);
  };
  Cast(threadpool)->runner()->Parallelize(range, tile, task);
}

static void Parallelize2DTile1D(pthreadpool_t threadpool,  // NOLINT
                                pthreadpool_task_2d_tile_1d_t function,
                                void* context, size_t range_i, size_t range_j,
                                size_t tile_j, uint32_t flags) {
  if (ABSL_PREDICT_FALSE(threadpool == nullptr)) {
    for (size_t i = 0; i < range_i; i++) {
      for (size_t j = 0; j < range_j; j += tile_j) {
        function(context, i, j, std::min(range_j - j, tile_j));
      }
    }
    return;
  }

  ParallelLoopRunner::Task2DTile1D task =
      [function, context](size_t offset_i, size_t offset_j, size_t extent_j) {
        (*function)(context, offset_i, offset_j, extent_j);
      };
  Cast(threadpool)->runner()->Parallelize(range_i, range_j, tile_j, task);
}

static void Parallelize3DTile2D(pthreadpool_t threadpool,  // NOLINT
                                pthreadpool_task_3d_tile_2d_t function,
                                void* context, size_t range_i, size_t range_j,
                                size_t range_k, size_t tile_j, size_t tile_k,
                                uint32_t flags) {
  if (ABSL_PREDICT_FALSE(threadpool == nullptr)) {
    for (size_t i = 0; i < range_i; i++) {
      for (size_t j = 0; j < range_j; j += tile_j) {
        for (size_t k = 0; k < range_k; k += tile_k) {
          function(context, i, j, k, std::min(range_j - j, tile_j),
                   std::min(range_k - k, tile_k));
        }
      }
    }
    return;
  }

  ParallelLoopRunner::Task3DTile2D task =
      [function, context](size_t offset_i, size_t offset_j, size_t offset_k,
                          size_t extent_j, size_t extent_k) {
        (*function)(context, offset_i, offset_j, offset_k, extent_j, extent_k);
      };
  Cast(threadpool)
      ->runner()
      ->Parallelize(range_i, range_j, range_k, tile_j, tile_k, task);
}

}  // namespace xla::cpu

#if defined(XLA_CPU_USE_CUSTOM_PTHREADPOOL)

extern "C" pthreadpool_t pthreadpool_create(size_t threads_count) {
  return xla::cpu::CreateCustomPthreadpool(threads_count);
}

extern "C" void pthreadpool_destroy(pthreadpool_t threadpool) {
  xla::cpu::DestroyCustomPthreadpool(threadpool);
}

extern "C" size_t pthreadpool_get_threads_count(pthreadpool_t threadpool) {
  return xla::cpu::GetThreadsCount(threadpool);
}

extern "C" void pthreadpool_parallelize_1d(pthreadpool_t threadpool,
                                           pthreadpool_task_1d_t function,
                                           void* context, size_t range,
                                           uint32_t flags) {
  xla::cpu::Parallelize1D(threadpool, function, context, range, flags);
}

extern "C" void pthreadpool_parallelize_1d_with_thread(
    pthreadpool_t threadpool, pthreadpool_task_1d_with_thread_t function,
    void* context, size_t range, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_1d_with_uarch(
    pthreadpool_t threadpool, pthreadpool_task_1d_with_id_t function,
    void* context, uint32_t default_uarch_index, uint32_t max_uarch_index,
    size_t range, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_1d_tile_1d(
    pthreadpool_t threadpool, pthreadpool_task_1d_tile_1d_t function,
    void* context, size_t range, size_t tile, uint32_t flags) {
  xla::cpu::Parallelize1DTile1D(threadpool, function, context, range, tile,
                                flags);
}

extern "C" void pthreadpool_parallelize_2d(pthreadpool_t threadpool,
                                           pthreadpool_task_2d_t function,
                                           void* context, size_t range_i,
                                           size_t range_j, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_2d_with_thread(
    pthreadpool_t threadpool, pthreadpool_task_2d_with_thread_t function,
    void* context, size_t range_i, size_t range_j, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_2d_tile_1d(
    pthreadpool_t threadpool, pthreadpool_task_2d_tile_1d_t function,
    void* context, size_t range_i, size_t range_j, size_t tile_j,
    uint32_t flags) {
  xla::cpu::Parallelize2DTile1D(threadpool, function, context, range_i, range_j,
                                tile_j, flags);
}

extern "C" void pthreadpool_parallelize_2d_tile_1d_with_uarch(
    pthreadpool_t threadpool, pthreadpool_task_2d_tile_1d_with_id_t function,
    void* context, uint32_t default_uarch_index, uint32_t max_uarch_index,
    size_t range_i, size_t range_j, size_t tile_j, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_2d_tile_1d_with_uarch_with_thread(
    pthreadpool_t threadpool,
    pthreadpool_task_2d_tile_1d_with_id_with_thread_t function, void* context,
    uint32_t default_uarch_index, uint32_t max_uarch_index, size_t range_i,
    size_t range_j, size_t tile_j, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_2d_tile_2d(
    pthreadpool_t threadpool, pthreadpool_task_2d_tile_2d_t function,
    void* context, size_t range_i, size_t range_j, size_t tile_i, size_t tile_j,
    uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_2d_tile_2d_with_uarch(
    pthreadpool_t threadpool, pthreadpool_task_2d_tile_2d_with_id_t function,
    void* context, uint32_t default_uarch_index, uint32_t max_uarch_index,
    size_t range_i, size_t range_j, size_t tile_i, size_t tile_j,
    uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_3d(pthreadpool_t threadpool,
                                           pthreadpool_task_3d_t function,
                                           void* context, size_t range_i,
                                           size_t range_j, size_t range_k,
                                           uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_3d_tile_1d(
    pthreadpool_t threadpool, pthreadpool_task_3d_tile_1d_t function,
    void* context, size_t range_i, size_t range_j, size_t range_k,
    size_t tile_k, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_3d_tile_1d_with_thread(
    pthreadpool_t threadpool,
    pthreadpool_task_3d_tile_1d_with_thread_t function, void* context,
    size_t range_i, size_t range_j, size_t range_k, size_t tile_k,
    uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_3d_tile_1d_with_uarch(
    pthreadpool_t threadpool, pthreadpool_task_3d_tile_1d_with_id_t function,
    void* context, uint32_t default_uarch_index, uint32_t max_uarch_index,
    size_t range_i, size_t range_j, size_t range_k, size_t tile_k,
    uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
    pthreadpool_t threadpool,
    pthreadpool_task_3d_tile_1d_with_id_with_thread_t function, void* context,
    uint32_t default_uarch_index, uint32_t max_uarch_index, size_t range_i,
    size_t range_j, size_t range_k, size_t tile_k, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_3d_tile_2d(
    pthreadpool_t threadpool, pthreadpool_task_3d_tile_2d_t function,
    void* context, size_t range_i, size_t range_j, size_t range_k,
    size_t tile_j, size_t tile_k, uint32_t flags) {
  xla::cpu::Parallelize3DTile2D(threadpool, function, context, range_i, range_j,
                                range_k, tile_j, tile_k, flags);
}

extern "C" void pthreadpool_parallelize_3d_tile_2d_with_uarch(
    pthreadpool_t threadpool, pthreadpool_task_3d_tile_2d_with_id_t function,
    void* context, uint32_t default_uarch_index, uint32_t max_uarch_index,
    size_t range_i, size_t range_j, size_t range_k, size_t tile_j,
    size_t tile_k, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_4d(pthreadpool_t threadpool,
                                           pthreadpool_task_4d_t function,
                                           void* context, size_t range_i,
                                           size_t range_j, size_t range_k,
                                           size_t range_l, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_4d_tile_1d(
    pthreadpool_t threadpool, pthreadpool_task_4d_tile_1d_t function,
    void* context, size_t range_i, size_t range_j, size_t range_k,
    size_t range_l, size_t tile_l, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_4d_tile_2d(
    pthreadpool_t threadpool, pthreadpool_task_4d_tile_2d_t function,
    void* context, size_t range_i, size_t range_j, size_t range_k,
    size_t range_l, size_t tile_k, size_t tile_l, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_4d_tile_2d_with_uarch(
    pthreadpool_t threadpool, pthreadpool_task_4d_tile_2d_with_id_t function,
    void* context, uint32_t default_uarch_index, uint32_t max_uarch_index,
    size_t range_i, size_t range_j, size_t range_k, size_t range_l,
    size_t tile_k, size_t tile_l, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_5d(pthreadpool_t threadpool,
                                           pthreadpool_task_5d_t function,
                                           void* context, size_t range_i,
                                           size_t range_j, size_t range_k,
                                           size_t range_l, size_t range_m,
                                           uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_5d_tile_1d(
    pthreadpool_t threadpool, pthreadpool_task_5d_tile_1d_t function,
    void* context, size_t range_i, size_t range_j, size_t range_k,
    size_t range_l, size_t range_m, size_t tile_m, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_5d_tile_2d(
    pthreadpool_t threadpool, pthreadpool_task_5d_tile_2d_t function,
    void* context, size_t range_i, size_t range_j, size_t range_k,
    size_t range_l, size_t range_m, size_t tile_l, size_t tile_m,
    uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_6d(pthreadpool_t threadpool,
                                           pthreadpool_task_6d_t function,
                                           void* context, size_t range_i,
                                           size_t range_j, size_t range_k,
                                           size_t range_l, size_t range_m,
                                           size_t range_n, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_6d_tile_1d(
    pthreadpool_t threadpool, pthreadpool_task_6d_tile_1d_t function,
    void* context, size_t range_i, size_t range_j, size_t range_k,
    size_t range_l, size_t range_m, size_t range_n, size_t tile_n,
    uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_6d_tile_2d(
    pthreadpool_t threadpool, pthreadpool_task_6d_tile_2d_t function,
    void* context, size_t range_i, size_t range_j, size_t range_k,
    size_t range_l, size_t range_m, size_t range_n, size_t tile_m,
    size_t tile_n, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

#endif  // XLA_CPU_USE_CUSTOM_PTHREADPOOL
