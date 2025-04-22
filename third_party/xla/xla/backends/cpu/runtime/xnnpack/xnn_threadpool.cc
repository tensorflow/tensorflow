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
#include <tuple>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "pthreadpool.h"
#include "xla/backends/cpu/runtime/parallel_loop_runner.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

// `pthreadpool` API implementation on top of ParallelLoopRunner.
//
// At link time `pthreadpool` symbols resolved to our own implementation. This
// is a temporary hack around the fact that it's impossible to customize
// `pthreadpool` implementation at run time. The downside is that it's
// impossible to have two `pthreadpool` implementations linked into the same
// binary.
//
// WARNING: This is under construction and implements only the subset of the API
// surface which is needed by XNNPACK uses inside XLA.

namespace xla::cpu {

static constexpr bool IsCustomPthreadpoolEnabled() {
#if defined(XLA_CPU_USE_CUSTOM_PTHREADPOOL)
  return true;
#else
  return false;
#endif  // XLA_CPU_USE_CUSTOM_PTHREADPOOL
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
  // If XLA was built without custom pthreadpool, we return a default threadpool
  // implementation. This should never be used in production jobs as it creates
  // and destroys a threadpool for each XNNPACK fusion. We enable this path only
  // for testing on platforms that do not support custom pthreadpool.
  if constexpr (!IsCustomPthreadpoolEnabled()) {
    LOG(WARNING) << absl::StrFormat(
        "Custom XLA pthreadpool is disabled. Create a default pthreadpool with "
        "%d threads.",
        runner->num_threads());
    return pthreadpool_create(runner->num_threads());
  }

  CHECK(IsCustomPthreadpoolEnabled()) << "Custom pthreadpool is not enabled";
  return reinterpret_cast<pthreadpool_t>(  // REINTERPRET_CAST_OK=perfectly safe
      std::make_unique<WrappedParallelLoopRunner>(runner).release());
}

static pthreadpool_t CreateCustomPthreadpool(size_t threads_count) {  // NOLINT
  CHECK(IsCustomPthreadpoolEnabled()) << "Custom pthreadpool is not enabled";
  return reinterpret_cast<pthreadpool_t>(  // REINTERPRET_CAST_OK=perfectly safe
      std::make_unique<OwnedParallelLoopRunner>(threads_count).release());
}

static Pthreadpool* Cast(pthreadpool_t threadpool) {
  CHECK(IsCustomPthreadpoolEnabled()) << "Custom pthreadpool is not enabled";
  return reinterpret_cast<Pthreadpool*>(threadpool);
}

xla::cpu::ParallelLoopRunner* GetParallelLoopRunner(pthreadpool_t threadpool) {
  return IsCustomPthreadpoolEnabled() ? Cast(threadpool)->runner() : nullptr;
}

//===----------------------------------------------------------------------===//
// C++ implementation of the subset of `pthreadpool` C API.
//===----------------------------------------------------------------------===//

using RangeDim = ParallelLoopRunner::RangeDim;
using TileDim = ParallelLoopRunner::TileDim;

using RangeIndex = ParallelLoopRunner::RangeIndex;
using TileIndex = ParallelLoopRunner::TileIndex;

static void DestroyCustomPthreadpool(pthreadpool_t threadpool) {  // NOLINT
  if (ABSL_PREDICT_FALSE(threadpool == nullptr)) {
    return;
  }

  tsl::BlockUntilReady(Cast(threadpool)->runner()->done_event());
  delete Cast(threadpool);
}

static size_t GetThreadsCount(pthreadpool_t threadpool) {  // NOLINT
  if (ABSL_PREDICT_FALSE(threadpool == nullptr)) {
    return 1;
  }

  return Cast(threadpool)->runner()->num_threads();
}

namespace internal {

// A little bit of a template metaprogramming to invoke XNNPACK function at the
// given task indices via recursive parameter pack expansion.

template <typename Fn, typename Offsets, typename Counts>
static void Invoke(Fn function, void* context, Offsets offsets, Counts counts) {
  std::apply([&](auto... args) { (*function)(context, args...); },
             std::tuple_cat(offsets, counts));
}

template <typename Fn, typename Offsets, typename Counts, typename Index,
          typename... Indices>
static void Invoke(Fn function, void* context, Offsets offsets, Counts counts,
                   Index index, Indices... indices) {
  if constexpr (std::is_same_v<Index, RangeIndex>) {
    Invoke(function, context,
           std::tuple_cat(offsets, std::make_tuple(index.offset)), counts,
           indices...);
  } else if constexpr (std::is_same_v<Index, TileIndex>) {
    Invoke(function, context,
           std::tuple_cat(offsets, std::make_tuple(index.offset)),
           std::tuple_cat(counts, std::make_tuple(index.count)), indices...);
  } else {
    static_assert(sizeof(Index) == 0, "Unsupported task index type");
  }
}

// A little bit of template metaprogramming to construct a loop nest to invoke
// XNNPACK function in the caller thread for all task indices.

template <bool dynamic, typename Fn, typename Indices>
static void InvokeAll(Fn function, void* context, Indices indices) {
  std::apply(
      [&](auto... indices) {
        Invoke(function, context, std::make_tuple(), std::make_tuple(),
               indices...);
      },
      indices);
}

template <bool dynamic, typename Fn, typename Indices, typename Dim,
          typename... Dims>
void InvokeAll(Fn function, void* context, Indices indices, Dim dim,
               Dims... dims) {
  // Appends index to the tuple of indices.
  auto index_cat = [&](auto index) {
    return std::tuple_cat(indices, std::make_tuple(index));
  };

  if constexpr (std::is_same_v<Dim, RangeDim>) {
    for (size_t d = 0; d < dim.range; ++d) {
      InvokeAll<dynamic>(function, context, index_cat(RangeIndex{d}), dims...);
    }

  } else if constexpr (std::is_same_v<Dim, TileDim>) {
    if constexpr (dynamic) {
      InvokeAll<dynamic>(function, context, index_cat(TileIndex{0, dim.range}),
                         dims...);
    } else {
      for (size_t d = 0; d < dim.range; d += dim.tile) {
        InvokeAll<dynamic>(
            function, context,
            index_cat(TileIndex{d, std::min(dim.range - d, dim.tile)}),
            dims...);
      }
    }
  } else {
    static_assert(sizeof(Dim) == 0, "Unsupported dimension type");
  }
}

}  // namespace internal

// Executes XNNPACK function in parallel over the given dimensions.
template <typename... Dims, typename Fn>
static void Parallelize(pthreadpool_t threadpool, Fn function, void* context,
                        Dims... dims) {
  if (ABSL_PREDICT_TRUE(threadpool)) {
    Pthreadpool* pthreadpool = Cast(threadpool);
    pthreadpool->runner()->Parallelize(
        dims..., [function, context](auto... indices) {
          internal::Invoke(function, context, std::make_tuple(),
                           std::make_tuple(), indices...);
        });

    // If pthreadpool is owned, it means it was created not by XLA, and the
    // caller expects parallel loops to be blocking.
    if (auto* owned = dynamic_cast<OwnedParallelLoopRunner*>(pthreadpool)) {
      tsl::BlockUntilReady(owned->runner()->done_event());
    }

  } else {
    internal::InvokeAll<false>(function, context, std::make_tuple(), dims...);
  }
}

template <typename... Dims, typename Fn>
static void ParallelizeDynamic(pthreadpool_t threadpool, Fn function,
                               void* context, Dims... dims) {
  if (ABSL_PREDICT_TRUE(threadpool)) {
    Pthreadpool* pthreadpool = Cast(threadpool);
    pthreadpool->runner()->ParallelizeDynamic(
        dims..., [function, context](auto... indices) {
          internal::Invoke(function, context, std::make_tuple(),
                           std::make_tuple(), indices...);
        });

    // If pthreadpool is owned, it means it was created not by XLA, and the
    // caller expects parallel loops to be blocking.
    if (auto* owned = dynamic_cast<OwnedParallelLoopRunner*>(pthreadpool)) {
      tsl::BlockUntilReady(owned->runner()->done_event());
    }

  } else {
    internal::InvokeAll<true>(function, context, std::make_tuple(), dims...);
  }
}

}  // namespace xla::cpu

//===----------------------------------------------------------------------===//
// pthreadpool C API implementation on top of the custom loop runner.
//===----------------------------------------------------------------------===//

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
  xla::cpu::Parallelize(threadpool, function, context,
                        xla::cpu::RangeDim{range});
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
  xla::cpu::Parallelize(threadpool, function, context,
                        xla::cpu::TileDim{range, tile});
}

extern "C" void pthreadpool_parallelize_1d_tile_1d_dynamic(
    pthreadpool_t threadpool, pthreadpool_task_1d_tile_1d_dynamic_t function,
    void* context, size_t range, size_t tile, uint32_t flags) {
  xla::cpu::ParallelizeDynamic(threadpool, function, context,
                               xla::cpu::TileDim{range, tile});
}

extern "C" void pthreadpool_parallelize_2d(pthreadpool_t threadpool,
                                           pthreadpool_task_2d_t function,
                                           void* context, size_t range_i,
                                           size_t range_j, uint32_t flags) {
  xla::cpu::Parallelize(threadpool, function, context,
                        xla::cpu::RangeDim{range_i},
                        xla::cpu::RangeDim{range_j});
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
  xla::cpu::Parallelize(threadpool, function, context,
                        xla::cpu::RangeDim{range_i},
                        xla::cpu::TileDim{range_j, tile_j});
}

extern "C" void pthreadpool_parallelize_2d_tile_1d_dynamic(
    pthreadpool_t threadpool, pthreadpool_task_2d_tile_1d_dynamic_t function,
    void* context, size_t range_i, size_t range_j, size_t tile_j,
    uint32_t flags) {
  xla::cpu::ParallelizeDynamic(threadpool, function, context,
                               xla::cpu::RangeDim{range_i},
                               xla::cpu::TileDim{range_j, tile_j});
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
  xla::cpu::Parallelize(threadpool, function, context,
                        xla::cpu::TileDim{range_i, tile_i},
                        xla::cpu::TileDim{range_j, tile_j});
}

extern "C" void pthreadpool_parallelize_2d_tile_2d_dynamic(
    pthreadpool_t threadpool, pthreadpool_task_2d_tile_2d_dynamic_t function,
    void* context, size_t range_i, size_t range_j, size_t tile_i, size_t tile_j,
    uint32_t flags) {
  xla::cpu::ParallelizeDynamic(threadpool, function, context,
                               xla::cpu::TileDim{range_i, tile_i},
                               xla::cpu::TileDim{range_j, tile_j});
}

extern "C" void pthreadpool_parallelize_2d_tile_2d_with_uarch(
    pthreadpool_t threadpool, pthreadpool_task_2d_tile_2d_with_id_t function,
    void* context, uint32_t default_uarch_index, uint32_t max_uarch_index,
    size_t range_i, size_t range_j, size_t tile_i, size_t tile_j,
    uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_2d_tile_2d_dynamic_with_uarch(
    pthreadpool_t threadpool,
    pthreadpool_task_2d_tile_2d_dynamic_with_id_t function, void* context,
    uint32_t default_uarch_index, uint32_t max_uarch_index, size_t range_i,
    size_t range_j, size_t tile_i, size_t tile_j, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_3d(pthreadpool_t threadpool,
                                           pthreadpool_task_3d_t function,
                                           void* context, size_t range_i,
                                           size_t range_j, size_t range_k,
                                           uint32_t flags) {
  xla::cpu::Parallelize(
      threadpool, function, context, xla::cpu::RangeDim{range_i},
      xla::cpu::RangeDim{range_j}, xla::cpu::RangeDim{range_k});
}

extern "C" void pthreadpool_parallelize_3d_tile_1d(
    pthreadpool_t threadpool, pthreadpool_task_3d_tile_1d_t function,
    void* context, size_t range_i, size_t range_j, size_t range_k,
    size_t tile_k, uint32_t flags) {
  xla::cpu::Parallelize(
      threadpool, function, context, xla::cpu::RangeDim{range_i},
      xla::cpu::RangeDim{range_j}, xla::cpu::TileDim{range_k, tile_k});
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
  xla::cpu::Parallelize(
      threadpool, function, context, xla::cpu::RangeDim{range_i},
      xla::cpu::TileDim{range_j, tile_j}, xla::cpu::TileDim{range_k, tile_k});
}

extern "C" void pthreadpool_parallelize_3d_tile_2d_dynamic(
    pthreadpool_t threadpool, pthreadpool_task_3d_tile_2d_dynamic_t function,
    void* context, size_t range_i, size_t range_j, size_t range_k,
    size_t tile_j, size_t tile_k, uint32_t flags) {
  xla::cpu::ParallelizeDynamic(
      threadpool, function, context, xla::cpu::RangeDim{range_i},
      xla::cpu::TileDim{range_j, tile_j}, xla::cpu::TileDim{range_k, tile_k});
}

extern "C" void pthreadpool_parallelize_3d_tile_2d_with_uarch(
    pthreadpool_t threadpool, pthreadpool_task_3d_tile_2d_with_id_t function,
    void* context, uint32_t default_uarch_index, uint32_t max_uarch_index,
    size_t range_i, size_t range_j, size_t range_k, size_t tile_j,
    size_t tile_k, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_3d_tile_2d_dynamic_with_uarch(
    pthreadpool_t threadpool,
    pthreadpool_task_3d_tile_2d_dynamic_with_id_t function, void* context,
    uint32_t default_uarch_index, uint32_t max_uarch_index, size_t range_i,
    size_t range_j, size_t range_k, size_t tile_j, size_t tile_k,
    uint32_t flags) {
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
  xla::cpu::Parallelize(
      threadpool, function, context, xla::cpu::RangeDim{range_i},
      xla::cpu::RangeDim{range_j}, xla::cpu::TileDim{range_k, tile_k},
      xla::cpu::TileDim{range_l, tile_l});
}

extern "C" void pthreadpool_parallelize_4d_tile_2d_with_uarch(
    pthreadpool_t threadpool, pthreadpool_task_4d_tile_2d_with_id_t function,
    void* context, uint32_t default_uarch_index, uint32_t max_uarch_index,
    size_t range_i, size_t range_j, size_t range_k, size_t range_l,
    size_t tile_k, size_t tile_l, uint32_t flags) {
  LOG(FATAL) << "Not implemented";
}

extern "C" void pthreadpool_parallelize_4d_tile_2d_dynamic(
    pthreadpool_t threadpool, pthreadpool_task_4d_tile_2d_dynamic_t function,
    void* context, size_t range_i, size_t range_j, size_t range_k,
    size_t range_l, size_t tile_k, size_t tile_l, uint32_t flags) {
  xla::cpu::ParallelizeDynamic(
      threadpool, function, context, xla::cpu::RangeDim{range_i},
      xla::cpu::RangeDim{range_j}, xla::cpu::TileDim{range_k, tile_k},
      xla::cpu::TileDim{range_l, tile_l});
}

extern "C" void pthreadpool_parallelize_5d(pthreadpool_t threadpool,
                                           pthreadpool_task_5d_t function,
                                           void* context, size_t range_i,
                                           size_t range_j, size_t range_k,
                                           size_t range_l, size_t range_m,
                                           uint32_t flags) {
  xla::cpu::Parallelize(
      threadpool, function, context, xla::cpu::RangeDim{range_i},
      xla::cpu::RangeDim{range_j}, xla::cpu::RangeDim{range_k},
      xla::cpu::RangeDim{range_l}, xla::cpu::RangeDim{range_m});
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
  xla::cpu::Parallelize(
      threadpool, function, context, xla::cpu::RangeDim{range_i},
      xla::cpu::RangeDim{range_j}, xla::cpu::RangeDim{range_k},
      xla::cpu::TileDim{range_l, tile_l}, xla::cpu::TileDim{range_m, tile_m});
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
