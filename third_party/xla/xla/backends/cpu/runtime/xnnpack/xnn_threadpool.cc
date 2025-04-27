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
#include <tuple>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "pthreadpool.h"
#include "xla/backends/cpu/runtime/parallel_loop_runner.h"
#include "xla/tsl/concurrency/async_value_ref.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

// `pthreadpool` API implementation on top of ParallelLoopRunner.
//
// At link time weak `pthreadpool` symbols resolved to our own implementation,
// and we use pointer tagging to distinguish between our custom pthreadpools and
// the native pthreadpools, and dispatch to strong pthreadpool symbol aliases.
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

// We rely on the pointer tagging to identify custom pthreadpools. We assume
// that native pthreadpool is at least std::max_align_t aligned and we can use
// the lowest bit to mark the custom pthreadpool.
static constexpr uintptr_t kCustomPthreadpoolTag = 1;

static bool IsCustomPthreadpool(pthreadpool_t threadpool) {
  return IsCustomPthreadpoolEnabled() &&
         (reinterpret_cast<uintptr_t>(threadpool) & kCustomPthreadpoolTag);
}

static ParallelLoopRunner* Cast(pthreadpool_t threadpool) {
  CHECK(IsCustomPthreadpoolEnabled()) << "Custom pthreadpool is not enabled";
  CHECK(IsCustomPthreadpool(threadpool)) << "Not a custom pthreadpool";

  return reinterpret_cast<ParallelLoopRunner*>(  // REINTERPRET_CAST_OK=ok
      reinterpret_cast<uintptr_t>(threadpool) &  // REINTERPRET_CAST_OK=ok
      ~kCustomPthreadpoolTag);
}

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
    pthreadpool_t threadpool = pthreadpool_create(runner->num_threads());
    CHECK(!IsCustomPthreadpool(threadpool))
        << "Default pthreadpool tagged as a custom pthreadpool";
    return threadpool;
  }

  return reinterpret_cast<pthreadpool_t>(    // REINTERPRET_CAST_OK=ok
      reinterpret_cast<uintptr_t>(runner) |  // REINTERPRET_CAST_OK=ok
      kCustomPthreadpoolTag);
}

void DestroyCustomPthreadpool(pthreadpool_t threadpool) {  // NOLINT
  if (ABSL_PREDICT_FALSE(threadpool == nullptr)) {
    return;
  }

  // If XLA was built without custom pthreadpool, then it must be the default
  // pthreadpool implementation that we should destroy.
  if constexpr (!IsCustomPthreadpoolEnabled()) {
    pthreadpool_destroy(threadpool);
    return;
  }

  tsl::BlockUntilReady(Cast(threadpool)->done_event());
}

ParallelLoopRunner* GetParallelLoopRunner(pthreadpool_t threadpool) {
  return IsCustomPthreadpool(threadpool) ? Cast(threadpool) : nullptr;
}

//===----------------------------------------------------------------------===//
// C++ implementation of the subset of `pthreadpool` C API.
//===----------------------------------------------------------------------===//

using RangeDim = ParallelLoopRunner::RangeDim;
using TileDim = ParallelLoopRunner::TileDim;

using RangeIndex = ParallelLoopRunner::RangeIndex;
using TileIndex = ParallelLoopRunner::TileIndex;

static size_t GetThreadsCount(pthreadpool_t threadpool) {  // NOLINT
  if (ABSL_PREDICT_FALSE(threadpool == nullptr)) {
    return 1;
  }

  return Cast(threadpool)->num_threads();
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
    ParallelLoopRunner* runner = Cast(threadpool);
    runner->Parallelize(dims..., [function, context](auto... indices) {
      internal::Invoke(function, context, std::make_tuple(), std::make_tuple(),
                       indices...);
    });
  } else {
    internal::InvokeAll<false>(function, context, std::make_tuple(), dims...);
  }
}

template <typename... Dims, typename Fn>
static void ParallelizeDynamic(pthreadpool_t threadpool, Fn function,
                               void* context, Dims... dims) {
  if (ABSL_PREDICT_TRUE(threadpool)) {
    ParallelLoopRunner* runner = Cast(threadpool);
    runner->ParallelizeDynamic(dims..., [function, context](auto... indices) {
      internal::Invoke(function, context, std::make_tuple(), std::make_tuple(),
                       indices...);
    });
  } else {
    internal::InvokeAll<true>(function, context, std::make_tuple(), dims...);
  }
}

}  // namespace xla::cpu

//===----------------------------------------------------------------------===//
// pthreadpool C API implementation on top of the custom loop runner.
//===----------------------------------------------------------------------===//

#if defined(XLA_CPU_USE_CUSTOM_PTHREADPOOL)

// In all APIs below we dispatch to XLA:CPU implementation if the threadpool is
// a custom pthreadpool (we use tagged pointers to detect custom pthreadpools),
// or ot the native pthreadpool implementation otherwise.
//
// IMPORTANT: We override only the small subset of pthreadpool API that we need
// for XLA + XNNPACK integration. The rest of the pthreadpool API calls will go
// to the default pthreadpool implementation.

using namespace xla::cpu;  // NOLINT

#define DEFINE_PTHREADPOOL_FUNCTION_R(result, name, ...)    \
  extern "C" result pthreadpool_##name##_private_impl(...); \
  extern "C" result pthreadpool_##name(__VA_ARGS__)

#define DEFINE_PTHREADPOOL_FUNCTION(name, ...)            \
  extern "C" void pthreadpool_##name##_private_impl(...); \
  extern "C" void pthreadpool_##name(__VA_ARGS__)

DEFINE_PTHREADPOOL_FUNCTION_R(pthreadpool_t, create, size_t num_threads) {
  return pthreadpool_create_private_impl(num_threads);
}

DEFINE_PTHREADPOOL_FUNCTION(destroy, pthreadpool_t threadpool) {
  pthreadpool_destroy_private_impl(threadpool);
}

DEFINE_PTHREADPOOL_FUNCTION_R(size_t, get_threads_count,
                              pthreadpool_t threadpool) {
  if (IsCustomPthreadpool(threadpool)) {
    return GetThreadsCount(threadpool);
  }
  return pthreadpool_get_threads_count_private_impl(threadpool);
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_1d, pthreadpool_t threadpool,
                            pthreadpool_task_1d_t function, void* context,
                            size_t range, uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    Parallelize(threadpool, function, context, RangeDim{range});
  } else {
    pthreadpool_parallelize_1d_private_impl(threadpool, function, context,
                                            range, flags);
  }
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_1d_tile_1d, pthreadpool_t threadpool,
                            pthreadpool_task_1d_tile_1d_t function,
                            void* context, size_t range, size_t tile,
                            uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    Parallelize(threadpool, function, context, TileDim{range, tile});
  } else {
    pthreadpool_parallelize_1d_tile_1d_private_impl(
        threadpool, function, context, range, tile, flags);
  }
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_1d_tile_1d_dynamic,
                            pthreadpool_t threadpool,
                            pthreadpool_task_1d_tile_1d_dynamic_t function,
                            void* context, size_t range, size_t tile,
                            uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    ParallelizeDynamic(threadpool, function, context, TileDim{range, tile});
  } else {
    pthreadpool_parallelize_1d_tile_1d_dynamic_private_impl(
        threadpool, function, context, range, tile, flags);
  }
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_2d, pthreadpool_t threadpool,
                            pthreadpool_task_2d_t function, void* context,
                            size_t range_i, size_t range_j, uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    Parallelize(threadpool, function, context, RangeDim{range_i},
                RangeDim{range_j});
  } else {
    pthreadpool_parallelize_2d_private_impl(threadpool, function, context,
                                            range_i, range_j, flags);
  }
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_2d_tile_1d, pthreadpool_t threadpool,
                            pthreadpool_task_2d_tile_1d_t function,
                            void* context, size_t range_i, size_t range_j,
                            size_t tile_j, uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    Parallelize(threadpool, function, context, RangeDim{range_i},
                TileDim{range_j, tile_j});
  } else {
    pthreadpool_parallelize_2d_tile_1d_private_impl(
        threadpool, function, context, range_i, range_j, tile_j, flags);
  }
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_2d_tile_1d_dynamic,
                            pthreadpool_t threadpool,
                            pthreadpool_task_2d_tile_1d_dynamic_t function,
                            void* context, size_t range_i, size_t range_j,
                            size_t tile_j, uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    ParallelizeDynamic(threadpool, function, context, RangeDim{range_i},
                       TileDim{range_j, tile_j});
  } else {
    pthreadpool_parallelize_2d_tile_1d_dynamic_private_impl(
        threadpool, function, context, range_i, range_j, tile_j, flags);
  }
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_2d_tile_2d, pthreadpool_t threadpool,
                            pthreadpool_task_2d_tile_2d_t function,
                            void* context, size_t range_i, size_t range_j,
                            size_t tile_i, size_t tile_j, uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    Parallelize(threadpool, function, context, TileDim{range_i, tile_i},
                TileDim{range_j, tile_j});
  } else {
    pthreadpool_parallelize_2d_tile_2d_private_impl(
        threadpool, function, context, range_i, range_j, tile_i, tile_j, flags);
  }
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_2d_tile_2d_dynamic,
                            pthreadpool_t threadpool,
                            pthreadpool_task_2d_tile_2d_dynamic_t function,
                            void* context, size_t range_i, size_t range_j,
                            size_t tile_i, size_t tile_j, uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    ParallelizeDynamic(threadpool, function, context, TileDim{range_i, tile_i},
                       TileDim{range_j, tile_j});
  } else {
    pthreadpool_parallelize_2d_tile_2d_dynamic_private_impl(
        threadpool, function, context, range_i, range_j, tile_i, tile_j, flags);
  }
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_3d, pthreadpool_t threadpool,
                            pthreadpool_task_3d_t function, void* context,
                            size_t range_i, size_t range_j, size_t range_k,
                            uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    Parallelize(threadpool, function, context, RangeDim{range_i},
                RangeDim{range_j}, RangeDim{range_k});
  } else {
    pthreadpool_parallelize_3d_private_impl(threadpool, function, context,
                                            range_i, range_j, range_k, flags);
  }
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_3d_tile_1d, pthreadpool_t threadpool,
                            pthreadpool_task_3d_tile_1d_t function,
                            void* context, size_t range_i, size_t range_j,
                            size_t range_k, size_t tile_k, uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    Parallelize(threadpool, function, context, RangeDim{range_i},
                RangeDim{range_j}, TileDim{range_k, tile_k});
  } else {
    pthreadpool_parallelize_3d_tile_1d_private_impl(threadpool, function,
                                                    context, range_i, range_j,
                                                    range_k, tile_k, flags);
  }
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_3d_tile_2d, pthreadpool_t threadpool,
                            pthreadpool_task_3d_tile_2d_t function,
                            void* context, size_t range_i, size_t range_j,
                            size_t range_k, size_t tile_j, size_t tile_k,
                            uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    Parallelize(threadpool, function, context, RangeDim{range_i},
                TileDim{range_j, tile_j}, TileDim{range_k, tile_k});
  } else {
    pthreadpool_parallelize_3d_tile_2d_private_impl(
        threadpool, function, context, range_i, range_j, range_k, tile_j,
        tile_k, flags);
  }
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_3d_tile_2d_dynamic,
                            pthreadpool_t threadpool,
                            pthreadpool_task_3d_tile_2d_dynamic_t function,
                            void* context, size_t range_i, size_t range_j,
                            size_t range_k, size_t tile_j, size_t tile_k,
                            uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    ParallelizeDynamic(threadpool, function, context, RangeDim{range_i},
                       TileDim{range_j, tile_j}, TileDim{range_k, tile_k});
  } else {
    pthreadpool_parallelize_3d_tile_2d_dynamic_private_impl(
        threadpool, function, context, range_i, range_j, range_k, tile_j,
        tile_k, flags);
  }
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_4d_tile_2d, pthreadpool_t threadpool,
                            pthreadpool_task_4d_tile_2d_t function,
                            void* context, size_t range_i, size_t range_j,
                            size_t range_k, size_t range_l, size_t tile_k,
                            size_t tile_l, uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    Parallelize(threadpool, function, context, RangeDim{range_i},
                RangeDim{range_j}, TileDim{range_k, tile_k},
                TileDim{range_l, tile_l});
  } else {
    pthreadpool_parallelize_4d_tile_2d_private_impl(
        threadpool, function, context, range_i, range_j, range_k, range_l,
        tile_k, tile_l, flags);
  }
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_4d_tile_2d_dynamic,
                            pthreadpool_t threadpool,
                            pthreadpool_task_4d_tile_2d_dynamic_t function,
                            void* context, size_t range_i, size_t range_j,
                            size_t range_k, size_t range_l, size_t tile_k,
                            size_t tile_l, uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    ParallelizeDynamic(threadpool, function, context, RangeDim{range_i},
                       RangeDim{range_j}, TileDim{range_k, tile_k},
                       TileDim{range_l, tile_l});
  } else {
    pthreadpool_parallelize_4d_tile_2d_dynamic_private_impl(
        threadpool, function, context, range_i, range_j, range_k, range_l,
        tile_k, tile_l, flags);
  }
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_5d, pthreadpool_t threadpool,
                            pthreadpool_task_5d_t function, void* context,
                            size_t range_i, size_t range_j, size_t range_k,
                            size_t range_l, size_t range_m, uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    Parallelize(threadpool, function, context, RangeDim{range_i},
                RangeDim{range_j}, RangeDim{range_k}, RangeDim{range_l},
                RangeDim{range_m});
  } else {
    pthreadpool_parallelize_5d_private_impl(threadpool, function, context,
                                            range_i, range_j, range_k, range_l,
                                            range_m, flags);
  }
}

DEFINE_PTHREADPOOL_FUNCTION(parallelize_5d_tile_2d, pthreadpool_t threadpool,
                            pthreadpool_task_5d_tile_2d_t function,
                            void* context, size_t range_i, size_t range_j,
                            size_t range_k, size_t range_l, size_t range_m,
                            size_t tile_l, size_t tile_m, uint32_t flags) {
  if (IsCustomPthreadpool(threadpool)) {
    Parallelize(threadpool, function, context, RangeDim{range_i},
                RangeDim{range_j}, RangeDim{range_k}, TileDim{range_l, tile_l},
                TileDim{range_m, tile_m});
  } else {
    pthreadpool_parallelize_5d_tile_2d_private_impl(
        threadpool, function, context, range_i, range_j, range_k, range_l,
        range_m, tile_l, tile_m, flags);
  }
}

#endif  // XLA_CPU_USE_CUSTOM_PTHREADPOOL
