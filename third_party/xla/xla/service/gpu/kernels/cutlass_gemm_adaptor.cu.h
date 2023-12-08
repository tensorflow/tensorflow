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

#ifndef XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_ADAPTOR_CU_H_
#define XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_ADAPTOR_CU_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"
#include "xla/service/gpu/kernels/cutlass_gemm.h"
#include "xla/statusor.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu::kernel::gemm_universal {

// This is a template library that implements an adaptor from a CUTLASS
// GemmUniversal kernel to StreamExecutor primitives for kernel arguments
// packing and kernel launching.
//
// This library is based on `GemmUniversalAdaptor` from CUTLASS itself, but
// instead of targeting CUDA runtime for launching kernels, it targets XLA
// StreamExecutor abstractions, but conceptually it has the same role: wrapping
// device kernels into C++ API to make them launchable on streams.

namespace se = ::stream_executor;

//===----------------------------------------------------------------------===//
// Gemm launch dimension computation.
//===----------------------------------------------------------------------===//

template <typename Gemm>
se::ThreadDim ThreadDim() {
  using Kernel = typename Gemm::GemmKernel;
  return se::ThreadDim(Kernel::kThreadCount, 1, 1);
}

template <typename Gemm>
se::BlockDim BlockDim(const cutlass::gemm::GemmCoord &problem_size) {
  using ThreadblockSwizzle = typename Gemm::ThreadblockSwizzle;
  using ThreadblockShape = typename Gemm::ThreadblockShape;

  cutlass::gemm::GemmCoord tile_size = {
      ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK};

  cutlass::gemm::GemmCoord grid_tiled_shape =
      ThreadblockSwizzle::get_tiled_shape(problem_size, tile_size,
                                          /*split_k_slices=*/1);

  auto grid = ThreadblockSwizzle().get_grid_shape(grid_tiled_shape);

  return se::BlockDim(grid.x, grid.y, grid.z);
}

//===----------------------------------------------------------------------===//
// Gemm strides computation.
//===----------------------------------------------------------------------===//

template <typename Gemm>
int64_t LdA(const cutlass::gemm::GemmCoord &problem_size) {
  using LayoutA = typename Gemm::LayoutA;

  if constexpr (std::is_same_v<LayoutA, cutlass::layout::RowMajor>) {
    return problem_size.k();
  } else {
    static_assert(sizeof(Gemm) == 0, "unsupported layout type");
  }
}

template <typename Gemm>
int64_t LdB(const cutlass::gemm::GemmCoord &problem_size) {
  using LayoutB = typename Gemm::LayoutB;

  if constexpr (std::is_same_v<LayoutB, cutlass::layout::RowMajor>) {
    return problem_size.n();
  } else {
    static_assert(sizeof(Gemm) == 0, "unsupported layout type");
  }
}

template <typename Gemm>
int64_t LdC(const cutlass::gemm::GemmCoord &problem_size) {
  using LayoutC = typename Gemm::LayoutA;

  if constexpr (std::is_same_v<LayoutC, cutlass::layout::RowMajor>) {
    return problem_size.n();
  } else {
    static_assert(sizeof(Gemm) == 0, "unsupported layout type");
  }
}

//===----------------------------------------------------------------------===//
// Packing kernel arguments to CUTLASS kernel parameters struct.
//===----------------------------------------------------------------------===//

using KernelArgsPacking = se::MultiKernelLoaderSpec::KernelArgsPacking;

template <typename Gemm, size_t index>
auto *ArgPtr(const se::KernelArgsDeviceMemoryArray *args,
             const ArgsIndices &indices) {
  if constexpr (index == 0) {
    const void *opaque = args->device_memory_ptr(indices.lhs);
    return static_cast<typename Gemm::ElementA *>(const_cast<void *>(opaque));
  } else if constexpr (index == 1) {
    const void *opaque = args->device_memory_ptr(indices.rhs);
    return static_cast<typename Gemm::ElementB *>(const_cast<void *>(opaque));
  } else if constexpr (index == 2) {
    const void *opaque = args->device_memory_ptr(indices.out);
    return static_cast<typename Gemm::ElementC *>(const_cast<void *>(opaque));
  } else {
    static_assert(sizeof(Gemm) == 0, "illegal Gemm argument index");
  }
}

inline int32_t *SlicePtr(const se::KernelArgsDeviceMemoryArray *args,
                         int64_t index) {
  const void *opaque = args->device_memory_ptr(index);
  return static_cast<int32_t *>(const_cast<void *>(opaque));
}

//===----------------------------------------------------------------------===//
// CUTLASS 2x arguments packing
//===----------------------------------------------------------------------===//

template <typename Gemm>
struct ArgsPacking {
  // CUTLASS operator type parameters.
  using Accumulator = typename Gemm::ElementAccumulator;
  using Arguments = typename Gemm::Arguments;
  using Kernel = typename Gemm::GemmKernel;

  // CUTLASS kernel type parameters.
  using Params = typename Kernel::Params;

  static KernelArgsPacking For(cutlass::gemm::GemmCoord problem_size,
                               const ArgsIndices &indices,
                               const DynamicSliceIndices &slices,
                               int32_t device_sms);
};

template <typename Gemm>
KernelArgsPacking ArgsPacking<Gemm>::For(cutlass::gemm::GemmCoord problem_size,
                                         const ArgsIndices &indices,
                                         const DynamicSliceIndices &slices,
                                         int32_t device_sms) {
  // Sanity check that we do not accidentally get a giant parameters struct.
  static_assert(sizeof(Params) < 512,
                "Params struct size is unexpectedly large");

  return [=](const se::Kernel &kernel, const se::KernelArgs &args)
             -> StatusOr<std::unique_ptr<se::KernelArgsPackedArrayBase>> {
    auto *mem_args = se::Cast<se::KernelArgsDeviceMemoryArray>(&args);

    cutlass::Status can_implement = Kernel::can_implement(problem_size);
    if (can_implement != cutlass::Status::kSuccess) {
      return absl::InternalError(absl::StrCat(
          "CUTLASS kernel can not implement gemm for a given problem size",
          ": m=", problem_size.m(), ", n=", problem_size.n(),
          ", k=", problem_size.k()));
    }

    auto lda = LdA<Gemm>(problem_size);
    auto ldb = LdB<Gemm>(problem_size);
    auto ldc = LdC<Gemm>(problem_size);

    auto ptr_a = ArgPtr<Gemm, 0>(mem_args, indices);
    auto ptr_b = ArgPtr<Gemm, 1>(mem_args, indices);
    auto ptr_c = ArgPtr<Gemm, 2>(mem_args, indices);

    auto mode = cutlass::gemm::GemmUniversalMode::kGemm;

    // TODO(ezhulenev): We hardcode parameters for `LinearCombination`
    // epilogue, however `Gemm` template can be compiled with arbitrary
    // epilogues. We have to support custom epilogues in a way that does not
    // leak cutlass types via the public API function signature.
    Accumulator alpha{1.0};
    Accumulator beta{0.0};

    // CUTLASS operation arguments.
    Arguments arguments(mode, problem_size,
                        1,                           // batch
                        {alpha, beta},               // epilogue
                        ptr_a, ptr_b, ptr_c, ptr_c,  // pointers
                        0, 0, 0, 0,                  // batch strides
                        lda, ldb, ldc, ldc           // strides
    );

    // We keep max_occupancy in a static variable as currently for all
    // practical purposes all stream executors in the process have identical
    // underlying devices, and there is no need to repeatedly query this
    // property.
    static int32_t shared_mem_bytes = sizeof(typename Kernel::SharedStorage);
    static int32_t sm_occupancy =
        kernel.GetMaxOccupiedBlocksPerCore(ThreadDim<Gemm>(), shared_mem_bytes)
            .value_or(1);

    // TODO(ezhulenev): In theory when sm_occupancy is 0 we should not be able
    // to run kernels, and we could return error here, however in practice
    // it's not true, and kernels with 0 occupancy run just fine! Figure out
    // where is the problem, and how we can reliably use sm occupancy numbers.
    //
    // TODO(ezhulenv): We need to set kernel dynamic shmem limit before asking
    // for sm occupancy, it's likely why we get 0 today.
    if (sm_occupancy == 0) {
      se::ThreadDim threads = ThreadDim<Gemm>();
      LOG_FIRST_N(WARNING, 1)
          << "CUTLASS gemm kernel reported 0 occupancy: threads_per_block="
          << (threads.x * threads.y * threads.z)
          << ", dynamic_shared_memory_bytes=" << shared_mem_bytes;
    }

    // Convert CUTLASS operation arguments to a device kernel parameters.
    Params params(arguments, device_sms, sm_occupancy);

    // Optionally set up dynamic slice parameters to allow kernel adjust
    // buffer pointers passed via `params`.
    DynamicSliceParams slice_params;
    if (slices.out.has_value()) {
      slice_params.out = SlicePtr(mem_args, *slices.out);
    }

    return se::PackKernelArgs<Params, DynamicSliceParams>(
        args.number_of_shared_bytes(), params, slice_params);
  };
}

}  // namespace xla::gpu::kernel::gemm_universal

#endif  // XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_ADAPTOR_CU_H_
