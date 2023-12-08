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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"
#include "xla/service/gpu/kernels/cutlass_gemm.h"

namespace xla::gpu::kernel::gemm_universal {

// This is a template library implementing adaptor from a CUTLASS kernel to
// StreamExecutor primitives for kernel arguments packing and kernel launching.
//
// This library is based on `GemmUniversalAdaptor` from CUTLASS itself, but
// instead of targeting CUDA runtime for launching kernels, it targets XLA
// StreamExecutor abstractions, but conceptually it has the same role: wrapping
// device kernels into C++ API to make them launchable on streams.

//===----------------------------------------------------------------------===//
// Gemm strides computation
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): CUTLASS already has functions in cute to compute strides for
// a GEMM operations/kernels. Remove custom LdA/B/C functions.

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
// CUTLASS 2x Adaptor
//===----------------------------------------------------------------------===//

template <typename Tag>
int32_t Adaptor<Tag>::shared_memory_bytes() {
  return sizeof(typename Traits<Tag>::Kernel::SharedStorage);
};

template <typename Tag>
std::optional<Dim3> Adaptor<Tag>::ClusterDim() {
  return std::nullopt;
}

template <typename Tag>
Dim3 Adaptor<Tag>::ThreadDim() {
  return Dim3{Traits<Tag>::Kernel::kThreadCount};
}

template <typename Tag>
Dim3 Adaptor<Tag>::BlockDim(int32_t m, int32_t n, int32_t k) {
  using Operation = typename Traits<Tag>::Operation;
  using ThreadblockSwizzle = typename Operation::ThreadblockSwizzle;
  using ThreadblockShape = typename Operation::ThreadblockShape;

  cutlass::gemm::GemmCoord problem_size(m, n, k);
  cutlass::gemm::GemmCoord tile_size(ThreadblockShape::kM, ThreadblockShape::kN,
                                     ThreadblockShape::kK);
  cutlass::gemm::GemmCoord grid_tiled_shape =
      ThreadblockSwizzle::get_tiled_shape(problem_size, tile_size,
                                          /*split_k_slices=*/1);

  auto grid = ThreadblockSwizzle().get_grid_shape(grid_tiled_shape);
  return Dim3{grid.x, grid.y, grid.z};
}

template <typename Tag>
bool Adaptor<Tag>::CanImplement(const Arguments &args) {
  cutlass::gemm::GemmCoord problem_size(args.m, args.n, args.k);
  return Traits<Tag>::Kernel::can_implement(problem_size) ==
         cutlass::Status::kSuccess;
}

template <typename Tag>
void Adaptor<Tag>::Initialize(void *params, const Arguments &args,
                              int32_t device_sms, int32_t sm_occupancy) {
  // Sanity check that parameters struct is compatible with parameters storage
  // defined by custom gemm kernel.
  static_assert(sizeof(typename Traits<Tag>::Params) <= 1024,
                "Params struct size is too large");
  static_assert(alignof(typename Traits<Tag>::Params) <= 32,
                "Params struct alignment is too large");

  cutlass::gemm::GemmCoord problem_size(args.m, args.n, args.k);

  // TODO(ezhulenev): Replace with cute::stride instead of custom templates.
  auto lda = LdA<typename Traits<Tag>::Operation>(problem_size);
  auto ldb = LdB<typename Traits<Tag>::Operation>(problem_size);
  auto ldc = LdC<typename Traits<Tag>::Operation>(problem_size);

  auto mode = cutlass::gemm::GemmUniversalMode::kGemm;

  // TODO(ezhulenev): We hardcode parameters for `LinearCombination`
  // epilogue, however `Gemm` template can be compiled with arbitrary
  // epilogues. We have to support custom epilogues in a way that does not
  // leak cutlass types via the public API function signature.
  using Accumulator = typename Traits<Tag>::Operation::ElementAccumulator;
  Accumulator alpha{1.0};
  Accumulator beta{0.0};

  typename Traits<Tag>::Arguments arguments(  // CUTLASS Operation arguments
      mode, problem_size,                     //
      1,                                      // batch
      {alpha, beta},                          // epilogue
      args.a, args.b, args.c, args.c,         // pointers
      0, 0, 0, 0,                             // batch strides
      lda, ldb, ldc, ldc                      // strides
  );

  // Convert CUTLASS operation arguments to a device kernel parameters.
  new (params)
      typename Traits<Tag>::Params(arguments, device_sms, sm_occupancy);
}

//===----------------------------------------------------------------------===//
// CUTLASS 2x Device Kernel Entry Point
//===----------------------------------------------------------------------===//

// This entry point is based on `cutlass::Kernel2` template with an extra
// parameter to pass dynamic slices.
template <typename Kernel>
__global__ void KernelEntryPoint(typename Kernel::Params params,
                                 DynamicSliceParams slices) {
  extern __shared__ int SharedStorageBase[];
  typename Kernel::SharedStorage *shared_storage =
      reinterpret_cast<typename Kernel::SharedStorage *>(SharedStorageBase);

  // Update output pointers to account for dynamic offsets.
  if (slices.out.has_value()) {
    auto m = params.problem_size.m();
    auto n = params.problem_size.n();

    int32_t out_offset = **slices.out;

    char *ptr_c = reinterpret_cast<char *>(params.ptr_C);
    char *ptr_d = reinterpret_cast<char *>(params.ptr_D);

    using ElementC = typename Kernel::ElementC;
    params.ptr_C = ptr_c + sizeof(ElementC) * out_offset * (m * n);
    params.ptr_D = ptr_d + sizeof(ElementC) * out_offset * (m * n);
  }

  Kernel::invoke(params, *shared_storage);
}

template <typename Tag>
void *DeviceKernel<Tag>::symbol() {
  return reinterpret_cast<void *>(
      KernelEntryPoint<typename Traits<Tag>::Kernel>);
};

//===----------------------------------------------------------------------===//
// CUTLASS kernel traits helper
//===----------------------------------------------------------------------===//

#define XLA_GPU_DEFINE_CUTLASS_GEMM_TRAITS(TAG, OPERATION) \
  template <>                                              \
  struct Traits<TAG> {                                     \
    using Operation = OPERATION;                           \
    using Arguments = typename Operation::Arguments;       \
    using Kernel = typename Operation::GemmKernel;         \
    using Params = typename Kernel::Params;                \
  }

}  // namespace xla::gpu::kernel::gemm_universal

#endif  // XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_ADAPTOR_CU_H_
