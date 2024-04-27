/* Copyright 2023 The OpenXLA Authors.

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

#include "cute/layout.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"
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
// CUTLASS 2x vs 3x
//===----------------------------------------------------------------------===//

// Cutlass 2x and 3x have slightly different APIs, with a little bit of template
// metaprogramming and constexpr ifs we dispatch to the correct version at
// compile time based on a kernel template.
template <typename Tag>
static constexpr bool is_cutlass_3x =
    cutlass::gemm::detail::IsCutlass3GemmKernel<
        typename Traits<Tag>::Kernel>::value;

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
// CUTLASS 2x host side adaptor
//===----------------------------------------------------------------------===//

namespace adaptor_2x {

template <typename Tag>
static std::optional<Dim3> ClusterDim() {
  return std::nullopt;
}

template <typename Tag>
static Dim3 BlockDim(int32_t m, int32_t n, int32_t k) {
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
static int32_t SharedMemoryBytes() {
  return sizeof(typename Traits<Tag>::Kernel::SharedStorage);
};

template <typename Tag>
static Dim3 ThreadDim() {
  return Dim3{Traits<Tag>::Kernel::kThreadCount, 1, 1};
}

template <typename Tag>
static bool CanImplement(const Arguments &args) {
  cutlass::gemm::GemmCoord problem_size(args.m, args.n, args.k);
  return Traits<Tag>::Kernel::can_implement(problem_size) ==
         cutlass::Status::kSuccess;
}

// Converts type-erased gemm arguments to the underlying CUTLASS operation
// arguments.
template <typename Tag>
static typename Traits<Tag>::Arguments OpArguments(const Arguments &args) {
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

  return typename Traits<Tag>::Arguments(      // CUTLASS Operation arguments
      mode, problem_size,                      //
      1,                                       // batch
      {alpha, beta},                           // epilogue
      args.lhs, args.rhs, args.out, args.out,  // pointers
      0, 0, 0, 0,                              // batch strides
      lda, ldb, ldc, ldc                       // strides
  );
}

template <typename Tag>
int64_t WorkspaceSize(const Arguments &args) {
  return Traits<Tag>::Operation::get_workspace_size(OpArguments<Tag>(args));
}

template <typename Tag>
void Initialize(void *params, const Arguments &args, int32_t device_sms,
                int32_t sm_occupancy) {
  // Sanity check that parameters struct is compatible with parameters storage
  // defined by custom gemm kernel.
  static_assert(sizeof(typename Traits<Tag>::Params) <= 1024,
                "Params struct size is too large");
  static_assert(alignof(typename Traits<Tag>::Params) <= 32,
                "Params struct alignment is too large");

  // Convert CUTLASS operation arguments to a device kernel parameters.
  new (params) typename Traits<Tag>::Params(OpArguments<Tag>(args), device_sms,
                                            sm_occupancy);
}

};  // namespace adaptor_2x

//===----------------------------------------------------------------------===//
// CUTLASS 3x host side adaptor
//===----------------------------------------------------------------------===//

namespace adaptor_3x {

template <typename Tag>
static std::optional<Dim3> ClusterDim() {
  typename Traits<Tag>::Kernel::DispatchPolicy::ClusterShape cluster;
  return Dim3{cute::get<0>(cluster), cute::get<1>(cluster),
              cute::get<2>(cluster)};
}

template <typename Tag>
static Dim3 BlockDim(int32_t m, int32_t n, int32_t k) {
  return adaptor_2x::BlockDim<Tag>(m, n, k);
}

template <typename Tag>
static Dim3 ThreadDim() {
  auto block_shape = Traits<Tag>::Kernel::get_block_shape();
  return Dim3{block_shape.x, block_shape.y, block_shape.z};
}

template <typename Tag>
static int32_t SharedMemoryBytes() {
  return Traits<Tag>::Kernel::SharedStorageSize;
};

template <typename Tag>
static typename Traits<Tag>::Arguments OpArguments(const Arguments &args) {
  using Kernel = typename Traits<Tag>::Kernel;
  using Operation = typename Traits<Tag>::Operation;

  auto stride_a = cutlass::make_cute_packed_stride(
      typename Kernel::StrideA{}, cute::make_shape(args.m, args.k, 1));
  auto stride_b = cutlass::make_cute_packed_stride(
      typename Kernel::StrideB{}, cute::make_shape(args.n, args.k, 1));
  auto stride_c = cutlass::make_cute_packed_stride(
      typename Kernel::StrideC{}, cute::make_shape(args.m, args.n, 1));
  auto stride_d = cutlass::make_cute_packed_stride(
      typename Kernel::StrideD{}, cute::make_shape(args.m, args.n, 1));

  // TODO(ezhulenev): Pass device id and sm_count in arguments.
  cutlass::KernelHardwareInfo hw_info{/*device_id=*/0, /*sm_count=*/128};

  auto mode = cutlass::gemm::GemmUniversalMode::kGemm;
  typename Kernel::ProblemShape problem_shape = {args.m, args.n, args.k,
                                                 /*batch=*/1};

  // TODO(ezhulenev): We hardcode parameters for `LinearCombination`
  // epilogue, however `Gemm` template can be compiled with arbitrary
  // epilogues. We have to support custom epilogues in a way that does not
  // leak cutlass types via the public API function signature.
  using Accumulator = typename Traits<Tag>::Operation::ElementAccumulator;
  Accumulator alpha{1.0};
  Accumulator beta{0.0};

  typename Kernel::MainloopArguments mainloop_args{
      reinterpret_cast<typename Operation::ElementA *>(args.lhs), stride_a,
      reinterpret_cast<typename Operation::ElementB *>(args.rhs), stride_b};

  typename Kernel::EpilogueArguments epilogue_args{
      {alpha, beta},
      reinterpret_cast<typename Operation::ElementC *>(args.out),
      stride_c,
      reinterpret_cast<typename Operation::ElementC *>(args.out),
      stride_d,
      {{args.slices.out}, {args.m * args.n}},  // dynamic offsets for C
      {{args.slices.out}, {args.m * args.n}},  // dynamic offsets for D
  };

  return typename Operation::Arguments{mode, problem_shape, mainloop_args,
                                       epilogue_args, hw_info};
}

template <typename Tag>
static bool CanImplement(const Arguments &args) {
  return Traits<Tag>::Kernel::can_implement(OpArguments<Tag>(args));
}

template <typename Tag>
static int64_t WorkspaceSize(const Arguments &args) {
  return Traits<Tag>::Operation::get_workspace_size(OpArguments<Tag>(args));
}

template <typename Tag>
static void Initialize(void *params, const Arguments &args, int32_t device_sms,
                       int32_t sm_occupancy) {
  // Sanity check that parameters struct is compatible with parameters storage
  // defined by custom gemm kernel.
  static_assert(sizeof(typename Traits<Tag>::Params) <= 1024,
                "Params struct size is too large");
  static_assert(alignof(typename Traits<Tag>::Params) <= 64,
                "Params struct alignment is too large");

  // Convert CUTLASS operation arguments to a device kernel parameters.
  using Kernel = typename Traits<Tag>::Kernel;
  new (params) typename Traits<Tag>::Params(
      Kernel::to_underlying_arguments(OpArguments<Tag>(args), args.workspace));
}

};  // namespace adaptor_3x

//===----------------------------------------------------------------------===//
// Dispatch between CUTLASS 2x and 3x host adaptors
//===----------------------------------------------------------------------===//

template <typename Tag>
std::optional<Dim3> Adaptor<Tag>::ClusterDim() const {
  if constexpr (is_cutlass_3x<Tag>) {
    return adaptor_3x::ClusterDim<Tag>();
  } else {
    return adaptor_2x::ClusterDim<Tag>();
  }
}

template <typename Tag>
Dim3 Adaptor<Tag>::ThreadDim() const {
  if constexpr (is_cutlass_3x<Tag>) {
    return adaptor_3x::ThreadDim<Tag>();
  } else {
    return adaptor_2x::ThreadDim<Tag>();
  }
}

template <typename Tag>
Dim3 Adaptor<Tag>::BlockDim(int32_t m, int32_t n, int32_t k) const {
  if constexpr (is_cutlass_3x<Tag>) {
    return adaptor_3x::BlockDim<Tag>(m, n, k);
  } else {
    return adaptor_2x::BlockDim<Tag>(m, n, k);
  }
}

template <typename Tag>
int32_t Adaptor<Tag>::SharedMemoryBytes() const {
  if constexpr (is_cutlass_3x<Tag>) {
    return adaptor_3x::SharedMemoryBytes<Tag>();
  } else {
    return adaptor_2x::SharedMemoryBytes<Tag>();
  }
};

template <typename Tag>
bool Adaptor<Tag>::CanImplement(const Arguments &args) const {
  if constexpr (is_cutlass_3x<Tag>) {
    return adaptor_3x::CanImplement<Tag>(args);
  } else {
    return adaptor_2x::CanImplement<Tag>(args);
  }
}

template <typename Tag>
int64_t Adaptor<Tag>::WorkspaceSize(const Arguments &args) const {
  if constexpr (is_cutlass_3x<Tag>) {
    return adaptor_3x::WorkspaceSize<Tag>(args);
  } else {
    return adaptor_2x::WorkspaceSize<Tag>(args);
  }
}

template <typename Tag>
void Adaptor<Tag>::Initialize(void *params, const Arguments &args,
                              int32_t device_sms, int32_t sm_occupancy) const {
  if constexpr (is_cutlass_3x<Tag>) {
    return adaptor_3x::Initialize<Tag>(params, args, device_sms, sm_occupancy);
  } else {
    return adaptor_2x::Initialize<Tag>(params, args, device_sms, sm_occupancy);
  }
}

//===----------------------------------------------------------------------===//
// CUTLASS 2x device kernel entry point
//===----------------------------------------------------------------------===//

// This entry point is based on `cutlass::Kernel2` template with an extra
// parameter to pass dynamic slices.
//
// TODO(ezhulenev): Dynamic slices should be encoded in kernel parameters.
template <typename Kernel>
__global__ void Kernel2EntryPoint(typename Kernel::Params params,
                                  DynamicSliceArguments dynamic_slices) {
  extern __shared__ int SharedStorageBase[];
  typename Kernel::SharedStorage *shared_storage =
      reinterpret_cast<typename Kernel::SharedStorage *>(SharedStorageBase);

  // Adjust output pointer to account for dynamic offsets.
  if (dynamic_slices.out) {
    auto m = params.problem_size.m();
    auto n = params.problem_size.n();

    using ElementC = typename Kernel::ElementC;
    int64_t offset = sizeof(ElementC) * *dynamic_slices.out * (m * n);

    char *ptr_c = reinterpret_cast<char *>(params.ptr_C);
    char *ptr_d = reinterpret_cast<char *>(params.ptr_D);

    params.ptr_C = ptr_c + offset;
    params.ptr_D = ptr_d + offset;
  }

  Kernel::invoke(params, *shared_storage);
}

//===----------------------------------------------------------------------===//
// CUTLASS 3x device kernel entry point
//===----------------------------------------------------------------------===//

template <typename Kernel>
__global__ void Kernel3EntryPoint(
    CUTLASS_GRID_CONSTANT const typename Kernel::Params params) {
  extern __shared__ char shared_memory[];

  Kernel kernel;
  kernel(params, shared_memory);
}

//===----------------------------------------------------------------------===//
// Dispatch between CUTLASS 2x and 3x kernel entry points
//===----------------------------------------------------------------------===//

template <typename Tag>
void *DeviceKernel<Tag>::symbol() const {
  using Kernel = typename Traits<Tag>::Kernel;

  if constexpr (is_cutlass_3x<Tag>) {
    return reinterpret_cast<void *>(Kernel3EntryPoint<Kernel>);
  } else {
    return reinterpret_cast<void *>(Kernel2EntryPoint<Kernel>);
  }
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
