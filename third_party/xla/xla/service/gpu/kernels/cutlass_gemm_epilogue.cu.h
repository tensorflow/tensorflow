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

#ifndef XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_EPILOGUE_CU_H_
#define XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_EPILOGUE_CU_H_

#include <cstddef>
#include <cstdint>

#include "cute/config.hpp"
#include "cute/container/array.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/int.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/tensor.hpp"
#include "cute/underscore.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/numeric_conversion.h"

namespace xla::gpu::kernel::gemm_universal {

using cutlass::epilogue::collective::detail::get_epilogue_stride;

//===----------------------------------------------------------------------===//
// Custom CUTLASS epilogue fusions
//===----------------------------------------------------------------------===//

template <typename ElementOutput, typename ElementCompute,
          unsigned dynamic_offset = 0,
          cutlass::FloatRoundStyle round_style =
              cutlass::FloatRoundStyle::round_to_nearest>
struct LinearCombinationWithDynamicSlice
    : cutlass::epilogue::fusion::ScaledAcc<ElementOutput, ElementCompute,
                                           ElementCompute, round_style> {
  static constexpr bool IsSourceSupported = true;  // NOLINT
};

//===----------------------------------------------------------------------===//
// CUTLASS gemm epilogue with an on-device offset support
//===----------------------------------------------------------------------===//

// This epilogue is derived from CUTLASS default epilogue with an additional
// support for dynamic slice offsets.
//
// Original: cutlass/epilogue/collective/default_epilogue.hpp

// Applies an element wise operation to all elements within the fragment
// and writes them out to destination storage. C and D storage can have
// optional dynamic offsets (offsets stored in a device memory).
template <typename StrideC_, typename StrideD_, typename ThreadEpilogueOp_,
          typename EpilogueSchedule_, unsigned dynamic_offset>
class DynamicSliceEpilogue {
 public:
  using EpilogueSchedule = EpilogueSchedule_;
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementC = typename ThreadEpilogueOp::ElementC;
  using StrideC = StrideC_;
  using ElementD = typename ThreadEpilogueOp::ElementD;
  using StrideD = StrideD_;

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  static const int kOutputAlignment = ThreadEpilogueOp::kCount;
  using AlignmentType =
      typename cute::uint_bit<cute::sizeof_bits<ElementOutput>::value *
                              kOutputAlignment>::type;

  static_assert(cute::rank(StrideC{}) == 3,
                "StrideCD must be rank-3: [M, N, L]");
  static_assert(cute::rank(StrideD{}) == 3,
                "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage {};

  // Offset into C and D computed as a dot product of `offset` and `stride`.
  struct DynamicOffset {
    cute::array<int32_t const*, dynamic_offset> offset{};
    cute::array<int64_t, dynamic_offset> stride{};
  };

  // Host side epilogue arguments
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const* ptr_c = nullptr;
    StrideC stride_c{};
    ElementD* ptr_d = nullptr;
    StrideD stride_d{};
    DynamicOffset offset_c{};
    DynamicOffset offset_d{};
  };

  // Device side epilogue params_
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const& _,
                                                  Arguments const& args,
                                                  void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape,
                                   Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const& problem_shape,
                                              Arguments const& args,
                                              void* workspace,
                                              cudaStream_t stream) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool can_implement(
      ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  explicit DynamicSliceEpilogue(Params const& params__)
      : params_(params__), epilogue_op_(params__.thread) {}

  CUTLASS_DEVICE
  bool is_source_needed() { return epilogue_op_.is_source_needed(); }

  template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL,
            class FrgEngine, class FrgLayout, class TiledMma, class ResidueMNK>
  CUTLASS_HOST_DEVICE void operator()(
      ProblemShapeMNKL problem_shape_mnkl, BlockShapeMNK blk_shape_mnk,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma, ResidueMNK residue_mnk, int thread_idx,
      char* smem_buf) {
    using cute::_;
    using cute::_1;
    using cute::local_tile;
    using cute::make_coord;
    using cute::make_gmem_ptr;
    using cute::make_identity_tensor;
    using cute::make_shape;
    using cute::make_tensor;
    using cute::shape;
    using cute::Tensor;
    using cute::unwrap;

    using X = cute::Underscore;

    static_assert(cute::rank(ProblemShapeMNKL{}) == 4,
                  "ProblemShapeMNKL must be rank 4");
    static_assert(cute::is_static<BlockShapeMNK>::value,
                  "ThreadBlock tile shape must be static");
    static_assert(cute::rank(BlockShapeMNK{}) == 3,
                  "BlockShapeMNK must be rank 3");
    static_assert(cute::rank(BlockCoordMNKL{}) == 4,
                  "BlockCoordMNKL must be rank 3");

    // Separate out problem shape for convenience
    auto m = cute::get<0>(problem_shape_mnkl);
    auto n = cute::get<1>(problem_shape_mnkl);
    auto l = cute::get<3>(problem_shape_mnkl);

    auto stride_c = get_epilogue_stride<EpilogueSchedule>(params_.stride_c);
    auto stride_d = get_epilogue_stride<EpilogueSchedule>(params_.stride_d);

    ElementC const* ptr_c = params_.ptr_c;
    ElementD* ptr_d = params_.ptr_d;

    // Apply dynamic offsets to base pointers.
    for (unsigned i = 0; i < dynamic_offset; ++i) {
      if (params_.offset_c.offset[i])
        ptr_c += *params_.offset_c.offset[i] * params_.offset_c.stride[i];
    }
    for (unsigned i = 0; i < dynamic_offset; ++i) {
      if (params_.offset_d.offset[i])
        ptr_d += *params_.offset_d.offset[i] * params_.offset_d.stride[i];
    }

    // Represent the full output tensor
    Tensor mC_mnl = make_tensor(make_gmem_ptr(ptr_c), make_shape(m, n, l),
                                stride_c);  // (m,n,l)
    Tensor mD_mnl = make_tensor(make_gmem_ptr(ptr_d), make_shape(m, n, l),
                                stride_d);  // (m,n,l)
    Tensor gC_mnl = local_tile(mC_mnl, blk_shape_mnk, make_coord(_, _, _),
                               cute::Step<_1, _1, X>{});  // (BLK_M,BLK_N,m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_mnk, make_coord(_, _, _),
                               cute::Step<_1, _1, X>{});  // (BLK_M,BLK_N,m,n,l)

    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    Tensor gC = gC_mnl(_, _, m_coord, n_coord, l_coord);  // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_, _, m_coord, n_coord, l_coord);  // (BLK_M,BLK_N)

    // Partition source and destination tiles to match the accumulator
    // partitioning
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCgD = thr_mma.partition_C(gD);  // (VEC,THR_M,THR_N)
    Tensor tCgC = thr_mma.partition_C(gC);  // (VEC,THR_M,THR_N)

    static_assert(cute::is_static<FrgLayout>::value,
                  "Accumulator layout must be static");
    CUTE_STATIC_ASSERT_V(
        size(tCgC) == size(tCgD),
        "Source and destination must have the same number of elements.");
    CUTE_STATIC_ASSERT_V(
        size(tCgD) == size(accumulators),
        "Accumulator count must have the same destination element count.");

    // Make an identity coordinate tensor for predicating our output MN tile
    auto cD = make_identity_tensor(
        make_shape(unwrap(shape<0>(gD)), unwrap(shape<1>(gD))));
    Tensor tCcD = thr_mma.partition_C(cD);

    if (epilogue_op_.is_source_needed()) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        if (elem_less(tCcD(i), make_coord(cute::get<0>(residue_mnk),
                                          cute::get<1>(residue_mnk)))) {
          tCgD(i) = epilogue_op_(accumulators(i), tCgC(i));
        }
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        if (elem_less(tCcD(i), make_coord(cute::get<0>(residue_mnk),
                                          cute::get<1>(residue_mnk)))) {
          tCgD(i) = epilogue_op_(accumulators(i));
        }
      }
    }
  }

 private:
  Params params_;
  ThreadEpilogueOp epilogue_op_;
};

}  // namespace xla::gpu::kernel::gemm_universal

namespace cutlass::epilogue::collective {

//===----------------------------------------------------------------------===//
// Collective builder specialization for LinearCombinationWithDynamicSlice
//===----------------------------------------------------------------------===//

// Specialization for `NoSmemWarpSpecialized` schedule.
template <typename TileShape_MNK, typename ClusterShape_MNK,
          typename EpilogueTileType, typename ElementAccumulator,
          typename ElementCompute, typename ElementC_, typename GmemLayoutTagC_,
          int AlignmentC, typename ElementD, typename GmemLayoutTagD,
          int AlignmentD, cutlass::FloatRoundStyle RoundStyle,
          unsigned dynamic_offset>
struct CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape_MNK,
    ClusterShape_MNK, EpilogueTileType, ElementAccumulator, ElementCompute,
    ElementC_, GmemLayoutTagC_, AlignmentC, ElementD, GmemLayoutTagD,
    AlignmentD, cutlass::epilogue::NoSmemWarpSpecialized,
    xla::gpu::kernel::gemm_universal::LinearCombinationWithDynamicSlice<
        ElementD, ElementCompute, dynamic_offset, RoundStyle>,
    void> {
  // Passing void C disables source load
  using ElementC =
      cute::conditional_t<cute::is_void_v<ElementC_>, ElementD, ElementC_>;
  using GmemLayoutTagC = cute::conditional_t<cute::is_void_v<ElementC_>,
                                             GmemLayoutTagD, GmemLayoutTagC_>;

  static constexpr cutlass::epilogue::thread::ScaleType::Kind ScaleType =
      cute::is_void_v<ElementC_>
          ? cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
          : cutlass::epilogue::thread::ScaleType::Default;

  static constexpr int FragmentSize = 1;
  using ThreadOp = cutlass::epilogue::thread::LinearCombination<
      ElementD, FragmentSize, ElementAccumulator, ElementCompute, ScaleType,
      RoundStyle, ElementC>;

  using CollectiveOp =
      cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
          xla::gpu::kernel::gemm_universal::DynamicSliceEpilogue<
              cutlass::detail::TagToStrideC_t<GmemLayoutTagC>,
              cutlass::detail::TagToStrideC_t<GmemLayoutTagD>, ThreadOp,
              cutlass::gemm::EpilogueDefault, dynamic_offset>>;
};

}  // namespace cutlass::epilogue::collective

#endif  // XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_EPILOGUE_CU_H_
