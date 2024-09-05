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

#ifndef TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_GPU_CU_H_
#define TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_GPU_CU_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/gpu_prim_helpers.h"
#include "tensorflow/core/kernels/segment_reduction_ops.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_solvers.h"  // For ScratchSpace
#include "tensorflow/core/util/permutation_input_iterator.h"

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
#include "xla/stream_executor/cuda/cuda_activation.h"
#elif (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#include "tensorflow/core/platform/rocm.h"
#endif

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

// Non/Atomic reduction functors for the gpu.
#define DEFINE_REDUCE_UPDATE_OP_GPU(name, func)                             \
  struct name##OpGpu {                                                      \
    template <typename T>                                                   \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(T* dest,          \
                                                          const T& value) { \
      func;                                                                 \
    }                                                                       \
  };
DEFINE_REDUCE_UPDATE_OP_GPU(AtomicSum, GpuAtomicAdd(dest, value))
DEFINE_REDUCE_UPDATE_OP_GPU(AtomicProd, GpuAtomicMul(dest, value))
DEFINE_REDUCE_UPDATE_OP_GPU(AtomicMax, GpuAtomicMax(dest, value))
DEFINE_REDUCE_UPDATE_OP_GPU(AtomicMin, GpuAtomicMin(dest, value))
DEFINE_REDUCE_UPDATE_OP_GPU(NonAtomicSum, *dest += value)
DEFINE_REDUCE_UPDATE_OP_GPU(NonAtomicProd, *dest *= value)
DEFINE_REDUCE_UPDATE_OP_GPU(NonAtomicMax, *dest = max(*dest, value))
DEFINE_REDUCE_UPDATE_OP_GPU(NonAtomicMin, *dest = min(*dest, value))
#undef DEFINE_REDUCE_UPDATE_OP_GPU

template <typename ReduceOp>
struct ReduceUpdateOpFor {};

#define DEFINE_REDUCE_UPDATE_OP_FOR(reduce_op, atomic, nonatomic) \
  template <>                                                     \
  struct ReduceUpdateOpFor<reduce_op> {                           \
    using atomic_op = atomic;                                     \
    using nonatomic_op = nonatomic;                               \
  };
DEFINE_REDUCE_UPDATE_OP_FOR(functor::Sum, AtomicSumOpGpu, NonAtomicSumOpGpu)
DEFINE_REDUCE_UPDATE_OP_FOR(functor::Prod, AtomicProdOpGpu, NonAtomicProdOpGpu)
DEFINE_REDUCE_UPDATE_OP_FOR(functor::Max, AtomicMaxOpGpu, NonAtomicMaxOpGpu)
DEFINE_REDUCE_UPDATE_OP_FOR(functor::Min, AtomicMinOpGpu, NonAtomicMinOpGpu)
#undef DEFINE_REDUCE_UPDATE_OP_FOR

// PR#61339: MSVC does not support compound-assignment operators on device

// SortedSegmentReductionFunctor kernel reduces input data just as
// UnsortedSegmentReductionCustomKernel does except that input data
// is partitioned along the outer reduction dimension. This is
// because consecutive rows (elements in a row share the same
// outer dimension index) in the flattened 2D input data likely
// belong to the same segment in sorted segment sum operation.
// Therefore such partitioning strategy has two advantages over
// the UnsortedSegmentReductionFunctor kernel:
// 1. Each thread reduces across multiple rows before writing
// answers to the global memory, we can therefore
// write reduction results to global memory less often.
// 2. We may know that the current thread is the only contributor
// to an output element because of the increasing nature of segment
// ids. In such cases, we do not need to use atomic operations
// to write results to global memory.
// In the flattened view of input data (with only outer and inner
// dimension), every thread processes a strip of input data of
// size OuterDimTileSize x 1. This strip runs across multiple
// rows of input data and all reduction elements share one inner
// dimension index.
template <typename T, typename Index, int OuterDimTileSize, typename ReductionF,
          typename AtomicReductionF>
__global__ void SortedSegmentReductionCustomKernel(
    const Index input_outer_dim_size, const Index inner_dim_size,
    const Index output_outer_dim_size, const Index* __restrict__ segment_ids,
    const T* __restrict__ input, T* __restrict__ output,
    const Index total_stripe_count, const T initial_value) {
  for (int stripe_index : GpuGridRangeX(total_stripe_count)) {
    const Index segment_offset = stripe_index % inner_dim_size;
    const Index input_outer_dim_index_base =
        stripe_index / inner_dim_size * Index(OuterDimTileSize);

    T reduce_res = initial_value;
    Index first_segment_id = segment_ids[input_outer_dim_index_base];
    Index last_output_segment_id = output_outer_dim_size;

    const Index actual_stripe_height =
        min(Index(OuterDimTileSize),
            input_outer_dim_size - input_outer_dim_index_base);
    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];
      // Decide whether to write result to global memory. Result is only written
      // to global memory if we move to another segment. Otherwise we can keep
      // accumulating locally.
      if (current_output_segment_id > last_output_segment_id) {
        const Index output_index =
            last_output_segment_id * inner_dim_size + segment_offset;
        // Decide whether to write result to global memory using atomic
        // operations.
        if (last_output_segment_id == first_segment_id) {
          AtomicReductionF()(output + output_index, reduce_res);
        } else {
          ReductionF()(output + output_index, reduce_res);
        }
        reduce_res = initial_value;
      }
      ReductionF()(
          &reduce_res,
          ldg(input + (input_outer_dim_index_base + j) * inner_dim_size +
              segment_offset));
      last_output_segment_id = current_output_segment_id;
    }
    // For the last result in a strip, always write using atomic operations
    // due to possible race conditions with threads computing
    // the following strip.
    const Index output_index =
        last_output_segment_id * inner_dim_size + segment_offset;
    AtomicReductionF()(output + output_index, reduce_res);
  }
}

template <typename SegmentId, typename Index, typename T>
__global__ void SegmentMeanNormalizeKernel(
    SegmentId nsegments, Index ninner,
    const Index* __restrict__ segment_offsets,  // [nsegments + 1]
    T* __restrict__ output) {                   // [nsegments, ninner]
  for (SegmentId seg : GpuGridRangeY(nsegments)) {
    SegmentId segment_size = segment_offsets[seg + 1] - segment_offsets[seg];
    segment_size = max(segment_size, Index(1));  // Avoid division by zero
    T inv_norm = T(1) / static_cast<T>(segment_size);
    for (Index i : GpuGridRangeX(ninner)) {
      output[seg * ninner + i] *= inv_norm;
    }
  }
}

template <typename SegmentId, typename Index, typename T>
Status LaunchSegmentMeanNormalizeKernel(
    const GPUDevice& d, SegmentId nsegments, Index ninner,
    const Index* __restrict__ segment_offsets,  // [nsegments + 1]
    T* __restrict__ output) {                   // [nsegments, ninner]
  Gpu2DLaunchConfig config = GetGpu2DLaunchConfig(
      ninner, nsegments, d, SegmentMeanNormalizeKernel<SegmentId, Index, T>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(SegmentMeanNormalizeKernel<SegmentId, Index, T>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), nsegments, ninner, segment_offsets,
                         output);
}

template <typename SegmentId, typename Index, typename T>
__global__ void SegmentSetEmptyKernel(
    SegmentId nsegments, Index ninner,
    const Index* __restrict__ segment_offsets,  // [nsegments + 1]
    const T empty_value,
    T* __restrict__ output) {  // [nsegments, ninner]
  for (SegmentId seg : GpuGridRangeY(nsegments)) {
    SegmentId segment_size = segment_offsets[seg + 1] - segment_offsets[seg];
    if (segment_size == 0) {
      for (Index i : GpuGridRangeX(ninner)) {
        output[seg * ninner + i] = empty_value;
      }
    }
  }
}

template <typename SegmentId, typename Index, typename T>
Status LaunchSegmentSetEmptyKernel(
    const GPUDevice& d, SegmentId nsegments, Index ninner,
    const Index* __restrict__ segment_offsets,  // [nsegments + 1]
    const T empty_value,
    T* __restrict__ output) {  // [nsegments, ninner]
  Gpu2DLaunchConfig config = GetGpu2DLaunchConfig(
      ninner, nsegments, d, SegmentSetEmptyKernel<SegmentId, Index, T>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(SegmentSetEmptyKernel<SegmentId, Index, T>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), nsegments, ninner, segment_offsets,
                         empty_value, output);
}

// UnsortedSegmentSumKernel processes 'input_total_size' elements.
// Each element is mapped from input to output by a combination of its
// 'segment_ids' mapping and 'inner_dim_size'.
template <typename T, typename Index, typename KernelReductionFunctor>
__global__ void UnsortedSegmentCustomKernel(
    const int64_t input_outer_dim_size, const int64_t inner_dim_size,
    const int64_t output_outer_dim_size, const Index* __restrict__ segment_ids,
    const T* __restrict__ input, T* __restrict__ output) {
  const int64_t input_total_size = input_outer_dim_size * inner_dim_size;
  for (int64_t input_index : GpuGridRangeX(input_total_size)) {
    const int64_t input_segment_index = input_index / inner_dim_size;
    const int64_t segment_offset = input_index % inner_dim_size;
    const Index output_segment_index = segment_ids[input_segment_index];
    if (output_segment_index < 0 ||
        output_segment_index >= output_outer_dim_size) {
      continue;
    }
    const int64_t output_index =
        output_segment_index * inner_dim_size + segment_offset;
    KernelReductionFunctor()(output + output_index, ldg(input + input_index));
  }
}

template <typename Toffsets, typename Tsegmentids>
__global__ void SegmentOffsetsKernel(
    Toffsets size, Tsegmentids nsegments,
    const Tsegmentids* __restrict__ segment_ids,  // [size]
    Toffsets* __restrict__ segment_offsets) {     // [nsegments + 1]
  GPU_1D_KERNEL_LOOP(i, size + 1) {
    // IDs are clipped to [-1, nsegments] so that out-of-bounds IDs are ignored.
    // Note that we can't report invalid IDs from the GPU without incurring
    // additional overhead.
    auto clip = [&](Tsegmentids id) {
      return min(max(Tsegmentids(-1), id), nsegments);
    };
    const Tsegmentids cur_id = (i < size) ? clip(segment_ids[i]) : nsegments;
    const Tsegmentids prev_id =
        (i == 0) ? Tsegmentids(-1) : clip(segment_ids[i - 1]);
    // At segment boundaries, write the offset for this ID and any missing IDs
    // since the previous one.
    for (Tsegmentids id = prev_id + 1; id <= cur_id; ++id) {
      segment_offsets[id] = i;
    }
  }
}

// Finds the start offset of each segment in the given sorted segment_ids
// vector. Missing IDs are given the same offset as the next ID so that they
// represent empty ranges. Invalid IDs (those that are outside the range
// [0, nsegments)) are ignored. The value at segment_offsets[0] is set to the
// start index of the first valid ID (e.g., 0 if all IDs are valid), and the
// value at segment_offsets[nsegments] is set to the end index of the last valid
// ID (e.g., nsegments if all IDs are valid).
template <typename Toffsets, typename Tsegmentids>
Status LaunchSegmentOffsetsKernel(
    const GPUDevice& d, Toffsets size, Tsegmentids nsegments,
    const Tsegmentids* segment_ids,  // [size]
    Toffsets* segment_offsets) {     // [nsegments + 1]
  GpuLaunchConfig config = GetGpuLaunchConfig(
      size + 1, d, &SegmentOffsetsKernel<Toffsets, Tsegmentids>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(SegmentOffsetsKernel<Toffsets, Tsegmentids>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), size, nsegments, segment_ids,
                         segment_offsets);
}

template <typename T>
struct RealTypeIfComplex {
  using type = T;
};

template <typename Real>
struct RealTypeIfComplex<std::complex<Real>> {
  using type = Real;
};

// Reduces along columns of the thread block, returning the result in the first
// row of threads.
template <typename T, typename ReduceOp>
__device__ T ReduceBlockAlongCols(ReduceOp reduce_op, const T& value,
                                  bool is_valid) {
  GPU_DYNAMIC_SHARED_MEM_DECL(/*ALIGN=*/16, char, shared_memory_raw);
  T* const shared_partial_reduction =
      reinterpret_cast<T*>(shared_memory_raw);  // [blockDim.y, blockDim.x]
  const int x = threadIdx.x;
  const int y = threadIdx.y;
  T reduced = value;
  // Reduce over the y dimension of the block.
  for (unsigned k = blockDim.y / 2; k > 0; k /= 2) {
    if (is_valid && y < 2 * k) {
      shared_partial_reduction[y * blockDim.x + x] = reduced;
    }
    __syncthreads();
    if (is_valid && y < k) {
      reduced = reduce_op(reduced,
                          shared_partial_reduction[(y + k) * blockDim.x + x]);
    }
    __syncthreads();
  }
  return reduced;
}

// This kernel uses a 2D thread decomposition. The x dimension maps to the inner
// dimension of the input/output. The y grid dimension maps to segments, and y
// threads within a block cooperate to reduce over the block's segment.
// Note that Tinit is needed because Tvec and Treducevec may be vector types,
// but Tinit is always a scalar type.
// Note that the first dimension of input_vec is nouter if indices is not
// provided; otherwise it is indexed indirectly via indices and can have any
// size (as long as it spans at least the maximum value in indices). This also
// applies to the weights vector.
template <typename Treducevec, typename Tvec, typename Toffsets,
          typename Tindices, typename Tsegmentids, typename ReduceOp,
          typename Tinit, typename Tweights>
__global__ void SegmentReduceVectorKernel(
    Toffsets nouter, Toffsets ninner_vec, Tsegmentids nsegments,
    ReduceOp reduce_op, Tinit initial_value, Tinit empty_segment_value,
    bool is_mean, bool is_sqrtn,
    const Tvec* __restrict__ input_vec,  // [nouter or any, ninner_vec]
    const Toffsets* __restrict__ segment_offsets,  // [nsegments + 1]
    const Tindices* __restrict__ indices,          // [nouter] (optional)
    const Tweights* __restrict__ weights,          // [nouter or any] (optional)
    Tvec* __restrict__ output_vec) {               // [nsegments, ninner_vec]
  const int num_blocks_x = (ninner_vec - 1) / blockDim.x + 1;
  // Grid-stride loop over inner dimension blocks.
  for (Toffsets blk_x = blockIdx.x; blk_x < num_blocks_x; blk_x += gridDim.x) {
    const Toffsets x = threadIdx.x + blk_x * blockDim.x;
    const Toffsets y = threadIdx.y;
    const bool x_ok = x < ninner_vec;
    // Grid-stride loop over segment blocks, each processing one segment.
    for (Tsegmentids seg = blockIdx.y; seg < nsegments; seg += gridDim.y) {
      // Load segment range.
      const Toffsets begin = segment_offsets[seg];
      const Toffsets end = segment_offsets[seg + 1];
      // Reduce over the segment.
      Treducevec result = Treducevec(initial_value);
      // Loop over the segment, reducing blockDim.y elements at a time.
      for (Toffsets y_offset = begin; y_offset < end; y_offset += blockDim.y) {
        const bool y_ok = (y_offset + y) < end;
        // Perform indirect lookup if required.
        const Toffsets y_idx =
            indices && y_ok ? indices[y_offset + y] : y_offset + y;
        const int64_t input_idx = static_cast<int64_t>(y_idx) * ninner_vec + x;
        // Load the input row from global mem.
        Treducevec block_result =
            x_ok && y_ok ? input_vec[input_idx] : Tvec(initial_value);
        // Apply weights if provided.
        if (weights && y_ok) block_result = block_result * Tvec(weights[y_idx]);
        // Reduce along the columns of the block, returning result in first row.
        block_result = ReduceBlockAlongCols(reduce_op, block_result, x_ok);
        if (y == 0 && x_ok) {
          result = reduce_op(result, block_result);
        }
      }
      // First row of the block stores the result to global memory.
      if (y == 0 && x_ok) {
        if (begin == end) {
          // Empty segment.
          result = Treducevec(empty_segment_value);
        } else {
          Tweights total_weight(end - begin);
          // Normalize the results if necessary.
          if (is_mean) {
            result = result / Treducevec(total_weight);
          } else if (is_sqrtn) {
            result =
                result / Treducevec(sqrt(static_cast<double>(total_weight)));
          }
        }
        // Cast from Treducevec to Tvec.
        const int64_t output_idx = static_cast<int64_t>(seg) * ninner_vec + x;
        output_vec[output_idx] = static_cast<Tvec>(result);
      }
    }
  }
}

// Reduces input matrix within segments over the outer dimension. Empty segments
// always output empty_segment_value.
// If is_mean or is_sqrtn is true, the results are normalized using the
// corresponding function.
// If indices is not nullptr, input rows are accessed indirectly as
// input[indices[i]], instead of input[i].
// Note: Treducevec is to allow reducing in higher precision than Tvec.
template <typename Treducevec, typename Tvec, typename Toffsets,
          typename Tindices, typename Tsegmentids, typename ReduceOp,
          typename Tinit, typename Tweights>
Status LaunchSegmentReduceVectorKernel(
    const GPUDevice& d, Toffsets nouter, Toffsets ninner_vec,
    Tsegmentids nsegments, ReduceOp reduce_op, Tinit initial_value,
    Tinit empty_segment_value, bool is_mean, bool is_sqrtn,
    const Tvec* input_vec,            // [nouter or any, ninner_vec]
    const Toffsets* segment_offsets,  // [nsegments + 1]
    const Tindices* indices,          // [nouter] (optional)
    const Tweights* weights,          // [nouter or any] (optional)
    Tvec* output_vec) {               // [nsegments, ninner_vec]
  static constexpr const int kMaxGridX = (1u << 31) - 1;
  static constexpr const int kMaxGridY = (1u << 16) - 1;
  const int max_block_size = 1024;  // Can be tuned for perf (<= 1024)
  const int min_block_size = 64;    // Can be tuned for perf
  const Toffsets ninner_pow2 = Toffsets(1) << Log2Ceiling64(ninner_vec);
  // This is a heuristic that first allocates threads in the block to the inner
  // (x) dimension (which is most efficient) and then allocates the rest to the
  // reduction (y) dimension (which is less efficient but increases
  // parallelism).
  int block_x = std::min(ninner_pow2, static_cast<Toffsets>(max_block_size));
  const Toffsets avg_reduce_size =
      Eigen::divup(nouter, static_cast<Toffsets>(nsegments));
  const Toffsets avg_reduce_size_pow2 = Toffsets(1)
                                        << Log2Ceiling64(avg_reduce_size);
  dim3 block(
      block_x,
      std::min(static_cast<Toffsets>(Eigen::divup(min_block_size, block_x)),
               avg_reduce_size_pow2));
  dim3 grid(std::min(Eigen::divup(ninner_vec, static_cast<Toffsets>(block.x)),
                     static_cast<Toffsets>(kMaxGridX)),
            std::min(nsegments, static_cast<Tsegmentids>(kMaxGridY)));
  unsigned shared_memory_bytes = block.x * block.y * sizeof(Treducevec);
  return GpuLaunchKernel(
      SegmentReduceVectorKernel<Treducevec, Tvec, Toffsets, Tindices,
                                Tsegmentids, ReduceOp, Tinit, Tweights>,
      grid, block, shared_memory_bytes, d.stream(), nouter, ninner_vec,
      nsegments, reduce_op, initial_value, empty_segment_value, is_mean,
      is_sqrtn, input_vec, segment_offsets, indices, weights, output_vec);
}

template <typename Tvec, typename Treducevec, typename Toffsets,
          typename Tsegmentids, typename Tinit>
__global__ void SegmentReduceEpilogueKernel(
    Tsegmentids nsegments, Tinit empty_segment_value, bool is_mean,
    bool is_sqrtn,
    const Treducevec* __restrict__ output_raw,     // [nsegments]
    const Toffsets* __restrict__ segment_offsets,  // [nsegments + 1]
    Tvec* __restrict__ output) {                   // [nsegments]
  GPU_1D_KERNEL_LOOP(seg, nsegments) {
    Toffsets segment_size = segment_offsets[seg + 1] - segment_offsets[seg];
    Treducevec val = output_raw[seg];
    if (segment_size == 0) {
      // Empty segment.
      val = Treducevec(empty_segment_value);
    } else if (is_mean) {
      val = val / Treducevec(segment_size);
    } else if (is_sqrtn) {
      val = val / Treducevec(sqrt(static_cast<double>(
                      typename RealTypeIfComplex<Tinit>::type(segment_size))));
    }
    // Cast from Treducevec to Tvec.
    output[seg] = static_cast<Tvec>(val);
  }
}

// Normalizes output_raw based on segment size and casts from Treducevec to
// Tvec. If Tvec == Treducevec, this is safe to call with output_raw == output.
// Note that Treducevec is the type that was used for the reduction, which may
// be a higher-precision type than the output type Tvec (e.g., float vs. half).
template <typename Tvec, typename Treducevec, typename Toffsets,
          typename Tsegmentids, typename Tinit>
Status LaunchSegmentReduceEpilogueKernel(
    const GPUDevice& d, Tsegmentids nsegments, Tinit empty_segment_value,
    bool is_mean, bool is_sqrtn,
    const Treducevec* output_raw,     // [nsegments]
    const Toffsets* segment_offsets,  // [nsegments + 1]
    Tvec* output) {                   // [nsegments]
  GpuLaunchConfig config = GetGpuLaunchConfig(
      nsegments, d,
      &SegmentReduceEpilogueKernel<Tvec, Treducevec, Toffsets, Tsegmentids,
                                   Tinit>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(SegmentReduceEpilogueKernel<Tvec, Treducevec, Toffsets,
                                                     Tsegmentids, Tinit>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), nsegments, empty_segment_value, is_mean,
                         is_sqrtn, output_raw, segment_offsets, output);
}

template <typename Tto>
struct CastFunctor {
  template <typename T>
  __device__ Tto operator()(const T& val) const {
    return static_cast<Tto>(val);
  }
};

template <typename Treducevec, typename Tvec, typename Tindices,
          typename Tweights>
struct LookupAndScaleAndCastInputsFunctor {
  LookupAndScaleAndCastInputsFunctor(const Tvec* input_vec,
                                     const Tindices* indices,
                                     const Tweights* weights)
      : input_vec_(input_vec), indices_(indices), weights_(weights) {}

  template <typename Toffsets>
  __device__ Treducevec operator()(Toffsets idx) const {
    if (indices_) idx = indices_[idx];
    Treducevec result = static_cast<Treducevec>(input_vec_[idx]);
    if (weights_) result = result * Tvec(weights_[idx]);
    return result;
  }

 private:
  const Tvec* __restrict__ input_vec_;
  const Tindices* __restrict__ indices_;
  const Tweights* __restrict__ weights_;
};

template <typename Treducevec, typename Tvec, typename Toffsets,
          typename Tindices, typename Tweights>
struct CastIterator {
  using FunctorTy =
      LookupAndScaleAndCastInputsFunctor<Treducevec, Tvec, Tindices, Tweights>;
  using InputIteratorTy = gpuprim::CountingInputIterator<Toffsets>;
  using IteratorTy =
      gpuprim::TransformInputIterator<Treducevec, FunctorTy, InputIteratorTy>;
};

template <typename Treducevec, typename Toffsets, typename Tvec,
          typename Tindices, typename Tweights>
typename CastIterator<Treducevec, Tvec, Toffsets, Tindices,
                      Tweights>::IteratorTy
MakeLookupAndScaleAndCastInputsIterator(const Tvec* input_vec,
                                        const Tindices* indices,
                                        const Tweights* weights) {
  using CastIteratorTy =
      CastIterator<Treducevec, Tvec, Toffsets, Tindices, Tweights>;
  typename CastIteratorTy::FunctorTy functor(input_vec, indices, weights);
  return typename CastIteratorTy::IteratorTy(
      typename CastIteratorTy::InputIteratorTy(Toffsets(0)), functor);
}

template <typename Treducevec, typename Tvec, typename Toffsets,
          typename Tindices, typename Tsegmentids, typename ReduceOp,
          typename Tinit, typename Tweights>
Status SegmentReduceGPUImplNoInnerDim(
    OpKernelContext* ctx, Toffsets nouter, Tsegmentids nsegments,
    ReduceOp reduce_op, Tinit initial_value, Tinit empty_segment_value,
    bool is_mean, bool is_sqrtn,
    const Tvec* input_vec,            // [nouter or any]
    const Toffsets* segment_offsets,  // [nsegments + 1]
    const Tindices* indices,          // [nouter] (optional)
    const Tweights* weights,          // [nouter or any] (optional)
    Tvec* output_vec) {               // [nsegments]
  // Here we use gpuprim::DeviceSegmentedReduce (which is optimized for this
  // shape) and add the additional required functionality using fancy input
  // iterators and an epilogue kernel.

  // Note: This reinterpret cast is only needed to avoid compilation error
  // when Tvec != Treducevec; the result is only used if Tvec == Treducevec.
  Treducevec* output_raw_ptr = reinterpret_cast<Treducevec*>(output_vec);
  Tensor output_raw;
  bool need_temp_output = !std::is_same<Tvec, Treducevec>::value;
  if (need_temp_output) {
    // Note: We must allocate and reinterpret as bytes because Treducevec may
    // be a vector type and they are not supported as Tensor dtypes.
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_INT8,
        TensorShape({static_cast<int64_t>(nsegments * sizeof(Treducevec))}),
        &output_raw));
    output_raw_ptr =
        reinterpret_cast<Treducevec*>(output_raw.flat<int8>().data());
  }
  auto input_iter =
      MakeLookupAndScaleAndCastInputsIterator<Treducevec, Toffsets>(
          input_vec, indices, weights);
  TF_RETURN_IF_ERROR(GpuSegmentedReduce(ctx, nsegments, reduce_op,
                                        Treducevec(initial_value), input_iter,
                                        segment_offsets, output_raw_ptr));
  bool need_epilogue = !std::is_same<Tvec, Treducevec>::value ||
                       initial_value != empty_segment_value || is_mean ||
                       is_sqrtn;
  if (need_epilogue) {
    const GPUDevice& device = ctx->eigen_gpu_device();
    // Normalize based on the segment size and cast results back to T.
    TF_RETURN_IF_ERROR(LaunchSegmentReduceEpilogueKernel(
        device, nsegments, empty_segment_value, is_mean, is_sqrtn,
        output_raw_ptr, segment_offsets, output_vec));
  }
  return OkStatus();
}

template <typename Treducevec, typename Tvec, typename Toffsets,
          typename Tindices, typename Tsegmentids, typename ReduceOp,
          typename Tinit, typename Tweights>
Status SegmentReduceGPUImpl(
    OpKernelContext* ctx, Toffsets nouter, Toffsets ninner_vec,
    Tsegmentids nsegments, ReduceOp reduce_op, Tinit initial_value,
    Tinit empty_segment_value, bool is_mean, bool is_sqrtn,
    const Tvec* input_vec,           // [nouter or any, ninner_vec]
    const Tsegmentids* segment_ids,  // [nouter]
    const Tindices* indices,         // [nouter] (optional)
    const Tweights* weights,         // [nouter or any] (optional)
    Tvec* output_vec) {              // [nsegments, ninner_vec]
  const GPUDevice& device = ctx->eigen_gpu_device();

  if (nouter == 0) {
    // Just set output to empty_segment_value.
    GPUDevice d = ctx->template eigen_device<GPUDevice>();
    int64_t output_size = static_cast<int64_t>(nsegments) * ninner_vec;
    GpuLaunchConfig config = GetGpuLaunchConfig(output_size, d);
    return GpuLaunchKernel(SetToValue<Tvec, Tinit>, config.block_count,
                           config.thread_per_block, 0, d.stream(), output_size,
                           output_vec, empty_segment_value);
  }

  // Allocate and compute segment_offsets.
  Tensor segment_offsets;
  TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<Toffsets>::value,
                                        TensorShape({nsegments + 1}),
                                        &segment_offsets));
  Toffsets* segment_offsets_ptr = segment_offsets.flat<Toffsets>().data();
  TF_RETURN_IF_ERROR(LaunchSegmentOffsetsKernel(
      device, nouter, nsegments, segment_ids, segment_offsets_ptr));

  const Toffsets avg_reduce_size =
      Eigen::divup(nouter, static_cast<Toffsets>(nsegments));
  // This avg_reduce_size threshold is a performance heuristic.
  if (ninner_vec == 1 && avg_reduce_size >= 512) {
    // Here we use a gpuprim-based implementation that doesn't support an
    // inner dimension but can be significantly faster for large reductions.
    return SegmentReduceGPUImplNoInnerDim<Treducevec>(
        ctx, nouter, nsegments, reduce_op, initial_value, empty_segment_value,
        is_mean, is_sqrtn, input_vec, segment_offsets_ptr, indices, weights,
        output_vec);
  }
  // Here we use a custom kernel that is optimized for ninner_vec >= ~64 and
  // gives decent performance for smaller cases. It also handles indices,
  // casting to/from Treducevec, and normalizing the output.
  return LaunchSegmentReduceVectorKernel<Treducevec>(
      device, nouter, ninner_vec, nsegments, reduce_op, initial_value,
      empty_segment_value, is_mean, is_sqrtn, input_vec, segment_offsets_ptr,
      indices, weights, output_vec);
}

template <typename Treduce>
struct SegmentReduceGPUVectorized {
  template <int vec_size>
  struct Impl {
    template <typename T, typename Toffsets, typename Tindices,
              typename Tsegmentids, typename ReduceOp, typename Tweights>
    Status operator()(OpKernelContext* ctx, Toffsets nouter, Toffsets ninner,
                      Tsegmentids nsegments, ReduceOp reduce_op,
                      T initial_value, T empty_segment_value, bool is_mean,
                      bool is_sqrtn, const T* input,
                      const Tsegmentids* segment_ids, const Tindices* indices,
                      const Tweights* weights, T* output) {
      DCHECK_EQ(ninner % vec_size, 0);
      DCHECK_EQ(reinterpret_cast<std::uintptr_t>(input) % vec_size, 0);
      DCHECK_EQ(reinterpret_cast<std::uintptr_t>(output) % vec_size, 0);
      Toffsets ninner_vec = ninner / vec_size;
      using Tvec = AlignedVector<T, vec_size>;
      using Treducevec = AlignedVector<Treduce, vec_size>;
      const Tvec* input_vec = reinterpret_cast<const Tvec*>(input);
      Tvec* output_vec = reinterpret_cast<Tvec*>(output);

      return SegmentReduceGPUImpl<Treducevec>(
          ctx, nouter, ninner_vec, nsegments, reduce_op, initial_value,
          empty_segment_value, is_mean, is_sqrtn, input_vec, segment_ids,
          indices, weights, output_vec);
    }
  };
};

// Reduces input matrix within segments over the outer dimension. Empty segments
// always output empty_segment_value.
// The segment_ids vector must be sorted.
// If is_mean or is_sqrtn is true, the results are normalized using the
// corresponding function.
// If indices is not nullptr, input rows are accessed indirectly as
// input[indices[i]], instead of input[i].
// The implementation is deterministic.
// Note: Treduce is to allow reducing in higher precision than T.
template <typename Treduce, typename T, typename Toffsets, typename Tindices,
          typename Tsegmentids, typename ReduceOp, typename Tweights>
Status SegmentReduceGPU(OpKernelContext* ctx, Toffsets nouter, Toffsets ninner,
                        Tsegmentids nsegments, ReduceOp reduce_op,
                        T initial_value, T empty_segment_value, bool is_mean,
                        bool is_sqrtn,
                        const T* input,  // [nouter or any, ninner]
                        const Tsegmentids* segment_ids,  // [nouter]
                        const Tindices* indices,         // [nouter] (optional)
                        const Tweights* weights,  // [nouter or any] (optional)
                        T* output) {              // [nsegments, ninner]
  if (ninner == 0 || nsegments == 0) return OkStatus();
  return DispatchToVectorized<
      T, SegmentReduceGPUVectorized<Treduce>::template Impl>(
      MinAlignmentOf(input, output, ninner), ctx, nouter, ninner, nsegments,
      reduce_op, initial_value, empty_segment_value, is_mean, is_sqrtn, input,
      segment_ids, indices, weights, output);
}

template <typename SegmentId, typename Index, typename Tweights>
__global__ void SegmentWeightsKernel(
    SegmentId nsegments, SparseSegmentReductionOperation operation,
    const Index* __restrict__ segment_offsets,  // [nsegments + 1]
    Tweights* __restrict__ weights) {           // [nsegments]
  GPU_1D_KERNEL_LOOP(i, nsegments) {
    Index segment_size = segment_offsets[i + 1] - segment_offsets[i];
    segment_size = max(segment_size, Index(1));  // Avoid division by zero
    if (operation == SparseSegmentReductionOperation::kMean) {
      weights[i] = Tweights(1) / static_cast<Tweights>(segment_size);
    } else if (operation == SparseSegmentReductionOperation::kSqrtN) {
      weights[i] = Tweights(1) / sqrt(static_cast<Tweights>(segment_size));
    }
  }
}

template <typename SegmentId, typename Index, typename Tweights>
Status LaunchSegmentWeightsKernel(
    const GPUDevice& d, SegmentId nsegments,
    SparseSegmentReductionOperation operation,
    const Index* segment_offsets,  // [nsegments + 1]
    Tweights* weights) {           // [nsegments]
  GpuLaunchConfig config = GetGpuLaunchConfig(
      nsegments, d, &SegmentWeightsKernel<SegmentId, Index, Tweights>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(SegmentWeightsKernel<SegmentId, Index, Tweights>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), nsegments, operation, segment_offsets,
                         weights);
}

template <typename ReduceOp, typename T>
struct ReduceType {
  using type = T;
};

// Sum fp16 values using an fp32 accumulator to avoid numerical issues.
template <>
struct ReduceType<functor::Sum, Eigen::half> {
  using type = float;
};

template <>
struct ReduceType<functor::Sum, Eigen::bfloat16> {
  using type = float;
};

namespace functor {

template <typename T, typename Index, typename InitialValueF,
          typename EmptySegmentValueF, typename ReductionF>
void SegmentReductionFunctor<
    T, Index, InitialValueF, EmptySegmentValueF,
    ReductionF>::operator()(OpKernelContext* ctx, const GPUDevice& d,
                            const Index output_rows,
                            const TensorShape& segment_ids_shape, bool is_mean,
                            typename TTypes<Index>::ConstFlat segment_ids,
                            const Index data_size, const T* data,
                            typename TTypes<T, 2>::Tensor output) {
  if (output.size() == 0) {
    return;
  }

  // Launch kernel(s) to compute sorted segment reduction.
  // Notes:
  // *) 'input_total_size' is the total number of elements to process.
  // *) 'segment_ids.shape' is a prefix of data's shape.
  // *) 'input_outer_dim_size' is the total number of segments to process.
  const Index input_total_size = data_size;
  const Index input_outer_dim_size = segment_ids.dimension(0);
  const Index input_inner_dim_size = input_total_size / input_outer_dim_size;
  const Index num_segments = output.size() / input_inner_dim_size;

  bool use_deterministic_kernels =
      UseDeterministicSegmentReductions() ||
      (OpDeterminismRequired() && !ReduceOpIsAssociative<ReductionF, T>::value);

  // TODO(benbarsdell): If there are no performance concerns with the new
  // deterministic kernels, remove this runtime check and the old
  // non-deterministic kernels.
  if (!use_deterministic_kernels) {
    // Set 'output' to initial value.
    GpuLaunchConfig config = GetGpuLaunchConfig(output.size(), d);
    const T initial_value = InitialValueF()();
    TF_CHECK_OK(GpuLaunchKernel(SetToValue<T>, config.block_count,
                                config.thread_per_block, 0, d.stream(),
                                output.size(), output.data(), initial_value));
    if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
      return;
    }

    const int OuterDimTileSize = 8;

    const Index input_outer_dim_num_stripe =
        Eigen::divup(input_outer_dim_size, Index(OuterDimTileSize));

    const Index total_stripe_count =
        input_inner_dim_size * input_outer_dim_num_stripe;

    config = GetGpuLaunchConfig(total_stripe_count, d);
    TF_CHECK_OK(GpuLaunchKernel(
        SortedSegmentReductionCustomKernel<
            T, Index, OuterDimTileSize,
            typename ReduceUpdateOpFor<ReductionF>::nonatomic_op,
            typename ReduceUpdateOpFor<ReductionF>::atomic_op>,
        config.block_count, config.thread_per_block, 0, d.stream(),
        input_outer_dim_size, input_inner_dim_size, output_rows,
        segment_ids.data(), data, output.data(), total_stripe_count,
        initial_value));

    const T empty_value = EmptySegmentValueF()();
    if (is_mean || initial_value != empty_value) {
      Tensor segment_offsets;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<Index>::value,
                                             TensorShape({num_segments + 1}),
                                             &segment_offsets));
      Index* segment_offsets_ptr = segment_offsets.flat<Index>().data();
      OP_REQUIRES_OK(ctx, LaunchSegmentOffsetsKernel(
                              d, input_outer_dim_size, num_segments,
                              segment_ids.data(), segment_offsets_ptr));

      if (is_mean) {
        OP_REQUIRES_OK(ctx, LaunchSegmentMeanNormalizeKernel(
                                d, num_segments, input_inner_dim_size,
                                segment_offsets_ptr, output.data()));
      }
      if (initial_value != empty_value) {
        OP_REQUIRES_OK(
            ctx, LaunchSegmentSetEmptyKernel(
                     d, num_segments, input_inner_dim_size, segment_offsets_ptr,
                     empty_value, output.data()));
      }
    }
  } else {
    using Treduce = typename ReduceType<ReductionF, T>::type;
    using Tweights = typename RealTypeIfComplex<T>::type;
    OP_REQUIRES_OK(
        ctx,
        SegmentReduceGPU<Treduce>(
            ctx, input_outer_dim_size, input_inner_dim_size, num_segments,
            ReductionF(), InitialValueF()(), EmptySegmentValueF()(),
            /*is_mean=*/is_mean, /*is_sqrtn=*/false, data, segment_ids.data(),
            /*indices=*/static_cast<const Index*>(nullptr),
            /*weights=*/static_cast<Tweights*>(nullptr), output.data()));
  }
}

template <typename T, typename Index, typename InitialValueF,
          typename ReductionF>
struct UnsortedSegmentFunctor<GPUDevice, T, Index, InitialValueF, ReductionF> {
  void operator()(OpKernelContext* ctx, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat unsorted_segment_ids,
                  typename TTypes<T, 2>::ConstTensor data,
                  typename TTypes<T, 2>::Tensor output) {
    if (output.size() == 0) {
      return;
    }

    bool use_deterministic_kernels =
        UseDeterministicSegmentReductions() ||
        (!ReduceOpIsAssociative<ReductionF, T>::value &&
         OpDeterminismRequired());

    bool determinism_requirement_met =
        use_deterministic_kernels ||
        ReduceOpIsAssociative<ReductionF, T>::value ||
        !OpDeterminismRequired() ||
        DisableSegmentReductionOpDeterminismExceptions();
    OP_REQUIRES(
        ctx, determinism_requirement_met,
        errors::Unimplemented(
            "Deterministic GPU implementation of unsorted segment reduction op"
            " not available."));

    // Launch kernel(s) to compute unsorted segment reduction.
    // Notes:
    // *) 'data_size' is the total number of elements to process.
    // *) 'segment_ids.shape' is a prefix of data's shape.
    // *) 'input_outer_dim_size' is the total number of segments to process.
    const Index input_outer_dim_size = unsorted_segment_ids.dimension(0);
    const Index input_inner_dim_size = data.dimension(1);
    const Index output_outer_dim_size = output.dimension(0);
    const Index num_segments = output.size() / input_inner_dim_size;

    // TODO(benbarsdell): If there are no performance concerns with the new
    // deterministic kernels, remove this runtime check and the old
    // non-deterministic kernels.
    if (!use_deterministic_kernels) {
      // Set 'output' to initial value.
      GPUDevice d = ctx->template eigen_device<GPUDevice>();
      GpuLaunchConfig config = GetGpuLaunchConfig(output.size(), d);
      TF_CHECK_OK(GpuLaunchKernel(
          SetToValue<T>, config.block_count, config.thread_per_block, 0,
          d.stream(), output.size(), output.data(), InitialValueF()()));
      const int64_t data_size = data.size();
      if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
        return;
      }
      config = GetGpuLaunchConfig(data_size, d);
      TF_CHECK_OK(GpuLaunchKernel(
          UnsortedSegmentCustomKernel<
              T, Index, typename ReduceUpdateOpFor<ReductionF>::atomic_op>,
          config.block_count, config.thread_per_block, 0, d.stream(),
          input_outer_dim_size, input_inner_dim_size, output_outer_dim_size,
          unsorted_segment_ids.data(), data.data(), output.data()));
    } else {
      // Allocate temporary space and sort segment_ids, then call the sorted
      // implem.
      Tensor segment_ids;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(
                   DataTypeToEnum<Index>::value,
                   TensorShape({static_cast<int64_t>(input_outer_dim_size)}),
                   &segment_ids));
      Index* segment_ids_ptr = segment_ids.flat<Index>().data();
      Tensor sorted_indices;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(
                   DataTypeToEnum<Index>::value,
                   TensorShape({static_cast<int64_t>(input_outer_dim_size)}),
                   &sorted_indices));
      Index* sorted_indices_ptr = sorted_indices.flat<Index>().data();
      // Note: We must sort using all bits here because unsorted_segment_ids
      // may contain negative values.
      OP_REQUIRES_OK(
          ctx, GpuRadixSort(ctx, input_outer_dim_size,
                            /*keys_in=*/unsorted_segment_ids.data(),
                            /*keys_out=*/segment_ids_ptr,
                            /*indices_in=*/static_cast<const Index*>(nullptr),
                            /*indices_out=*/sorted_indices_ptr));
      using Treduce = typename ReduceType<ReductionF, T>::type;
      using Tweights = typename RealTypeIfComplex<T>::type;
      OP_REQUIRES_OK(
          ctx,
          SegmentReduceGPU<Treduce>(
              ctx, input_outer_dim_size, input_inner_dim_size, num_segments,
              ReductionF(), /*initial_value=*/InitialValueF()(),
              /*empty_segment_value=*/InitialValueF()(), /*is_mean=*/false,
              /*is_sqrtn=*/false, /*input=*/data.data(),
              /*segment_ids=*/segment_ids_ptr, /*indices=*/sorted_indices_ptr,
              /*weights=*/static_cast<Tweights*>(nullptr), output.data()));
    }
  }
};

template <typename T, typename Index, typename SegmentId>
Status SparseSegmentReductionFunctor<T, Index, SegmentId>::operator()(
    OpKernelContext* context, bool is_mean, bool is_sqrtn, T default_value,
    typename TTypes<T, 2>::ConstTensor input,
    typename TTypes<Index>::ConstVec indices,
    typename TTypes<SegmentId>::ConstVec segment_ids,
    typename TTypes<T, 2>::Tensor output) {
  using ReduceOp = functor::Sum;
  using Treduce = typename ReduceType<ReduceOp, T>::type;
  using Tweights = typename RealTypeIfComplex<T>::type;
  Index nouter = segment_ids.size();
  Index ninner = input.dimension(1);
  SegmentId nsegments = output.dimension(0);
  return SegmentReduceGPU<Treduce>(
      context, /*nouter=*/nouter, /*ninner=*/ninner,
      /*nsegments=*/nsegments, /*reduce_op=*/ReduceOp(),
      /*initial_value=*/T(0),
      /*empty_segment_value=*/default_value,
      /*is_mean=*/is_mean, /*is_sqrtn=*/is_sqrtn,
      /*input=*/input.data(), /*segment_ids=*/segment_ids.data(),
      /*indices=*/indices.data(), /*weights=*/static_cast<Tweights*>(nullptr),
      /*output=*/output.data());
}

template <typename T, typename Index, typename SegmentId>
struct SparseSegmentGradFunctor<GPUDevice, T, Index, SegmentId> {
  void operator()(OpKernelContext* context,
                  SparseSegmentReductionOperation operation,
                  typename TTypes<T>::ConstMatrix input_flat,
                  typename TTypes<Index>::ConstVec indices_vec,
                  typename TTypes<SegmentId>::ConstVec segment_vec,
                  Tensor* output) {
    const GPUDevice& device = context->eigen_gpu_device();

    auto output_flat = output->flat_outer_dims<T>();
    const SegmentId nsegments = input_flat.dimension(0);
    const Index ninner = input_flat.dimension(1);
    const Index nouter = indices_vec.dimension(0);
    const Index noutput = output_flat.dimension(0);

    // Allocate and compute segment weights (for Mean/SqrtN operations only).
    Tensor weights;
    using Tweights = typename RealTypeIfComplex<T>::type;
    Tweights* weights_ptr = nullptr;
    if (operation != SparseSegmentReductionOperation::kSum) {
      OP_REQUIRES_OK(
          context, context->allocate_temp(DataTypeToEnum<Tweights>::value,
                                          TensorShape({nsegments}), &weights));
      weights_ptr = weights.flat<Tweights>().data();
      // Allocate and compute segment_offsets.
      Tensor segment_offsets;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<Index>::value,
                                            TensorShape({nsegments + 1}),
                                            &segment_offsets));
      Index* segment_offsets_ptr = segment_offsets.flat<Index>().data();
      OP_REQUIRES_OK(context, LaunchSegmentOffsetsKernel(
                                  device, nouter, nsegments, segment_vec.data(),
                                  segment_offsets_ptr));
      // Compute the weights based on the segment sizes using segment_offsets.
      OP_REQUIRES_OK(context, LaunchSegmentWeightsKernel(
                                  device, nsegments, operation,
                                  segment_offsets_ptr, weights_ptr));
    }

    const Index* sorted_indices_ptr = indices_vec.data();
    const SegmentId* sorted_segment_ptr = segment_vec.data();
    Tensor tmp_sorted_indices;
    Tensor tmp_sorted_segment;
    if (noutput > 1) {
      // Sort indices and permute segments.
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<Index>::value,
                                  TensorShape({nouter}), &tmp_sorted_indices));
      Index* tmp_sorted_indices_ptr = tmp_sorted_indices.flat<Index>().data();
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<SegmentId>::value,
                                  TensorShape({nouter}), &tmp_sorted_segment));
      SegmentId* tmp_sorted_segment_ptr =
          tmp_sorted_segment.flat<SegmentId>().data();
      OP_REQUIRES_OK(context,
                     GpuRadixSort(context, nouter,
                                  /*keys_in=*/indices_vec.data(),
                                  /*keys_out=*/tmp_sorted_indices_ptr,
                                  /*indices_in=*/segment_vec.data(),
                                  /*indices_out=*/tmp_sorted_segment_ptr,
                                  /*num_bits=*/Log2Ceiling64(noutput)));
      sorted_indices_ptr = tmp_sorted_indices_ptr;
      sorted_segment_ptr = tmp_sorted_segment_ptr;
    }

    // Compute the gradient using a weighted SegmentReduceGPU with the segment
    // IDs and indices swapped.
    using ReduceOp = functor::Sum;
    using Treduce = typename ReduceType<ReduceOp, T>::type;
    OP_REQUIRES_OK(
        context,
        SegmentReduceGPU<Treduce>(
            context, /*nouter=*/static_cast<SegmentId>(nouter),
            /*ninner=*/static_cast<SegmentId>(ninner),
            /*nsegments=*/noutput,
            /*reduce_op=*/ReduceOp(),
            /*initial_value=*/T(0),
            /*empty_segment_value=*/T(0),
            /*is_mean=*/false, /*is_sqrtn=*/false,
            /*input=*/input_flat.data(), /*segment_ids=*/sorted_indices_ptr,
            /*indices=*/sorted_segment_ptr, /*weights=*/weights_ptr,
            /*output=*/output_flat.data()));
  }
};

template <typename TindicesCompact>
struct EdgeIndicatorFunctor {
  EdgeIndicatorFunctor(const TindicesCompact* sorted_indices)
      : sorted_indices_(sorted_indices) {}

  template <typename Idx>
  __device__ bool operator()(Idx idx) const {
    return idx == 0 ? false : sorted_indices_[idx] != sorted_indices_[idx - 1];
  }

 private:
  const TindicesCompact* __restrict__ sorted_indices_;
};

template <typename Toffsets, typename EdgeIndicatorIter,
          typename TindicesCompact, typename Tindices>
__global__ void ScatterUniqueIndicesKernel(
    Toffsets nouter,
    EdgeIndicatorIter sorted_indices_edge_indicator,     // [nouter]
    const TindicesCompact* __restrict__ sorted_indices,  // [nouter]
    const Toffsets* __restrict__ sorted_indices_ids,     // [nouter]
    Tindices* __restrict__ sorted_unique_indices) {      // [num_unique]
  for (int i : GpuGridRangeX(nouter)) {
    if (i == 0 || sorted_indices_edge_indicator[i]) {
      sorted_unique_indices[sorted_indices_ids[i]] =
          static_cast<Tindices>(sorted_indices[i]);
    }
  }
}

template <typename Toffsets, typename EdgeIndicatorIter,
          typename TindicesCompact, typename Tindices>
Status LaunchScatterUniqueIndicesKernel(
    const GPUDevice& d, Toffsets nouter,
    EdgeIndicatorIter sorted_indices_edge_indicator,     // [nouter]
    const TindicesCompact* __restrict__ sorted_indices,  // [nouter]
    const Toffsets* __restrict__ sorted_indices_ids,     // [nouter]
    Tindices* __restrict__ sorted_unique_indices) {      // [num_unique]
  GpuLaunchConfig config = GetGpuLaunchConfig(
      nouter, d,
      &ScatterUniqueIndicesKernel<Toffsets, EdgeIndicatorIter, TindicesCompact,
                                  Tindices>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(ScatterUniqueIndicesKernel<Toffsets, EdgeIndicatorIter,
                                                    TindicesCompact, Tindices>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), nouter, sorted_indices_edge_indicator,
                         sorted_indices, sorted_indices_ids,
                         sorted_unique_indices);
}

template <typename T, typename Tindices, typename Tsegmentids>
struct SparseSegmentGradV2Functor<GPUDevice, T, Tindices, Tsegmentids> {
  void operator()(OpKernelContext* context,
                  SparseSegmentReductionOperation operation,
                  typename TTypes<T>::ConstMatrix input_flat,
                  typename TTypes<Tindices>::ConstVec indices_vec,
                  typename TTypes<Tsegmentids>::ConstVec segment_vec,
                  const TensorShape& dense_output_shape,
                  typename AsyncOpKernel::DoneCallback done) {
    const GPUDevice& device = context->eigen_gpu_device();

    const int64_t nsegments = input_flat.dimension(0);
    const int64_t ninner64 = input_flat.dimension(1);
    const int64_t nouter64 = indices_vec.dimension(0);
    // Note: nouter and ninner are not expected to be huge, so we use int32 to
    // save memory bandwidth.
    using Toffsets = int32;
    OP_REQUIRES_ASYNC(context, nouter64 <= std::numeric_limits<Toffsets>::max(),
                      absl::InvalidArgumentError(
                          absl::StrCat("Indices vector of length ", nouter64,
                                       " is too large to fit in int32.")),
                      done);
    const Toffsets nouter = static_cast<Toffsets>(nouter64);
    OP_REQUIRES_ASYNC(context, ninner64 <= std::numeric_limits<Toffsets>::max(),
                      absl::InvalidArgumentError(absl::StrCat(
                          "Inner data dimension of size ", ninner64,
                          " is too large to fit in int32.")),
                      done);
    const Toffsets ninner = static_cast<Toffsets>(ninner64);

    // Cast indices to 32-bit to save memory bandwidth (the cost of the cast is
    // worth it because the vector is used multiple times).
    // Note that we can currently assume int32 is safe because the op's dense
    // output_dim0 input is always int32.
    using TindicesCompact = int32;
    Tensor tmp_indices_internal;
    const TindicesCompact* indices_internal_ptr;
    if constexpr (std::is_same<Tindices, TindicesCompact>::value) {
      indices_internal_ptr = indices_vec.data();
    } else {
      OP_REQUIRES_OK_ASYNC(
          context,
          context->allocate_temp(DataTypeToEnum<TindicesCompact>::value,
                                 TensorShape({nouter}), &tmp_indices_internal),
          done);
      auto indices_vec_internal = tmp_indices_internal.flat<TindicesCompact>();
      indices_vec_internal.device(device) =
          indices_vec.template cast<TindicesCompact>();
      indices_internal_ptr = indices_vec_internal.data();
    }

    // Cast segment IDs to smallest possible type to save memory bandwidth.
    if (nsegments <= std::numeric_limits<int16_t>::max()) {
      CastSegmentIdsThenImpl<Toffsets, TindicesCompact, int16_t>(
          context, operation, nouter, ninner, nsegments, input_flat.data(),
          tmp_indices_internal, indices_internal_ptr, segment_vec,
          dense_output_shape, done);
    } else if (sizeof(Tsegmentids) > sizeof(int32) &&
               nsegments <= std::numeric_limits<int32>::max()) {
      CastSegmentIdsThenImpl<Toffsets, TindicesCompact, int32>(
          context, operation, nouter, ninner, nsegments, input_flat.data(),
          tmp_indices_internal, indices_internal_ptr, segment_vec,
          dense_output_shape, done);
    } else {
      Impl<Toffsets, TindicesCompact, Tsegmentids>(
          context, operation, nouter, ninner, nsegments, input_flat.data(),
          tmp_indices_internal, indices_internal_ptr, Tensor(),
          segment_vec.data(), dense_output_shape, done);
    }
  }

 private:
  using Tweights = typename RealTypeIfComplex<T>::type;

  template <typename Toffsets, typename TindicesCompact,
            typename Tsegmentids_internal>
  void CastSegmentIdsThenImpl(
      OpKernelContext* context, SparseSegmentReductionOperation operation,
      Toffsets nouter, Toffsets ninner, Tsegmentids_internal nsegments,
      const T* input, Tensor indices_tensor, const TindicesCompact* indices,
      typename TTypes<Tsegmentids>::ConstVec segment_vec,
      const TensorShape& dense_output_shape,
      typename AsyncOpKernel::DoneCallback done) {
    const GPUDevice& device = context->eigen_gpu_device();
    Tensor tmp_segment_internal;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_temp(DataTypeToEnum<Tsegmentids_internal>::value,
                               TensorShape({nouter}), &tmp_segment_internal),
        done);
    auto segment_vec_internal =
        tmp_segment_internal.flat<Tsegmentids_internal>();
    segment_vec_internal.device(device) =
        segment_vec.template cast<Tsegmentids_internal>();

    Impl<Toffsets, TindicesCompact, Tsegmentids_internal>(
        context, operation, nouter, ninner, nsegments, input, indices_tensor,
        indices, tmp_segment_internal, segment_vec_internal.data(),
        dense_output_shape, done);
  }

  template <typename Toffsets, typename TindicesCompact,
            typename Tsegmentids_internal>
  void Impl(OpKernelContext* context, SparseSegmentReductionOperation operation,
            Toffsets nouter, Toffsets ninner, Tsegmentids_internal nsegments,
            const T* input, Tensor indices_tensor,
            const TindicesCompact* indices, Tensor segment_ids_tensor,
            const Tsegmentids_internal* segment_ids,
            const TensorShape& dense_output_shape,
            typename AsyncOpKernel::DoneCallback done) {
    const int64_t dense_output_dim0 = dense_output_shape.dim_size(0);

    // Allocate and compute segment weights (for Mean/SqrtN operations only).
    Tensor tmp_weights;
    Tweights* weights_ptr = nullptr;
    if (operation != SparseSegmentReductionOperation::kSum) {
      ComputeSegmentWeights(context, operation, nsegments, nouter, segment_ids,
                            &tmp_weights, done);
      weights_ptr = tmp_weights.flat<Tweights>().data();
    }

    const TindicesCompact* sorted_indices_ptr = indices;
    const Tsegmentids_internal* permuted_segment_ptr = segment_ids;
    Tensor tmp_sorted_indices;
    Tensor tmp_permuted_segment;
    if (dense_output_dim0 > 1) {
      // Sort indices and permute segments.
      OP_REQUIRES_OK_ASYNC(
          context,
          context->allocate_temp(DataTypeToEnum<TindicesCompact>::value,
                                 TensorShape({nouter}), &tmp_sorted_indices),
          done);
      TindicesCompact* tmp_sorted_indices_ptr =
          tmp_sorted_indices.flat<TindicesCompact>().data();
      OP_REQUIRES_OK_ASYNC(
          context,
          context->allocate_temp(DataTypeToEnum<Tsegmentids_internal>::value,
                                 TensorShape({nouter}), &tmp_permuted_segment),
          done);
      Tsegmentids_internal* tmp_permuted_segment_ptr =
          tmp_permuted_segment.flat<Tsegmentids_internal>().data();
      OP_REQUIRES_OK_ASYNC(
          context,
          GpuRadixSort(context, nouter,
                       /*keys_in=*/indices,
                       /*keys_out=*/tmp_sorted_indices_ptr,
                       /*indices_in=*/segment_ids,
                       /*indices_out=*/tmp_permuted_segment_ptr,
                       /*num_bits=*/Log2Ceiling64(dense_output_dim0)),
          done);
      sorted_indices_ptr = tmp_sorted_indices_ptr;
      permuted_segment_ptr = tmp_permuted_segment_ptr;
      // The original tensors are no longer needed.
      indices_tensor = Tensor();
      indices = nullptr;
      segment_ids_tensor = Tensor();
      segment_ids = nullptr;
    }

    using CountIter = gpuprim::CountingInputIterator<Toffsets>;
    using EdgeIndicatorIter = gpuprim::TransformInputIterator<
        Toffsets, EdgeIndicatorFunctor<TindicesCompact>, CountIter>;
    EdgeIndicatorIter sorted_indices_edge_indicator(
        CountIter(0),
        EdgeIndicatorFunctor<TindicesCompact>(sorted_indices_ptr));

    Tensor tmp_sorted_indices_unique_ids;
    OP_REQUIRES_OK_ASYNC(context,
                         context->allocate_temp(DataTypeToEnum<Toffsets>::value,
                                                TensorShape({nouter}),
                                                &tmp_sorted_indices_unique_ids),
                         done);
    Toffsets* sorted_indices_unique_ids_ptr =
        tmp_sorted_indices_unique_ids.flat<Toffsets>().data();
    OP_REQUIRES_OK_ASYNC(
        context,
        GpuInclusivePrefixSum(context, nouter, sorted_indices_edge_indicator,
                              sorted_indices_unique_ids_ptr),
        done);

    se::Stream* stream = context->op_device_context()->stream();
    OP_REQUIRES_ASYNC(context, stream,
                      absl::InternalError("No GPU stream available."), done);

    // Copy the last element of sorted_indices_unique_ids back to the host to
    // obtain num_unique.
    ScratchSpace<Toffsets> last_idx_host(context, 1, /*on_host=*/true);
    OP_REQUIRES_OK_ASYNC(
        context,
        stream->Memcpy(last_idx_host.mutable_data(),
                       se::DeviceMemoryBase(const_cast<Toffsets*>(
                                                sorted_indices_unique_ids_ptr) +
                                                (nouter - 1),
                                            sizeof(*last_idx_host.data())),
                       sizeof(*last_idx_host.data())),
        done);

    auto async_finish_computation =
        [this, context, dense_output_shape, nouter, ninner, input,
         indices_tensor, tmp_sorted_indices, sorted_indices_ptr,
         tmp_sorted_indices_unique_ids, sorted_indices_unique_ids_ptr,
         segment_ids_tensor, tmp_permuted_segment, permuted_segment_ptr,
         sorted_indices_edge_indicator, tmp_weights, weights_ptr, last_idx_host,
         done]() -> void {
      const GPUDevice& device = context->eigen_gpu_device();
      Toffsets num_unique = (*last_idx_host.data()) + 1;

      se::gpu::ScopedActivateExecutorContext scoped_activation{
          context->op_device_context()->stream()->parent()};

      TensorShape output_shape = dense_output_shape;
      OP_REQUIRES_OK_ASYNC(context,
                           output_shape.SetDimWithStatus(0, num_unique), done);
      Tensor* output = nullptr;
      T* output_ptr;
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(0, output_shape, &output), done);
      output_ptr = output->flat<T>().data();

      // Compute the gradient using a weighted SegmentReduceGPU with the segment
      // IDs and indices swapped.
      using ReduceOp = functor::Sum;
      using Treduce = typename ReduceType<ReduceOp, T>::type;
      OP_REQUIRES_OK_ASYNC(context,
                           SegmentReduceGPU<Treduce>(
                               context, /*nouter=*/nouter,
                               /*ninner=*/ninner,
                               /*nsegments=*/num_unique,
                               /*reduce_op=*/ReduceOp(),
                               /*initial_value=*/T(0),
                               /*empty_segment_value=*/T(0),
                               /*is_mean=*/false, /*is_sqrtn=*/false,
                               /*input=*/input,
                               /*segment_ids=*/sorted_indices_unique_ids_ptr,
                               /*indices=*/permuted_segment_ptr,
                               /*weights=*/weights_ptr,
                               /*output=*/output_ptr),
                           done);

      Tensor* sorted_unique_indices = nullptr;
      Tindices* sorted_unique_indices_ptr;
      OP_REQUIRES_OK_ASYNC(
          context,
          context->allocate_output(1, TensorShape({num_unique}),
                                   &sorted_unique_indices),
          done);
      sorted_unique_indices_ptr =
          sorted_unique_indices->flat<Tindices>().data();

      OP_REQUIRES_OK_ASYNC(
          context,
          LaunchScatterUniqueIndicesKernel(
              device, nouter, sorted_indices_edge_indicator, sorted_indices_ptr,
              sorted_indices_unique_ids_ptr, sorted_unique_indices_ptr),
          done);

      done();
    };

    context->device()
        ->tensorflow_accelerator_device_info()
        ->event_mgr->ThenExecute(stream, async_finish_computation);
  }

  template <typename Tsegmentids_internal, typename Toffsets>
  void ComputeSegmentWeights(OpKernelContext* context,
                             SparseSegmentReductionOperation operation,
                             Tsegmentids_internal nsegments, Toffsets nouter,
                             const Tsegmentids_internal* segment_ids,
                             Tensor* tmp_weights,
                             typename AsyncOpKernel::DoneCallback done) {
    const GPUDevice& device = context->eigen_gpu_device();
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_temp(DataTypeToEnum<Tweights>::value,
                               TensorShape({nsegments}), tmp_weights),
        done);
    Tweights* weights_ptr = tmp_weights->flat<Tweights>().data();
    // Allocate and compute segment_offsets.
    Tensor tmp_segment_offsets;
    OP_REQUIRES_OK_ASYNC(context,
                         context->allocate_temp(DataTypeToEnum<Toffsets>::value,
                                                TensorShape({nsegments + 1}),
                                                &tmp_segment_offsets),
                         done);
    Toffsets* segment_offsets_ptr = tmp_segment_offsets.flat<Toffsets>().data();
    OP_REQUIRES_OK_ASYNC(
        context,
        LaunchSegmentOffsetsKernel(device, nouter, nsegments, segment_ids,
                                   segment_offsets_ptr),
        done);
    // Compute the weights based on the segment sizes using segment_offsets.
    OP_REQUIRES_OK_ASYNC(
        context,
        LaunchSegmentWeightsKernel(device, nsegments, operation,
                                   segment_offsets_ptr, weights_ptr),
        done);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#endif  // TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_GPU_CU_H_
