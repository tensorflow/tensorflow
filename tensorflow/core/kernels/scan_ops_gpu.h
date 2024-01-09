/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SCAN_OPS_GPU_H_
#define TENSORFLOW_CORE_KERNELS_SCAN_OPS_GPU_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#define CUB_USE_COOPERATIVE_GROUPS

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/scan_ops.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "tensorflow/core/util/permutation_input_iterator.h"
#include "tensorflow/core/util/permutation_output_iterator.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::Index Index;

namespace functor {

// Map a contiguous range to the actual memory locations depending on which
// axis the scan is taking place over and whether or not reversed.
struct MapIndexToLocation {
  __host__ __device__ MapIndexToLocation(int dimx, int dimy, int dimz,
                                         bool reverse = false)
      : dimx_(dimx), dimy_(dimy), dimz_(dimz), reverse_(reverse) {}

  __host__ __device__ int operator()(int id) const {
    if (dimx_ == 1) {
      int row = id % dimy_;
      int col = id / dimy_;

      if (reverse_) return (dimy_ - row - 1) * dimz_ + col;

      return row * dimz_ + col;
    } else if (dimz_ == 1) {
      if (reverse_) {
        int row = id / dimy_;
        int col = id % dimy_;
        return row * dimy_ + (dimy_ - col - 1);
      }
      return id;
    } else {
      int col = id % dimy_;
      int tmp = id / dimy_;

      int row1 = id / (dimy_ * dimz_);
      int col1 = tmp % dimz_;

      if (reverse_)
        return row1 * dimy_ * dimz_ + (dimy_ - col - 1) * dimz_ + col1;

      return row1 * dimy_ * dimz_ + col * dimz_ + col1;
    }
  }

  int dimx_;
  int dimy_;
  int dimz_;
  bool reverse_;
};

template <typename T, typename Op>
struct BlockPrefixCallbackOp {
  // Running prefix
  T running_total_;
  Op op_;

  __device__ BlockPrefixCallbackOp(T running_total, Op op)
      : running_total_(running_total), op_(op) {}

  // Callback operator to be entered by the first warp of threads in the block.
  // tid 0 is responsible for returning a value for seeding the block-wide scan.
  __device__ T operator()(T block_aggregate) {
    T old_prefix = running_total_;
    running_total_ = op_(old_prefix, block_aggregate);
    return old_prefix;
  }
};

template <typename T>
struct Sum {
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

template <typename T>
struct Prod {
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return a * b;
  }
};

template <typename T, typename Op>
struct IsSum {
  constexpr static bool value =
      (std::is_same<Op, Sum<T>>::value ||
       std::is_same<Op, Eigen::internal::SumReducer<T>>::value);
};

template <typename T, typename Op>
struct IsProd {
  constexpr static bool value =
      (std::is_same<Op, Prod<T>>::value ||
       std::is_same<Op, Eigen::internal::ProdReducer<T>>::value);
};

template <typename T, typename Op>
struct IsLogSumExp {
  constexpr static bool value = (std::is_same<Op, LogSumExp<T>>::value ||
                                 std::is_same<Op, LogSumExpReducer<T>>::value);
};

template <typename T, typename Op>
struct IdentityValue {
  static_assert(IsSum<T, Op>::value || IsProd<T, Op>::value ||
                    IsLogSumExp<T, Op>::value,
                "IdentityValue not yet defined for this type.");

  template <typename U = T, typename OpCopy = Op>
  __host__ __device__ U operator()(
      typename std::enable_if<IsSum<U, OpCopy>::value, U>::type t = U(0)) {
    return t;
  }

  template <typename U = T, typename OpCopy = Op>
  __host__ __device__ U operator()(
      typename std::enable_if<IsProd<U, OpCopy>::value, U>::type t = U(1)) {
    return t;
  }

  template <typename U = T, typename OpCopy = Op>
  __host__ __device__ U
  operator()(typename std::enable_if<IsLogSumExp<U, OpCopy>::value, U>::type t =
                 U(Eigen::NumTraits<U>::lowest())) {
    return t;
  }
};

// Each block is mapped to one sequence.  A contiguous range is mapped to the
// appropriate locations in memory by the permutation iterators.  This is
// ideal for 1-D and row based scans.  Column scans would be better if they
// did a block load and then locally transposed.  CUB's device wide scan is not
// used in the large 1D case, even though it would be more efficient, because
// it is not deterministic.
template <typename T, typename Op, int BlockDim = 128, int ItemsPerThread = 4>
__launch_bounds__(BlockDim) __global__
    void scan_kernel(const T* in, T* out, int dimx, int dimy, int dimz,
                     bool exclusive, bool reverse, Op op) {
  typedef gpuprim::BlockLoad<T, BlockDim, ItemsPerThread,
                             gpuprim::BLOCK_LOAD_TRANSPOSE>
      BlockLoad;
  typedef gpuprim::BlockStore<T, BlockDim, ItemsPerThread,
                              gpuprim::BLOCK_STORE_TRANSPOSE>
      BlockStore;
  typedef gpuprim::BlockScan<T, BlockDim> BlockScan;

  // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockScan::TempStorage scan;
    typename BlockStore::TempStorage store;
  } temp_storage;

  int problem_length = dimy;

  // Initialize running total
  BlockPrefixCallbackOp<T, Op> prefix_op(IdentityValue<T, Op>()(), op);

  MapIndexToLocation map_op(dimx, dimy, dimz, reverse);
  int block_start = problem_length * blockIdx.x;
  // Have the block iterate over segments of items
  for (int block_offset = block_start;
       block_offset < block_start + problem_length;
       block_offset += BlockDim * ItemsPerThread) {
    int valid_items = min(BlockDim * ItemsPerThread,
                          problem_length - (block_offset % problem_length));

    // first construct a counting iterator that has the desired start point
    typedef gpuprim::TransformInputIterator<int, MapIndexToLocation,
                                            gpuprim::CountingInputIterator<int>>
        MapIterType;

    gpuprim::CountingInputIterator<int> counting_iter(block_offset);

    // Next map the iterator to the actual locations in memory
    MapIterType map_iter(counting_iter, map_op);

    PermutationInputIterator<T, const T*, MapIterType> permutein_iter(in,
                                                                      map_iter);
    PermutationOutputIterator<T, T*, MapIterType> permuteout_iter(out,
                                                                  map_iter);

    // Load a segment of consecutive items that are blocked across threads
    T thread_data[ItemsPerThread];
    BlockLoad(temp_storage.load).Load(permutein_iter, thread_data, valid_items);
    __syncthreads();

    // Collectively compute the block-wide scan
    if (exclusive) {
      BlockScan(temp_storage.scan)
          .ExclusiveScan(thread_data, thread_data, op, prefix_op);
    } else {
      BlockScan(temp_storage.scan)
          .InclusiveScan(thread_data, thread_data, op, prefix_op);
    }
    __syncthreads();

    // Store scanned items to output segment
    BlockStore(temp_storage.store)
        .Store(permuteout_iter, thread_data, valid_items);
    __syncthreads();
  }
}

template <typename T, typename Op>
void LaunchScan(const GPUDevice& d, typename TTypes<T, 3>::ConstTensor in,
                typename TTypes<T, 3>::Tensor out, Op op, const bool reverse,
                const bool exclusive) {
  const int items_per_thread = 4;

  int dimx = in.dimension(0);
  int dimy = in.dimension(1);
  int dimz = in.dimension(2);
  int num_blocks = dimx * dimz;

  int ideal_block_size = dimy / items_per_thread;
  const int rocm_threads_per_warp = 64;
  ideal_block_size = std::max(ideal_block_size, rocm_threads_per_warp);

  // There seems to be a bug when the type is not float and block_size 1024.
  // Launch on the smallest power of 2 block size that we can.
  if (ideal_block_size >= 1024 && std::is_same<T, float>::value) {
    const int block_size = 1024;
    TF_CHECK_OK(
        GpuLaunchKernel(scan_kernel<T, Op, block_size, items_per_thread>,
                        num_blocks, block_size, 0, d.stream(), in.data(),
                        out.data(), dimx, dimy, dimz, exclusive, reverse, op));
  } else if (ideal_block_size >= 512) {
    const int block_size = 512;
    TF_CHECK_OK(
        GpuLaunchKernel(scan_kernel<T, Op, block_size, items_per_thread>,
                        num_blocks, block_size, 0, d.stream(), in.data(),
                        out.data(), dimx, dimy, dimz, exclusive, reverse, op));
  } else if (ideal_block_size >= 256) {
    const int block_size = 256;
    TF_CHECK_OK(
        GpuLaunchKernel(scan_kernel<T, Op, block_size, items_per_thread>,
                        num_blocks, block_size, 0, d.stream(), in.data(),
                        out.data(), dimx, dimy, dimz, exclusive, reverse, op));
  } else if (ideal_block_size >= 128) {
    const int block_size = 128;
    TF_CHECK_OK(
        GpuLaunchKernel(scan_kernel<T, Op, block_size, items_per_thread>,
                        num_blocks, block_size, 0, d.stream(), in.data(),
                        out.data(), dimx, dimy, dimz, exclusive, reverse, op));
#if TENSORFLOW_COMPILER_IS_HIP_CLANG
    // HIP-CLANG has some kind of problem here with 32 threads (possibly because
    // the warpsize is 64). Reenable when working properly
  } else if (true) {
#else
  } else if (ideal_block_size >= 64) {
#endif
    const int block_size = 64;
    TF_CHECK_OK(
        GpuLaunchKernel(scan_kernel<T, Op, block_size, items_per_thread>,
                        num_blocks, block_size, 0, d.stream(), in.data(),
                        out.data(), dimx, dimy, dimz, exclusive, reverse, op));
  } else {
    const int block_size = 32;
    TF_CHECK_OK(
        GpuLaunchKernel(scan_kernel<T, Op, block_size, items_per_thread>,
                        num_blocks, block_size, 0, d.stream(), in.data(),
                        out.data(), dimx, dimy, dimz, exclusive, reverse, op));
  }
}

template <typename T>
struct Scan<GPUDevice, Eigen::internal::SumReducer<T>, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 3>::ConstTensor in,
                  typename TTypes<T, 3>::Tensor out,
                  const Eigen::internal::SumReducer<T>& reducer,
                  const bool reverse, const bool exclusive) {
    LaunchScan<T, Sum<T>>(d, in, out, Sum<T>(), reverse, exclusive);
  }
};

template <typename T>
struct Scan<GPUDevice, Eigen::internal::ProdReducer<T>, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 3>::ConstTensor in,
                  typename TTypes<T, 3>::Tensor out,
                  const Eigen::internal::ProdReducer<T>& reducer,
                  const bool reverse, const bool exclusive) {
    LaunchScan<T, Prod<T>>(d, in, out, Prod<T>(), reverse, exclusive);
  }
};

template <typename T>
struct Scan<GPUDevice, LogSumExpReducer<T>, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 3>::ConstTensor in,
                  typename TTypes<T, 3>::Tensor out,
                  const LogSumExpReducer<T>& reducer, const bool reverse,
                  const bool exclusive) {
    LaunchScan<T, LogSumExp<T>>(d, in, out, LogSumExp<T>(), reverse, exclusive);
  }
};

}  // namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_SCAN_OPS_GPU_H_
