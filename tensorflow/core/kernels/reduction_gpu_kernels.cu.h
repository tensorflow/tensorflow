/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_REDUCTION_GPU_KERNELS_CU_H_
#define TENSORFLOW_CORE_KERNELS_REDUCTION_GPU_KERNELS_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "third_party/cub/device/device_reduce.cuh"
#include "third_party/cub/device/device_segmented_reduce.cuh"
#include "third_party/cub/iterator/counting_input_iterator.cuh"
#include "third_party/cub/iterator/transform_input_iterator.cuh"
#include "third_party/cub/warp/warp_reduce.cuh"
#include "cuda/include/cuComplex.h"
#include "tensorflow/core/kernels/reduction_ops.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/permutation_input_iterator.h"
#include "tensorflow/core/util/transform_output_iterator.h"

#include <sstream>

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct Sum {
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

// needed to work around a compiler bug in nvcc - it doesn't seem to like
// the overloaded addition op for std::complex
template <>
struct Sum<std::complex<float>> {
  __host__ __device__ std::complex<float> operator()(
      const std::complex<float>& a, const std::complex<float>& b) const {
    auto result = cuCaddf(make_cuComplex(a.real(), a.imag()),
                          make_cuComplex(b.real(), b.imag()));
    return std::complex<float>(result.x, result.y);
  }
};

template <>
struct Sum<std::complex<double>> {
  __host__ __device__ std::complex<double> operator()(
      const std::complex<double>& a, const std::complex<double>& b) const {
    auto result = cuCadd(make_cuDoubleComplex(a.real(), a.imag()),
                         make_cuDoubleComplex(b.real(), b.imag()));
    return std::complex<double>(result.x, result.y);
  }
};

template <typename T>
struct Prod {
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return a * b;
  }
};

// needed to work around a compiler bug in nvcc - it doesn't seem to like
// the overloaded multiply op for std::complex
template <>
struct Prod<std::complex<float>> {
  __host__ __device__ std::complex<float> operator()(
      const std::complex<float>& a, const std::complex<float>& b) const {
    auto result = cuCmulf(make_cuComplex(a.real(), a.imag()),
                          make_cuComplex(b.real(), b.imag()));
    return std::complex<float>(result.x, result.y);
  }
};

template <>
struct Prod<std::complex<double>> {
  __host__ __device__ std::complex<double> operator()(
      const std::complex<double>& a, const std::complex<double>& b) const {
    auto result = cuCmul(make_cuDoubleComplex(a.real(), a.imag()),
                         make_cuDoubleComplex(b.real(), b.imag()));
    return std::complex<double>(result.x, result.y);
  }
};

template <typename T, typename outT = T>
struct DividesBy {
  T divisor;

  __host__ __device__ explicit DividesBy(T divisor) : divisor(divisor) {}

  __host__ __device__ outT operator()(const T& x) const { return x / divisor; }
};

// needed to work around a compiler bug in nvcc - it doesn't seem to like
// the overloaded ops for std::complex
template <>
struct DividesBy<std::complex<float>> {
  cuFloatComplex divisor;

  __host__ __device__ explicit DividesBy(std::complex<float> divisor)
      : divisor(make_cuComplex(divisor.real(), divisor.imag())) {}

  // implements
  __host__ __device__ std::complex<float> operator()(
      const std::complex<float>& x) const {
    auto result = cuCdivf(make_cuComplex(x.real(), x.imag()), divisor);
    return std::complex<float>(result.x, result.y);
  }
};

template <>
struct DividesBy<std::complex<double>> {
  cuDoubleComplex divisor;

  __host__ __device__ explicit DividesBy(std::complex<double> divisor)
      : divisor(make_cuDoubleComplex(divisor.real(), divisor.imag())) {}

  // implements
  __host__ __device__ std::complex<double> operator()(
      const std::complex<double>& x) const {
    auto result = cuCdiv(make_cuDoubleComplex(x.real(), x.imag()), divisor);
    return std::complex<double>(result.x, result.y);
  }
};

template <>
struct DividesBy<float, Eigen::half> {
  float divisor;

  __host__ __device__ explicit DividesBy(float divisor) : divisor(divisor) {}

  __host__ __device__ Eigen::half operator()(const float& x) const {
    return Eigen::half(x / divisor);
  }
};

struct HalfToFloat {
  __host__ __device__ float operator()(const Eigen::half& x) const {
    return Eigen::half_impl::half_to_float(x);
  }
};

struct FloatToHalf {
  __host__ __device__ Eigen::half operator()(const float& x) const {
    return Eigen::half_impl::float_to_half_rtne(x);
  }
};

struct And {
  __host__ __device__ bool operator()(const bool& a, const bool& b) const {
    return a && b;
  }
};

struct Or {
  __host__ __device__ bool operator()(const bool& a, const bool& b) const {
    return a || b;
  }
};

// each block does a grid strided loop and reduces its values locally
// the case of one block is used for low latency small reductions to scalars
template <typename T, typename outT, int num_threads, typename Op>
__global__ void BlockReduceKernel(
    T in, outT out, int num_elems, Op op,
    typename std::iterator_traits<T>::value_type initVal) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  const int gid = bid * blockDim.x + tid;
  const int stride = blockDim.x * gridDim.x;

  typedef typename std::iterator_traits<T>::value_type value_type;

  value_type sum = initVal;
  if (gid < num_elems) {
    sum = in[gid];
    for (int pos = gid + stride; pos < num_elems; pos += stride) {
      sum = op(sum, in[pos]);
    }
  }

  typedef cub::BlockReduce<value_type, num_threads> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  // only include input values in the reduction
  //
  // elements: -----------------
  // grid:     |====|====|====|====|====|
  const int num_elements_to_reduce =
      max(min(num_elems - bid * blockDim.x, num_threads), 0);

  sum = BlockReduce(temp_storage).Reduce(sum, op, num_elements_to_reduce);

  if (tid == 0) out[bid] = sum;
}

// maps a warp to each row
template <typename T, typename outT, typename Op>
__global__ void RowReduceKernel(
    T in, outT out, int num_rows, int num_cols, Op op,
    typename std::iterator_traits<T>::value_type initVal) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  // Defensive index computation to avoid integer overflow.
  assert(blockDim.x % 32 == 0);
  int warps_per_block = blockDim.x / 32;
  int warp_index = threadIdx.x / 32;
  const int row = blockIdx.x * warps_per_block + warp_index;
  const int lane = threadIdx.x % 32;

  if (num_cols == 1) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < num_rows) out[gid] = in[gid];
    return;
  }

  value_type sum = initVal;
  int col = lane;

  if (row < num_rows && col < num_cols) {
    sum = in[row * num_cols + col];
    col += 32;
    for (; col < num_cols; col += 32) {
      sum = op(sum, in[row * num_cols + col]);
    }
  }

  typedef cub::WarpReduce<value_type> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp_storage;

  sum = WarpReduce(temp_storage).Reduce(sum, op, min(num_cols, 32));

  if (row < num_rows && lane == 0) out[row] = sum;
}

template <typename T1>
struct storage_type {
  T1 val;
  __host__ __device__ storage_type() {}
  __host__ __device__ operator T1() { return val; }
  __host__ __device__ storage_type<T1>& operator=(const T1& in) {
    val = in;
    return *this;
  }
};

template <typename T2>
struct storage_type<std::complex<T2>> {
  T2 real;
  T2 imag;
  __host__ __device__ storage_type() {}
  __host__ __device__ operator std::complex<T2>() {
    return std::complex<T2>(real, imag);
  }
  __host__ __device__ storage_type<std::complex<T2>>& operator=(
      const std::complex<T2>& in) {
    real = in.real();
    imag = in.imag();
    return *this;
  }
};

// Works only if there are <= 16 columns
// each warps sums over multiple rows at once
template <typename T, typename outT, typename Op>
__global__ void ColumnReduceMax16ColumnsKernel(
    T in, outT out, int num_rows, int num_cols, Op op,
    typename std::iterator_traits<T>::value_type initVal) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  int rows_per_warp = 32 / num_cols;

  const int lane = threadIdx.x % 32;
  const int lane_row = lane / num_cols;

  const int start_row_warp =
      rows_per_warp * (blockIdx.y * blockDim.y + threadIdx.y);
  const int start_row_lane = start_row_warp + lane_row;
  int row = start_row_lane;
  int col = lane % num_cols;

  value_type sum = initVal;
  if (row * num_cols + col < num_rows * num_cols)
    sum = in[row * num_cols + col];

  // 1D array necessary due to bug in CUDA 9 compiler.
  // TODO(nluehr) revert to 2D array when compiler is ready.
  // This is to mimic the following, but without any constructors:
  //   __shared__ storage_type<value_type> partial_sums[32 * 33];
  __shared__ __align__(
      alignof(value_type)) char partial_sums_raw[32 * 33 * sizeof(value_type)];
  value_type* partial_sums = reinterpret_cast<value_type*>(partial_sums_raw);

  row += rows_per_warp * gridDim.y * blockDim.y;
  for (; row < num_rows; row += rows_per_warp * gridDim.y * blockDim.y) {
    int global_pos = row * num_cols + col;
    if (global_pos < (num_rows * num_cols))
      sum = op(sum, in[row * num_cols + col]);
  }

  const int rows_in_this_warp = min(rows_per_warp, num_rows - start_row_warp);
  // not the most efficient way to do this sum
  for (int i = 1; i < rows_in_this_warp; ++i) {
    value_type tmp = cub::ShuffleIndex<32, value_type>(
        sum, static_cast<int>(threadIdx.x + i * num_cols), 0xffffffff);
    if (lane < num_cols) sum = op(sum, tmp);
  }

  if (lane < num_cols) partial_sums[lane * 33 + threadIdx.y] = sum;

  __syncthreads();

  if (threadIdx.y == 0 && threadIdx.x < num_cols) {
    value_type s = partial_sums[threadIdx.x * 33];

    if (blockDim.y > 1) {
      for (int row = 1; row < blockDim.y; ++row) {
        value_type t = partial_sums[threadIdx.x * 33 + row];
        s = op(s, t);
      }
    }

    out[col * gridDim.y + blockIdx.y] = s;
  }
}

// Maps each block to a column range 32 wide
template <typename T, typename outT, typename Op>
__global__ void ColumnReduceKernel(
    T in, outT out, int num_rows, int num_cols, Op op,
    typename std::iterator_traits<T>::value_type initVal) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * 32 + threadIdx.x;

  value_type sum = initVal;
  if (row < num_rows && col < num_cols) sum = in[row * num_cols + col];

  // 1D array necessary due to bug in CUDA 9 compiler.
  // TODO(nluehr) revert to 2D array when compiler is ready.
  // This is to mimic the following, but without constructors:
  //     __shared__ storage_type<value_type> partial_sums[32 * 33];
  __shared__ __align__(
      alignof(value_type)) char partial_sums_raw[32 * 33 * sizeof(value_type)];
  value_type* partial_sums = reinterpret_cast<value_type*>(partial_sums_raw);

  row += gridDim.y * blockDim.y;

  if (col < num_cols) {
    for (; row < num_rows; row += gridDim.y * blockDim.y) {
      sum = op(sum, in[row * num_cols + col]);
    }
  }

  partial_sums[threadIdx.x * 33 + threadIdx.y] = sum;

  __syncthreads();

  if (threadIdx.y == 0 && col < num_cols) {
    value_type s = partial_sums[threadIdx.x * 33];

    // only include input values in the reduction
    // elem   block_rows
    //  -         =
    //  -         =
    //  #         #  block boundary
    //  -         =
    //  -         =
    //  #         #  block boundary
    //  -         =
    //            =
    const int numRowsThisBlock =
        min(blockDim.y, num_rows - blockIdx.y * blockDim.y);

    for (int row = 1; row < numRowsThisBlock; ++row) {
      value_type t = partial_sums[threadIdx.x * 33 + row];
      s = op(s, t);
    }

    out[col * gridDim.y + blockIdx.y] = s;
  }
}

// does multiple warp size segmented reductions in parallel
// segments cannot cross warp boundaries (mainly used for reducing the segments
// that come from the Max16Columns column reduction kernel)
template <typename T, typename outT, typename Op>
__global__ void CleanupSegments(
    T partial_sums, outT out, int num_rows, int num_cols, int segment_size,
    Op op, typename std::iterator_traits<T>::value_type initVal) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  value_type val = initVal;
  if (tid < segment_size * num_cols) val = partial_sums[tid];

  typedef cub::WarpReduce<value_type> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp_storage;

  const bool head_flag = (threadIdx.x % segment_size) == 0;
  value_type sum =
      WarpReduce(temp_storage).HeadSegmentedReduce(val, head_flag, op);

  if (head_flag && tid < segment_size * num_cols) {
    out[tid / segment_size] = sum;
  }
}

// assigns one thread to a column
template <typename T, typename outT, typename Op>
__global__ void ColumnReduceSimpleKernel(T in, outT out, int num_planes,
                                         int num_rows, int num_cols, Op op) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int elems_per_plane = num_rows * num_cols;

  const int plane = gid / num_cols;
  const int col = gid % num_cols;

  if (plane >= num_planes) return;

  if (num_rows == 1) {
    out[plane * elems_per_plane + col] = in[plane * elems_per_plane + col];
    return;
  }

  value_type sum = op(in[plane * elems_per_plane + col],
                      in[plane * elems_per_plane + num_cols + col]);
  for (int row = 2; row < num_rows; ++row) {
    sum = op(sum, in[plane * elems_per_plane + row * num_cols + col]);
  }

  out[plane * num_cols + col] = sum;
}

struct RowOffset {
  __host__ __device__ explicit RowOffset(const int& cols) : cols_(cols) {}

  __host__ __device__ int operator()(const int& x) const { return cols_ * x; }

  int cols_;
};

struct GatherOp {
  __host__ __device__ GatherOp(const int& extent_x, const int& extent_y,
                               const int& extent_z, bool kOne)
      : extent_x_(extent_x),
        extent_y_(extent_y),
        extent_z_(extent_z),
        kOne_(kOne) {
    if (kOne_)
      group_size_ = extent_y_;
    else
      group_size_ = extent_x_ * extent_z_;
  }

  __host__ __device__ int operator()(const int& ind) const {
    const int group = kOne_ ? ind / group_size_ : ind % group_size_;
    const int offset = kOne_ ? ind % group_size_ : ind / group_size_;

    const int x = group / extent_z_;
    const int z = group % extent_z_;

    return x * extent_y_ * extent_z_ + z + offset * extent_z_;
  }

  int extent_x_;
  int extent_y_;
  int extent_z_;
  bool kOne_;
  int group_size_;
};

template <typename T, typename Op, typename OUT_T, typename IN_T>
void LaunchScalarReduction(OpKernelContext* ctx, OUT_T out, IN_T in,
                           int in_size, Op op, T init,
                           const cudaStream_t& cu_stream) {
  // handle situations where low latency is important better than CUB
  if (in_size <= 4096) {
    const int num_blocks = 1;
    const int num_threads = 256;
    BlockReduceKernel<IN_T, OUT_T, num_threads>
        <<<num_blocks, num_threads, 0, cu_stream>>>(in, out, in_size, op, init);
    return;
  } else if (in_size <= 1 << 19) {
    const int num_threads = 256;
    const int num_blocks = std::min(32, Eigen::divup(in_size, num_threads));
    // it seems like tailoring this to the GPU
    // would be more effective, but all attempts
    // at making this a multiple of the number of
    // multiprocessors have lead to lower perf
    // in general
    // TODO(eriche) investigate this more

    Tensor temp_storage;
    OP_REQUIRES_OK(
        ctx,
        ctx->allocate_temp(
            DT_INT8, TensorShape({static_cast<int64>(num_blocks * sizeof(T))}),
            &temp_storage));

    BlockReduceKernel<IN_T, T*, num_threads>
        <<<num_blocks, num_threads, 0, cu_stream>>>(
            in, (T*)temp_storage.flat<int8_t>().data(), in_size, op, init);

    // take care that we only reduce blocks that had some valid elements in them
    // TODO(eriche): CUB currently has a bug in HeadSegmentedReduce that
    // requires it to be used with a full warp.  Can reduce 32 -> num_blocks
    // when this is fixed.
    CleanupSegments<<<1, 32, 0, cu_stream>>>(
        (T*)temp_storage.flat<int8_t>().data(), out, 1, 1, num_blocks, op,
        init);
    return;
  }

  size_t temp_storage_bytes = 0;
  auto reduce = [&](void* temp_storage_ptr) {
    auto success =
        cub::DeviceReduce::Reduce(temp_storage_ptr, temp_storage_bytes, in, out,
                                  in_size, op, init, cu_stream);

    OP_REQUIRES(
        ctx, success == 0,
        errors::Internal("CUB reduce error", cudaGetErrorString(success)));
  };

  reduce(nullptr);  // Get required amount of temp storage.

  Tensor temp_storage;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(
               DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
               &temp_storage));

  reduce(temp_storage.flat<int8_t>().data());  // Do reduction.
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void LaunchRowReduction(OpKernelContext* ctx, OUT_T out, IN_T in, int num_rows,
                        int num_cols, Op op, T init,
                        const cudaStream_t& cu_stream) {
  if (num_cols < 1024) {
    const int threads_per_block = 128;
    const int warps_per_block = threads_per_block / 32;
    int num_blocks = (num_rows + warps_per_block - 1) / warps_per_block;

    RowReduceKernel<<<num_blocks, threads_per_block, 0, cu_stream>>>(
        in, out, num_rows, num_cols, op, init);
    return;
  }

  // setup segment offsets with counting and transform iterator
  RowOffset row_offset_op(num_cols);
  cub::CountingInputIterator<int> counting_iter(0);
  cub::TransformInputIterator<int, RowOffset, cub::CountingInputIterator<int>>
      transform_iter(counting_iter, row_offset_op);

  size_t temp_storage_bytes = 0;
  auto reduce = [&](void* temp_storage_ptr) {
    auto success = cub::DeviceSegmentedReduce::Reduce(
        temp_storage_ptr, temp_storage_bytes, in, out, num_rows, transform_iter,
        transform_iter + 1, op, init, cu_stream);

    OP_REQUIRES(ctx, success == 0,
                errors::Internal("CUB segmented reduce error",
                                 cudaGetErrorString(success)));
  };

  reduce(nullptr);  // Get required amount of temp storage.

  Tensor temp_storage;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(
               DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
               &temp_storage));

  reduce(temp_storage.flat<int8_t>().data());  // Do reduction.
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void LaunchColumnReduction_LTE16Cols(OpKernelContext* ctx, OUT_T out, IN_T in,
                                     int extent_x, int extent_y, Op op, T init,
                                     const cudaStream_t& cu_stream) {
  int rows_per_warp = 32 / extent_y;
  dim3 block_dim(32, std::min(Eigen::divup(extent_x, rows_per_warp), 32), 1);
  dim3 grid_dim(1,
                Eigen::divup(static_cast<unsigned int>(extent_x),
                             rows_per_warp * block_dim.y),
                1);

  grid_dim.y = std::min((int)grid_dim.y, 32);

  if (grid_dim.y > 2 && grid_dim.y < 32) {
    int log2 = Log2Floor(grid_dim.y);
    grid_dim.y = 1 << log2;
  }

  if (grid_dim.y == 1) {
    ColumnReduceMax16ColumnsKernel<<<grid_dim, block_dim, 0, cu_stream>>>(
        in, out, extent_x, extent_y, op, init);
  } else {
    Tensor temp_storage;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_INT8,
                                      TensorShape({static_cast<int64>(
                                          sizeof(T) * extent_y * grid_dim.y)}),
                                      &temp_storage));
    ColumnReduceMax16ColumnsKernel<<<grid_dim, block_dim, 0, cu_stream>>>(
        in, (T*)temp_storage.flat<int8_t>().data(), extent_x, extent_y, op,
        init);

    dim3 new_grid_dim((grid_dim.y * extent_y + 31) / 32, 1, 1);
    dim3 num_threads(128, 1, 1);
    CleanupSegments<<<new_grid_dim, num_threads, 0, cu_stream>>>(
        (T*)temp_storage.flat<int8_t>().data(), out, extent_x, extent_y,
        grid_dim.y, op, init);
  }
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void LaunchColumnReduction_LTE4096Cols(OpKernelContext* ctx, OUT_T out, IN_T in,
                                       int extent_x, int extent_y, Op op,
                                       T init, const cudaStream_t& cu_stream) {
  dim3 block_dim(32, std::min(extent_x, 32), 1);
  dim3 grid_dim((extent_y + 31) / 32, 1, 1);

  if (grid_dim.x < 16) grid_dim.y = std::min((extent_x + 31) / 32, 32);

  if (grid_dim.y > 2 && grid_dim.y < 32) {
    int log2 = Log2Floor(grid_dim.y);
    grid_dim.y = 1 << log2;
  }

  if (grid_dim.y == 1) {
    ColumnReduceKernel<<<grid_dim, block_dim, 0, cu_stream>>>(
        in, out, extent_x, extent_y, op, init);
  } else {
    Tensor temp_storage;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_INT8,
                                      TensorShape({static_cast<int64>(
                                          sizeof(T) * extent_y * grid_dim.y)}),
                                      &temp_storage));

    ColumnReduceKernel<<<grid_dim, block_dim, 0, cu_stream>>>(
        in, (T*)temp_storage.flat<int8_t>().data(), extent_x, extent_y, op,
        init);

    dim3 new_grid_dim((grid_dim.y * extent_y + 31) / 32, 1, 1);
    dim3 num_threads(128, 1, 1);
    CleanupSegments<<<new_grid_dim, block_dim, 0, cu_stream>>>(
        (T*)temp_storage.flat<int8_t>().data(), out, extent_x, extent_y,
        grid_dim.y, op, init);
  }
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void LaunchColumnReduction(OpKernelContext* ctx, OUT_T out, IN_T in,
                           int extent_x, int extent_y, Op op, T init,
                           const cudaStream_t& cu_stream) {
  if (extent_y <= 16) {
    LaunchColumnReduction_LTE16Cols(ctx, out, in, extent_x, extent_y, op, init,
                                    cu_stream);
  } else if (extent_y <= 4096) {
    LaunchColumnReduction_LTE4096Cols(ctx, out, in, extent_x, extent_y, op,
                                      init, cu_stream);
  } else {
    int threads_per_block = 128;
    int num_blocks = Eigen::divup(extent_y, threads_per_block);

    ColumnReduceSimpleKernel<<<num_blocks, threads_per_block, 0, cu_stream>>>(
        in, out, 1, extent_x, extent_y, op);
  }
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void Launch3DYReduction(OpKernelContext* ctx, OUT_T out, IN_T in, int extent_x,
                        int extent_y, int extent_z, Op op, T init,
                        const cudaStream_t& cu_stream) {
  int threads_per_block = 128;
  int num_blocks =
      (extent_x * extent_z + threads_per_block - 1) / threads_per_block;

  // TODO(eriche): this won't be very good in the case of small x
  //                small z and large y.
  ColumnReduceSimpleKernel<<<num_blocks, threads_per_block, 0, cu_stream>>>(
      in, out, extent_x, extent_y, extent_z, op);
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void Launch3DXZReduction(OpKernelContext* ctx, OUT_T out, IN_T in, int extent_x,
                         int extent_y, int extent_z, Op op, T init,
                         const cudaStream_t& cu_stream) {
  // setup segment offsets with counting and transform iterator
  RowOffset row_offset_op(extent_x * extent_z);
  cub::CountingInputIterator<int> counting_iter(0);
  cub::TransformInputIterator<int, RowOffset, cub::CountingInputIterator<int>>
      transform_iter(counting_iter, row_offset_op);

  GatherOp gather_op(extent_x, extent_y, extent_z, false);
  typedef cub::TransformInputIterator<int, GatherOp,
                                      cub::CountingInputIterator<int>>
      gatherIterType;
  gatherIterType gather_iter(counting_iter, gather_op);

  PermutationInputIterator<T, IN_T, gatherIterType> permute_iter(in,
                                                                 gather_iter);

  std::size_t temp_storage_bytes = 0;
  auto reduce = [&](void* temp_storage_ptr) {
    auto success = cub::DeviceSegmentedReduce::Reduce(
        temp_storage_ptr, temp_storage_bytes, permute_iter, out, extent_y,
        transform_iter, transform_iter + 1, op, init, cu_stream);

    OP_REQUIRES(ctx, success == 0,
                errors::Internal("CUB segmented reduce error",
                                 cudaGetErrorString(success)));
  };

  reduce(nullptr);  // Get required amount of temp storage.

  Tensor temp_storage;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(
               DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
               &temp_storage));

  reduce(temp_storage.flat<int8_t>().data());  // Do reduction.
}

namespace reduction_op_helper {

template <typename T, typename Op>
struct IsSum {
  constexpr static bool value =
      (std::is_same<Op, cub::Sum>::value ||
       std::is_same<Op, Eigen::internal::SumReducer<T>>::value ||
       std::is_same<Op, Sum<T>>::value);
};

template <typename T, typename Op>
struct IsMax {
  constexpr static bool value =
      (std::is_same<Op, cub::Max>::value ||
       std::is_same<Op, Eigen::internal::MaxReducer<T>>::value);
};

template <typename T, typename Op>
struct IsMin {
  constexpr static bool value =
      (std::is_same<Op, cub::Min>::value ||
       std::is_same<Op, Eigen::internal::MinReducer<T>>::value);
};

template <typename T, typename Op>
struct IsProd {
  constexpr static bool value =
      (std::is_same<Op, Prod<T>>::value ||
       std::is_same<Op, Eigen::internal::ProdReducer<T>>::value);
};

template <typename T, typename Op>
struct IdentityValue {
  static_assert(IsSum<T, Op>::value || IsMax<T, Op>::value ||
                    IsMin<T, Op>::value || IsProd<T, Op>::value ||
                    std::is_same<Op, And>::value || std::is_same<Op, Or>::value,
                "IdentityValue not yet defined for this type");

  template <typename U = T, typename OpCopy = Op>
  U operator()(
      typename std::enable_if<IsSum<U, OpCopy>::value, U>::type t = U(0)) {
    return t;
  }

  template <typename U = T, typename OpCopy = Op>
  U operator()(typename std::enable_if<IsMax<U, OpCopy>::value, U>::type t =
                   Eigen::NumTraits<U>::lowest()) {
    return t;
  }

  template <typename U = T, typename OpCopy = Op>
  U operator()(typename std::enable_if<IsMin<U, OpCopy>::value, U>::type t =
                   Eigen::NumTraits<U>::highest()) {
    return t;
  }

  template <typename U = T, typename OpCopy = Op>
  U operator()(
      typename std::enable_if<IsProd<U, OpCopy>::value, U>::type t = U(1)) {
    return t;
  }

  template <typename U = T, typename OpCopy = Op>
  U operator()(typename std::enable_if<std::is_same<OpCopy, And>::value,
                                       bool>::type t = true) {
    return t;
  }

  template <typename U = T, typename OpCopy = Op>
  U operator()(typename std::enable_if<std::is_same<OpCopy, Or>::value,
                                       bool>::type t = false) {
    return t;
  }
};

}  // namespace reduction_op_helper

template <typename T, typename Op, typename OUT_T, typename IN_T,
          typename ReductionAxes>
void ReduceImpl(OpKernelContext* ctx, OUT_T out, IN_T in, int in_rank,
                int in_dim0, int in_dim1, int in_dim2, int out_rank,
                const ReductionAxes& reduction_axes, Op op) {
  T init = reduction_op_helper::IdentityValue<T, Op>()();
  const cudaStream_t& cu_stream = GetCudaStream(ctx);
  if (out_rank == 0) {
    const int in_size = in_dim0 * in_dim1 * in_dim2;
    LaunchScalarReduction(ctx, out, in, in_size, op, init, cu_stream);
  } else if (in_rank == 2 && out_rank == 1 &&
             reduction_axes[0] == 1) {  // row reduction
    LaunchRowReduction(ctx, out, in, in_dim0, in_dim1, op, init, cu_stream);
  } else if (in_rank == 2 && out_rank == 1 &&
             reduction_axes[0] == 0) {  // column reduction
    LaunchColumnReduction(ctx, out, in, in_dim0, in_dim1, op, init, cu_stream);
  } else if (in_rank == 3 && out_rank == 2 && reduction_axes[0] == 1) {
    Launch3DYReduction(ctx, out, in, in_dim0, in_dim1, in_dim2, op, init,
                       cu_stream);
  } else if (in_rank == 3 && out_rank == 1 && reduction_axes[0] == 0 &&
             reduction_axes[1] == 2) {
    Launch3DXZReduction(ctx, out, in, in_dim0, in_dim1, in_dim2, op, init,
                        cu_stream);
  } else {
    std::stringstream ss;
    ss << "Invalid reduction requested: in_rank, out_rank, axes " << in_rank
       << " " << out_rank;
    if (out_rank == 1) ss << " " << reduction_axes[0];
    if (out_rank == 2) ss << " " << reduction_axes[1];
    LOG(FATAL) << ss.str();
  }
}

template <typename Reducer>
struct ReduceFunctor<GPUDevice, Reducer> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer);
};

template <typename T>
struct ReduceFunctor<GPUDevice, Eigen::internal::SumReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::SumReducer<T>& reducer) {
    ReduceImpl<T, Sum<T>, T*, T*, ReductionAxes>(
        ctx, (T*)out.data(), (T*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        Sum<T>());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::SumReducer<T>& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <typename T>
struct ReduceFunctor<GPUDevice, functor::MeanReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const functor::MeanReducer<T>& reducer) {
    int divisor = 1;
    if (out.rank() == 0)
      divisor = in.size();
    else if (out.rank() == 1 && in.rank() == 2 && reduction_axes[0] == 0)
      divisor = in.dimension(0);
    else if (out.rank() == 1 && in.rank() == 2 && reduction_axes[0] == 1)
      divisor = in.dimension(1);
    else if (out.rank() == 1 && in.rank() == 3 && reduction_axes[0] == 0 &&
             reduction_axes[1] == 2)
      divisor = in.dimension(0) * in.dimension(2);
    else if (out.rank() == 2 && in.rank() == 3 && reduction_axes[0] == 1)
      divisor = in.dimension(1);

    DividesBy<T> div_op(static_cast<T>(divisor));
    TransformOutputIterator<T, T, DividesBy<T>> itr((T*)out.data(), div_op);
    ReduceImpl<T, Sum<T>, TransformOutputIterator<T, T, DividesBy<T>>, T*,
               ReductionAxes>(ctx, itr, (T*)in.data(), in.rank(),
                              in.dimension(0),
                              in.rank() >= 2 ? in.dimension(1) : 1,
                              in.rank() >= 3 ? in.dimension(2) : 1, out.rank(),
                              reduction_axes, Sum<T>());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const functor::MeanReducer<T>& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <>
struct ReduceFunctor<GPUDevice, functor::MeanReducer<Eigen::half>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const functor::MeanReducer<Eigen::half>& reducer) {
    float divisor = 1.f;
    if (out.rank() == 0)
      divisor = in.size();
    else if (out.rank() == 1 && in.rank() == 2 && reduction_axes[0] == 0)
      divisor = in.dimension(0);
    else if (out.rank() == 1 && in.rank() == 2 && reduction_axes[0] == 1)
      divisor = in.dimension(1);
    else if (out.rank() == 1 && in.rank() == 3 && reduction_axes[0] == 0 &&
             reduction_axes[1] == 2)
      divisor = in.dimension(0) * in.dimension(2);
    else if (out.rank() == 2 && in.rank() == 3 && reduction_axes[0] == 1)
      divisor = in.dimension(1);
    DividesBy<float, Eigen::half> div_op(divisor);

    typedef cub::TransformInputIterator<float, HalfToFloat, Eigen::half*>
        inputIterType;
    inputIterType input_itr((Eigen::half*)in.data(), HalfToFloat());

    typedef TransformOutputIterator<Eigen::half, float,
                                    DividesBy<float, Eigen::half>>
        outputIterType;
    outputIterType itr((Eigen::half*)out.data(), div_op);

    ReduceImpl<float, cub::Sum, outputIterType, inputIterType, ReductionAxes>(
        ctx, itr, input_itr, in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        cub::Sum());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const functor::MeanReducer<Eigen::half>& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <typename T>
struct ReduceFunctor<GPUDevice, Eigen::internal::MaxReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::MaxReducer<T>& reducer) {
    ReduceImpl<T, cub::Max, T*, T*, ReductionAxes>(
        ctx, (T*)out.data(), (T*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        cub::Max());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::MaxReducer<T>& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <typename T>
struct ReduceFunctor<GPUDevice, Eigen::internal::MinReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::MinReducer<T>& reducer) {
    ReduceImpl<T, cub::Min, T*, T*, ReductionAxes>(
        ctx, (T*)out.data(), (T*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        cub::Min());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::MinReducer<T>& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <typename T>
struct ReduceFunctor<GPUDevice, Eigen::internal::ProdReducer<T>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::ProdReducer<T>& reducer) {
    ReduceImpl<T, Prod<T>, T*, T*, ReductionAxes>(
        ctx, (T*)out.data(), (T*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        Prod<T>());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::ProdReducer<T>& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <>
struct ReduceFunctor<GPUDevice, Eigen::internal::AndReducer> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::AndReducer& reducer) {
    ReduceImpl<bool, And, bool*, bool*, ReductionAxes>(
        ctx, (bool*)out.data(), (bool*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        And());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::AndReducer& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <>
struct ReduceFunctor<GPUDevice, Eigen::internal::OrReducer> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::OrReducer& reducer) {
    ReduceImpl<bool, Or, bool*, bool*, ReductionAxes>(
        ctx, (bool*)out.data(), (bool*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes, Or());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::OrReducer& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_REDUCTION_GPU_KERNELS_CU_H_
