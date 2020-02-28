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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <sstream>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "gpu_prim.h"
#include "tensorflow/core/kernels/reduction_ops.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/permutation_input_iterator.h"
#include "tensorflow/core/util/transform_output_iterator.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct SqrtOfReal {
  __host__ __device__ T operator()(const T& a) const {
    return T(Eigen::numext::sqrt(Eigen::numext::real(a)));
  }
};

#if TENSORFLOW_USE_ROCM
template <>
struct SqrtOfReal<hipFloatComplex> {
  __host__ __device__ hipFloatComplex operator()(const hipFloatComplex& a) const {
    return hipFloatComplex(sqrt(float(a.x)), 0.0f);
  }
};

template <>
struct SqrtOfReal<hipDoubleComplex> {
  __host__ __device__ hipDoubleComplex operator()(const hipDoubleComplex& a) const {
    return hipDoubleComplex(sqrt(double(a.x)), 0.0);
  }
};
#endif


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

template <typename T>
struct Square {
  __host__ __device__ T operator()(const T& a) const {
    return Prod<T>()(a, Eigen::numext::conj(a));
  }
};

#if TENSORFLOW_USE_ROCM
template <>
struct Square<hipFloatComplex> {
  __host__ __device__ hipFloatComplex operator()(const hipFloatComplex& a) const {
    return hipFloatComplex(a.x*a.x+a.y*a.y, 0.0f);
  }
};

template <>
struct Square<hipDoubleComplex> {
  __host__ __device__ hipDoubleComplex operator()(const hipDoubleComplex& a) const {
    return hipDoubleComplex(a.x*a.x+a.y*a.y, 0.0);
  }
};
#endif

template <typename T> struct inner_float {
  typedef float IT;
};

template <> struct inner_float<double> {
  typedef double IT;
};

template <> struct inner_float<std::complex<double> > {
  typedef double IT;
};

#if TENSORFLOW_USE_ROCM
template <> struct inner_float<hipDoubleComplex> {
  typedef double IT;
};
#endif

//divisor_type was previously same as T, but it's wasteful since the constructor
//argument is always integer (and it introduces troubles with complex division)
template <typename T, typename OUT_T = T>
struct DividesBy {
  // Do the division in float unless the type is double or double complex
  typedef typename inner_float<T>::IT divisor_type;
  divisor_type divisor;

  __host__ __device__ explicit DividesBy(uint64 divisor) 
    : divisor(divisor_type(1.0)/divisor) {}

  __host__ __device__ OUT_T operator()(const T& x) const { return x*divisor; }
};

#if GOOGLE_CUDA
// needed to work around a compiler bug in nvcc - it doesn't seem to like
// the overloaded ops for std::complex
// (TODO: check if this is still needed)
template <>
struct DividesBy<std::complex<float>> {
  float divisor;

  __host__ __device__ explicit DividesBy(uint64 divisor)
      : divisor(1.0f/divisor) {}

  // implements
  __host__ __device__ std::complex<float> operator()(
      const std::complex<float>& x) const {
    return std::complex<float>(x.real()*divisor, x.imag()*divisor);
  }
};

template <>
struct DividesBy<std::complex<double>> {
  double divisor;

  __host__ __device__ explicit DividesBy(uint64 divisor)
      : divisor(1./divisor) {}

  // implements
  __host__ __device__ std::complex<double> operator()(
      const std::complex<double>& x) const {
    return std::complex<double>(x.real()*divisor, x.imag()*divisor);
  }
};
#endif

template <>
struct DividesBy<float, Eigen::half> {
  float divisor;

  __host__ __device__ explicit DividesBy(uint64 divisor) : divisor(1.0f/divisor) {}

  __host__ __device__ Eigen::half operator()(const float& x) const {
    return Eigen::half(x * divisor);
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
template <typename T, typename OUT_T, int num_threads, typename Op>
__global__ void BlockReduceKernel(
    T in, OUT_T out, int num_elems, Op op,
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

  typedef gpuprim::BlockReduce<value_type, num_threads> BlockReduce;

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
template <typename T, typename OUT_T, typename Op, int WARPSIZE>
__global__ void RowReduceKernel(
    T in, OUT_T out, int num_rows, int num_cols, Op op,
    typename std::iterator_traits<T>::value_type initVal) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  // Defensive index computation to avoid integer overflow.
  assert(blockDim.x % WARPSIZE == 0);
  int warps_per_block = blockDim.x / WARPSIZE;
  int warp_index = threadIdx.x / WARPSIZE;
  const int row = blockIdx.x * warps_per_block + warp_index;
  const int lane = threadIdx.x % WARPSIZE;

  if (num_cols == 1) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < num_rows) out[gid] = in[gid];
    return;
  }

  value_type sum = initVal;
  int col = lane;

  if (row < num_rows && col < num_cols) {
    sum = in[row * num_cols + col];
    col += WARPSIZE;
    for (; col < num_cols; col += WARPSIZE) {
      sum = op(sum, in[row * num_cols + col]);
    }
  }

  typedef gpuprim::WarpReduce<value_type> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp_storage;

  sum =
      WarpReduce(temp_storage).Reduce(sum, op, min(num_cols, WARPSIZE));

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
template <typename T, typename OUT_T, typename Op, int WARPSIZE>
__global__ void ColumnReduceMax16ColumnsKernel(
    T in, OUT_T out, int num_rows, int num_cols, Op op,
    typename std::iterator_traits<T>::value_type initVal) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  int rows_per_warp = WARPSIZE / num_cols;

  const int lane = threadIdx.x % WARPSIZE;
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
    //   __shared__ storage_type<value_type> partial_sums[TF_RED_WARPSIZE *
    //   (TF_RED_WARPSIZE+1)];
#if GOOGLE_CUDA || TENSORFLOW_COMPILER_IS_HIP_CLANG
  __shared__ __align__(alignof(value_type)) char
      partial_sums_raw[WARPSIZE * (WARPSIZE + 1) *
                       sizeof(value_type)];
  value_type* partial_sums = reinterpret_cast<value_type*>(partial_sums_raw);
#elif TENSORFLOW_USE_ROCM
  __shared__ storage_type<value_type>
      partial_sums[WARPSIZE * (WARPSIZE + 1)];
#endif

  row += rows_per_warp * gridDim.y * blockDim.y;
  for (; row < num_rows; row += rows_per_warp * gridDim.y * blockDim.y) {
    int global_pos = row * num_cols + col;
    if (global_pos < (num_rows * num_cols))
      sum = op(sum, in[row * num_cols + col]);
  }

  const int rows_in_this_warp = min(rows_per_warp, num_rows - start_row_warp);
  // not the most efficient way to do this sum
  uint64_t warp_mask = (WARPSIZE==32) ? 0xffffffff : 0xffffffffffffffffll;
  for (int i = 1; i < rows_in_this_warp; ++i) {
    value_type tmp = gpuprim::ShuffleIndex<WARPSIZE, value_type>(
        sum, static_cast<int>(threadIdx.x + i * num_cols), warp_mask);
    if (lane < num_cols) sum = op(sum, tmp);
  }

  if (lane < num_cols)
    partial_sums[lane * (WARPSIZE + 1) + threadIdx.y] = sum;

  __syncthreads();

  if (threadIdx.y == 0 && threadIdx.x < num_cols) {
    value_type s = partial_sums[threadIdx.x * (WARPSIZE + 1)];

    if (blockDim.y > 1) {
      for (int row = 1; row < blockDim.y; ++row) {
        value_type t = partial_sums[threadIdx.x * (WARPSIZE + 1) + row];
        s = op(s, t);
      }
    }

    out[col * gridDim.y + blockIdx.y] = s;
  }
}

// Maps each block to a column range TF_RED_WARPSIZE wide
template <typename T, typename OUT_T, typename Op, int WARPSIZE>
__global__ void ColumnReduceKernel(
    T in, OUT_T out, int num_rows, int num_cols, Op op,
    typename std::iterator_traits<T>::value_type initVal) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * WARPSIZE + threadIdx.x;

  value_type sum = initVal;
  if (row < num_rows && col < num_cols) sum = in[row * num_cols + col];

    // 1D array necessary due to bug in CUDA 9 compiler.
    // TODO(nluehr) revert to 2D array when compiler is ready.
    // This is to mimic the following, but without constructors:
    //     __shared__ storage_type<value_type> partial_sums[WARPSIZE *
    //     (WARPSIZE + 1)];
#if GOOGLE_CUDA || TENSORFLOW_COMPILER_IS_HIP_CLANG
  __shared__ __align__(alignof(value_type)) char
      partial_sums_raw[WARPSIZE * (WARPSIZE + 1) *
                       sizeof(value_type)];
  value_type* partial_sums = reinterpret_cast<value_type*>(partial_sums_raw);
#elif TENSORFLOW_USE_ROCM
  __shared__ storage_type<value_type>
      partial_sums[WARPSIZE * (WARPSIZE + 1)];
#endif

  row += gridDim.y * blockDim.y;

  if (col < num_cols) {
    for (; row < num_rows; row += gridDim.y * blockDim.y) {
      sum = op(sum, in[row * num_cols + col]);
    }
  }

  partial_sums[threadIdx.x * (WARPSIZE + 1) + threadIdx.y] = sum;

  __syncthreads();

  if (threadIdx.y == 0 && col < num_cols) {
    value_type s = partial_sums[threadIdx.x * (WARPSIZE + 1)];

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
      value_type t = partial_sums[threadIdx.x * (WARPSIZE + 1) + row];
      s = op(s, t);
    }

    out[col * gridDim.y + blockIdx.y] = s;
  }
}

// does multiple warp size segmented reductions in parallel
// segments cannot cross warp boundaries (mainly used for reducing the segments
// that come from the Max16Columns column reduction kernel)
template <typename T, typename OUT_T, typename Op>
__global__ void CleanupSegments(
    T partial_sums, OUT_T out, int num_rows, int num_cols, int segment_size,
    Op op, typename std::iterator_traits<T>::value_type initVal) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  value_type val = initVal;
  if (tid < segment_size * num_cols) val = partial_sums[tid];

  typedef gpuprim::WarpReduce<value_type> WarpReduce;

  __shared__ typename WarpReduce::TempStorage temp_storage;

  const bool head_flag = (threadIdx.x % segment_size) == 0;
  value_type sum =
      WarpReduce(temp_storage).HeadSegmentedReduce(val, head_flag, op);

  if (head_flag && tid < segment_size * num_cols) {
    out[tid / segment_size] = sum;
  }
}

// assigns one thread to a column
template <typename T, typename OUT_T, typename Op>
__global__ void ColumnReduceSimpleKernel(T in, OUT_T out, int num_planes,
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

namespace {
constexpr int kUnroll = 8;
}

template <typename T, typename IN_T, typename Op>
__device__ __inline__ T ComputeSum(IN_T in_, const int plane,
                                   const int num_out_rows, int num_rows,
                                   int num_cols, const int col, Op op) {
  const int out_rows = num_rows / (2 * kUnroll);
  const int num_rem_rows = num_rows % (2 * kUnroll);
  const int elems_per_plane = num_rows * num_cols;
  T reg[2 * kUnroll];
  T sum;
  int offset = 0;
  if (out_rows != 0) {
    for (int i = 0; i < 2 * kUnroll; i++) {
      reg[i] =
          in_[plane * elems_per_plane + i * (num_out_rows * num_cols) + col];
    }
    sum = reg[0];
    for (int i = 1; i < 2 * kUnroll; i++) {
      sum = op(sum, reg[i]);
    }
    offset = 2 * kUnroll * (num_out_rows * num_cols);
  }

  if (col < num_cols && num_rem_rows > 0) {
    reg[0] = in_[plane * elems_per_plane + offset + 0 * num_cols + col];
    if (out_rows != 0) {
      sum = op(sum, reg[0]);
    } else {
      sum = reg[0];
    }
    for (int i = 1; i < num_rem_rows; i++) {
      reg[0] = in_[plane * elems_per_plane + offset + i * num_cols + col];
      sum = op(sum, reg[0]);
    }
  }
  return sum;
}

template <typename IN_T, typename Op>
__global__ void ColumnReduceInToTempKernel(void* __restrict__ temp,
                                           int temp_in_offset,
                                           int temp_out_offset, IN_T in,
                                           int num_planes, int num_rows,
                                           int num_cols, Op op) {
  typedef typename std::iterator_traits<IN_T>::value_type value_type;

  value_type* t = (value_type*)temp;
  value_type* out_ = t + temp_out_offset;

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_out_rows = max(1, num_rows / (2 * kUnroll));
  const int plane = gid / (num_out_rows * num_cols);
  const int col = gid % (num_out_rows * num_cols);

  if (plane >= num_planes) return;

  value_type sum;
  if (temp_in_offset == -1) {
    auto in_ = in;
    sum = ComputeSum<value_type, IN_T, Op>(in_, plane, num_out_rows, num_rows,
                                           num_cols, col, op);
  } else {
    auto in_ = t + temp_in_offset;
    sum = ComputeSum<value_type, value_type*, Op>(in_, plane, num_out_rows,
                                                  num_rows, num_cols, col, op);
  }
  out_[plane * num_out_rows * num_cols + col] = sum;
}

template <typename T, typename OUT_T, typename Op>
__global__ void ColumnReduceTempToOutKernel(void* __restrict__ temp,
                                            int temp_in_offset, T in, OUT_T out,
                                            int num_planes, int num_rows,
                                            int num_cols, Op op) {
  typedef typename std::iterator_traits<T>::value_type value_type;
  value_type* t = (value_type*)temp;
  const int tid = threadIdx.x;
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  int elems_per_plane = num_rows * num_cols;

  if (num_rows == 1) {
    if (gid >= num_planes * num_cols) return;
    if (temp_in_offset == -1) {
      auto in_ = in;
      out[gid] = in_[gid];
    } else {
      auto in_ = t + temp_in_offset;
      out[gid] = in_[gid];
    }
    return;
  }

  const int planes_per_block = 1;
  const int plane = blockIdx.x * planes_per_block + tid / elems_per_plane;
  // A thread block contains one or multiple plane(s),
  // i.e. num_rows * num_cols <= blockDim.x
  const int col = tid % elems_per_plane;
  const int local_plane = plane % planes_per_block;

  if (tid >= planes_per_block * elems_per_plane || plane >= num_planes) return;

  GPU_DYNAMIC_SHARED_MEM_DECL(8, char, ss);
  value_type* const smem = reinterpret_cast<value_type*>(ss);

  if (temp_in_offset == -1) {
    auto in_ = in;
    smem[local_plane * elems_per_plane + col] =
        in_[plane * elems_per_plane + col];
  } else {
    auto in_ = t + temp_in_offset;
    smem[local_plane * elems_per_plane + col] =
        in_[plane * elems_per_plane + col];
  }
  __syncthreads();

  int num_in_rows = num_rows;
  int num_out_rows;
  int num_rem_rows;

  int in_offset = 0;
  int out_offset = blockDim.x;

  int in_elems_per_plane = elems_per_plane;
  int out_elems_per_plane;

  while (num_in_rows > 1) {
    num_out_rows = num_in_rows / 2;
    num_rem_rows = num_in_rows % 2;
    out_elems_per_plane = num_out_rows * num_cols;

    if (col < out_elems_per_plane) {
      value_type sum;
      sum = op(smem[in_offset + local_plane * in_elems_per_plane + col],
               smem[in_offset + local_plane * in_elems_per_plane +
                    out_elems_per_plane + col]);
      if (num_rem_rows == 1 && col < num_cols) {
        sum = op(sum, smem[in_offset + local_plane * in_elems_per_plane +
                           2 * out_elems_per_plane + col]);
      }
      smem[out_offset + local_plane * out_elems_per_plane + col] = sum;
    }

    num_in_rows = num_out_rows;
    in_elems_per_plane = out_elems_per_plane;
    int t_offset = in_offset;
    in_offset = out_offset;
    out_offset = t_offset;
    __syncthreads();
  }

  if (col < num_cols) {
    out[plane * num_cols + col] =
        smem[in_offset + local_plane * out_elems_per_plane + col];
  }
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
                           const gpuStream_t& cu_stream) {
  // handle situations where low latency is important better than CUB
  if (in_size <= 4096) {
    const int num_blocks = 1;
    const int num_threads = 256;
    TF_CHECK_OK(GpuLaunchKernel(BlockReduceKernel<IN_T, OUT_T, num_threads, Op>,
                                num_blocks, num_threads, 0, cu_stream, in, out,
                                in_size, op, init));
    return;
  } else if (in_size <= 1 << 18) {
    const int num_threads = 256;
    const int num_blocks =
        std::min(TF_RED_WARPSIZE, Eigen::divup(in_size, num_threads));
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

    TF_CHECK_OK(GpuLaunchKernel(BlockReduceKernel<IN_T, T*, num_threads, Op>,
                                num_blocks, num_threads, 0, cu_stream, in,
                                (T*)temp_storage.flat<int8_t>().data(), in_size,
                                op, init));

    // take care that we only reduce blocks that had some valid elements in them
    // TODO(eriche): CUB currently has a bug in HeadSegmentedReduce that
    // requires it to be used with a full warp.  Can reduce TF_RED_WARPSIZE ->
    // num_blocks when this is fixed.
    TF_CHECK_OK(GpuLaunchKernel(CleanupSegments<T*, OUT_T, Op>, 1,
                                TF_RED_WARPSIZE, 0, cu_stream,
                                (T*)temp_storage.flat<int8_t>().data(), out, 1,
                                1, num_blocks, op, init));
    return;
  }
  size_t temp_storage_bytes = 0;
  auto reduce = [&](void* temp_storage_ptr) {
    auto success =
        gpuprim::DeviceReduce::Reduce(temp_storage_ptr, temp_storage_bytes, in,
                                      out, in_size, op, init, cu_stream);

    OP_REQUIRES(
        ctx, success == 0,
        errors::Internal("CUB reduce error ", GpuGetErrorString(success)));
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
                        const gpuStream_t& cu_stream) {
  if (num_cols < 1024) {
    const int threads_per_block = 128;
    const int warps_per_block = threads_per_block / TF_RED_WARPSIZE;
    int num_blocks = (num_rows + warps_per_block - 1) / warps_per_block;

    TF_CHECK_OK(GpuLaunchKernel(RowReduceKernel<IN_T, OUT_T, Op, TF_RED_WARPSIZE>, num_blocks,
                                threads_per_block, 0, cu_stream, in, out,
                                num_rows, num_cols, op, init));
    return;
  }
  // setup segment offsets with counting and transform iterator
  RowOffset row_offset_op(num_cols);
  gpuprim::CountingInputIterator<int> counting_iter(0);
  gpuprim::TransformInputIterator<int, RowOffset,
                                  gpuprim::CountingInputIterator<int>>
      transform_iter(counting_iter, row_offset_op);

  size_t temp_storage_bytes = 0;
  auto reduce = [&](void* temp_storage_ptr) {
    auto success = gpuprim::DeviceSegmentedReduce::Reduce(
        temp_storage_ptr, temp_storage_bytes, in, out, num_rows, transform_iter,
        transform_iter + 1, op, init, cu_stream);

    OP_REQUIRES(ctx, success == 0,
                errors::Internal("CUB segmented reduce error",
                                 GpuGetErrorString(success)));
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
                                     const gpuStream_t& cu_stream) {
#if TENSORFLOW_USE_ROCM  
  constexpr int WARPSIZE = std::is_same<T, hipDoubleComplex>::value 
      ? (TF_RED_WARPSIZE/2) 
      : TF_RED_WARPSIZE;
#else
  constexpr int WARPSIZE = TF_RED_WARPSIZE;
#endif
  int rows_per_warp = WARPSIZE / extent_y;
  dim3 block_dim(
      WARPSIZE,
      std::min(Eigen::divup(extent_x, rows_per_warp), (1024 / WARPSIZE)),
      1);
  dim3 grid_dim(1,
                Eigen::divup(static_cast<unsigned int>(extent_x),
                             rows_per_warp * block_dim.y),
                1);

  grid_dim.y = std::min((int)grid_dim.y, WARPSIZE);

  if (grid_dim.y > 2 && grid_dim.y < WARPSIZE) {
    int log2 = Log2Floor(grid_dim.y);
    grid_dim.y = 1 << log2;
  }

  if (grid_dim.y == 1) {
    TF_CHECK_OK(GpuLaunchKernel(ColumnReduceMax16ColumnsKernel<IN_T, OUT_T, Op, WARPSIZE>,
                                grid_dim, block_dim, 0, cu_stream, in, out,
                                extent_x, extent_y, op, init));
  } else {
    Tensor temp_storage;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_INT8,
                                      TensorShape({static_cast<int64>(
                                          sizeof(T) * extent_y * grid_dim.y)}),
                                      &temp_storage));
    TF_CHECK_OK(GpuLaunchKernel(ColumnReduceMax16ColumnsKernel<IN_T, T*, Op, WARPSIZE>,
                                grid_dim, block_dim, 0, cu_stream, in,
                                (T*)temp_storage.flat<int8_t>().data(),
                                extent_x, extent_y, op, init));

    dim3 new_grid_dim(
        (grid_dim.y * extent_y + (WARPSIZE - 1)) / WARPSIZE, 1,
        1);
    dim3 num_threads(128, 1, 1);
    TF_CHECK_OK(GpuLaunchKernel(CleanupSegments<T*, OUT_T, Op>, new_grid_dim,
                                num_threads, 0, cu_stream,
                                (T*)temp_storage.flat<int8_t>().data(), out,
                                extent_x, extent_y, grid_dim.y, op, init));
  }
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void LaunchColumnReduction_LTE4096Cols(OpKernelContext* ctx, OUT_T out, IN_T in,
                                       int extent_x, int extent_y, Op op,
                                       T init, const gpuStream_t& cu_stream) {
#if TENSORFLOW_USE_ROCM  
  // On ROCm, TF_RED_WARPSIZE is 64 and the default value would require
  // 66 kB of shared memory with double complex - more than actually
  // available in the GPU.
  constexpr int WARPSIZE = std::is_same<T, hipDoubleComplex>::value 
    ? (TF_RED_WARPSIZE/2) 
    : TF_RED_WARPSIZE;
#else
  constexpr int WARPSIZE = TF_RED_WARPSIZE;
#endif
  dim3 block_dim(WARPSIZE, std::min(extent_x, (1024 / WARPSIZE)),
                 1);
  dim3 grid_dim((extent_y + (WARPSIZE - 1)) / WARPSIZE, 1, 1);

  if (grid_dim.x < 16)
    grid_dim.y = std::min((extent_x + (WARPSIZE - 1)) / WARPSIZE,
                          WARPSIZE);

  if (grid_dim.y > 2 && grid_dim.y < WARPSIZE) {
    int log2 = Log2Floor(grid_dim.y);
    grid_dim.y = 1 << log2;
  }

  if (grid_dim.y == 1) {
    TF_CHECK_OK(GpuLaunchKernel(ColumnReduceKernel<IN_T, OUT_T, Op, WARPSIZE>, grid_dim,
                                block_dim, 0, cu_stream, in, out, extent_x,
                                extent_y, op, init));
  } else {
    Tensor temp_storage;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_INT8,
                                      TensorShape({static_cast<int64>(
                                          sizeof(T) * extent_y * grid_dim.y)}),
                                      &temp_storage));

    TF_CHECK_OK(GpuLaunchKernel(
        ColumnReduceKernel<IN_T, T*, Op, WARPSIZE>, grid_dim, block_dim, 0, cu_stream, in,
        (T*)temp_storage.flat<int8_t>().data(), extent_x, extent_y, op, init));

    dim3 new_grid_dim(
        (grid_dim.y * extent_y + (WARPSIZE - 1)) / WARPSIZE, 1,
        1);
    TF_CHECK_OK(GpuLaunchKernel(CleanupSegments<T*, OUT_T, Op>, new_grid_dim,
                                block_dim, 0, cu_stream,
                                (T*)temp_storage.flat<int8_t>().data(), out,
                                extent_x, extent_y, grid_dim.y, op, init));
  }
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void LaunchColumnReduction(OpKernelContext* ctx, OUT_T out, IN_T in,
                           int extent_x, int extent_y, Op op, T init,
                           const gpuStream_t& cu_stream) {
  if (extent_y <= 16) {
    LaunchColumnReduction_LTE16Cols(ctx, out, in, extent_x, extent_y, op, init,
                                    cu_stream);
  } else if (extent_y <= 4096) {
    LaunchColumnReduction_LTE4096Cols(ctx, out, in, extent_x, extent_y, op,
                                      init, cu_stream);
  } else {
    int threads_per_block = 128;
    int num_blocks = Eigen::divup(extent_y, threads_per_block);

    TF_CHECK_OK(GpuLaunchKernel(ColumnReduceSimpleKernel<IN_T, OUT_T, Op>,
                                num_blocks, threads_per_block, 0, cu_stream, in,
                                out, 1, extent_x, extent_y, op));
  }
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void Launch3DYReductionSimple(OpKernelContext* ctx, OUT_T out, IN_T in,
                              int extent_x, int extent_y, int extent_z, Op op,
                              T init, const gpuStream_t& cu_stream) {
  int threads_per_block = 128;
  int num_blocks =
      (extent_x * extent_z + threads_per_block - 1) / threads_per_block;

  // TODO(eriche): this won't be very good in the case of small x
  //                small z and large y.
  TF_CHECK_OK(GpuLaunchKernel(ColumnReduceSimpleKernel<IN_T, OUT_T, Op>,
                              num_blocks, threads_per_block, 0, cu_stream, in,
                              out, extent_x, extent_y, extent_z, op));
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void Launch3DYReduction(OpKernelContext* ctx, OUT_T out, IN_T in, int extent_x,
                        int extent_y, int extent_z, Op op, T init,
                        const gpuStream_t& cu_stream) {
  int threads_per_block = 128;

  int n_group_in = extent_y;
  int n_size = extent_z;

  // Calculate and allocate temporary space
  std::size_t temp_storage_bytes = 0;
  // A plane's size is n_group_in * n_size. We make sure no single plane crosses
  // more than one thread block, meaning a thread block will handle one whole
  // plane or multiple planes in the second stage. Also, It may handle a partial
  // plane when n_size is too large and the while-loop will stop at
  // n_group_in = 1, where we directly copy the temp to output in the next
  // stage.
  while (n_group_in >= 2 && n_group_in * n_size > threads_per_block) {
    int n_group_out = std::max(1, n_group_in / (2 * kUnroll));
    temp_storage_bytes += n_group_out * n_size;
    n_group_in = n_group_out;
  }
  temp_storage_bytes *= extent_x * sizeof(T);
  Tensor temp_storage;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(
               DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
               &temp_storage));

  // Reduction
  n_group_in = extent_y;
  int temp_in_offset = -1;
  int temp_out_offset = 0;
  int num_blocks;
  while (n_group_in >= 2 && n_group_in * n_size > threads_per_block) {
    int n_group_out = std::max(1, n_group_in / (2 * kUnroll));
    num_blocks =
        Eigen::divup(extent_x * n_group_out * n_size, threads_per_block);
    TF_CHECK_OK(GpuLaunchKernel(
        ColumnReduceInToTempKernel<IN_T, Op>, num_blocks, threads_per_block, 0,
        cu_stream, (void*)(temp_storage.flat<int8_t>().data()), temp_in_offset,
        temp_out_offset, in, extent_x, n_group_in, extent_z, op));

    n_group_in = n_group_out;
    temp_in_offset = temp_out_offset;
    temp_out_offset = temp_in_offset + extent_x * n_group_out * n_size;
  }

  if (n_group_in * n_size <= threads_per_block) {
    num_blocks = extent_x;
  } else {
    DCHECK_EQ(1, n_group_in);
    num_blocks = Eigen::divup(extent_x * n_size, threads_per_block);
  }

  TF_CHECK_OK(GpuLaunchKernel(
      ColumnReduceTempToOutKernel<IN_T, OUT_T, Op>, num_blocks,
      threads_per_block, 2 * sizeof(T) * threads_per_block, cu_stream,
      (void*)(temp_storage.flat<int8_t>().data()), temp_in_offset, in, out,
      extent_x, n_group_in, extent_z, op));
}

template <typename T, typename Op, typename OUT_T, typename IN_T>
void Launch3DXZReduction(OpKernelContext* ctx, OUT_T out, IN_T in, int extent_x,
                         int extent_y, int extent_z, Op op, T init,
                         const gpuStream_t& cu_stream) {
  // setup segment offsets with counting and transform iterator
  RowOffset row_offset_op(extent_x * extent_z);
  gpuprim::CountingInputIterator<int> counting_iter(0);
  gpuprim::TransformInputIterator<int, RowOffset,
                                  gpuprim::CountingInputIterator<int>>
      transform_iter(counting_iter, row_offset_op);

  GatherOp gather_op(extent_x, extent_y, extent_z, false);
  typedef gpuprim::TransformInputIterator<int, GatherOp,
                                          gpuprim::CountingInputIterator<int>>
      gatherIterType;
  gatherIterType gather_iter(counting_iter, gather_op);

  PermutationInputIterator<T, IN_T, gatherIterType> permute_iter(in,
                                                                 gather_iter);

  std::size_t temp_storage_bytes = 0;
  auto reduce = [&](void* temp_storage_ptr) {
    auto success = gpuprim::DeviceSegmentedReduce::Reduce(
        temp_storage_ptr, temp_storage_bytes, permute_iter, out, extent_y,
        transform_iter, transform_iter + 1, op, init, cu_stream);

    OP_REQUIRES(ctx, success == 0,
                errors::Internal("CUB segmented reduce error",
                                 GpuGetErrorString(success)));
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
      (std::is_same<Op, gpuprim::Sum>::value ||
       std::is_same<Op, Eigen::internal::SumReducer<T>>::value ||
       std::is_same<Op, Sum<T>>::value);
};

template <typename T, typename Op>
struct IsMax {
  constexpr static bool value =
      (std::is_same<Op, gpuprim::Max>::value ||
       std::is_same<Op, Eigen::internal::MaxReducer<T>>::value);
};

template <typename T, typename Op>
struct IsMin {
  constexpr static bool value =
      (std::is_same<Op, gpuprim::Min>::value ||
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

#if TENSORFLOW_USE_ROCM
// the generic template produces identity value 1+i (header bug?)
template <>
struct IdentityValue<hipFloatComplex, Prod<hipFloatComplex> > {
  hipFloatComplex operator()() {
    return hipFloatComplex(1.0, 0.0);
  }
};

template <>
struct IdentityValue<hipDoubleComplex, Prod<hipDoubleComplex> > {
  hipDoubleComplex operator()() {
    return hipDoubleComplex(1.0, 0.0);
  }
};
#endif

}  // namespace reduction_op_helper

template <typename T, typename Op, typename OUT_T, typename IN_T,
          typename ReductionAxes>
void ReduceImpl(OpKernelContext* ctx, OUT_T out, IN_T in, int in_rank,
                int in_dim0, int in_dim1, int in_dim2, int out_rank,
                const ReductionAxes& reduction_axes, Op op) {
  T init = reduction_op_helper::IdentityValue<T, Op>()();
  const gpuStream_t& cu_stream = GetGpuStream(ctx);
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
    int elems_per_thread = in_dim1 / (in_dim0 * in_dim2);
    if (elems_per_thread >= 16) {
      Launch3DYReduction(ctx, out, in, in_dim0, in_dim1, in_dim2, op, init,
                         cu_stream);
    } else {
      Launch3DYReductionSimple(ctx, out, in, in_dim0, in_dim1, in_dim2, op,
                               init, cu_stream);
    }
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

template <typename T>
struct ReduceFunctor<GPUDevice, Eigen::internal::SumReducer<T>> {
  using TM = typename MapComplexToHipComplex<T>::TM;
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::SumReducer<T>& reducer) {
    ReduceImpl<TM, Sum<TM>, TM*, TM*, ReductionAxes>(
        ctx, (TM*)out.data(), (TM*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        Sum<TM>());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::SumReducer<T>& reducer) {
    FillIdentityEigenImplWithCast<T>(d, To32Bit(out), Eigen::internal::SumReducer<TM>());
  }
};

// TODO(rmlarsen): Specialize for float16.
template <typename T>
struct ReduceFunctor<GPUDevice, functor::EuclideanNormReducer<T>> {
  using TM = typename MapComplexToHipComplex<T>::TM;
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const functor::EuclideanNormReducer<T>& reducer) {
    typedef gpuprim::TransformInputIterator<TM, Square<TM>, TM*> inputIterType;
    inputIterType input_itr((TM*)in.data(), Square<TM>());
    typedef TransformOutputIterator<TM, TM, SqrtOfReal<TM>> outputIterType;
    outputIterType output_itr((TM*)out.data(), SqrtOfReal<TM>());
    ReduceImpl<TM, Sum<TM>, outputIterType, inputIterType, ReductionAxes>(
        ctx, output_itr, input_itr, in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        Sum<TM>());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const functor::EuclideanNormReducer<T>& reducer) {
    FillIdentityEigenImplWithCast<T>(d, To32Bit(out), functor::EuclideanNormReducer<TM>());
  }
};

template <typename T>
struct ReduceFunctor<GPUDevice, functor::MeanReducer<T>> {
  using TM = typename MapComplexToHipComplex<T>::TM;
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const functor::MeanReducer<T>& reducer) {
    uint64 divisor = 1;
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

    DividesBy<TM> div_op(divisor);
    TransformOutputIterator<TM, TM, DividesBy<TM>> itr((TM*)out.data(), div_op);
    ReduceImpl<TM, Sum<TM>, TransformOutputIterator<TM, TM, DividesBy<TM>>, TM*,
               ReductionAxes>(ctx, itr, (TM*)in.data(), in.rank(),
                              in.dimension(0),
                              in.rank() >= 2 ? in.dimension(1) : 1,
                              in.rank() >= 3 ? in.dimension(2) : 1, out.rank(),
                              reduction_axes, Sum<TM>());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const functor::MeanReducer<T>& reducer) {
    FillIdentityEigenImplWithCast<T>(d, To32Bit(out), functor::MeanReducer<TM>());
  }
};

template <>
struct ReduceFunctor<GPUDevice, functor::MeanReducer<Eigen::half>> {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const functor::MeanReducer<Eigen::half>& reducer) {
    uint64 divisor = 1;
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

    typedef gpuprim::TransformInputIterator<float, HalfToFloat, Eigen::half*>
        inputIterType;
    inputIterType input_itr((Eigen::half*)in.data(), HalfToFloat());

    typedef TransformOutputIterator<Eigen::half, float,
                                    DividesBy<float, Eigen::half>>
        outputIterType;
    outputIterType itr((Eigen::half*)out.data(), div_op);

    ReduceImpl<float, gpuprim::Sum, outputIterType, inputIterType,
               ReductionAxes>(ctx, itr, input_itr, in.rank(), in.dimension(0),
                              in.rank() >= 2 ? in.dimension(1) : 1,
                              in.rank() >= 3 ? in.dimension(2) : 1, out.rank(),
                              reduction_axes, gpuprim::Sum());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const functor::MeanReducer<Eigen::half>& reducer) {
    FillIdentityEigenImpl(d, To32Bit(out), reducer);
  }
};

template <typename T>
struct ReduceFunctor<GPUDevice, Eigen::internal::MaxReducer<T>> {
  using TM = typename MapComplexToHipComplex<T>::TM;
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::MaxReducer<T>& reducer) {
    ReduceImpl<TM, gpuprim::Max, TM*, TM*, ReductionAxes>(
        ctx, (TM*)out.data(), (TM*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        gpuprim::Max());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::MaxReducer<T>& reducer) {
    FillIdentityEigenImplWithCast<T>(d, To32Bit(out), Eigen::internal::MaxReducer<TM>());
  }
};

template <typename T>
struct ReduceFunctor<GPUDevice, Eigen::internal::MinReducer<T>> {
  using TM = typename MapComplexToHipComplex<T>::TM;
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::MinReducer<T>& reducer) {
    ReduceImpl<TM, gpuprim::Min, TM*, TM*, ReductionAxes>(
        ctx, (TM*)out.data(), (TM*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        gpuprim::Min());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::MinReducer<T>& reducer) {
    FillIdentityEigenImplWithCast<T>(d, To32Bit(out), Eigen::internal::MinReducer<TM>());
  }
};

template <typename T>
struct ReduceFunctor<GPUDevice, Eigen::internal::ProdReducer<T>> {
  using TM = typename MapComplexToHipComplex<T>::TM;  
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(OpKernelContext* ctx, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Eigen::internal::ProdReducer<T>& reducer) {
    ReduceImpl<TM, Prod<TM>, TM*, TM*, ReductionAxes>(
        ctx, (TM*)out.data(), (TM*)in.data(), in.rank(), in.dimension(0),
        in.rank() >= 2 ? in.dimension(1) : 1,
        in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), reduction_axes,
        Prod<TM>());
  }

  template <typename OUT_T>
  static void FillIdentity(const GPUDevice& d, OUT_T out,
                           const Eigen::internal::ProdReducer<T>& reducer) {
    FillIdentityEigenImplWithCast<T>(d, To32Bit(out), Eigen::internal::ProdReducer<TM>());
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

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_REDUCTION_GPU_KERNELS_CU_H_
