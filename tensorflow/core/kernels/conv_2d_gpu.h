/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_CONV_2D_GPU_H_
#define TENSORFLOW_CORE_KERNELS_CONV_2D_GPU_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T, bool conjugate>
struct maybe_conj {
  __device__ static __inline__ T run(T x) {
    if (conjugate) {
      return Eigen::numext::conj(x);
    } else {
      return x;
    }
  }
};

// Partial specializations for Gpu types used to store complex numbers.
template <bool conjugate>
struct maybe_conj<float2, conjugate> {
  __device__ static __inline__ float2 run(float2 c) {
    if (conjugate) {
      float2 c_conj;
      c_conj.x = c.x;
      c_conj.y = -c.y;
      return c_conj;
    } else {
      return c;
    }
  }
};

template <bool conjugate>
struct maybe_conj<double2, conjugate> {
  __device__ static __inline__ double2 run(double2 c) {
    if (conjugate) {
      double2 c_conj;
      c_conj.x = c.x;
      c_conj.y = -c.y;
      return c_conj;
    } else {
      return c;
    }
  }
};

// TODO(mjanusz): Move this to a shared util file.
// A simple array that contains data that can be passed between CPU and GPU.
template <typename T, int IndexCount, T DefaultValue>
struct Array {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T& operator[](int index) const {
    return data[index];
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T& operator[](int index) {
    return data[index];
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Array() {
    for (int i = 0; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Array(T a0) {
    data[0] = a0;
    for (int i = 1; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Array(T a0, T a1) {
    data[0] = a0;
    data[1] = a1;
    for (int i = 2; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Array(T a0, T a1, T a2) {
    data[0] = a0;
    data[1] = a1;
    data[2] = a2;
    for (int i = 3; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_STRONG_INLINE Array(const std::array<T, IndexCount>& array) {
    for (int i = 0; i < IndexCount; i++) {
      data[i] = array[i];
    }
  }
  T data[IndexCount];
};

// A dimension type with compile-time known size.
template <int IndexCount>
struct Dimension : Array<int, IndexCount, 1> {
  typedef Array<int, IndexCount, 1> Base;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dimension() : Base() {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dimension(int a0) : Base(a0) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dimension(int a0, int a1)
      : Base(a0, a1) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dimension(int a0, int a1, int a2)
      : Base(a0, a1, a2) {}
  EIGEN_STRONG_INLINE Dimension(const std::array<int, IndexCount>& array)
      : Base(array) {}
};

// An index type with compile-time known size.
template <int IndexCount>
struct Index : Array<int, IndexCount, 0> {
  typedef Array<int, IndexCount, 0> Base;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index() : Base() {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index(int a0) : Base(a0) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index(int a0, int a1) : Base(a0, a1) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index(int a0, int a1, int a2)
      : Base(a0, a1, a2) {}
};

// A helper function that converts a tensor index into a flat array index.
template <int IndexCount>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int TensorIndexToFlat(
    const Index<IndexCount>& index, const Dimension<IndexCount>& dims) {
  int flat_index = index[0];
  for (int i = 1; i < IndexCount; i++) {
    flat_index = flat_index * dims[i] + index[i];
  }
  return flat_index;
}

// A helper function that converts a flat array index into a tensor index.
template <int IndexCount>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index<IndexCount> FlatToTensorIndex(
    int index, const Dimension<IndexCount>& dims) {
  Index<IndexCount> tensor_index;
  for (int i = IndexCount - 1; i >= 0; i--) {
    int new_index = index / dims[i];
    tensor_index[i] = index - dims[i] * new_index;
    index = new_index;
  }
  return tensor_index;
}

// A simple CUDA custom kernel to shuffle dimensions of a 3D tensor according to
// the given shuffle permutation in template parameters. Shuffle permutation
// <sp0, sp1, sp2> shuffles dimensions such that input dimension 0 goes to sp0,
// 1 goes to sp1 and 2 goes to sp2. For example, shuffle permutation <2, 0, 1>
// will populate output so that input[x][y][z] is equal to (*output)[y][z][x].
//
// Requires that nthreads is equal to the total number of elements in the input
// tensor.
template <typename T, int sp0, int sp1, int sp2, bool conjugate = false>
__global__ void ShuffleInTensor3Simple(int nthreads,
                                       const T* __restrict__ input,
                                       Dimension<3> input_dims,
                                       T* __restrict__ output) {
  Dimension<3> output_dims;
  output_dims[sp0] = input_dims[0];
  output_dims[sp1] = input_dims[1];
  output_dims[sp2] = input_dims[2];

  // Iterate over output as opposed to iterating over input for better
  // performance. Iterating over output will generate sequential writes and
  // random reads that performs better compared to sequential reads and random
  // writes.
  GPU_1D_KERNEL_LOOP(output_index, nthreads) {
    Index<3> output_tensor_index = FlatToTensorIndex(output_index, output_dims);

    Index<3> input_tensor_index;
    input_tensor_index[0] = output_tensor_index[sp0];
    input_tensor_index[1] = output_tensor_index[sp1];
    input_tensor_index[2] = output_tensor_index[sp2];

    int input_index = TensorIndexToFlat(input_tensor_index, input_dims);

    output[output_index] =
        maybe_conj<T, conjugate>::run(ldg(input + input_index));
  }
}

static constexpr int kUnroll = 4;

template <typename T, int sp0, int sp1, int sp2, bool conjugate = false>
__global__ void ShuffleInTensor3SimpleVector(int nthreads,
                                             const T* __restrict__ input,
                                             Dimension<3> input_dims,
                                             T* __restrict__ output) {
  Dimension<3> output_dims;
  output_dims[sp0] = input_dims[0];
  output_dims[sp1] = input_dims[1];
  output_dims[sp2] = input_dims[2];

  const int stride = blockDim.x * gridDim.x * kUnroll;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  T buf[kUnroll];

  int output_index;
  for (output_index = tid * kUnroll; output_index + kUnroll - 1 < nthreads;
       output_index += stride) {
#pragma unroll
    for (int i = 0; i < kUnroll; i++) {
      int output_index_i = output_index + i;
      Index<3> output_tensor_index =
          FlatToTensorIndex(output_index_i, output_dims);
      Index<3> input_tensor_index;
      input_tensor_index[0] = output_tensor_index[sp0];
      input_tensor_index[1] = output_tensor_index[sp1];
      input_tensor_index[2] = output_tensor_index[sp2];

      int input_index_i = TensorIndexToFlat(input_tensor_index, input_dims);
      buf[i] = maybe_conj<T, conjugate>::run(ldg(input + input_index_i));
    }
    float2* out = reinterpret_cast<float2*>(output + output_index);
    *out = *reinterpret_cast<float2*>(buf);
  }

  for (; output_index < nthreads; ++output_index) {
    Index<3> output_tensor_index = FlatToTensorIndex(output_index, output_dims);

    Index<3> input_tensor_index;
    input_tensor_index[0] = output_tensor_index[sp0];
    input_tensor_index[1] = output_tensor_index[sp1];
    input_tensor_index[2] = output_tensor_index[sp2];

    int input_index = TensorIndexToFlat(input_tensor_index, input_dims);

    output[output_index] =
        maybe_conj<T, conjugate>::run(ldg(input + input_index));
  }
}

// Use shared memory tiles to swap dimension-1 and dimension-2 of a 3D tensor,
// where dimensions are zero-based: output[i][j][k] = input[i][k][j].
//
// Each thread block operates on a single tile, a rectangle of dimensions
// TileSizeI x TileSizeJ.
//
// In general, for best performance, you should probably set TileSizeI,
// TileSizeJ equal to the number of threads in a warp (32 in nvidia GPUs).
// With a TileSizeI, TileSizeJ of 32, NumThreads of 128 or 256 seems to get
// the best performance on K40 GPUs.
template <typename T, int NumThreads, int TileSizeI, int TileSizeJ,
          bool conjugate = false>
__global__ void SwapDimension1And2InTensor3UsingTiles(
    const T* __restrict__ input, Dimension<3> input_dims,
    T* __restrict__ output) {
  eigen_assert(blockDim.x == NumThreads);
  eigen_assert(blockDim.y == 1);
  eigen_assert(blockDim.z == 1);
  eigen_assert(gridDim.y == 1);
  eigen_assert(gridDim.z == 1);

  constexpr int ReadRowPerPass = NumThreads / TileSizeJ;
  constexpr int WriteRowPerPass = NumThreads / TileSizeI;
  // One extra line in the inner dimension to avoid share memory bank conflict.
  // This is to mimic the following, but no constructor of T can be invoked.
  //     __shared__ T shared_memory_tile[TileSizeI][TileSizeJ + 1];
#if GOOGLE_CUDA
  __shared__ __align__(
      alignof(T)) char shared_mem_raw[TileSizeI * (TileSizeJ + 1) * sizeof(T)];
  typedef T(*SharedMemoryTile)[TileSizeJ + 1];
  SharedMemoryTile shared_memory_tile =
      reinterpret_cast<SharedMemoryTile>(shared_mem_raw);
#elif TENSORFLOW_USE_ROCM
  __shared__ T shared_memory_tile[TileSizeI][TileSizeJ + 1];
#endif

  int x = threadIdx.x;

  Dimension<3> output_dims = {
      input_dims[0],
      input_dims[2],
      input_dims[1],
  };

  Dimension<3> input_dims_in_tiles = {
      input_dims[0],
      (input_dims[1] + TileSizeI - 1) / TileSizeI,
      (input_dims[2] + TileSizeJ - 1) / TileSizeJ,
  };

  Index<3> input_tile_index =
      FlatToTensorIndex(blockIdx.x, input_dims_in_tiles);

  Index<3> input_tile_origin = {
      input_tile_index[0],
      input_tile_index[1] * TileSizeI,
      input_tile_index[2] * TileSizeJ,
  };

  int input_origin_flat_index =
      TensorIndexToFlat(input_tile_origin, input_dims);

  bool full_tile = true;
  int tile_width = TileSizeJ;

  // Only the last row or column may not have the full size.
  if (input_tile_index[2] == input_dims_in_tiles[2] - 1) {
    tile_width = input_dims[2] - (input_dims_in_tiles[2] - 1) * TileSizeJ;
    full_tile &= false;
  }

  int tile_height = TileSizeI;

  if (input_tile_index[1] == input_dims_in_tiles[1] - 1) {
    tile_height = input_dims[1] - (input_dims_in_tiles[1] - 1) * TileSizeI;
    full_tile &= false;
  }

  // Calculate effective thread number. This ensures that we use the largest
  // number of threads available to form a regular thread block with no
  // trailing incomplete lines.
  constexpr int in_effective_thread_num = NumThreads / TileSizeJ * TileSizeJ;

  if (x < in_effective_thread_num) {
    // Orient the logical thread block with respect to the input array.
    // ie. align the contiguous dimension of thread blocks with the contiguous
    // dimension of the input array.
    int ti = x / TileSizeJ;
    int tj = x % TileSizeJ;
    int input_index = input_origin_flat_index + ti * input_dims[2] + tj;
    int input_increment = ReadRowPerPass * input_dims[2];

    if (full_tile) {
#pragma unroll
      for (int i_loc = ti; i_loc < (TileSizeI); i_loc += ReadRowPerPass) {
        shared_memory_tile[i_loc][tj] =
            maybe_conj<T, conjugate>::run(input[input_index]);
        input_index += input_increment;
      }
    } else {
      if (tj < tile_width) {
        for (int i_loc = ti; i_loc < (tile_height); i_loc += ReadRowPerPass) {
          shared_memory_tile[i_loc][tj] =
              maybe_conj<T, conjugate>::run(input[input_index]);
          input_index += input_increment;
        }
      }
    }
  }

  __syncthreads();

  Index<3> output_tile_index = {
      input_tile_index[0],
      input_tile_index[2],
      input_tile_index[1],
  };

  Index<3> output_tile_origin = {
      output_tile_index[0],
      output_tile_index[1] * TileSizeJ,
      output_tile_index[2] * TileSizeI,
  };

  int output_origin_flat_index =
      TensorIndexToFlat(output_tile_origin, output_dims);

  constexpr int out_effective_thread_num = NumThreads / TileSizeI * TileSizeI;

  if (x < out_effective_thread_num) {
    // Re-orient the logical thread block with respect to the output array.
    // ie. align the contiguous dimension of thread blocks with contiguous
    // dimension of the output array.
    int ti = x / TileSizeI;
    int tj = x % TileSizeI;
    int output_index = output_origin_flat_index + ti * output_dims[2] + tj;
    int output_increment = WriteRowPerPass * output_dims[2];

    if (full_tile) {
#pragma unroll
      for (int i_loc = ti; i_loc < (TileSizeJ); i_loc += WriteRowPerPass) {
        output[output_index] = shared_memory_tile[tj][i_loc];
        output_index += output_increment;
      }
    } else {
      if (tj < tile_height) {
        for (int i_loc = ti; i_loc < (tile_width); i_loc += WriteRowPerPass) {
          output[output_index] = shared_memory_tile[tj][i_loc];
          output_index += output_increment;
        }
      }
    }
  }
}

// A Gpu custom kernel that convert input to output, given proper padding on
// the left and the top.
template <typename T, int NDIMS>
__global__ void PadInputCustomKernelNHWC(
    int nthreads, const T* __restrict__ input, Dimension<NDIMS> input_dims,
    T* __restrict__ output, Dimension<NDIMS> output_dims,
    Dimension<NDIMS - 2> padding_left, T padding_value) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int output_index = index;
    Index<NDIMS> output_tensor_index =
        FlatToTensorIndex(output_index, output_dims);

    Index<NDIMS> input_tensor_index;
    input_tensor_index[0] = output_tensor_index[0];  // batch
    bool ok = true;
    for (int i = 1; i < NDIMS - 1; i++) {
      input_tensor_index[i] = output_tensor_index[i] - padding_left[i - 1];
      ok &=
          (input_tensor_index[i] >= 0 && input_tensor_index[i] < input_dims[i]);
    }
    input_tensor_index[NDIMS - 1] = output_tensor_index[NDIMS - 1];  // channels

    if (ok) {
      const int input_index = TensorIndexToFlat(input_tensor_index, input_dims);
      output[output_index] = input[input_index];
    } else {
      output[output_index] = padding_value;
    }
  }
}

template <typename T, int NDIMS>
__global__ void PadInputCustomKernelNCHW(
    int nthreads, const T* __restrict__ input, Dimension<NDIMS> input_dims,
    T* __restrict__ output, Dimension<NDIMS> output_dims,
    Dimension<NDIMS - 2> padding_left, T padding_value) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int output_index = index;
    Index<NDIMS> output_tensor_index =
        FlatToTensorIndex(output_index, output_dims);

    Index<NDIMS> input_tensor_index;
    input_tensor_index[0] = output_tensor_index[0];  // batch
    input_tensor_index[1] = output_tensor_index[1];  // channels
    bool ok = true;
    for (int i = 2; i < NDIMS; i++) {
      input_tensor_index[i] = output_tensor_index[i] - padding_left[i - 2];
      ok &=
          (input_tensor_index[i] >= 0 && input_tensor_index[i] < input_dims[i]);
    }

    if (ok) {
      const int input_index = TensorIndexToFlat(input_tensor_index, input_dims);
      output[output_index] = input[input_index];
    } else {
      output[output_index] = padding_value;
    }
  }
}

// A GPU helper function that converts TensorFlow filter format to Cudnn filter
// format.
template <typename T, int NDIMS>
struct TransformFilter<GPUDevice, T, int, NDIMS> {
  typedef GPUDevice Device;
  void operator()(const Device& d, FilterTensorFormat dst_filter_format,
                  typename TTypes<T, NDIMS, int>::ConstTensor in,
                  typename TTypes<T, NDIMS, int>::Tensor out) {
    Dimension<3> combined_dims;
    combined_dims[0] = in.dimension(0);  // spatial dimensions
    for (int i = 1; i < NDIMS - 2; i++) {
      combined_dims[0] *= in.dimension(i);
    }
    combined_dims[1] = in.dimension(NDIMS - 2);  // input filters
    combined_dims[2] = in.dimension(NDIMS - 1);  // output filters
    GpuLaunchConfig config = GetGpuLaunchConfig(out.size(), d);

    if (dst_filter_format == FORMAT_OIHW) {
      TF_CHECK_OK(GpuLaunchKernel(ShuffleInTensor3Simple<T, 2, 1, 0>,
                                  config.block_count, config.thread_per_block,
                                  0, d.stream(), config.virtual_thread_count,
                                  in.data(), combined_dims, out.data()));

    } else if (dst_filter_format == FORMAT_OHWI) {
      TF_CHECK_OK(GpuLaunchKernel(ShuffleInTensor3Simple<T, 1, 2, 0>,
                                  config.block_count, config.thread_per_block,
                                  0, d.stream(), config.virtual_thread_count,
                                  in.data(), combined_dims, out.data()));

    } else {
      LOG(ERROR) << "Unsupported filter format: "
                 << ToString(dst_filter_format);
    }
  }
};

// Converts Cudnn filter format OIHW or OHWI back to TensorFlow filter format
// HWIO.
template <typename T, int NDIMS>
struct ReverseTransformFilter<GPUDevice, T, NDIMS> {
  typedef GPUDevice Device;
  void operator()(const Device& d, FilterTensorFormat src_filter_format,
                  typename TTypes<T, NDIMS>::ConstTensor in,
                  typename TTypes<T, NDIMS>::Tensor out) {
    Dimension<3> combined_dims;

    if (src_filter_format == FORMAT_OIHW) {
      combined_dims[0] = in.dimension(0);  // output filters
      combined_dims[1] = in.dimension(1);  // input filters
      combined_dims[2] = in.dimension(2);  // spatial dimensions
      for (int i = 3; i < NDIMS; ++i) {
        combined_dims[2] *= in.dimension(i);
      }

      GpuLaunchConfig config = GetGpuLaunchConfig(out.size(), d);
      TF_CHECK_OK(GpuLaunchKernel(ShuffleInTensor3Simple<T, 2, 1, 0>,
                                  config.block_count, config.thread_per_block,
                                  0, d.stream(), config.virtual_thread_count,
                                  in.data(), combined_dims, out.data()));

    } else if (src_filter_format == FORMAT_OHWI) {
      combined_dims[0] = in.dimension(0);  // output filters
      combined_dims[1] = in.dimension(1);  // spatial dimensions
      for (int i = 2; i < NDIMS - 1; i++) {
        combined_dims[1] *= in.dimension(i);
      }
      combined_dims[2] = in.dimension(NDIMS - 1);  // input filters

      GpuLaunchConfig config = GetGpuLaunchConfig(out.size(), d);
      TF_CHECK_OK(GpuLaunchKernel(ShuffleInTensor3Simple<T, 2, 0, 1>,
                                  config.block_count, config.thread_per_block,
                                  0, d.stream(), config.virtual_thread_count,
                                  in.data(), combined_dims, out.data()));

    } else {
      // TODO(ezhulenev): Set error status in OpKernelContext instead.
      LOG(FATAL) << "Unsupported filter format: "
                 << ToString(src_filter_format);
    }
  }
};

// A GPU helper function that converts input tensor to a larger output tensor,
// given proper padding values. The padded value is zero.
template <typename T, int NDIMS>
struct PadInput<GPUDevice, T, int, NDIMS> {
  typedef GPUDevice Device;
  void operator()(const Device& d,
                  typename TTypes<T, NDIMS, int>::ConstTensor in,
                  const std::array<int, NDIMS - 2>& padding_left,
                  const std::array<int, NDIMS - 2>& padding_right,
                  typename TTypes<T, NDIMS, int>::Tensor out,
                  TensorFormat format, const T& padding_value) {
    GpuLaunchConfig config = GetGpuLaunchConfig(out.size(), d);
    Dimension<NDIMS> input_dims;
    for (int i = 0; i < NDIMS; ++i) {
      input_dims[i] = in.dimension(i);
    }
    Dimension<NDIMS> output_dims;
    for (int i = 0; i < NDIMS; ++i) {
      output_dims[i] = out.dimension(i);
    }

    const Dimension<NDIMS - 2> padding_left_dim(padding_left);

    if (format == FORMAT_NHWC) {
      TF_CHECK_OK(GpuLaunchKernel(
          PadInputCustomKernelNHWC<T, NDIMS>, config.block_count,
          config.thread_per_block, 0, d.stream(), config.virtual_thread_count,
          in.data(), input_dims, out.data(), output_dims, padding_left_dim,
          padding_value));
    } else if (format == FORMAT_NCHW) {
      TF_CHECK_OK(GpuLaunchKernel(
          PadInputCustomKernelNCHW<T, NDIMS>, config.block_count,
          config.thread_per_block, 0, d.stream(), config.virtual_thread_count,
          in.data(), input_dims, out.data(), output_dims, padding_left_dim,
          padding_value));
    } else {
      LOG(FATAL) << "Invalid data format: " << format;
    }
  }
};

// We want std::equal_to and std::greater, but they're not constexpr until
// C++14.
struct EqualTo {
  constexpr bool operator()(int a, int b) const { return a == b; }
};

struct GreaterThan {
  constexpr bool operator()(int a, int b) const { return a > b; }
};

// For each data type, the tile size possibility frontier denotes the tile size
// combinations that consume the most computational resources constrained by
// - number of threads per SM limit,
// - limit on size of the short dimension (<=15) due to the definition of
//   narrow matrix,
// - shared memory limit and
// - some experimentally determined, type-specific constraint on the product of
//   two side lengths to increase grid-level parallelism.
//
// A tile size combination lies on the frontier if and only if one or more
// constraint mentioned above is hit. Tile size combinations lying outside this
// frontier are either not possible, or are slower than the alternatives.
//
// It is instrumental to consider, for each data type, two subsets of the
// corresponding frontier:
// - long side frontier: the union of the biggest tile size combination for
//   each legal long side len.
// - non long side frontier: the frontier set minus the long side frontier.
//
// TileSizePossibilityFrontierCheck defines the frontier using only the long
// side frontier tile size combinations (since one can easily extrapolate
// the entire frontier from this subset). It serves as a utility function
// to help us determine where a tile size combination of interest lies with
// resepect to the frontier.
template <typename Op>
constexpr bool TileSizePossibilityFrontierCheck(int TileLongSide,
                                                int TileShortSide,
                                                int size_of_t, Op op) {
  // clang-format off

  return (size_of_t == 16 && ((TileLongSide == 32   && op(TileShortSide, 4))  ||
                             (TileLongSide == 64   && op(TileShortSide, 4))  ||
                             (TileLongSide == 128  && op(TileShortSide, 4))  ||
                             (TileLongSide == 256  && op(TileShortSide, 2)))) ||
          (size_of_t == 8 && ((TileLongSide == 32   && op(TileShortSide, 15)) ||
                             (TileLongSide == 64   && op(TileShortSide, 15)) ||
                             (TileLongSide == 128  && op(TileShortSide, 8))  ||
                             (TileLongSide == 256  && op(TileShortSide, 4))  ||
                             (TileLongSide == 512  && op(TileShortSide, 2)))) ||
          (size_of_t == 4 && ((TileLongSide == 32   && op(TileShortSide, 15)) ||
                             (TileLongSide == 64   && op(TileShortSide, 15)) ||
                             (TileLongSide == 128  && op(TileShortSide, 15)) ||
                             (TileLongSide == 256  && op(TileShortSide, 8))  ||
                             (TileLongSide == 512  && op(TileShortSide, 4))  ||
                             (TileLongSide == 1024 && op(TileShortSide, 2)))) ||
          (size_of_t == 2 && ((TileLongSide == 32   && op(TileShortSide, 15)) ||
                             (TileLongSide == 64   && op(TileShortSide, 15)) ||
                             (TileLongSide == 128  && op(TileShortSide, 15)) ||
                             (TileLongSide == 256  && op(TileShortSide, 8))  ||
                             (TileLongSide == 512  && op(TileShortSide, 4))  ||
                             (TileLongSide == 1024 && op(TileShortSide, 2)))) ||
          (size_of_t == 1 && ((TileLongSide == 32   && op(TileShortSide, 15)) ||
                             (TileLongSide == 64   && op(TileShortSide, 15)) ||
                             (TileLongSide == 128  && op(TileShortSide, 15)) ||
                             (TileLongSide == 256  && op(TileShortSide, 8))  ||
                             (TileLongSide == 512  && op(TileShortSide, 4))  ||
                             (TileLongSide == 1024 && op(TileShortSide, 2))));

  // clang-format on
}

constexpr bool TileSizeOnLongSideFrontier(int TileLongSide, int TileShortSide,
                                          int size_of_t) {
  return TileSizePossibilityFrontierCheck(TileLongSide, TileShortSide,
                                          size_of_t, EqualTo());
}
constexpr bool TileSizeOutsideFrontier(int TileLongSide, int TileShortSide,
                                       int size_of_t) {
  return TileSizePossibilityFrontierCheck(TileLongSide, TileShortSide,
                                          size_of_t, GreaterThan());
}
constexpr bool TileSizeOnNonLongSideFrontier(int TileLongSide,
                                             int TileShortSide, int size_of_t) {
  // For a tile size combination (longside, shortside), lying on the frontier
  // implies that (longside, shortside) is on or within the frontier but
  // (longside*2, shortside) or (longside, shortside+1) is not. With the above
  // criterion, we simply need to use !TileSizeOnLongSideFrontier to ensure that
  // it is not on the long side frontier.
  return !TileSizeOutsideFrontier(TileLongSide, TileShortSide, size_of_t) &&
         (TileSizeOutsideFrontier(TileLongSide * 2, TileShortSide, size_of_t) ||
          TileSizeOutsideFrontier(TileLongSide, TileShortSide + 1,
                                  size_of_t)) &&
         !TileSizeOnLongSideFrontier(TileLongSide, TileShortSide, size_of_t);
}

// Helper function to launch a batch narrow matirx transpose kernel.
template <typename T, int TileLongSide, int TileShortSide, bool conjugate>
void LaunchBatchNarrowMatrixTransposeKernel(
    const GPUDevice& d, int tile_size_i, int tile_size_j, int total_tiles_count,
    const T* input, const Dimension<3>& input_dims, T* output) {
  constexpr int NumThreads = TileLongSide;
  if (tile_size_i <= TileLongSide && tile_size_j <= TileShortSide) {
    TF_CHECK_OK(GpuLaunchKernel(
        SwapDimension1And2InTensor3UsingTiles<T, NumThreads, TileLongSide,
                                              TileShortSide, conjugate>,
        total_tiles_count, NumThreads, 0, d.stream(), input, input_dims,
        output));
  } else {
    TF_CHECK_OK(GpuLaunchKernel(
        SwapDimension1And2InTensor3UsingTiles<T, NumThreads, TileShortSide,
                                              TileLongSide, conjugate>,
        total_tiles_count, NumThreads, 0, d.stream(), input, input_dims,
        output));
  }
}

// Recursive template function to search, in a trial-and-error manner, for the
// minimum tile size configuration satisfying the requested tile side lengths.
// An important invariant of this search procedure is that for an unsatisfied
// request, we always try doubling the long side len first, and only after
// the request is satisfied for the long side len do we begin incrementing
// the short side len.
//
// We have three specializations of this search function depending on where the
// current tile size combination lies with respect to the frontier.
// - It lies within the frontier. If request is not satisfied, for the next tile
// size combination, we first try doubling the long side len and if that does
// not work, we then increment the short side len.
// - It lies on the non long side frontier. If the request is not satisfied, we
// can only increment the short side len.
// - It lies on the long side frontier. We launch the kernel without checking if
// the request is satisfied or not.
template <typename T, int TileLongSide, int TileShortSide, bool conjugate,
          typename dummy = void>
struct BatchNarrowMatrixTransposeDispatcher {
  static void DoIt(const GPUDevice& d, int tile_size_i, int tile_size_j,
                   int total_tiles_count, const T* input,
                   const Dimension<3>& input_dims, T* output) {
    static_assert(
        (TileLongSide & (TileLongSide - 1)) == 0,
        "The length of the longer side of the tile is always a power of 2.");
    bool request_satisfied =
        std::max(tile_size_i, tile_size_j) <= TileLongSide &&
        std::min(tile_size_i, tile_size_j) <= TileShortSide;

    if (request_satisfied) {
      LaunchBatchNarrowMatrixTransposeKernel<T, TileLongSide, TileShortSide,
                                             conjugate>(
          d, tile_size_i, tile_size_j, total_tiles_count, input, input_dims,
          output);
      return;
    }

    // If the execution reaches here, then the kernel was not launched; we then
    // determine whether it is the long side or the short side that falls short
    // of the request and increase that parameter accordingly.
    const bool long_side_request_not_satisfied =
        std::max(tile_size_i, tile_size_j) > TileLongSide;

    if (long_side_request_not_satisfied) {
      BatchNarrowMatrixTransposeDispatcher<T, TileLongSide * 2, TileShortSide,
                                           conjugate>::DoIt(d, tile_size_i,
                                                            tile_size_j,
                                                            total_tiles_count,
                                                            input, input_dims,
                                                            output);
    } else {
      BatchNarrowMatrixTransposeDispatcher<T, TileLongSide, TileShortSide + 1,
                                           conjugate>::DoIt(d, tile_size_i,
                                                            tile_size_j,
                                                            total_tiles_count,
                                                            input, input_dims,
                                                            output);
    }
  }
};

template <typename T, int TileLongSide, int TileShortSide, bool conjugate>
struct BatchNarrowMatrixTransposeDispatcher<
    T, TileLongSide, TileShortSide, conjugate,
    typename std::enable_if<TileSizeOnNonLongSideFrontier(
                                TileLongSide, TileShortSide, sizeof(T)),
                            void>::type> {
  static void DoIt(const GPUDevice& d, int tile_size_i, int tile_size_j,
                   int total_tiles_count, const T* input,
                   const Dimension<3>& input_dims, T* output) {
    static_assert(
        (TileLongSide & (TileLongSide - 1)) == 0,
        "The length of the longer side of the tile is always a power of 2.");
    bool request_satisfied =
        std::max(tile_size_i, tile_size_j) <= TileLongSide &&
        std::min(tile_size_i, tile_size_j) <= TileShortSide;

    if (request_satisfied) {
      LaunchBatchNarrowMatrixTransposeKernel<T, TileLongSide, TileShortSide,
                                             conjugate>(
          d, tile_size_i, tile_size_j, total_tiles_count, input, input_dims,
          output);
      return;
    }

    // If the execution reaches here, then the kernel was not launched; since
    // we are on the non long side frontier, we increment the short dimension
    // and try again.
    BatchNarrowMatrixTransposeDispatcher<T, TileLongSide, TileShortSide + 1,
                                         conjugate>::DoIt(d, tile_size_i,
                                                          tile_size_j,
                                                          total_tiles_count,
                                                          input, input_dims,
                                                          output);
  }
};

template <typename T, int TileLongSide, int TileShortSide, bool conjugate>
struct BatchNarrowMatrixTransposeDispatcher<
    T, TileLongSide, TileShortSide, conjugate,
    typename std::enable_if<TileSizeOnLongSideFrontier(
                                TileLongSide, TileShortSide, sizeof(T)),
                            void>::type> {
  static void DoIt(const GPUDevice& d, int tile_size_i, int tile_size_j,
                   int total_tiles_count, const T* input,
                   const Dimension<3>& input_dims, T* output) {
    static_assert(
        (TileLongSide & (TileLongSide - 1)) == 0,
        "The length of the longer side of the tile is always a power of 2.");

    LaunchBatchNarrowMatrixTransposeKernel<T, TileLongSide, TileShortSide,
                                           conjugate>(
        d, tile_size_i, tile_size_j, total_tiles_count, input, input_dims,
        output);
  }
};

// This function tries to recover, in a brute force way, the frontier defined in
// TileSizePossibilityFrontierCheck as a vector of tile size combinations lying
// on the long side frontier. This vector is sufficient to determine the entire
// frontier.
//
// Note that if one changes the frontier definition in
// TileSizePossibilityFrontierCheck and forgets to set the largest short
// side len of the largest legal long side len to 2, this function will fail
// and crash the program.
template <int SizeOfT>
const std::vector<std::pair<int, int>>& GetTileSizesFrontier() {
  static_assert(
      SizeOfT <= 16,
      "Currently, only data types of sizes 16 bytes or less are supported.");
  static_assert((SizeOfT & (SizeOfT - 1)) == 0,
                "Data types must have sizes that are powers of 2.");

  // Expensive work to populate sizes, lazily run in a thread-safe
  // manner the first time GetTileSizesFrontier<N> is called.
  static auto* frontier = [] {
    auto* frontier = new std::vector<std::pair<int, int>>();
    const int kMaxLongSideLen = 1024;
    const int kMaxShortSideLen = 15;
    for (int long_side = 32; long_side <= kMaxLongSideLen; long_side *= 2) {
      for (int short_side = 2; short_side <= kMaxShortSideLen;
           short_side += 1) {
        if (TileSizeOnLongSideFrontier(long_side, short_side, SizeOfT)) {
          // The current combination lies on the frontier, thus we
          // add it to the frontier definition.
          frontier->push_back(std::make_pair(long_side, short_side));

          // The long side length is the largest one allowed iff its
          // corresponding short side length is 2.
          if (short_side == 2) return frontier;

          // We have exhausted all the possibilities in the frontier
          // with the given long side length.
          break;
        }
      }
    }
    LOG(FATAL)
        << "The corresponding short side length of the largest long side "
           "length has to be 2.";
  }();
  return *frontier;
}

// Helper structs to help determine which data type to use given the size of
// the matrix data type. A transpose of elements of size N will use a kernel
// which operates on an array of TransposeElemType<N>::type.
template <int ElemBytes>
struct TransposeElemType;
template <>
struct TransposeElemType<1> {
  using type = uint8;
};
template <>
struct TransposeElemType<2> {
  using type = uint16;
};
template <>
struct TransposeElemType<4> {
  using type = uint32;
};
template <>
struct TransposeElemType<8> {
  using type = float2;
};
template <>
struct TransposeElemType<16> {
  using type = double2;
};

// A helper function to make RunSwapDimension1And2InTensor3 concise. This
// helper function looks at the data type and input matrix sizes and decides
// the thread numbers and tile sizes to use.
template <typename T, bool conjugate = false>
void SwapDimension1And2InTensor3WithNarrowMatrices(
    const GPUDevice& d, const T* input, const Dimension<3>& input_dims,
    T* output, const int kMinDimensionToUseTiles) {
  // Get available tile sizes here for the data type requested:
  const auto& tile_spec = GetTileSizesFrontier<sizeof(T)>();

  int tile_long_side_len = 0;
  int tile_short_side_len = 0;
  float lowest_cost = std::numeric_limits<float>::max();
  int data_long_side = std::max(input_dims[1], input_dims[2]);

  for (auto tile_size_pair : tile_spec) {
    int proposed_tile_long_side_len = tile_size_pair.first;

    // Number of threads that will not be doing anything useful when reading
    // the matrix because the thread block size is bigger than the data block
    // size.
    int num_wasted_threads =
        data_long_side - MathUtil::FloorOfRatio<int>(
                             data_long_side, proposed_tile_long_side_len) *
                             proposed_tile_long_side_len;

    int num_full_tiles = MathUtil::FloorOfRatio<int>(
        data_long_side, proposed_tile_long_side_len);

    float cost = 0;

    // However, if we can execute two or more full tiles, then we gladly
    // accept any number of wasted threads and ignore its cost.
    if (num_full_tiles <= 1) cost = num_wasted_threads;

    // Using less than or equal to here because given the same cost, we
    // would like to launch as many threads as possible.
    if (cost <= lowest_cost) {
      tile_long_side_len = proposed_tile_long_side_len;
      tile_short_side_len = tile_size_pair.second;
      lowest_cost = cost;
    }
  }

  // Request tile sizes such that the longer side of threadblock aligns with
  // the longer side of input data block to maximize read throughput.
  // The ideal tile shape is one where the length of the shorter side of the
  // tile is equal to the length of the shorter side of the input matrix.
  int requested_tile_size_i = input_dims[1] >= kMinDimensionToUseTiles
                                  ? tile_long_side_len
                                  : input_dims[1];
  int requested_tile_size_j = input_dims[1] >= kMinDimensionToUseTiles
                                  ? input_dims[2]
                                  : tile_long_side_len;

  // Truncate the shorter size requested according to the manual limit set in
  // tile_spec to make sure that we do not launch configurations violating
  // hardware limits.
  requested_tile_size_i =
      requested_tile_size_i == tile_long_side_len
          ? tile_long_side_len
          : std::min(requested_tile_size_i, tile_short_side_len);
  requested_tile_size_j =
      requested_tile_size_j == tile_long_side_len
          ? tile_long_side_len
          : std::min(requested_tile_size_j, tile_short_side_len);

  Dimension<3> input_dims_in_tiles = {
      input_dims[0],
      MathUtil::CeilOfRatio<int>(input_dims[1], requested_tile_size_i),
      MathUtil::CeilOfRatio<int>(input_dims[2], requested_tile_size_j),
  };

  int total_tiles_count =
      input_dims_in_tiles[0] * input_dims_in_tiles[1] * input_dims_in_tiles[2];

  using ElemType = typename TransposeElemType<sizeof(T)>::type;
  static_assert(alignof(T) >= alignof(ElemType), "Unexpected data alignment.");
  BatchNarrowMatrixTransposeDispatcher<ElemType, 32, 2, conjugate>::DoIt(
      d, requested_tile_size_i, requested_tile_size_j, total_tiles_count,
      reinterpret_cast<const ElemType*>(input), input_dims,
      reinterpret_cast<ElemType*>(output));
}

// Launch the GPU kernel that would swap dimension-1 and dimension-2 in a
// 3D tensor. It looks at the shape of the incoming data, and decides the best
// strategy to launch.
template <typename T, bool conjugate = false>
void RunSwapDimension1And2InTensor3(const GPUDevice& d, const T* input,
                                    const Dimension<3>& input_dims, T* output) {
  // If both dimensions are not trivial, use tiles for the actual swapping.
  // If one dimension is trivial, use SmallDim kernel for swapping.
  // Otherwise, the trivial swapping relying on the ldg cache is more efficient.
  static const int kMinDimensionToUseTiles = 16;
  static const int kMinDimensionToUseRectTiles = 96;

  bool large_matrix = input_dims[1] >= kMinDimensionToUseTiles &&
                      input_dims[2] >= kMinDimensionToUseTiles;
  bool narrow_matrix = input_dims[1] >= kMinDimensionToUseRectTiles ||
                       input_dims[2] >= kMinDimensionToUseRectTiles;
  if (large_matrix) {
    // We get best performance when kTileSize is the number of threads in a warp
    // (32 on our GPUs) and NumSubTiles is 8, so our block size is 8 * 32 = 256
    // threads.
    constexpr int kTileSize = 32;
    constexpr int kNumThreads = 256;

    Dimension<3> input_dims_in_tiles = {
        input_dims[0],
        MathUtil::CeilOfRatio<int>(input_dims[1], kTileSize),
        MathUtil::CeilOfRatio<int>(input_dims[2], kTileSize),
    };

    int total_tiles_count = input_dims_in_tiles[0] * input_dims_in_tiles[1] *
                            input_dims_in_tiles[2];
    TF_CHECK_OK(GpuLaunchKernel(
        SwapDimension1And2InTensor3UsingTiles<T, kNumThreads, kTileSize,
                                              kTileSize, conjugate>,
        total_tiles_count, kNumThreads, 0, d.stream(), input, input_dims,
        output));

  } else if (narrow_matrix) {
    SwapDimension1And2InTensor3WithNarrowMatrices<T, conjugate>(
        d, input, input_dims, output, kMinDimensionToUseTiles);
  } else {
    int total_element_count = input_dims[0] * input_dims[1] * input_dims[2];
    GpuLaunchConfig config = GetGpuLaunchConfig(total_element_count, d);
    TF_CHECK_OK(GpuLaunchKernel(ShuffleInTensor3Simple<T, 0, 2, 1, conjugate>,
                                config.block_count, config.thread_per_block, 0,
                                d.stream(), config.virtual_thread_count, input,
                                input_dims, output));
  }
}

// A GPU helper functor that does general dimension 1 and 2 switch for 3D
// tensor.
template <typename T, bool conjugate>
struct SwapDimension1And2InTensor3<GPUDevice, T, conjugate> {
  typedef GPUDevice Device;
  void operator()(const Device& d, const T* in,
                  const gtl::ArraySlice<int64_t>& combined_dims, T* out) {
    Dimension<3> input_dims = {static_cast<int>(combined_dims[0]),
                               static_cast<int>(combined_dims[1]),
                               static_cast<int>(combined_dims[2])};
    RunSwapDimension1And2InTensor3<T, conjugate>(d, in, input_dims, out);
  }
};

// A GPU helper functor that does general dimension 0 and 2 switch for 3D
// tensor.
template <typename T, bool conjugate>
struct SwapDimension0And2InTensor3<GPUDevice, T, conjugate> {
  typedef GPUDevice Device;
  void operator()(const Device& d, const T* in,
                  const gtl::ArraySlice<int64_t>& combined_dims, T* out) {
    Dimension<3> input_dims = {static_cast<int>(combined_dims[0]),
                               static_cast<int>(combined_dims[1]),
                               static_cast<int>(combined_dims[2])};
    size_t total_size = combined_dims[0] * combined_dims[1] * combined_dims[2];
    GpuLaunchConfig config = GetGpuLaunchConfig(total_size, d);

    auto out_ptr = reinterpret_cast<uintptr_t>(out);
    bool aligned = out_ptr % 16 == 0;

    bool use_vector = false;
    bool use_custom_config = false;
    if ((input_dims[0] <= 128 && input_dims[2] <= 128) ||
        input_dims[0] * input_dims[1] <= 128 ||
        input_dims[1] * input_dims[2] <= 8) {
      use_vector = true;
      use_custom_config = true;
    } else if (input_dims[1] * input_dims[2] <= 16384) {
      use_vector = true;
    }

    if (sizeof(T) == 2 && aligned && use_vector) {
      int block_count;
      if (use_custom_config) {
        block_count = (total_size + config.thread_per_block - 1) /
                      config.thread_per_block;
      } else {
        block_count = config.block_count;
      }

      TF_CHECK_OK(
          GpuLaunchKernel(ShuffleInTensor3SimpleVector<T, 2, 1, 0, conjugate>,
                          block_count, config.thread_per_block / kUnroll, 0,
                          d.stream(), total_size, in, input_dims, out));
    } else {
      TF_CHECK_OK(GpuLaunchKernel(ShuffleInTensor3Simple<T, 2, 1, 0, conjugate>,
                                  config.block_count, config.thread_per_block,
                                  0, d.stream(), config.virtual_thread_count,
                                  in, input_dims, out));
    }
  }
};

// A GPU helper functor that converts NHWC TensorFlow data format to
// NCHW format that is accepted by Cudnn.
template <typename T, int NDIMS>
struct NHWCToNCHW<GPUDevice, T, NDIMS> {
  typedef GPUDevice Device;
  void operator()(const Device& d, typename TTypes<T, NDIMS>::ConstTensor in,
                  typename TTypes<T, NDIMS>::Tensor out) {
    Dimension<3> combined_dims;
    combined_dims[0] = in.dimension(0);  // N (batch)
    combined_dims[1] = in.dimension(1);  // spatial dimensions (HW)
    for (int i = 2; i < NDIMS - 1; ++i) {
      combined_dims[1] *= in.dimension(i);
    }
    combined_dims[2] = in.dimension(NDIMS - 1);  // C (channels)
    RunSwapDimension1And2InTensor3(d, in.data(), combined_dims, out.data());
  }
};

// A GPU helper functor that converts NCHW Cudnn data format to NHWC TensorFlow
// Format.
template <typename T, int NDIMS>
struct NCHWToNHWC<GPUDevice, T, NDIMS> {
  typedef GPUDevice Device;
  void operator()(const Device& d, typename TTypes<T, NDIMS>::ConstTensor in,
                  typename TTypes<T, NDIMS>::Tensor out) {
    Dimension<3> combined_dims;
    combined_dims[0] = in.dimension(0);  // N (batch)
    combined_dims[1] = in.dimension(1);  // C (channel)
    combined_dims[2] = in.dimension(2);  // spatial dimensions (HW)
    for (int i = 3; i < NDIMS; ++i) {
      combined_dims[2] *= in.dimension(i);
    }
    RunSwapDimension1And2InTensor3(d, in.data(), combined_dims, out.data());
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_CONV_2D_GPU_H_
