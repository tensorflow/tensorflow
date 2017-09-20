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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <algorithm>
#include <array>
#include <limits>

#include "cuda/include/cuda.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

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
    tensor_index[i] = index % dims[i];
    index /= dims[i];
  }
  return tensor_index;
}

// A Cuda custom kernel that swaps dimension-0 and dimension-2 of a 3D tensor.
template <typename T>
__global__ void SwapDimension0And2InTensor3Simple(int nthreads, const T* input,
                                                  Dimension<3> input_dims,
                                                  T* output) {
  Dimension<3> output_dims;
  output_dims[0] = input_dims[2];
  output_dims[1] = input_dims[1];
  output_dims[2] = input_dims[0];

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int output_index = index;

    Index<3> output_tensor_index = FlatToTensorIndex(output_index, output_dims);

    Index<3> input_tensor_index;
    input_tensor_index[0] = output_tensor_index[2];
    input_tensor_index[1] = output_tensor_index[1];
    input_tensor_index[2] = output_tensor_index[0];

    int input_index = TensorIndexToFlat(input_tensor_index, input_dims);

    output[output_index] = ldg(input + input_index);
  }
}

// A Cuda custom kernel that swaps dimension-1 and dimension-2 of a 3D tensor.
template <typename T>
__global__ void SwapDimension1And2InTensor3Simple(int nthreads, const T* input,
                                                  Dimension<3> input_dims,
                                                  T* output) {
  Dimension<3> output_dims;
  output_dims[0] = input_dims[0];
  output_dims[1] = input_dims[2];
  output_dims[2] = input_dims[1];

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int output_index = index;
    Index<3> output_tensor_index = FlatToTensorIndex(output_index, output_dims);

    Index<3> input_tensor_index;
    input_tensor_index[0] = output_tensor_index[0];
    input_tensor_index[1] = output_tensor_index[2];
    input_tensor_index[2] = output_tensor_index[1];

    int input_index = TensorIndexToFlat(input_tensor_index, input_dims);

    output[output_index] = ldg(input + input_index);
  }
}

// Use shared memory tiles to swap dimension-1 and dimension-2 of a 3D tensor,
// where dimensions are zero-based: output[i][j][k] = input[i][k][j].
//
// Each thread block operates on a single tile, a rectangle of dimensions TileSizeI
// x TileSizeJ.
//
// In general, for best performance, you should probably set TileSizeI,
// TileSizeJ equal to the number of threads in a warp (32 in nvidia GPUs).
// With a TileSizeI, TileSizeJ of 32, ThreadNum of 128 or 256 seems to get
// the best performance on K40 GPUs.
template <typename T, int ThreadNum, int TileSizeI, int TileSizeJ>
__global__ void SwapDimension1And2InTensor3UsingTiles(
    const T* __restrict__ input, Dimension<3> input_dims,
    T* __restrict__ output) {

  eigen_assert(blockDim.x == ThreadNum);
  eigen_assert(blockDim.y == 1);
  eigen_assert(blockDim.z == 1);
  eigen_assert(gridDim.y == 1);
  eigen_assert(gridDim.z == 1);

  const int ReadRowPerPass = (ThreadNum / TileSizeJ);
  const int WriteRowPerPass = (ThreadNum / TileSizeI);
  // One extra line in the inner dimension to avoid share memory bank conflict.
  __shared__ T shared_memory_tile[TileSizeI][TileSizeJ + 1];

// Memory access macros:
#define SHARED(i, j) shared_memory_tile[i][j]

#define INPUT(i, j) input[input_origin_flat_index + (i)*input_dims[2] + (j)]

#define OUTPUT(i, j) output[output_origin_flat_index + (i)*output_dims[2] + (j)]

  int x = threadIdx.x;

  Dimension<3> output_dims = {
      input_dims[0], input_dims[2], input_dims[1],
  };

  Dimension<3> input_dims_in_tiles = {
      input_dims[0], (input_dims[1] + TileSizeI - 1) / TileSizeI,
      (input_dims[2] + TileSizeJ - 1) / TileSizeJ,
  };

  Index<3> input_tile_index =
      FlatToTensorIndex(blockIdx.x, input_dims_in_tiles);

  Index<3> input_tile_origin = {
      input_tile_index[0], input_tile_index[1] * TileSizeI,
      input_tile_index[2] * TileSizeJ,
  };

  int input_origin_flat_index =
      TensorIndexToFlat(input_tile_origin, input_dims);

  int tile_width = TileSizeJ;

  // Only the last row or column may not have the full size.
  if (input_tile_index[2] == input_dims_in_tiles[2] - 1) {
    tile_width = input_dims[2] - (input_dims_in_tiles[2] - 1) * TileSizeJ;
  }

  int tile_height = TileSizeI;

  if (input_tile_index[1] == input_dims_in_tiles[1] - 1) {
    tile_height = input_dims[1] - (input_dims_in_tiles[1] - 1) * TileSizeI;
  }

  // Calculate effective thread number. This ensures that we use the largest
  // number of threads available to form a regular thread block with no
  // trailing incomplete lines.
  int effective_thread_num = ThreadNum / TileSizeJ * TileSizeJ;

  if (x < effective_thread_num) {
    // Orient the logical thread block with respect to the input array.
    // ie. align the contiguous dimension of thread blocks with contiguous
    // dimension of the input array.
    int ti = x / TileSizeJ;
    int tj = x % TileSizeJ;
    if (tj < tile_width)
      for (int i_loc = ti; i_loc < (tile_height); i_loc += ReadRowPerPass) {
        SHARED(i_loc, tj) = INPUT(i_loc, tj);
      }
  }

  __syncthreads();

  Index<3> output_tile_index = {
      input_tile_index[0], input_tile_index[2], input_tile_index[1],
  };

  Index<3> output_tile_origin = {
      output_tile_index[0], output_tile_index[1] * TileSizeJ,
      output_tile_index[2] * TileSizeI,
  };

  int output_origin_flat_index =
      TensorIndexToFlat(output_tile_origin, output_dims);

  effective_thread_num = ThreadNum / TileSizeI * TileSizeI;

  if (x < effective_thread_num) {
    // Re-oriente the logical thread block with respect to the output array.
    // ie. align the contiguous dimension of thread blocks with contiguous
    // dimension of the output array.
    int ti = x / TileSizeI;
    int tj = x % TileSizeI;

    if (tj < tile_height)
      for (int i_loc = ti; i_loc < (tile_width); i_loc += WriteRowPerPass) {
        OUTPUT(i_loc, tj) = SHARED(tj, i_loc);
      }
  }
}

// A Cuda custom kernel that convert input to output, given proper padding on
// the left and the top. The padded value is zero.
template <typename T, int NDIMS>
__global__ void PadInputCustomKernelNHWC(int nthreads, const T* input,
                                         Dimension<NDIMS> input_dims, T* output,
                                         Dimension<NDIMS> output_dims,
                                         Dimension<NDIMS - 2> padding_left) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
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
      output[output_index] = T(0);
    }
  }
}

template <typename T, int NDIMS>
__global__ void PadInputCustomKernelNCHW(int nthreads, const T* input,
                                         Dimension<NDIMS> input_dims, T* output,
                                         Dimension<NDIMS> output_dims,
                                         Dimension<NDIMS - 2> padding_left) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
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
      output[output_index] = T(0);
    }
  }
}

// A GPU helper function that converts TensorFlow filter format to Cudnn filter
// format.
template <typename T, int NDIMS>
struct TransformFilter<GPUDevice, T, int, NDIMS> {
  typedef GPUDevice Device;
  void operator()(const Device& d,
                  typename TTypes<T, NDIMS, int>::ConstTensor in,
                  typename TTypes<T, NDIMS, int>::Tensor out) {
    Dimension<3> combined_dims;
    combined_dims[0] = in.dimension(0);  // spatial dimensions
    for (int i = 1; i < NDIMS - 2; i++) {
      combined_dims[0] *= in.dimension(i);
    }
    combined_dims[1] = in.dimension(NDIMS - 2);  // input filters
    combined_dims[2] = in.dimension(NDIMS - 1);  // output filters
    CudaLaunchConfig config = GetCudaLaunchConfig(out.size(), d);
    SwapDimension0And2InTensor3Simple<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            config.virtual_thread_count, in.data(), combined_dims, out.data());
  }
};

// Converts Cudnn filter format back to TensorFlow filter format.
template <typename T, int NDIMS>
struct ReverseTransformFilter<GPUDevice, T, NDIMS> {
  typedef GPUDevice Device;
  void operator()(const Device& d, typename TTypes<T, NDIMS>::ConstTensor in,
                  typename TTypes<T, NDIMS>::Tensor out) {
    Dimension<3> combined_dims;
    combined_dims[0] = in.dimension(0);  // output filters
    combined_dims[1] = in.dimension(1);  // input filters
    combined_dims[2] = in.dimension(2);  // spatial dimensions
    for (int i = 3; i < NDIMS; ++i) {
      combined_dims[2] *= in.dimension(i);
    }
    CudaLaunchConfig config = GetCudaLaunchConfig(out.size(), d);
    SwapDimension0And2InTensor3Simple<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            config.virtual_thread_count, in.data(), combined_dims, out.data());
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
                  TensorFormat format) {
    CudaLaunchConfig config = GetCudaLaunchConfig(out.size(), d);
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
      PadInputCustomKernelNHWC<T, NDIMS><<<
          config.block_count, config.thread_per_block, 0, d.stream()>>>(
          config.virtual_thread_count, in.data(), input_dims, out.data(),
          output_dims, padding_left_dim);
    } else if (format == FORMAT_NCHW) {
      PadInputCustomKernelNCHW<T, NDIMS><<<
          config.block_count, config.thread_per_block, 0, d.stream()>>>(
          config.virtual_thread_count, in.data(), input_dims, out.data(),
          output_dims, padding_left_dim);
    } else {
      LOG(FATAL) << "Invalid data format: " << format;
    }
  }
};

// Recursive template function to search for the minimum tile size configuration
// satisfying the requested tile side lengths.
template <typename T, int TileLongSide, int TileShortSide>
struct BatchNarrowMatrixTransposeDispatcher {
  static void DoBatchNarrowMatrixTranspose(const GPUDevice& d, int tile_size_i,
                                           int tile_size_j,
                                           int total_tiles_count,
                                           const T* input,
                                           const Dimension<3>& input_dims,
                                           T* output) {
    bool request_satisfied = (max(tile_size_i, tile_size_j) <= TileLongSide) &&
                             (min(tile_size_i, tile_size_j) <= TileShortSide);

    if (request_satisfied) {
      const int ThreadNum = TileLongSide;
      if (tile_size_i <= TileLongSide && tile_size_j <= TileShortSide)
        SwapDimension1And2InTensor3UsingTiles<
            T, ThreadNum, TileLongSide,
            TileShortSide><<<total_tiles_count, ThreadNum, 0, d.stream()>>>(
            input, input_dims, output);
      else if (tile_size_j <= TileLongSide && tile_size_i <= TileShortSide)
        SwapDimension1And2InTensor3UsingTiles<
            T, ThreadNum, TileShortSide,
            TileLongSide><<<total_tiles_count, ThreadNum, 0, d.stream()>>>(
            input, input_dims, output);
      return;
    }

    // Kernel is not launched, meaning the launch configuration is not
    // satisfied.
    const bool long_side_request_not_satisfied =
        max(tile_size_i, tile_size_j) > TileLongSide;

    // Increase launch parameters and try again.
    if (long_side_request_not_satisfied) {
      BatchNarrowMatrixTransposeDispatcher<T, TileLongSide * 2, TileShortSide>::
          DoBatchNarrowMatrixTranspose(d, tile_size_i, tile_size_j,
                                       total_tiles_count, input, input_dims,
                                       output);
    } else {
      BatchNarrowMatrixTransposeDispatcher<T, TileLongSide, TileShortSide + 1>::
          DoBatchNarrowMatrixTranspose(d, tile_size_i, tile_size_j,
                                       total_tiles_count, input, input_dims,
                                       output);
    }
  }
};

#define BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_OVERALL(TYPE, LONG_SIDE,           \
                                                    SHORT_SIDE)                \
  template <int TileSizeI>                                                     \
  struct BatchNarrowMatrixTransposeDispatcher<TYPE, TileSizeI, SHORT_SIDE> {   \
    static void DoBatchNarrowMatrixTranspose(const GPUDevice& d,               \
                                             int tile_size_i, int tile_size_j, \
                                             int total_tiles_count,            \
                                             const TYPE* input,                \
                                             const Dimension<3>& input_dims,   \
                                             TYPE* output) {                   \
      assert(                                                                  \
          false &&                                                             \
          "BatchNarrowMatrixTransposeDispatcher has requested an unexpected "  \
          "launch configuration. ");                                           \
    }                                                                          \
  };                                                                           \
  template <int TileSizeJ>                                                     \
  struct BatchNarrowMatrixTransposeDispatcher<TYPE, LONG_SIDE, TileSizeJ> {    \
    static void DoBatchNarrowMatrixTranspose(const GPUDevice& d,               \
                                             int tile_size_i, int tile_size_j, \
                                             int total_tiles_count,            \
                                             const TYPE* input,                \
                                             const Dimension<3>& input_dims,   \
                                             TYPE* output) {                   \
      assert(                                                                  \
          false &&                                                             \
          "BatchNarrowMatrixTransposeDispatcher has requested an unexpected "  \
          "launch configuration. ");                                           \
    }                                                                          \
  };

#define BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, LONG_SIDE, SHORT_SIDE)       \
  template <>                                                                  \
  struct BatchNarrowMatrixTransposeDispatcher<TYPE, LONG_SIDE, SHORT_SIDE> {   \
    static void DoBatchNarrowMatrixTranspose(const GPUDevice& d,               \
                                             int tile_size_i, int tile_size_j, \
                                             int total_tiles_count,            \
                                             const TYPE* input,                \
                                             const Dimension<3>& input_dims,   \
                                             TYPE* output) {                   \
      const int ThreadNum = LONG_SIDE;                                         \
      if (tile_size_i <= LONG_SIDE && tile_size_j <= SHORT_SIDE)               \
        SwapDimension1And2InTensor3UsingTiles<                                 \
            TYPE, ThreadNum, LONG_SIDE,                                        \
            SHORT_SIDE><<<total_tiles_count, ThreadNum, 0, d.stream()>>>(      \
            input, input_dims, output);                                        \
      else if (tile_size_j <= LONG_SIDE && tile_size_i <= SHORT_SIDE)          \
        SwapDimension1And2InTensor3UsingTiles<                                 \
            TYPE, ThreadNum, SHORT_SIDE,                                       \
            LONG_SIDE><<<total_tiles_count, ThreadNum, 0, d.stream()>>>(       \
            input, input_dims, output);                                        \
      return;                                                                  \
    }                                                                          \
  };

#define BATCH_NARROW_MATRIX_TRANSPOSE_128(TYPE)               \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_OVERALL(TYPE, 256, 16); \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 32, 15);          \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 64, 15);          \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 128, 15);         \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 256, 2);

#define BATCH_NARROW_MATRIX_TRANSPOSE_64(TYPE)                \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_OVERALL(TYPE, 512, 16); \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 32, 15);          \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 64, 15);          \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 128, 15);         \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 256, 8);          \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 512, 2);

#define BATCH_NARROW_MATRIX_TRANSPOSE_32(TYPE)                 \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_OVERALL(TYPE, 1024, 16); \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 32, 15);           \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 64, 15);           \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 128, 15);          \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 256, 10);          \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 512, 4);           \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 1024, 2);

#define BATCH_NARROW_MATRIX_TRANSPOSE_16(TYPE)                 \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_OVERALL(TYPE, 1024, 16); \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 32, 15);           \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 64, 15);           \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 128, 15);          \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 256, 10);          \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 512, 4);           \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 1024, 2);

#define BATCH_NARROW_MATRIX_TRANSPOSE_8(TYPE)                  \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_OVERALL(TYPE, 1024, 16); \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 32, 15);           \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 64, 15);           \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 128, 15);          \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 256, 10);          \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 512, 4);           \
  BATCH_NARROW_MATRIX_TRANSPOSE_LIMIT_PER_LONG_SIDE_LEN(TYPE, 1024, 2);

BATCH_NARROW_MATRIX_TRANSPOSE_128(float4);
BATCH_NARROW_MATRIX_TRANSPOSE_64(double);
BATCH_NARROW_MATRIX_TRANSPOSE_64(uint64);
BATCH_NARROW_MATRIX_TRANSPOSE_32(float);
BATCH_NARROW_MATRIX_TRANSPOSE_32(uint32);
BATCH_NARROW_MATRIX_TRANSPOSE_16(Eigen::half);
BATCH_NARROW_MATRIX_TRANSPOSE_16(uint16);
BATCH_NARROW_MATRIX_TRANSPOSE_8(uint8);

// Launch the GPU kernel that would swap dimension-1 and dimension-2 in a
// 3D tensor. It looks at the shape of the incoming data, and decides the best
// strategy to launch.
template <typename T>
void RunSwapDimension1And2InTensor3(const GPUDevice& d, const T* input,
                                    const Dimension<3>& input_dims, T* output) {
  // If both dimensions are not trivial, use tiles for the actual swapping.
  // Otherwise, the trivial swapping relying on the ldg cache is more efficient.
  static const int kMinDimensionToUseTiles = 16;
  static const int kMinDimensionToUseRectTiles = 96;

  bool large_matrix = (input_dims[1] >= kMinDimensionToUseTiles &&
                       input_dims[2] >= kMinDimensionToUseTiles);
  bool narrow_matrix = (input_dims[1] >= kMinDimensionToUseRectTiles ||
                      input_dims[2] >= kMinDimensionToUseRectTiles);
  if (large_matrix) {
    // We get best performance when TileSize is the number of threads in a warp
    // (32 on our GPUs) and NumSubTiles is 8, so our block size is 8 * 32 = 256
    // threads.
    static const int TileSize = 32;
    static const int ThreadNum = 256;

    Dimension<3> input_dims_in_tiles = {
        input_dims[0], (input_dims[1] + TileSize - 1) / TileSize,
        (input_dims[2] + TileSize - 1) / TileSize,
    };

    int total_tiles_count = input_dims_in_tiles[0] * input_dims_in_tiles[1] *
                            input_dims_in_tiles[2];
    SwapDimension1And2InTensor3UsingTiles<
        T, ThreadNum, TileSize,
        TileSize><<<total_tiles_count, ThreadNum, 0, d.stream()>>>(
        input, input_dims, output);

  } else if (narrow_matrix) {
    // Define available tile sizes here for each size of data type supported:
    std::map<int, int> tile_spec_128 = {{32, 15}, {64, 15}, {128, 15},
                                        {256, 2}};
    std::map<int, int> tile_spec_64  = {{32, 15},  {64, 15}, {128, 15},
                                        {256, 8},  {512, 2}};
    std::map<int, int> tile_spec_32  = {{32, 15},  {64, 15}, {128, 15},
                                        {256, 10}, {512, 4}, {1024, 2}};
    std::map<int, int> tile_spec_16  = {{32, 15},  {64, 15}, {128, 15},
                                        {256, 10}, {512, 4}, {1024, 2}};
    std::map<int, int> tile_spec_8   = {{32, 15},  {64, 15}, {128, 15},
                                        {256, 10}, {512, 4}, {1024, 2}};

    // Organize these tile size specifications into a map that maps from data
    // type sizes to their specifications.
    std::map<int, std::map<int, int>> tile_spec_map = {{128, tile_spec_128},
                                                       {64, tile_spec_64},
                                                       {32, tile_spec_32},
                                                       {16, tile_spec_16},
                                                       {8, tile_spec_8}};

    std::map<int, int> tile_spec = tile_spec_map[8 * sizeof(T)];

    int tile_long_side_len = 0;
    int tile_short_side_len = 0;
    float lowest_cost = std::numeric_limits<float>::max();
    int data_long_side = max(input_dims[1], input_dims[2]);

    for (std::map<int, int>::iterator it = tile_spec.begin();
         it != tile_spec.end(); ++it) {
      int proposed_tile_long_side_len = it->first;

      // Threads that will not be doing anything useful when reading the matrix
      // because the thread block size is bigger than the data block size.
      float wasted_threads = (data_long_side -
                              data_long_side / proposed_tile_long_side_len *
                                  proposed_tile_long_side_len);
      int num_full_tiles = data_long_side / proposed_tile_long_side_len;

      float cost = 0;

      // However, if we can execute two or more full tiles, then we gladly
      // accept any number of wasted thread and ignore its cost.
      if (num_full_tiles <= 1) cost = wasted_threads;

      // Using less and equal here because given the same cost, we would like to
      // launch as many threads
      // as possible.
      if (cost <= lowest_cost) {
        tile_long_side_len = proposed_tile_long_side_len;
        tile_short_side_len = it->second;
        lowest_cost = cost;
      }
    }

    // Request tile sizes such that the longer side of threadblock align with
    // the longer side of input data block to maximize read throughput.
    // The ideal tile shape to request is one with its length of the shorter
    // side of the tile being equal to the length of the shorter side of the
    // input matrix.
    int requested_tile_size_i = input_dims[1] >= kMinDimensionToUseTiles
                                    ? tile_long_side_len
                                    : input_dims[1];
    int requested_tile_size_j = input_dims[1] >= kMinDimensionToUseTiles
                                    ? input_dims[2]
                                    : tile_long_side_len;

    // Truncate the shorter size requested according to the manual limit set in
    // tile_spec to make sure that we do not
    // launch configurations violating hardware limits.
    requested_tile_size_i =
        requested_tile_size_i == tile_long_side_len
            ? tile_long_side_len
            : min(requested_tile_size_i, tile_short_side_len);
    requested_tile_size_j =
        requested_tile_size_j == tile_long_side_len
            ? tile_long_side_len
            : min(requested_tile_size_j, tile_short_side_len);

    Dimension<3> input_dims_in_tiles = {
        input_dims[0],
        (input_dims[1] + requested_tile_size_i - 1) / requested_tile_size_i,
        (input_dims[2] + requested_tile_size_j - 1) / requested_tile_size_j,
    };

    int total_tiles_count = input_dims_in_tiles[0] * input_dims_in_tiles[1] *
                            input_dims_in_tiles[2];

    // We recusively search for the minimum pre-compiled configuration that
    // satisfies the requested tile sizes.
    BatchNarrowMatrixTransposeDispatcher<T, 32, 2>::DoBatchNarrowMatrixTranspose(
        d, requested_tile_size_i, requested_tile_size_j, total_tiles_count,
        input, input_dims, output);

  } else {
    int total_element_count = input_dims[0] * input_dims[1] * input_dims[2];
    CudaLaunchConfig config = GetCudaLaunchConfig(total_element_count, d);
    SwapDimension1And2InTensor3Simple<
        T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, input, input_dims, output);
  }
}

// A GPU helper functor that does general dimension 1 and 2 switch for 3D
// tensor.
template <typename T>
struct SwapDimension1And2InTensor3<GPUDevice, T> {
  typedef GPUDevice Device;
  void operator()(const Device& d, const T* in,
                  const gtl::ArraySlice<int64>& combined_dims, T* out) {
    Dimension<3> input_dims = {static_cast<int>(combined_dims[0]),
                               static_cast<int>(combined_dims[1]),
                               static_cast<int>(combined_dims[2])};
    RunSwapDimension1And2InTensor3(d, in, input_dims, out);
  }
};

// A GPU helper functor that does general dimension 0 and 2 switch for 3D
// tensor.
template <typename T>
struct SwapDimension0And2InTensor3<GPUDevice, T> {
  typedef GPUDevice Device;
  void operator()(const Device& d, const T* in,
                  const gtl::ArraySlice<int64>& combined_dims, T* out) {
    Dimension<3> input_dims = {static_cast<int>(combined_dims[0]),
                               static_cast<int>(combined_dims[1]),
                               static_cast<int>(combined_dims[2])};
    size_t total_size = combined_dims[0] * combined_dims[1] * combined_dims[2];
    CudaLaunchConfig config = GetCudaLaunchConfig(total_size, d);
    SwapDimension0And2InTensor3Simple<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            config.virtual_thread_count, in, input_dims, out);
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

template struct functor::ShuffleAndReverse<GPUDevice, float, 4, int>;
template struct functor::ShuffleAndReverse<GPUDevice, Eigen::half, 4, int>;

template struct functor::ShuffleAndReverse<GPUDevice, float, 4,
                                           Eigen::DenseIndex>;
template struct functor::ShuffleAndReverse<GPUDevice, Eigen::half, 4,
                                           Eigen::DenseIndex>;

template struct functor::TransformDepth<GPUDevice, float, int>;
template struct functor::TransformDepth<GPUDevice, Eigen::half, int>;

template struct functor::SwapDimension1And2InTensor3<GPUDevice, uint8>;
template struct functor::SwapDimension1And2InTensor3<GPUDevice, uint16>;
template struct functor::SwapDimension1And2InTensor3<GPUDevice, uint32>;
template struct functor::SwapDimension1And2InTensor3<GPUDevice, uint64>;
template struct functor::SwapDimension1And2InTensor3<GPUDevice, float4>;

template struct functor::SwapDimension0And2InTensor3<GPUDevice, uint8>;
template struct functor::SwapDimension0And2InTensor3<GPUDevice, uint16>;
template struct functor::SwapDimension0And2InTensor3<GPUDevice, uint32>;
template struct functor::SwapDimension0And2InTensor3<GPUDevice, uint64>;
template struct functor::SwapDimension0And2InTensor3<GPUDevice, float4>;

// For 2d ops.
template struct functor::TransformFilter<GPUDevice, float, int, 4>;
template struct functor::TransformFilter<GPUDevice, Eigen::half, int, 4>;

template struct functor::ReverseTransformFilter<GPUDevice, float, 4>;
template struct functor::ReverseTransformFilter<GPUDevice, Eigen::half, 4>;

template struct functor::NHWCToNCHW<GPUDevice, double, 4>;
template struct functor::NHWCToNCHW<GPUDevice, float, 4>;
template struct functor::NHWCToNCHW<GPUDevice, Eigen::half, 4>;

template struct functor::NCHWToNHWC<GPUDevice, double, 4>;
template struct functor::NCHWToNHWC<GPUDevice, float, 4>;
template struct functor::NCHWToNHWC<GPUDevice, Eigen::half, 4>;

template struct functor::PadInput<GPUDevice, int, int, 4>;
template struct functor::PadInput<GPUDevice, float, int, 4>;
template struct functor::PadInput<GPUDevice, Eigen::half, int, 4>;

// For 3d ops.
template struct functor::TransformFilter<GPUDevice, float, int, 5>;
template struct functor::TransformFilter<GPUDevice, Eigen::half, int, 5>;

template struct functor::ReverseTransformFilter<GPUDevice, float, 5>;
template struct functor::ReverseTransformFilter<GPUDevice, Eigen::half, 5>;

template struct functor::NHWCToNCHW<GPUDevice, float, 5>;
template struct functor::NHWCToNCHW<GPUDevice, Eigen::half, 5>;

template struct functor::NCHWToNHWC<GPUDevice, float, 5>;
template struct functor::NCHWToNHWC<GPUDevice, Eigen::half, 5>;

template struct functor::PadInput<GPUDevice, float, int, 5>;
template struct functor::PadInput<GPUDevice, Eigen::half, int, 5>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
