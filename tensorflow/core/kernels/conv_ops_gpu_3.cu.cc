/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

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
__global__ void SwapDimension0And2InTensor3(int nthreads, const T* input,
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
__global__ void SwapDimension1And2InTensor3(int nthreads, const T* input,
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
// TileSize could be arbitrary. But for best performance, it is better to be
// the same as number of threads in a warp, which is 32 for almost all GPU
// architectures.
template <typename T, int TileSize>
__global__ void SwapDimension1And2InTensor3UsingTiles(const T* input,
                                                      Dimension<3> input_dims,
                                                      T* output) {
  // One extra line in the inner dimension to avoid share memory bank conflict.
  __shared__ T shared_memory_tile[TileSize][TileSize + 1];

  int x = threadIdx.x;
  if (x >= TileSize) {
    return;
  }

  Dimension<3> output_dims = {
      input_dims[0], input_dims[2], input_dims[1],
  };

  Dimension<3> input_dims_in_tiles = {
      input_dims[0], (input_dims[1] + TileSize - 1) / TileSize,
      (input_dims[2] + TileSize - 1) / TileSize,
  };

  Index<3> input_tile_index =
      FlatToTensorIndex(blockIdx.x, input_dims_in_tiles);

  Index<3> input_tile_origin = {
      input_tile_index[0], input_tile_index[1] * TileSize,
      input_tile_index[2] * TileSize,
  };

  int input_origin_flat_index =
      TensorIndexToFlat(input_tile_origin, input_dims);

  int tile_width = TileSize;
  // Only the last row or column may not have the full size.
  if (input_tile_index[2] == input_dims_in_tiles[2] - 1) {
    tile_width = input_dims[2] - (input_dims_in_tiles[2] - 1) * TileSize;
  }
  int tile_height = TileSize;
  if (input_tile_index[1] == input_dims_in_tiles[1] - 1) {
    tile_height = input_dims[1] - (input_dims_in_tiles[1] - 1) * TileSize;
  }

  // Load the data from input memory to the shared memory tile.
  if (x < tile_width) {
    int input_flat_index = input_origin_flat_index + x;
    for (int y = 0; y < tile_height; y++) {
      shared_memory_tile[y][x] = input[input_flat_index];
      input_flat_index += input_dims[2];
    }
  }

  __syncthreads();

  Index<3> output_tile_index = {
      input_tile_index[0], input_tile_index[2], input_tile_index[1],
  };

  Index<3> output_tile_origin = {
      output_tile_index[0], output_tile_index[1] * TileSize,
      output_tile_index[2] * TileSize,
  };

  int output_origin_flat_index =
      TensorIndexToFlat(output_tile_origin, output_dims);

  int output_flat_index = output_origin_flat_index + x;

  // Load the data from the shared memory tile to the output memory.
  if (x < tile_height) {
    for (int y = 0; y < tile_width; y++) {
      output[output_flat_index] = shared_memory_tile[x][y];
      output_flat_index += output_dims[2];
    }
  }
}

// A Cuda custom kernel that convert input to output, given proper padding on
// the left and the top. The padded value is zero.
template <typename T>
__global__ void PadInputCustomKernelNHWC(int nthreads, const T* input,
                                         Dimension<4> input_dims, T* output,
                                         Dimension<4> output_dims,
                                         int padding_rows_left,
                                         int padding_cols_left) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int output_index = index;
    Index<4> output_tensor_index = FlatToTensorIndex(output_index, output_dims);

    Index<4> input_tensor_index;
    input_tensor_index[0] = output_tensor_index[0];
    input_tensor_index[1] = output_tensor_index[1] - padding_rows_left;
    input_tensor_index[2] = output_tensor_index[2] - padding_cols_left;
    input_tensor_index[3] = output_tensor_index[3];

    if (input_tensor_index[1] >= 0 && input_tensor_index[1] < input_dims[1] &&
        input_tensor_index[2] >= 0 && input_tensor_index[2] < input_dims[2]) {
      int input_index = TensorIndexToFlat(input_tensor_index, input_dims);
      output[output_index] = input[input_index];
    } else {
      output[output_index] = T(0);
    }
  }
}

template <typename T>
__global__ void PadInputCustomKernelNCHW(int nthreads, const T* input,
                                         Dimension<4> input_dims, T* output,
                                         Dimension<4> output_dims,
                                         int padding_rows_left,
                                         int padding_cols_left) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int output_index = index;
    Index<4> output_tensor_index = FlatToTensorIndex(output_index, output_dims);

    Index<4> input_tensor_index;
    input_tensor_index[0] = output_tensor_index[0];
    input_tensor_index[1] = output_tensor_index[1];
    input_tensor_index[2] = output_tensor_index[2] - padding_rows_left;
    input_tensor_index[3] = output_tensor_index[3] - padding_cols_left;

    if (input_tensor_index[2] >= 0 && input_tensor_index[2] < input_dims[2] &&
        input_tensor_index[3] >= 0 && input_tensor_index[3] < input_dims[3]) {
      int input_index = TensorIndexToFlat(input_tensor_index, input_dims);
      output[output_index] = input[input_index];
    } else {
      output[output_index] = T(0);
    }
  }
}

// A GPU helper function that converts TensorFlow filter format to Cudnn filter
// format.
template <typename T>
struct TransformFilter<GPUDevice, T, int> {
  typedef GPUDevice Device;
  void operator()(const Device& d, typename TTypes<T, 4, int>::ConstTensor in,
                  typename TTypes<T, 4, int>::Tensor out) {
    Dimension<3> combined_dims;
    combined_dims[0] = in.dimension(0) * in.dimension(1);
    combined_dims[1] = in.dimension(2);
    combined_dims[2] = in.dimension(3);
    CudaLaunchConfig config = GetCudaLaunchConfig(out.size(), d);
    SwapDimension0And2InTensor3<
        T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, in.data(), combined_dims, out.data());
  }
};

// Converts Cudnn filter format back to TensorFlow filter format.
template <typename T>
struct ReverseTransformFilter<GPUDevice, T> {
  typedef GPUDevice Device;
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor in,
                  typename TTypes<T, 4>::Tensor out) {
    Dimension<3> combined_dims;
    combined_dims[0] = in.dimension(0);
    combined_dims[1] = in.dimension(1);
    combined_dims[2] = in.dimension(2) * in.dimension(3);
    CudaLaunchConfig config = GetCudaLaunchConfig(out.size(), d);
    SwapDimension0And2InTensor3<
        T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, in.data(), combined_dims, out.data());
  }
};

// A GPU helper function that converts input tensor to a larger output tensor,
// given proper padding values. The padded value is zero.
template <typename T>
struct PadInput<GPUDevice, T, int> {
  typedef GPUDevice Device;
  void operator()(const Device& d, typename TTypes<T, 4, int>::ConstTensor in,
                  int padding_rows_left, int padding_rows_right,
                  int padding_cols_left, int padding_cols_right,
                  typename TTypes<T, 4, int>::Tensor out, TensorFormat format) {
    CudaLaunchConfig config = GetCudaLaunchConfig(out.size(), d);
    Dimension<4> input_dims;
    for (int i = 0; i < 4; i++) {
      input_dims[i] = in.dimension(i);
    }
    Dimension<4> output_dims;
    for (int i = 0; i < 4; i++) {
      output_dims[i] = out.dimension(i);
    }

    if (format == FORMAT_NHWC) {
      PadInputCustomKernelNHWC<
          T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          config.virtual_thread_count, in.data(), input_dims, out.data(),
          output_dims, padding_rows_left, padding_cols_left);
    } else if (format == FORMAT_NCHW) {
      PadInputCustomKernelNCHW<
          T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          config.virtual_thread_count, in.data(), input_dims, out.data(),
          output_dims, padding_rows_left, padding_cols_left);
    } else {
      LOG(FATAL) << "Invalid data format: " << format;
    }
  }
};

// Launch the GPU kernel that would swap dimension-1 and dimension-2 in a
// 3D tensor. It looks at the shape of the incoming data, and decides the best
// strategy to launch.
template <typename T>
void RunSwapDimension1And2InTensor3(const GPUDevice& d, const T* input,
                                    const Dimension<3>& input_dims, T* output) {
  // If both dimensions are not trivial, use tiles for the actual swapping.
  // Otherwise, the trivial swapping relying on the ldg cache is more efficient.
  static const int kMinDimensionToUseTiles = 16;
  bool use_tiles = (input_dims[1] >= kMinDimensionToUseTiles &&
                    input_dims[2] >= kMinDimensionToUseTiles);
  if (use_tiles) {
    // The tile-size can be chosen to be arbitrary number. But it is better to
    // be the same as number of threads in a warp, which is 32.
    static const int TileSize = 32;
    Dimension<3> input_dims_in_tiles = {
        input_dims[0], (input_dims[1] + TileSize - 1) / TileSize,
        (input_dims[2] + TileSize - 1) / TileSize,
    };
    int total_tiles_count = input_dims_in_tiles[0] * input_dims_in_tiles[1] *
                            input_dims_in_tiles[2];
    SwapDimension1And2InTensor3UsingTiles<
        T, TileSize><<<total_tiles_count, TileSize, 0, d.stream()>>>(
        input, input_dims, output);
  } else {
    int total_element_count = input_dims[0] * input_dims[1] * input_dims[2];
    CudaLaunchConfig config = GetCudaLaunchConfig(total_element_count, d);
    SwapDimension1And2InTensor3<
        T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, input, input_dims, output);
  }
}

// A GPU helper functor that converts NHWC TensorFlow data format to
// NCHW format that is accepted by Cudnn.
template <typename T>
struct NHWCToNCHW<GPUDevice, T> {
  typedef GPUDevice Device;
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor in,
                  typename TTypes<T, 4>::Tensor out) {
    Dimension<3> combined_dims;
    combined_dims[0] = in.dimension(0);
    combined_dims[1] = in.dimension(1) * in.dimension(2);
    combined_dims[2] = in.dimension(3);
    RunSwapDimension1And2InTensor3(d, in.data(), combined_dims, out.data());
  }
};

// A GPU helper functor that converts NCHW Cudnn data format to NHWC TensorFlow
// Format.
template <typename T>
struct NCHWToNHWC<GPUDevice, T> {
  typedef GPUDevice Device;
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor in,
                  typename TTypes<T, 4>::Tensor out) {
    Dimension<3> combined_dims;
    combined_dims[0] = in.dimension(0);
    combined_dims[1] = in.dimension(1);
    combined_dims[2] = in.dimension(2) * in.dimension(3);
    RunSwapDimension1And2InTensor3(d, in.data(), combined_dims, out.data());
  }
};

}  // namespace functor

template struct functor::ShuffleAndReverse<GPUDevice, float, 4, int>;

template struct functor::ShuffleAndReverse<GPUDevice, float, 4,
                                           Eigen::DenseIndex>;

template struct functor::TransformFilter<GPUDevice, float, int>;

template struct functor::ReverseTransformFilter<GPUDevice, float>;

template struct functor::PadInput<GPUDevice, float, int>;

template struct functor::TransformDepth<GPUDevice, float, int>;

template struct functor::NHWCToNCHW<GPUDevice, float>;

template struct functor::NCHWToNHWC<GPUDevice, float>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
