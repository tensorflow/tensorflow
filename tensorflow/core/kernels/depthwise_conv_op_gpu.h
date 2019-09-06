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

#ifndef TENSORFLOW_CORE_KERNELS_DEPTHWISE_CONV_OP_GPU_H_
#define TENSORFLOW_CORE_KERNELS_DEPTHWISE_CONV_OP_GPU_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/depthwise_conv_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/tensor_format.h"

#if defined(_MSC_VER) && !defined(__clang__)
#define UNROLL
#define NOUNROLL
#else
#define UNROLL _Pragma("unroll")
#define NOUNROLL _Pragma("nounroll")
#endif

namespace tensorflow {

namespace detail {
template <typename T>
struct PseudoHalfType {
  using Type = T;
};
template <>
struct PseudoHalfType<Eigen::half> {
  using Type = float;
};
}  // namespace detail

using Eigen::GpuDevice;

// Returns whether depthwise convolution forward or backward input pass can be
// performed using the faster ('Small') variant of the kernel.
inline EIGEN_DEVICE_FUNC bool CanLaunchDepthwiseConv2dGPUSmall(
    const DepthwiseArgs& args) {
  return args.depth_multiplier == 1 && args.stride == 1 && args.in_rows <= 32 &&
         args.in_cols <= 32 && args.in_rows == args.out_rows &&
         args.in_cols == args.out_cols && args.pad_rows >= 0 &&
         args.pad_rows < args.filter_rows && args.pad_cols >= 0 &&
         args.pad_cols < args.filter_cols &&
         args.filter_rows * args.filter_cols <=
             (args.in_rows + 1) / 2 * args.in_cols;
}

// Returns whether depthwise convolution backward filter pass can be performed
// using the faster ('Small') variant of the kernel.
inline EIGEN_DEVICE_FUNC bool CanLaunchDepthwiseConv2dBackpropFilterGPUSmall(
    const DepthwiseArgs& args, const int block_height) {
  return args.depth_multiplier == 1 && args.stride == 1 && args.in_rows <= 32 &&
         args.in_cols <= 32 && args.in_rows == args.out_rows &&
         args.in_cols == args.out_cols && args.pad_rows >= 0 &&
         args.pad_rows < args.filter_rows && args.pad_cols >= 0 &&
         args.pad_cols < args.filter_cols && block_height <= args.in_rows &&
         args.filter_rows * args.filter_cols <= args.in_cols * block_height;
}

// The DepthwiseConv2dGPUKernels perform either forward or backprop input
// convolution depending on a template argument of this enum.
enum DepthwiseConv2dDirection { DIRECTION_FORWARD, DIRECTION_BACKWARD };

// A GPU kernel to compute the depthwise convolution forward pass
// in NHWC format.
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(1024, 2)
    DepthwiseConv2dGPUKernelNHWC(const DepthwiseArgs args,
                                 const T* __restrict__ input,
                                 const T* __restrict__ filter,
                                 T* __restrict__ output, int num_outputs) {
  typedef typename detail::PseudoHalfType<T>::Type S;
  const int in_height = args.in_rows;
  const int in_width = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_height =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_width =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int depth_multiplier =
      kKnownDepthMultiplier < 0 ? args.depth_multiplier : kKnownDepthMultiplier;
  const int stride = args.stride;
  const int pad_height = args.pad_rows;
  const int pad_width = args.pad_cols;
  const int out_height = args.out_rows;
  const int out_width = args.out_cols;
  const int out_depth = args.out_depth;

  GPU_1D_KERNEL_LOOP(thread_id, num_outputs) {
    // Compute the indexes of this thread in the output.
    const int out_channel = thread_id % out_depth;
    const int out_col = (thread_id / out_depth) % out_width;
    const int out_row = (thread_id / out_depth / out_width) % out_height;
    const int batch = thread_id / out_depth / out_width / out_height;
    // Compute the input depth and the index of depth multiplier.
    const int in_channel = out_channel / depth_multiplier;
    const int multiplier = out_channel % depth_multiplier;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int input_row_start = out_row * stride - pad_height;
    const int input_col_start = out_col * stride - pad_width;
    const int input_row_end = input_row_start + filter_height;
    const int input_col_end = input_col_start + filter_width;

    S sum = static_cast<S>(0);

    const int input_offset_temp = in_height * batch;
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_row_end < in_height && input_col_end < in_width) {
      UNROLL for (int filter_row = 0; filter_row < filter_height;
                  ++filter_row) {
        const int in_row = input_row_start + filter_row;
        const int filter_offset_temp = filter_width * filter_row;
        UNROLL for (int filter_col = 0; filter_col < filter_width;
                    ++filter_col) {
          const int in_col = input_col_start + filter_col;

          const int input_offset =
              in_channel +
              in_depth * (in_col + in_width * (in_row + input_offset_temp));
          const int filter_offset =
              multiplier +
              depth_multiplier *
                  (in_channel + in_depth * (filter_col + filter_offset_temp));
          sum += static_cast<S>(ldg(input + input_offset)) *
                 static_cast<S>(ldg(filter + filter_offset));
        }
      }
    } else {
      UNROLL for (int filter_row = 0; filter_row < filter_height;
                  ++filter_row) {
        const int in_row = input_row_start + filter_row;
        const int filter_offset_temp = filter_width * filter_row;
        UNROLL for (int filter_col = 0; filter_col < filter_width;
                    ++filter_col) {
          const int in_col = input_col_start + filter_col;
          if (in_row >= 0 && in_row < in_height && in_col >= 0 &&
              in_col < in_width) {
            const int in_col = input_col_start + filter_col;

            const int input_offset =
                in_channel +
                in_depth * (in_col + in_width * (in_row + input_offset_temp));
            const int filter_offset =
                multiplier +
                depth_multiplier *
                    (in_channel + in_depth * (filter_col + filter_offset_temp));
            sum += static_cast<S>(ldg(input + input_offset)) *
                   static_cast<S>(ldg(filter + filter_offset));
          }
        }
      }
    }
    output[thread_id] = static_cast<T>(sum);
  }
}

// CUDA kernel to compute the depthwise convolution forward pass in NHWC format,
// tailored for small images up to 32x32. Stride and depth multiplier must be 1.
// Padding must be 'SAME', which allows to reuse the index computation. Only
// use this kernel if CanLaunchDepthwiseConv2dGPUSmall(args) returns true.
// Tiles of the input and filter tensors are loaded into shared memory before
// performing the convolution. Each thread handles two elements per iteration,
// one each in the lower and upper half of a tile.
// Backprop input direction is the same as forward direction with the filter
// rotated by 180°.
// T is the tensors' data type. S is the math type the kernel uses. This is the
// same as T for all cases but pseudo half (which has T=Eigen::half, S=float).
template <typename T, DepthwiseConv2dDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight, int kBlockDepth,
          bool kKnownEvenHeight>
__global__ __launch_bounds__(1024, 2) void DepthwiseConv2dGPUKernelNHWCSmall(
    const DepthwiseArgs args, const T* __restrict__ input,
    const T* __restrict__ filter, T* __restrict__ output) {
  typedef typename detail::PseudoHalfType<T>::Type S;
  assert(CanLaunchDepthwiseConv2dGPUSmall(args));
  // Holds block plus halo and filter data for blockDim.x depths.
  GPU_DYNAMIC_SHARED_MEM_DECL(8, unsigned char, shared_memory);
  static_assert(sizeof(S) <= 8, "Insufficient alignment detected");
  S* const shared_data = reinterpret_cast<S*>(shared_memory);

  const int num_batches = args.batch;
  const int in_height = args.in_rows;
  const int in_width = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_height =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_width =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int pad_height = args.pad_rows;
  const int pad_width = args.pad_cols;

  assert(blockDim.x == kBlockDepth);
  assert(blockDim.y == args.in_cols);
  const int block_height = blockDim.z;

  // These values are the same for all threads and could
  // be precomputed on the CPU.
  const int block_size = block_height * in_width * kBlockDepth;
  const int in_row_size = in_width * in_depth;
  const int in_size = in_height * in_row_size;
  const int in_increment = (in_width - 1) * kBlockDepth;
  const int filter_pixels = filter_height * filter_width;
  const int tile_width = in_width + filter_width - 1;
  const int even_height = kKnownEvenHeight || (1 & ~in_height);
  const int tile_height = in_height + filter_height - even_height;
  const int tile_row_size = tile_width * kBlockDepth;
  const int tile_size = tile_height * tile_row_size;
  const int tile_offset = block_height * tile_row_size;
  const int pad_offset = pad_height * tile_width + pad_width;
  const int batch_blocks = (in_depth + kBlockDepth - 1) / kBlockDepth;
  const int in_blocks = batch_blocks * num_batches;
  const int tensor_offset =
      kKnownEvenHeight ? in_size / 2 : block_height * in_row_size;

  const int thread_depth = threadIdx.x;
  const int thread_col = threadIdx.y;
  const int thread_row = threadIdx.z;

  // Position in block.
  const int thread_pix = thread_row * in_width + thread_col;
  const int thread_idx = thread_pix * kBlockDepth + thread_depth;

  // Initialize tile, in particular the padding.
  for (int i = thread_idx; i < tile_size; i += block_size) {
    shared_data[i] = S();
  }
  __syncthreads();

  // Position in tensors.
  const int tensor_idx = thread_pix * in_depth + thread_depth;

  // Position in (padded) shared memory.
  const int data_pix = thread_row * tile_width + thread_col;
  const int data_idx = data_pix * kBlockDepth + thread_depth;

  // Position in shared memory, offset by pad_height / pad_width.
  const int tile_pix = data_pix + pad_offset;
  const int tile_idx = tile_pix * kBlockDepth + thread_depth;

  const int max_channel = in_depth - thread_depth;
  const int filter_write_offset =
      thread_pix < filter_pixels ? tile_size + thread_idx : 0;
  const int filter_read_offset =
      tile_size + thread_depth +
      (kDirection == DIRECTION_FORWARD ? 0 : filter_pixels * kBlockDepth);
  const bool skip_second =
      !kKnownEvenHeight && thread_row + (in_height & 1) == block_height;

  for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
    const int batch = b / batch_blocks;
    const int block = b - batch * batch_blocks;

    const int start_channel = block * kBlockDepth;
    const int filter_offset = tensor_idx + start_channel;
    const int inout_offset = batch * in_size + filter_offset;
    const bool channel_in_range = start_channel < max_channel;

    if (channel_in_range) {
      const T* const in_ptr = inout_offset + input;
      S* const tile_ptr = tile_idx + shared_data;
      tile_ptr[0] = static_cast<S>(ldg(in_ptr));
      if (!skip_second) {
        tile_ptr[tile_offset] = static_cast<S>(ldg(tensor_offset + in_ptr));
      }

      if (filter_write_offset != 0) {
        shared_data[filter_write_offset] =
            static_cast<S>(ldg(filter_offset + filter));
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();

    if (channel_in_range) {
      S sum1 = S();
      S sum2 = S();
      int shared_offset = data_idx;
      const S* filter_ptr = filter_read_offset + shared_data;
      UNROLL for (int r = 0; r < filter_height; ++r) {
        UNROLL for (int c = 0; c < filter_width; ++c) {
          if (kDirection == DIRECTION_BACKWARD) {
            filter_ptr -= kBlockDepth;
          }
          const S filter_value = *filter_ptr;
          const S* const tile_ptr = shared_offset + shared_data;
          sum1 += filter_value * tile_ptr[0];
          sum2 += filter_value * tile_ptr[tile_offset];
          shared_offset += kBlockDepth;
          if (kDirection == DIRECTION_FORWARD) {
            filter_ptr += kBlockDepth;
          }
        }
        shared_offset += in_increment;
      }
      T* const out_ptr = inout_offset + output;
      out_ptr[0] = static_cast<T>(sum1);
      if (!skip_second) {
        out_ptr[tensor_offset] = static_cast<T>(sum2);
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();
  }
}

// A GPU kernel to compute the depthwise convolution forward pass
// in NCHW format.
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(1024, 2)
    DepthwiseConv2dGPUKernelNCHW(const DepthwiseArgs args,
                                 const T* __restrict__ input,
                                 const T* __restrict__ filter,
                                 T* __restrict__ output, int num_outputs) {
  typedef typename detail::PseudoHalfType<T>::Type S;
  const int in_height = args.in_rows;
  const int in_width = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_height =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_width =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int depth_multiplier =
      kKnownDepthMultiplier < 0 ? args.depth_multiplier : kKnownDepthMultiplier;
  const int stride = args.stride;
  const int pad_height = args.pad_rows;
  const int pad_width = args.pad_cols;
  const int out_height = args.out_rows;
  const int out_width = args.out_cols;
  const int out_depth = args.out_depth;

  GPU_1D_KERNEL_LOOP(thread_id, num_outputs) {
    // Compute the indexes of this thread in the output.
    //
    // We want coalesced reads so we make sure that each warp reads
    // a contiguous chunk of memory.
    //
    // THIS IS PROBABLY WRONG, we are not doing coalesced reads
    // into the input, because of the depth multiplier division...
    const int out_col = thread_id % out_width;
    const int out_row = (thread_id / out_width) % out_height;
    const int out_channel = (thread_id / out_width / out_height) % out_depth;
    const int batch = thread_id / out_width / out_height / out_depth;

    // Compute the input depth and the index of depth multiplier
    // based off the output depth index that this thread is
    // computing n.
    const int in_channel = out_channel / depth_multiplier;
    const int multiplier = out_channel % depth_multiplier;

    // Data is stored in the following format (let's assume we
    // flatten the height and width into one contiguous dimension
    // called "P".
    //
    // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
    //
    // Each row contains in_depth * in_height * in_width values
    // for each sample in the batch.
    //
    // We can further flatten it into:
    //
    // B1C1P1 B1C1P2 .....
    // B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 .....
    // B2C2P1 B2C2P2 ....
    //
    // where each row is a contiguous array of all of the spatial
    // pixels for a given batch and input depth.  The following
    // loop unrolls across the filter dimensions for a given thread,
    // indexing into the filter value and the corresponding input
    // patch.
    //
    // We can compute the index into the patch once right here.
    const int input_offset_temp =
        (batch * in_depth + in_channel) * (in_height * in_width);

    // Finally, we can iterate over the spatial dimensions and perform the
    // convolution, writing into the output at the end.
    //
    // We perform an additional optimization, where we can determine
    // whether the patch fits within the image indices statically, and
    // avoid boundary checking within the loop.
    const int input_row_start = out_row * stride - pad_height;
    const int input_col_start = out_col * stride - pad_width;
    const int input_row_end = input_row_start + filter_height;
    const int input_col_end = input_col_start + filter_width;

    S sum = static_cast<S>(0);
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_row_end < in_height && input_col_end < in_width) {
      // Loop that doesn't need to check for boundary conditions.
      UNROLL for (int filter_row = 0; filter_row < filter_height;
                  ++filter_row) {
        const int in_row = input_row_start + filter_row;
        const int filter_offset_temp = filter_width * filter_row;
        UNROLL for (int filter_col = 0; filter_col < filter_width;
                    ++filter_col) {
          const int in_col = input_col_start + filter_col;

          const int input_offset =
              (input_offset_temp) + (in_row * in_width) + in_col;
          const int filter_offset =
              multiplier +
              depth_multiplier *
                  (in_channel + in_depth * (filter_col + filter_offset_temp));
          sum += static_cast<S>(ldg(input + input_offset)) *
                 static_cast<S>(ldg(filter + filter_offset));
        }
      }
    } else {
      // Loop that needs to check for boundary conditions.
      UNROLL for (int filter_row = 0; filter_row < filter_height;
                  ++filter_row) {
        const int in_row = input_row_start + filter_row;
        const int filter_offset_temp = filter_width * filter_row;
        UNROLL for (int filter_col = 0; filter_col < filter_width;
                    ++filter_col) {
          const int in_col = input_col_start + filter_col;
          // TODO(vrv): the in_row check can be done outside of this loop;
          // benchmark both methods to determine the better decision.
          if (in_row >= 0 && in_row < in_height && in_col >= 0 &&
              in_col < in_width) {
            const int in_col = input_col_start + filter_col;

            // input_offset_temp indexes into the start of memory
            // where the spatial data starts.
            const int input_offset =
                (input_offset_temp) + (in_row * in_width) + in_col;

            const int filter_offset =
                multiplier +
                depth_multiplier *
                    (in_channel + in_depth * (filter_col + filter_offset_temp));
            sum += static_cast<S>(ldg(input + input_offset)) *
                   static_cast<S>(ldg(filter + filter_offset));
          }
        }
      }
    }

    output[thread_id] = static_cast<T>(sum);
  }
}

// CUDA kernel to compute the depthwise convolution forward pass in NCHW format,
// tailored for small images up to 32x32. Stride and depth multiplier must be 1.
// Padding must be 'SAME', which allows to reuse the index computation. Only
// use this kernel if CanLaunchDepthwiseConv2dGPUSmall(args) returns true.
// Tiles of the input and filter tensors are loaded into shared memory before
// performing the convolution. Each thread handles two elements per iteration,
// one each in the lower and upper half of a tile.
// Backprop input direction is the same as forward direction with the filter
// rotated by 180°.
// T is the tensors' data type. S is the math type the kernel uses. This is the
// same as T for all cases but pseudo half (which has T=Eigen::half, S=float).
template <typename T, DepthwiseConv2dDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight, int kBlockDepth,
          bool kKnownEvenHeight>
__global__ __launch_bounds__(1024, 2) void DepthwiseConv2dGPUKernelNCHWSmall(
    const DepthwiseArgs args, const T* __restrict__ input,
    const T* __restrict__ filter, T* __restrict__ output) {
  typedef typename detail::PseudoHalfType<T>::Type S;
  assert(CanLaunchDepthwiseConv2dGPUSmall(args));
  // Holds block plus halo and filter data for blockDim.z depths.
  GPU_DYNAMIC_SHARED_MEM_DECL(8, unsigned char, shared_memory);
  static_assert(sizeof(S) <= 8, "Insufficient alignment detected");
  S* const shared_data = reinterpret_cast<S*>(shared_memory);

  const int num_batches = args.batch;
  const int in_height = args.in_rows;
  const int in_width = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_height =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_width =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int pad_height = args.pad_rows;
  const int pad_width = args.pad_cols;

  // Fixed blockDim.z, tailored for maximum grid size for images of size 16x16.
  assert(blockDim.x == args.in_cols);
  assert(blockDim.z == kBlockDepth);
  const int block_height = blockDim.y;

  // These values are the same for all threads and could
  // be precomputed on the CPU.
  const int block_pixels = in_width * block_height;
  const int block_size = block_pixels * kBlockDepth;
  const int in_pixels = in_width * in_height;
  const int in_increment = in_width - 1;
  const int filter_pixels = filter_height * filter_width;
  const int tile_width = in_width + filter_width - 1;
  const int even_height = kKnownEvenHeight || (1 & ~in_height);
  const int tile_height = in_height + filter_height - even_height;
  const int tile_pixels = tile_width * tile_height;
  const int tile_size = tile_pixels * kBlockDepth;
  const int tile_offset = block_height * tile_width;
  const int pad_offset = pad_height * tile_width + pad_width;
  const int in_total_depth = in_depth * num_batches;
  const int in_blocks = (in_total_depth + kBlockDepth - 1) / kBlockDepth;

  const int thread_col = threadIdx.x;
  const int thread_row = threadIdx.y;
  const int thread_depth = threadIdx.z;

  // Position in block.
  const int thread_pix = thread_row * in_width + thread_col;
  const int thread_idx = thread_depth * block_pixels + thread_pix;

  // Initialize tile, in particular the padding.
  for (int i = thread_idx; i < tile_size; i += block_size) {
    shared_data[i] = S();
  }
  __syncthreads();

  // Position in tensors.
  const int tensor_idx = thread_depth * in_pixels + thread_pix;

  // Position in (padded) shared memory.
  const int data_pix = thread_row * tile_width + thread_col;
  const int data_idx = thread_depth * tile_pixels + data_pix;

  // Position in shared memory, offset by pad_height / pad_width.
  const int tile_idx = data_idx + pad_offset;

  // Filter is always in HWCK format, irrespective of the input/output format.
  const int filter_pix = thread_idx / kBlockDepth;
  const int filter_channel = thread_idx % kBlockDepth;
  const int filter_idx = filter_pix * in_depth;

  const int max_channel = in_total_depth - thread_depth;
  const int filter_write_offset =
      filter_pix < filter_pixels ? tile_size + thread_idx : 0;
  const int filter_read_offset =
      tile_size + thread_depth +
      (kDirection == DIRECTION_FORWARD ? 0 : filter_pixels * kBlockDepth);
  const bool skip_second =
      !kKnownEvenHeight && thread_row + (in_height & 1) == block_height;

  for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
    const int channel = b * kBlockDepth;

    const int inout_offset = channel * in_pixels + tensor_idx;
    const bool channel_in_range = channel < max_channel;

    if (channel_in_range) {
      const T* const in_ptr = inout_offset + input;
      S* const tile_ptr = tile_idx + shared_data;
      tile_ptr[0] = static_cast<S>(ldg(in_ptr));
      if (!skip_second) {
        tile_ptr[tile_offset] = static_cast<S>(ldg(block_pixels + in_ptr));
      }
    }

    if (filter_write_offset != 0) {
      const int filter_offset =
          filter_idx + (channel + filter_channel) % in_depth;
      shared_data[filter_write_offset] =
          static_cast<S>(ldg(filter_offset + filter));
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();

    if (channel_in_range) {
      S sum1 = S();
      S sum2 = S();
      int shared_offset = data_idx;
      const S* filter_ptr = filter_read_offset + shared_data;
      UNROLL for (int r = 0; r < filter_height; ++r) {
        UNROLL for (int c = 0; c < filter_width; ++c) {
          if (kDirection == DIRECTION_BACKWARD) {
            filter_ptr -= kBlockDepth;
          }
          const S filter_value = *filter_ptr;
          const S* const tile_ptr = shared_offset + shared_data;
          sum1 += filter_value * tile_ptr[0];
          sum2 += filter_value * tile_ptr[tile_offset];
          ++shared_offset;
          if (kDirection == DIRECTION_FORWARD) {
            filter_ptr += kBlockDepth;
          }
        }
        shared_offset += in_increment;
      }
      T* const out_ptr = inout_offset + output;
      out_ptr[0] = static_cast<T>(sum1);
      if (!skip_second) {
        out_ptr[block_pixels] = static_cast<T>(sum2);
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();
  }
}

template <typename T, DepthwiseConv2dDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight, int kBlockDepth,
          bool kKnownEvenHeight>
Status LaunchDepthwiseConv2dGPUSmall(OpKernelContext* ctx,
                                     const DepthwiseArgs& args, const T* input,
                                     const T* filter, T* output,
                                     TensorFormat data_format) {
  typedef typename detail::PseudoHalfType<T>::Type S;
  const int block_height = (args.in_rows + 1) / 2;
  dim3 block_dim;
  int block_count;
  void (*kernel)(const DepthwiseArgs, const T*, const T*, T*);
  switch (data_format) {
    case FORMAT_NHWC:
      block_dim = dim3(kBlockDepth, args.in_cols, block_height);
      block_count =
          args.batch * DivUp(args.out_depth, kBlockDepth) * kBlockDepth;
      kernel =
          DepthwiseConv2dGPUKernelNHWCSmall<T, kDirection, kKnownFilterWidth,
                                            kKnownFilterHeight, kBlockDepth,
                                            kKnownEvenHeight>;
      break;
    case FORMAT_NCHW:
      block_dim = dim3(args.in_cols, block_height, kBlockDepth);
      block_count =
          DivUp(args.batch * args.out_depth, kBlockDepth) * kBlockDepth;
      kernel =
          DepthwiseConv2dGPUKernelNCHWSmall<T, kDirection, kKnownFilterWidth,
                                            kKnownFilterHeight, kBlockDepth,
                                            kKnownEvenHeight>;
      break;
    default:
      return errors::InvalidArgument("FORMAT_", ToString(data_format),
                                     " is not supported");
  }
  const int tile_width = args.in_cols + args.filter_cols - 1;
  const int tile_height = block_height * 2 + args.filter_rows - 1;
  const int tile_pixels = tile_height * tile_width;
  const int filter_pixels = args.filter_rows * args.filter_cols;
  const int shared_memory_size =
      kBlockDepth * (tile_pixels + filter_pixels) * sizeof(S);
  const int num_outputs = args.out_rows * args.out_cols * block_count;
  auto device = ctx->eigen_gpu_device();
  GpuLaunchConfig config = GetGpuLaunchConfigFixedBlockSize(
      num_outputs, device, kernel, shared_memory_size,
      block_dim.x * block_dim.y * block_dim.z);
  TF_CHECK_OK(GpuLaunchKernel(kernel, config.block_count, block_dim,
                              shared_memory_size, device.stream(), args, input,
                              filter, output));
  return Status::OK();
}

// Returns whether the context's GPU supports efficient fp16 math.
inline bool HasFastHalfMath(OpKernelContext* ctx) {
  int major, minor;
  ctx->op_device_context()
      ->stream()
      ->parent()
      ->GetDeviceDescription()
      .cuda_compute_capability(&major, &minor);
  auto cuda_arch = major * 100 + minor * 10;
  // GPUs before sm_53 don't support fp16 math, and sm_61's fp16 math is slow.
  return cuda_arch >= 530 && cuda_arch != 610;
}

template <typename T, DepthwiseConv2dDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight, int kBlockDepth>
Status LaunchDepthwiseConv2dGPUSmall(OpKernelContext* ctx,
                                     const DepthwiseArgs& args, const T* input,
                                     const T* filter, T* output,
                                     TensorFormat data_format) {
  if (args.in_rows & 1) {
    return LaunchDepthwiseConv2dGPUSmall<T, kDirection, kKnownFilterWidth,
                                         kKnownFilterHeight, kBlockDepth,
                                         false>(ctx, args, input, filter,
                                                output, data_format);
  } else {
    return LaunchDepthwiseConv2dGPUSmall<T, kDirection, kKnownFilterWidth,
                                         kKnownFilterHeight, kBlockDepth, true>(
        ctx, args, input, filter, output, data_format);
  }
}

template <typename T, DepthwiseConv2dDirection kDirection,
          int kKnownFilterWidth, int kKnownFilterHeight>
Status LaunchDepthwiseConv2dGPUSmall(OpKernelContext* ctx,
                                     const DepthwiseArgs& args, const T* input,
                                     const T* filter, T* output,
                                     TensorFormat data_format) {
  // Maximize (power of two) kBlockDepth while keeping a block within 1024
  // threads (2 pixels per thread).
  const int block_pixels = (args.in_rows + 1) / 2 * args.in_cols;
  if (block_pixels > 256) {
    return LaunchDepthwiseConv2dGPUSmall<T, kDirection, kKnownFilterWidth,
                                         kKnownFilterHeight, 2>(
        ctx, args, input, filter, output, data_format);
  } else if (block_pixels > 128) {
    return LaunchDepthwiseConv2dGPUSmall<T, kDirection, kKnownFilterWidth,
                                         kKnownFilterHeight, 4>(
        ctx, args, input, filter, output, data_format);
  } else {
    return LaunchDepthwiseConv2dGPUSmall<T, kDirection, kKnownFilterWidth,
                                         kKnownFilterHeight, 8>(
        ctx, args, input, filter, output, data_format);
  }
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
Status LaunchDepthwiseConv2dGPU(OpKernelContext* ctx, const DepthwiseArgs& args,
                                const T* input, const T* filter, T* output,
                                TensorFormat data_format) {
  void (*kernel)(const DepthwiseArgs, const T*, const T*, T*, int);
  switch (data_format) {
    case FORMAT_NHWC:
      kernel =
          DepthwiseConv2dGPUKernelNHWC<T, kKnownFilterWidth, kKnownFilterHeight,
                                       kKnownDepthMultiplier>;
      break;
    case FORMAT_NCHW:
      kernel =
          DepthwiseConv2dGPUKernelNCHW<T, kKnownFilterWidth, kKnownFilterHeight,
                                       kKnownDepthMultiplier>;
      break;
    default:
      return errors::InvalidArgument("FORMAT_", ToString(data_format),
                                     " is not supported");
  }
  const int num_outputs =
      args.batch * args.out_rows * args.out_cols * args.out_depth;
  auto device = ctx->eigen_gpu_device();
  GpuLaunchConfig config =
      GetGpuLaunchConfig(num_outputs, device, kernel, 0, 0);
  // The compile-time constant version runs faster with a single block.
  const int max_block_count = kKnownFilterWidth < 0 || kKnownFilterHeight < 0 ||
                                      kKnownDepthMultiplier < 0
                                  ? std::numeric_limits<int>::max()
                                  : device.getNumGpuMultiProcessors();
  TF_CHECK_OK(GpuLaunchKernel(kernel,
                              std::min(max_block_count, config.block_count),
                              config.thread_per_block, 0, device.stream(), args,
                              input, filter, output, num_outputs));
  return Status::OK();
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
Status LaunchDepthwiseConv2dGPU(OpKernelContext* ctx, const DepthwiseArgs& args,
                                const T* input, const T* filter, T* output,
                                TensorFormat data_format) {
  if (args.depth_multiplier == 1) {
    if (CanLaunchDepthwiseConv2dGPUSmall(args)) {
      return LaunchDepthwiseConv2dGPUSmall<
          T, DIRECTION_FORWARD, kKnownFilterWidth, kKnownFilterHeight>(
          ctx, args, input, filter, output, data_format);
    }

    return LaunchDepthwiseConv2dGPU<T, kKnownFilterWidth, kKnownFilterHeight,
                                    1>(ctx, args, input, filter, output,
                                       data_format);
  } else {
    return LaunchDepthwiseConv2dGPU<T, kKnownFilterWidth, kKnownFilterHeight,
                                    -1>(ctx, args, input, filter, output,
                                        data_format);
  }
}

// A simple launch pad to launch the GPU kernel for depthwise convolution.
template <typename T>
void LaunchDepthwiseConvOp<GpuDevice, T>::operator()(OpKernelContext* ctx,
                                                     const DepthwiseArgs& args,
                                                     const T* input,
                                                     const T* filter, T* output,
                                                     TensorFormat data_format) {
  if (args.filter_rows == 3 && args.filter_cols == 3) {
    OP_REQUIRES_OK(ctx, LaunchDepthwiseConv2dGPU<T, 3, 3>(
                            ctx, args, input, filter, output, data_format));
  } else {
    OP_REQUIRES_OK(ctx, LaunchDepthwiseConv2dGPU<T, -1, -1>(
                            ctx, args, input, filter, output, data_format));
  }
}

// A GPU kernel to compute the depthwise convolution backprop w.r.t. input.
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(640, 2)
    DepthwiseConv2dBackpropInputGPUKernelNHWC(
        const DepthwiseArgs args, const T* __restrict__ out_backprop,
        const T* __restrict__ filter, T* __restrict__ in_backprop,
        int num_in_backprop) {
  const int in_height = args.in_rows;
  const int in_width = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_height =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_width =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int depth_multiplier =
      kKnownDepthMultiplier < 0 ? args.depth_multiplier : kKnownDepthMultiplier;
  const int stride = args.stride;
  const int pad_height = args.pad_rows;
  const int pad_width = args.pad_cols;
  const int out_height = args.out_rows;
  const int out_width = args.out_cols;
  const int out_depth = args.out_depth;

  GPU_1D_KERNEL_LOOP(thread_id, num_in_backprop) {
    // Compute the indexes of this thread in the output.
    const int in_channel = thread_id % in_depth;
    const int in_col = (thread_id / in_depth) % in_width;
    const int in_row = (thread_id / in_depth / in_width) % in_height;
    const int batch = thread_id / in_depth / in_width / in_height;

    T sum = static_cast<T>(0);

    const int out_row_start =
        tf_max<int>(0, (in_row - filter_height + pad_height + stride) / stride);
    const int out_row_end =
        tf_min(out_height - 1, (in_row + pad_height) / stride);
    const int out_col_start =
        tf_max(0, (in_col - filter_width + pad_width + stride) / stride);
    const int out_col_end =
        tf_min(out_width - 1, (in_col + pad_width) / stride);

    NOUNROLL for (int out_row = out_row_start; out_row <= out_row_end;
                  ++out_row) {
      const int filter_row = in_row + pad_height - out_row * stride;
      const int temp_out_backprop_offset =
          out_depth * out_width * (out_row + out_height * batch);
      const int temp_filter_offset = filter_width * filter_row;
      NOUNROLL for (int out_col = out_col_start; out_col <= out_col_end;
                    ++out_col) {
        const int filter_col = in_col + pad_width - out_col * stride;
        int filter_offset =
            depth_multiplier *
            (in_channel + in_depth * (filter_col + temp_filter_offset));
        const int out_backprop_offset =
            out_depth * out_col + temp_out_backprop_offset;
#pragma unroll 6
        for (int i = 0; i < depth_multiplier; ++i) {
          sum += ldg(out_backprop + out_backprop_offset +
                     in_channel * depth_multiplier + i) *
                 ldg(filter + filter_offset + i);
        }
      }
    }
    const int in_backprop_offset =
        in_channel +
        in_depth * (in_col + in_width * (in_row + in_height * batch));
    in_backprop[in_backprop_offset] = sum;
  }
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(640, 2)
    DepthwiseConv2dBackpropInputGPUKernelNCHW(
        const DepthwiseArgs args, const T* __restrict__ out_backprop,
        const T* __restrict__ filter, T* __restrict__ in_backprop,
        int num_in_backprop) {
  const int in_height = args.in_rows;
  const int in_width = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_height =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_width =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int depth_multiplier =
      kKnownDepthMultiplier < 0 ? args.depth_multiplier : kKnownDepthMultiplier;
  const int stride = args.stride;
  const int pad_height = args.pad_rows;
  const int pad_width = args.pad_cols;
  const int out_height = args.out_rows;
  const int out_width = args.out_cols;
  const int out_depth = args.out_depth;

  // TODO(vrv): Consider assigning threads to output and using
  // atomics for accumulation, similar to the filter case.
  GPU_1D_KERNEL_LOOP(thread_id, num_in_backprop) {
    // Compute the indexes of this thread in the input.
    const int in_col = thread_id % in_width;
    const int in_row = (thread_id / in_width) % in_height;
    const int in_channel = (thread_id / in_width / in_height) % in_depth;
    const int batch = thread_id / in_depth / in_width / in_height;

    T sum = static_cast<T>(0);
    const int out_channel_start = in_channel * depth_multiplier;
    const int out_channel_end = out_channel_start + depth_multiplier;

    const int out_row_start =
        tf_max<int>(0, (in_row - filter_height + pad_height + stride) / stride);
    const int out_row_end =
        tf_min(out_height - 1, (in_row + pad_height) / stride);
    const int out_col_start =
        tf_max(0, (in_col - filter_width + pad_width + stride) / stride);
    const int out_col_end =
        tf_min(out_width - 1, (in_col + pad_width) / stride);

    UNROLL for (int out_channel = out_channel_start;
                out_channel < out_channel_end; ++out_channel) {
      UNROLL for (int out_row = out_row_start; out_row <= out_row_end;
                  ++out_row) {
        const int filter_row = in_row + pad_height - out_row * stride;
        const int filter_dm = out_channel - out_channel_start;

        const int temp_filter_offset = filter_width * filter_row;
        for (int out_col = out_col_start; out_col <= out_col_end; ++out_col) {
          const int filter_col = in_col + pad_width - out_col * stride;
          const int filter_offset =
              filter_dm +
              args.depth_multiplier *
                  (in_channel + in_depth * (filter_col + temp_filter_offset));

          const int out_backprop_offset =
              (batch * out_depth * out_height * out_width) +
              (out_channel * out_height * out_width) + (out_row * out_width) +
              (out_col);

          sum += ldg(out_backprop + out_backprop_offset) *
                 ldg(filter + filter_offset);
        }
      }
    }
    const int in_backprop_offset = (batch * in_height * in_width * in_depth) +
                                   (in_channel * in_height * in_width) +
                                   (in_row * in_width) + (in_col);
    in_backprop[in_backprop_offset] = sum;
  }
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
Status LaunchDepthwiseConv2dBackpropInputGPU(OpKernelContext* ctx,
                                             const DepthwiseArgs& args,
                                             const T* out_backprop,
                                             const T* filter, T* in_backprop,
                                             TensorFormat data_format) {
  void (*kernel)(const DepthwiseArgs, const T*, const T*, T*, int);
  switch (data_format) {
    case FORMAT_NHWC:
      kernel = DepthwiseConv2dBackpropInputGPUKernelNHWC<
          T, kKnownFilterWidth, kKnownFilterHeight, kKnownDepthMultiplier>;
      break;
    case FORMAT_NCHW:
      kernel = DepthwiseConv2dBackpropInputGPUKernelNCHW<
          T, kKnownFilterWidth, kKnownFilterHeight, kKnownDepthMultiplier>;
      break;
    default:
      return errors::InvalidArgument("FORMAT_", ToString(data_format),
                                     " is not supported");
  }
  const int num_in_backprop =
      args.batch * args.in_rows * args.in_cols * args.in_depth;
  auto device = ctx->eigen_gpu_device();
  GpuLaunchConfig config =
      GetGpuLaunchConfig(num_in_backprop, device, kernel, 0, 0);
  TF_CHECK_OK(GpuLaunchKernel(
      kernel, config.block_count, config.thread_per_block, 0, device.stream(),
      args, out_backprop, filter, in_backprop, num_in_backprop));
  return Status::OK();
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
Status LaunchDepthwiseConv2dBackpropInputGPU(OpKernelContext* ctx,
                                             const DepthwiseArgs& args,
                                             const T* out_backprop,
                                             const T* filter, T* in_backprop,
                                             TensorFormat data_format) {
  if (args.depth_multiplier == 1) {
    if (CanLaunchDepthwiseConv2dGPUSmall(args)) {
      return LaunchDepthwiseConv2dGPUSmall<
          T, DIRECTION_BACKWARD, kKnownFilterWidth, kKnownFilterHeight>(
          ctx, args, out_backprop, filter, in_backprop, data_format);
    }

    return LaunchDepthwiseConv2dBackpropInputGPU<T, kKnownFilterWidth,
                                                 kKnownFilterHeight, 1>(
        ctx, args, out_backprop, filter, in_backprop, data_format);
  } else {
    return LaunchDepthwiseConv2dBackpropInputGPU<T, kKnownFilterWidth,
                                                 kKnownFilterHeight, -1>(
        ctx, args, out_backprop, filter, in_backprop, data_format);
  }
}

// A simple launch pad to launch the GPU kernel for depthwise convolution.
template <typename T>
void LaunchDepthwiseConvBackpropInputOp<GpuDevice, T>::operator()(
    OpKernelContext* ctx, const DepthwiseArgs& args, const T* out_backprop,
    const T* filter, T* in_backprop, TensorFormat data_format) {
  if (args.filter_rows == 3 && args.filter_cols == 3) {
    OP_REQUIRES_OK(
        ctx, LaunchDepthwiseConv2dBackpropInputGPU<T, 3, 3>(
                 ctx, args, out_backprop, filter, in_backprop, data_format));
  } else {
    OP_REQUIRES_OK(
        ctx, LaunchDepthwiseConv2dBackpropInputGPU<T, -1, -1>(
                 ctx, args, out_backprop, filter, in_backprop, data_format));
  }
}

// A GPU kernel to compute the depthwise convolution backprop w.r.t. filter.
// TODO: Add fp32 accumulation to half calls of this function. This addition
// is non-trivial as the partial sums are added directly to the output
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(640, 2)
    DepthwiseConv2dBackpropFilterGPUKernelNHWC(
        const DepthwiseArgs args, const T* __restrict__ out_backprop,
        const T* __restrict__ input, T* __restrict__ filter_backprop,
        int num_out_backprop) {
  const int in_height = args.in_rows;
  const int in_width = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_height =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_width =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int depth_multiplier =
      kKnownDepthMultiplier < 0 ? args.depth_multiplier : kKnownDepthMultiplier;
  const int stride = args.stride;
  const int pad_height = args.pad_rows;
  const int pad_width = args.pad_cols;
  const int out_height = args.out_rows;
  const int out_width = args.out_cols;
  const int out_depth = args.out_depth;

  GPU_1D_KERNEL_LOOP(thread_id, num_out_backprop) {
    // Compute the indexes of this thread in the output.
    const int out_channel = thread_id % out_depth;
    const int out_col = (thread_id / out_depth) % out_width;
    const int out_row = (thread_id / out_depth / out_width) % out_height;
    const int batch = thread_id / out_depth / out_width / out_height;
    // Compute the input depth and the index of depth multiplier.
    const int in_channel = out_channel / depth_multiplier;
    const int dm = out_channel % depth_multiplier;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_row_start = out_row * stride - pad_height;
    const int in_col_start = out_col * stride - pad_width;
    const int in_row_end = in_row_start + filter_height;
    const int in_col_end = in_col_start + filter_width;

    const int out_backprop_offset =
        out_channel +
        out_depth * (out_col + out_width * (out_row + out_height * batch));
    const T out_bp = ldg(out_backprop + out_backprop_offset);
    if (in_row_start >= 0 && in_col_start >= 0 && in_row_end < in_height &&
        in_col_end < in_width) {
      UNROLL for (int filter_row = 0; filter_row < filter_height;
                  ++filter_row) {
        const int in_row = in_row_start + filter_row;
        // Avoid repeated computation.
        const int input_offset_temp = in_width * (in_row + in_height * batch);
        UNROLL for (int filter_col = 0; filter_col < filter_width;
                    ++filter_col) {
          const int in_col = in_col_start + filter_col;

          const int input_offset =
              in_channel + in_depth * (in_col + input_offset_temp);
          T partial_sum = ldg(input + input_offset) * out_bp;
          T* addr =
              filter_backprop +
              (dm + depth_multiplier *
                        (in_channel +
                         in_depth * (filter_col + filter_width * filter_row)));
          GpuAtomicAdd(addr, partial_sum);
        }
      }
    } else {
      UNROLL for (int filter_row = 0; filter_row < filter_height;
                  ++filter_row) {
        const int in_row = in_row_start + filter_row;
        // Avoid repeated computation.
        const int input_offset_temp = in_width * (in_row + in_height * batch);
        UNROLL for (int filter_col = 0; filter_col < filter_width;
                    ++filter_col) {
          const int in_col = in_col_start + filter_col;
          const int addr_temp = filter_width * filter_row;

          if (in_row >= 0 && in_row < in_height && in_col >= 0 &&
              in_col < in_width) {
            const int input_offset =
                in_channel + in_depth * (in_col + input_offset_temp);
            T partial_sum = ldg(input + input_offset) * out_bp;
            T* addr =
                filter_backprop +
                (dm + depth_multiplier *
                          (in_channel + in_depth * (filter_col + addr_temp)));
            // Potentially many threads can add to the same address so we have
            // to use atomic add here.
            // TODO(jmchen): If atomic add turns out to be slow, we can:
            // 1. allocate multiple buffers for the gradients (one for each
            // example in a batch, for example). This can reduce the
            // contention on the destination; 2. Have each thread compute one
            // gradient for an element in the filters. This should work well
            // when the input depth is big and filter size is not too small.
            GpuAtomicAdd(addr, partial_sum);
          }
        }
      }
    }
  }
}

// Device function to compute sub-warp sum reduction for a power-of-two group of
// neighboring threads.
template <int kWidth, typename T>
__device__ __forceinline__ T WarpSumReduce(T val) {
  // support only power-of-two widths.
  assert(__popc(kWidth) == 1);
  int sub_warp = GpuLaneId() / kWidth;
  int zeros = sub_warp * kWidth;
  unsigned mask = ((1UL << kWidth) - 1) << zeros;
  for (int delta = kWidth / 2; delta > 0; delta /= 2) {
    val += GpuShuffleXorSync(mask, val, delta);
  }
  return val;
}

// CUDA kernel to compute the depthwise convolution backward w.r.t. filter in
// NHWC format, tailored for small images up to 32x32. Stride and depth
// multiplier must be 1. Padding must be 'SAME'. Only use this kernel if
// CanLaunchDepthwiseConv2dGPUSmall(args) returns true.
// Tiles of the input tensor are loaded into shared memory before performing the
// convolution. Per iteration and filter element, each thread first performs
// a partial convolution for two elements, one each in the lower and upper half
// of a tile. The intermediate result of all pixels of a warp are then
// accumulated and written to shared memory. Finally, the values in shared
// memory are warp-accumulated (in chunks of kAccumPixels elements) and summed
// up in global memory using atomics.
// Requirements: threads per block must be multiple of 32 and <= launch_bounds,
// kAccumPixels * 64 >= args.in_rows * args.in_cols * kBlockDepth.
// T is the tensors' data type. S is the math type the kernel uses. This is the
// same as T for all cases but pseudo half (which has T=Eigen::half, S=float).
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kBlockDepth, int kAccumPixels>
__global__
__launch_bounds__(1024, 2) void DepthwiseConv2dBackpropFilterGPUKernelNHWCSmall(
    const DepthwiseArgs args, const T* __restrict__ output,
    const T* __restrict__ input, T* __restrict__ filter) {
  typedef typename detail::PseudoHalfType<T>::Type S;
  assert(CanLaunchDepthwiseConv2dBackpropFilterGPUSmall(args, blockDim.z));
  // Holds block plus halo and filter data for blockDim.x depths.
  GPU_DYNAMIC_SHARED_MEM_DECL(8, unsigned char, shared_memory);
  static_assert(sizeof(S) <= 8, "Insufficient alignment detected");
  S* const shared_data = reinterpret_cast<S*>(shared_memory);

  const int num_batches = args.batch;
  const int in_height = args.in_rows;
  const int in_width = blockDim.y;  // slower (see b/62280718): args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_height =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_width =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int pad_height = args.pad_rows;
  const int pad_width = args.pad_cols;

  assert(blockDim.x == kBlockDepth);
  assert(blockDim.y == args.in_cols);
  const int block_height = blockDim.z;

  // These values are the same for all threads and could
  // be precomputed on the CPU.
  const int block_size = block_height * in_width * kBlockDepth;
  assert((block_size & 31) == 0);
  const int in_row_size = in_width * in_depth;
  const int in_size = in_height * in_row_size;
  const int in_increment = (in_width - 1) * kBlockDepth;
  const int filter_pixels = filter_height * filter_width;
  const int tile_width = in_width + filter_width - 1;
  const int tile_height = 2 * block_height + filter_height - 1;
  const int tile_row_size = tile_width * kBlockDepth;
  const int tile_size = tile_height * tile_row_size;
  const int tile_offset = block_height * tile_row_size;
  const int pad_offset = pad_height * tile_width + pad_width;
  const int batch_blocks = (in_depth + kBlockDepth - 1) / kBlockDepth;
  const int in_blocks = batch_blocks * num_batches;
  const int tensor_offset = block_height * in_row_size;
  // The accumulator has a fixed number of pixels that can be reduced by one
  // warp. Pixels beyond ceil(in_pixels * kBlockDepth / 64) are never written.
  assert(kAccumPixels * 64 >= in_height * in_width * kBlockDepth);
  const int accum_increment = kAccumPixels * kBlockDepth;
  const int accum_size = filter_pixels * accum_increment;

  const int thread_depth = threadIdx.x;
  const int thread_col = threadIdx.y;
  const int thread_row = threadIdx.z;

  // Position in block.
  const int thread_pix = thread_row * in_width + thread_col;
  const int thread_idx = thread_pix * kBlockDepth + thread_depth;

  // Initialize tile, in particular the padding and accumulator.
  for (int i = thread_idx; i < tile_size + accum_size; i += block_size) {
    shared_data[i] = S();
  }
  __syncthreads();

  // Position in tensors.
  const int tensor_idx = thread_pix * in_depth + thread_depth;

  // Position in (padded) shared memory.
  const int data_pix = thread_row * tile_width + thread_col;
  const int data_idx = data_pix * kBlockDepth + thread_depth;

  // Position in shared memory, offset by pad_height / pad_width.
  const int tile_pix = data_pix + pad_offset;
  const int tile_idx = tile_pix * kBlockDepth + thread_depth;

  // Position in accumulator (kBlockDepth per warp, depth major).
  const int accum_pix = thread_pix / (32 / kBlockDepth);
  const int accum_idx = thread_depth * kAccumPixels + accum_pix;

  const int max_channel = in_depth - thread_depth;
  const int accum_offset = tile_size + accum_idx;
  const bool skip_second = block_height + thread_row >= in_height;

  for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
    const int batch = b / batch_blocks;
    const int block = b - batch * batch_blocks;

    const int start_channel = block * kBlockDepth;
    const int filter_offset = tensor_idx + start_channel;
    const int inout_offset = batch * in_size + filter_offset;
    const bool channel_in_range = start_channel < max_channel;

    if (channel_in_range) {
      const T* const in_ptr = inout_offset + input;
      S* const tile_ptr = tile_idx + shared_data;
      tile_ptr[0] = static_cast<S>(ldg(in_ptr));
      if (!skip_second) {
        tile_ptr[tile_offset] = static_cast<S>(ldg(tensor_offset + in_ptr));
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();
    unsigned active_threads = GpuBallotSync(kCudaWarpAll, channel_in_range);

    if (channel_in_range) {
      const T* const out_ptr = inout_offset + output;
      const S out1 = static_cast<S>(ldg(out_ptr));
      const S out2 =
          skip_second ? S() : static_cast<S>(ldg(tensor_offset + out_ptr));
      int shared_offset = data_idx;
      S* accum_ptr = accum_offset + shared_data;
      UNROLL for (int r = 0; r < filter_height; ++r) {
        UNROLL for (int c = 0; c < filter_width; ++c) {
          const S* const tile_ptr = shared_offset + shared_data;
          S val = out1 * tile_ptr[0] + out2 * tile_ptr[tile_offset];
          // Warp-accumulate pixels of the same depth and write to accumulator.
          for (int delta = 16; delta >= kBlockDepth; delta /= 2) {
            val += GpuShuffleXorSync(active_threads, val, delta);
          }
          if (!(thread_idx & 32 - kBlockDepth) /* lane_idx < kBlockDepth */) {
            *accum_ptr = val;
          }
          shared_offset += kBlockDepth;
          accum_ptr += accum_increment;
        }
        shared_offset += in_increment;
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();

    const S* const accum_data = tile_size + shared_data;
    for (int i = thread_idx; i < accum_size; i += block_size) {
      const int filter_idx = i / kAccumPixels;
      const int filter_pix = filter_idx / kBlockDepth;
      const int filter_channel = filter_idx % kBlockDepth + start_channel;
      const int filter_offset = filter_pix * in_depth + filter_channel;
      if (filter_channel < in_depth) {
        S val = accum_data[i];
        // Warp-accumulate the pixels of the same depth from the accumulator.
        val = WarpSumReduce<kAccumPixels>(val);
        if (!(thread_idx & kAccumPixels - 1)) {
          GpuAtomicAdd(filter_offset + filter, static_cast<T>(val));
        }
      }
    }
  }
}

// A GPU kernel to compute the depthwise convolution backprop w.r.t. filter.
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(640, 2)
    DepthwiseConv2dBackpropFilterGPUKernelNCHW(
        const DepthwiseArgs args, const T* __restrict__ out_backprop,
        const T* __restrict__ input, T* __restrict__ filter_backprop,
        int num_out_backprop) {
  const int in_height = args.in_rows;
  const int in_width = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_height =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_width =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int depth_multiplier =
      kKnownDepthMultiplier < 0 ? args.depth_multiplier : kKnownDepthMultiplier;
  const int stride = args.stride;
  const int pad_height = args.pad_rows;
  const int pad_width = args.pad_cols;
  const int out_height = args.out_rows;
  const int out_width = args.out_cols;
  const int out_depth = args.out_depth;

  GPU_1D_KERNEL_LOOP(thread_id, num_out_backprop) {
    // Compute the indexes of this thread in the output.
    const int out_col = thread_id % out_width;
    const int out_row = (thread_id / out_width) % out_height;
    const int out_channel = (thread_id / out_width / out_height) % out_depth;

    const int batch = thread_id / out_depth / out_width / out_height;
    // Compute the input depth and the index of depth multiplier.
    const int in_channel = out_channel / depth_multiplier;
    const int dm = out_channel % depth_multiplier;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_row_start = out_row * stride - pad_height;
    const int in_col_start = out_col * stride - pad_width;
    const int in_row_end = in_row_start + filter_height;
    const int in_col_end = in_col_start + filter_width;

    const int out_backprop_offset =
        (batch * out_depth * out_height * out_width) +
        (out_channel * out_height * out_width) + (out_row * out_width) +
        (out_col);

    const T out_bp = ldg(out_backprop + out_backprop_offset);
    if (in_row_start >= 0 && in_col_start >= 0 && in_row_end < in_height &&
        in_col_end < in_width) {
      UNROLL for (int filter_row = 0; filter_row < filter_height;
                  ++filter_row) {
        const int in_row = in_row_start + filter_row;
        // Avoid repeated computation.
        const int input_offset_temp =
            (batch * in_depth * in_height * in_width) +
            (in_channel * in_height * in_width) + (in_row * in_width);

        UNROLL for (int filter_col = 0; filter_col < filter_width;
                    ++filter_col) {
          const int in_col = in_col_start + filter_col;
          const int input_offset = input_offset_temp + in_col;
          T partial_sum = ldg(input + input_offset) * out_bp;
          T* addr =
              filter_backprop +
              (dm + depth_multiplier *
                        (in_channel +
                         in_depth * (filter_col + filter_width * filter_row)));
          GpuAtomicAdd(addr, partial_sum);
        }
      }
    } else {
      UNROLL for (int filter_row = 0; filter_row < filter_height;
                  ++filter_row) {
        const int in_row = in_row_start + filter_row;
        // Avoid repeated computation.
        const int input_offset_temp =
            (batch * in_depth * in_height * in_width) +
            (in_channel * in_height * in_width) + (in_row * in_width);
        UNROLL for (int filter_col = 0; filter_col < filter_width;
                    ++filter_col) {
          const int in_col = in_col_start + filter_col;
          const int addr_temp = filter_width * filter_row;

          if (in_row >= 0 && in_row < in_height && in_col >= 0 &&
              in_col < in_width) {
            const int input_offset = input_offset_temp + in_col;
            T partial_sum = ldg(input + input_offset) * out_bp;
            T* addr =
                filter_backprop +
                (dm + depth_multiplier *
                          (in_channel + in_depth * (filter_col + addr_temp)));
            // Potentially many threads can add to the same address so we have
            // to use atomic add here.
            // TODO(jmchen): If atomic add turns out to be slow, we can:
            // 1. allocate multiple buffers for the gradients (one for each
            // example in a batch, for example). This can reduce the
            // contention on the destination; 2. Have each thread compute one
            // gradient for an element in the filters. This should work well
            // when the input depth is big and filter size is not too small.
            GpuAtomicAdd(addr, partial_sum);
          }
        }
      }
    }
  }
}

// CUDA kernel to compute the depthwise convolution backward w.r.t. filter in
// NCHW format, tailored for small images up to 32x32. Stride and depth
// multiplier must be 1. Padding must be 'SAME'. Only use this kernel if
// CanLaunchDepthwiseConv2dGPUSmall(args) returns true.
// Tiles of the input tensor are loaded into shared memory before performing the
// convolution. Per iteration and filter element, each thread first performs
// a partial convolution for two elements, one each in the lower and upper half
// of a tile. The intermediate result of all pixels of a warp are then
// accumulated and written to shared memory. Finally, the values in shared
// memory are warp-accumulated (in chunks of kAccumPixels elements) and summed
// up in global memory using atomics.
// Requirements: threads per block must be multiple of 32 and <= launch_bounds,
// kAccumPixels * 64 >= args.in_rows * args.in_cols * kBlockDepth.
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kBlockDepth, int kAccumPixels>
__global__
__launch_bounds__(1024, 2) void DepthwiseConv2dBackpropFilterGPUKernelNCHWSmall(
    const DepthwiseArgs args, const T* __restrict__ output,
    const T* __restrict__ input, T* __restrict__ filter) {
  typedef typename detail::PseudoHalfType<T>::Type S;
  assert(CanLaunchDepthwiseConv2dBackpropFilterGPUSmall(args, blockDim.x));
  // Holds block plus halo and filter data for blockDim.z depths.
  GPU_DYNAMIC_SHARED_MEM_DECL(8, unsigned char, shared_memory);
  static_assert(sizeof(S) <= 8, "Insufficient alignment detected");
  S* const shared_data = reinterpret_cast<S*>(shared_memory);

  const int num_batches = args.batch;
  const int in_height = args.in_rows;
  const int in_width = blockDim.x;  // slower (see b/62280718): args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_height =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_width =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int pad_height = args.pad_rows;
  const int pad_width = args.pad_cols;

  assert(blockDim.x == args.in_cols);
  assert(blockDim.z == kBlockDepth);
  const int block_height = blockDim.y;

  // These values are the same for all threads and could
  // be precomputed on the CPU.
  const int block_pixels = in_width * block_height;
  const int block_size = block_pixels * kBlockDepth;
  assert((block_size & 31) == 0);
  const int in_pixels = in_width * in_height;
  const int in_increment = in_width - 1;
  const int filter_pixels = filter_height * filter_width;
  const int tile_width = in_width + filter_width - 1;
  const int tile_height = 2 * block_height + filter_height - 1;
  const int tile_pixels = tile_width * tile_height;
  const int tile_size = tile_pixels * kBlockDepth;
  const int tile_offset = block_height * tile_width;
  const int pad_offset = pad_height * tile_width + pad_width;
  const int in_total_depth = in_depth * num_batches;
  const int in_blocks = (in_total_depth + kBlockDepth - 1) / kBlockDepth;
  // The accumulator has a fixed number of pixels that can be reduced by one
  // warp. Pixels beyond ceil(in_pixels * kBlockDepth / 64) are never written.
  assert(kAccumPixels * 64 >= in_height * in_width * kBlockDepth);
  const int accum_increment = kAccumPixels * kBlockDepth;
  const int accum_size = filter_pixels * accum_increment;

  const int thread_col = threadIdx.x;
  const int thread_row = threadIdx.y;
  const int thread_depth = threadIdx.z;

  // Position in block.
  const int thread_pix = thread_row * in_width + thread_col;
  const int thread_idx = thread_depth * block_pixels + thread_pix;

  // Initialize tile, in particular the padding and accumulator.
  for (int i = thread_idx; i < tile_size + accum_size; i += block_size) {
    shared_data[i] = S();
  }
  __syncthreads();

  // Position in tensors.
  const int tensor_idx = thread_depth * in_pixels + thread_pix;

  // Position in (padded) shared memory.
  const int data_pix = thread_row * tile_width + thread_col;
  const int data_idx = thread_depth * tile_pixels + data_pix;

  // Position in shared memory, offset by pad_height / pad_width.
  const int tile_idx = data_idx + pad_offset;

  // Position in accumulator (kBlockDepth per warp, depth major).
  const int accum_pix = thread_pix / (32 / kBlockDepth);
  const int accum_idx = thread_depth * kAccumPixels + accum_pix;

  const int max_channel = in_total_depth - thread_depth;
  const int accum_offset = tile_size + accum_idx;
  const bool skip_second = block_height + thread_row >= in_height;

  for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
    const int channel = b * kBlockDepth;

    const int inout_offset = channel * in_pixels + tensor_idx;
    const bool channel_in_range = channel < max_channel;

    if (channel_in_range) {
      const T* const in_ptr = inout_offset + input;
      S* const tile_ptr = tile_idx + shared_data;
      tile_ptr[0] = static_cast<S>(ldg(in_ptr));
      if (!skip_second) {
        tile_ptr[tile_offset] = static_cast<S>(ldg(block_pixels + in_ptr));
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();
    unsigned active_threads = GpuBallotSync(kCudaWarpAll, channel_in_range);

    if (channel_in_range) {
      const T* const out_ptr = inout_offset + output;
      const S out1 = static_cast<S>(ldg(out_ptr));
      const S out2 =
          skip_second ? S() : static_cast<S>(ldg(block_pixels + out_ptr));
      int shared_offset = data_idx;
      S* accum_ptr = accum_offset + shared_data;
      UNROLL for (int r = 0; r < filter_height; ++r) {
        UNROLL for (int c = 0; c < filter_width; ++c) {
          const S* const tile_ptr = shared_offset + shared_data;
          S val = out1 * tile_ptr[0] + out2 * tile_ptr[tile_offset];
          // Warp-accumulate pixels of the same depth and write to accumulator.
          for (int delta = 16 / kBlockDepth; delta > 0; delta /= 2) {
            val += GpuShuffleXorSync(active_threads, val, delta);
          }
          if (!(thread_idx & 32 / kBlockDepth - 1)) {
            *accum_ptr = val;  // kBlockDepth threads per warp.
          }
          ++shared_offset;
          accum_ptr += accum_increment;
        }
        shared_offset += in_increment;
      }
    }

    // Note: the condition to reach this is uniform across the entire block.
    __syncthreads();

    const S* const accum_data = tile_size + shared_data;
    for (int i = thread_idx; i < accum_size; i += block_size) {
      const int filter_idx = i / kAccumPixels;
      const int filter_pix = filter_idx / kBlockDepth;
      const int filter_channel =
          (channel + filter_idx % kBlockDepth) % in_depth;
      const int filter_offset = filter_pix * in_depth + filter_channel;
      if (filter_channel < in_depth) {
        S val = accum_data[i];
        // Warp-accumulate pixels of the same depth from the accumulator.
        val = WarpSumReduce<kAccumPixels>(val);
        if (!(thread_idx & kAccumPixels - 1)) {
          GpuAtomicAdd(filter_offset + filter, static_cast<T>(val));
        }
      }
    }
  }
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kBlockDepth, int kAccumPixels>
Status TryLaunchDepthwiseConv2dBackpropFilterGPUSmall(
    OpKernelContext* ctx, const DepthwiseArgs& args, const int block_height,
    const T* out_backprop, const T* input, T* filter_backprop,
    TensorFormat data_format) {
  typedef typename detail::PseudoHalfType<T>::Type S;
  auto device = ctx->eigen_gpu_device();
  const int tile_width = args.in_cols + args.filter_cols - 1;
  const int tile_height = block_height * 2 + args.filter_rows - 1;
  const int tile_pixels = tile_height * tile_width;
  const int filter_pixels = args.filter_rows * args.filter_cols;
  const int shared_memory_size =
      kBlockDepth * (tile_pixels + filter_pixels * kAccumPixels) * sizeof(S);
  if (shared_memory_size > device.sharedMemPerBlock()) {
    return errors::FailedPrecondition("Not enough shared memory");
  }

  dim3 block_dim;
  int block_count;
  void (*kernel)(const DepthwiseArgs, const T*, const T*, T*);
  switch (data_format) {
    case FORMAT_NHWC:
      block_dim = dim3(kBlockDepth, args.in_cols, block_height);
      block_count =
          args.batch * DivUp(args.out_depth, kBlockDepth) * kBlockDepth;
      kernel = DepthwiseConv2dBackpropFilterGPUKernelNHWCSmall<
          T, kKnownFilterWidth, kKnownFilterHeight, kBlockDepth, kAccumPixels>;
      break;
    case FORMAT_NCHW:
      block_dim = dim3(args.in_cols, block_height, kBlockDepth);
      block_count =
          DivUp(args.batch * args.out_depth, kBlockDepth) * kBlockDepth;
      kernel = DepthwiseConv2dBackpropFilterGPUKernelNCHWSmall<
          T, kKnownFilterWidth, kKnownFilterHeight, kBlockDepth, kAccumPixels>;
      break;
    default:
      return errors::InvalidArgument("FORMAT_", ToString(data_format),
                                     " is not supported");
  }
  const int num_out_backprop = args.out_rows * args.out_cols * block_count;
  GpuLaunchConfig config = GetGpuLaunchConfigFixedBlockSize(
      num_out_backprop, device, kernel, shared_memory_size,
      block_dim.x * block_dim.y * block_dim.z);
  TF_CHECK_OK(GpuLaunchKernel(kernel, config.block_count, block_dim,
                              shared_memory_size, device.stream(), args,
                              out_backprop, input, filter_backprop));
  return Status::OK();
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kBlockDepth>
Status TryLaunchDepthwiseConv2dBackpropFilterGPUSmall(
    OpKernelContext* ctx, const DepthwiseArgs& args, const int block_height,
    const T* out_backprop, const T* input, T* filter_backprop,
    TensorFormat data_format) {
  // Minimize (power of two) kAccumPixels, while satisfying
  // kAccumPixels * 32 >= block_height * in_width * kBlockDepth.
  const int block_pixels = block_height * args.in_cols * kBlockDepth;
  if (block_pixels > 512) {
    return TryLaunchDepthwiseConv2dBackpropFilterGPUSmall<
        T, kKnownFilterWidth, kKnownFilterHeight, kBlockDepth, 32>(
        ctx, args, block_height, out_backprop, input, filter_backprop,
        data_format);
  } else if (block_pixels > 256) {
    return TryLaunchDepthwiseConv2dBackpropFilterGPUSmall<
        T, kKnownFilterWidth, kKnownFilterHeight, kBlockDepth, 16>(
        ctx, args, block_height, out_backprop, input, filter_backprop,
        data_format);
  } else {
    return TryLaunchDepthwiseConv2dBackpropFilterGPUSmall<
        T, kKnownFilterWidth, kKnownFilterHeight, kBlockDepth, 8>(
        ctx, args, block_height, out_backprop, input, filter_backprop,
        data_format);
  }
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
Status TryLaunchDepthwiseConv2dBackpropFilterGPUSmall(
    OpKernelContext* ctx, const DepthwiseArgs& args, const T* out_backprop,
    const T* input, T* filter_backprop, TensorFormat data_format) {
  // Maximize (power of two) kBlockDepth while keeping a block within 1024
  // threads (2 pixels per thread).
  int block_depth = 8;
  int block_height = (args.in_rows + 1) / 2;
  int round_mask = 1;
  for (; block_depth > 1; block_depth /= 2) {
    // args.in_cols * block_height * kBlockDepth must be multiple of 32.
    for (; block_height * args.in_cols * block_depth & 31;
         round_mask = round_mask * 2 + 1) {
      block_height = block_height + round_mask & ~round_mask;
    }
    int block_size = block_height * args.in_cols * block_depth;
    if (block_size <= 1024) {
      break;
    }
  }

  if (!CanLaunchDepthwiseConv2dBackpropFilterGPUSmall(args, block_height)) {
    return errors::FailedPrecondition("Cannot launch this configuration");
  }

  switch (block_depth) {
    case 8:
      return TryLaunchDepthwiseConv2dBackpropFilterGPUSmall<
          T, kKnownFilterWidth, kKnownFilterHeight, 8>(
          ctx, args, block_height, out_backprop, input, filter_backprop,
          data_format);
    case 4:
      return TryLaunchDepthwiseConv2dBackpropFilterGPUSmall<
          T, kKnownFilterWidth, kKnownFilterHeight, 4>(
          ctx, args, block_height, out_backprop, input, filter_backprop,
          data_format);
    case 2:
      return TryLaunchDepthwiseConv2dBackpropFilterGPUSmall<
          T, kKnownFilterWidth, kKnownFilterHeight, 2>(
          ctx, args, block_height, out_backprop, input, filter_backprop,
          data_format);
    default:
      return errors::InvalidArgument("Unexpected block depth");
  }
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
Status LaunchDepthwiseConv2dBackpropFilterGPU(
    OpKernelContext* ctx, const DepthwiseArgs& args, const T* out_backprop,
    const T* input, T* filter_backprop, TensorFormat data_format) {
  void (*kernel)(const DepthwiseArgs, const T*, const T*, T*, int);
  switch (data_format) {
    case FORMAT_NHWC:
      kernel = DepthwiseConv2dBackpropFilterGPUKernelNHWC<
          T, kKnownFilterWidth, kKnownFilterHeight, kKnownDepthMultiplier>;
      break;
    case FORMAT_NCHW:
      kernel = DepthwiseConv2dBackpropFilterGPUKernelNCHW<
          T, kKnownFilterWidth, kKnownFilterHeight, kKnownDepthMultiplier>;
      break;
    default:
      return errors::InvalidArgument("FORMAT_", ToString(data_format),
                                     " is not supported");
  }
  const int num_out_backprop =
      args.batch * args.out_rows * args.out_cols * args.out_depth;
  auto device = ctx->eigen_gpu_device();
  GpuLaunchConfig config =
      GetGpuLaunchConfig(num_out_backprop, device, kernel, 0, 0);
  TF_CHECK_OK(GpuLaunchKernel(
      kernel, config.block_count, config.thread_per_block, 0, device.stream(),
      args, out_backprop, input, filter_backprop, num_out_backprop));
  return Status::OK();
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
Status LaunchDepthwiseConv2dBackpropFilterGPU(
    OpKernelContext* ctx, const DepthwiseArgs& args, const T* out_backprop,
    const T* input, T* filter_backprop, TensorFormat data_format) {
  if (args.depth_multiplier == 1) {
    if (TryLaunchDepthwiseConv2dBackpropFilterGPUSmall<T, kKnownFilterWidth,
                                                       kKnownFilterHeight>(
            ctx, args, out_backprop, input, filter_backprop, data_format)
            .ok()) {
      return Status::OK();
    }

    return LaunchDepthwiseConv2dBackpropFilterGPU<T, kKnownFilterWidth,
                                                  kKnownFilterHeight, 1>(
        ctx, args, out_backprop, input, filter_backprop, data_format);
  } else {
    return LaunchDepthwiseConv2dBackpropFilterGPU<T, kKnownFilterWidth,
                                                  kKnownFilterHeight, -1>(
        ctx, args, out_backprop, input, filter_backprop, data_format);
  }
}

// A simple launch pad to launch the GPU kernel for depthwise convolution.
template <typename T>
void LaunchDepthwiseConvBackpropFilterOp<GpuDevice, T>::operator()(
    OpKernelContext* ctx, const DepthwiseArgs& args, const T* out_backprop,
    const T* input, T* filter_backprop, TensorFormat data_format) {
  auto stream = ctx->op_device_context()->stream();

  // Initialize the results to 0.
  int num_filter_backprop =
      args.filter_rows * args.filter_cols * args.out_depth;
  se::DeviceMemoryBase filter_bp_ptr(filter_backprop, num_filter_backprop);
  stream->ThenMemset32(&filter_bp_ptr, 0, num_filter_backprop * sizeof(T));

  if (args.filter_rows == 3 && args.filter_cols == 3) {
    OP_REQUIRES_OK(
        ctx, LaunchDepthwiseConv2dBackpropFilterGPU<T, 3, 3>(
                 ctx, args, out_backprop, input, filter_backprop, data_format));
  } else {
    OP_REQUIRES_OK(
        ctx, LaunchDepthwiseConv2dBackpropFilterGPU<T, -1, -1>(
                 ctx, args, out_backprop, input, filter_backprop, data_format));
  }
}
}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_DEPTHWISE_CONV_OP_GPU_H_
