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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/depthwise_conv_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/tensor_format.h"

#if !defined(_MSC_VER)
#define UNROLL _Pragma("unroll")
#else
#define UNROLL
#endif

namespace tensorflow {

namespace {

typedef Eigen::GpuDevice GPUDevice;

// A Cuda kernel to compute the depthwise convolution forward pass
// in NHWC format.
template <typename T>
__global__ void DepthwiseConv2dGPUKernelNHWC(const DepthwiseArgs args,
                                             const T* input, const T* filter,
                                             T* output, int num_outputs) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows = args.filter_rows;
  const int filter_cols = args.filter_cols;
  const int depth_multiplier = args.depth_multiplier;
  const int stride = args.stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_depth = args.out_depth;

  CUDA_1D_KERNEL_LOOP(thread_id, num_outputs) {
    // Compute the indexes of this thread in the output.
    const int OD = thread_id % out_depth;
    const int OC = (thread_id / out_depth) % out_cols;
    const int OR = (thread_id / out_depth / out_cols) % out_rows;
    const int OB = thread_id / out_depth / out_cols / out_rows;
    // Compute the input depth and the index of depth multiplier.
    const int in_d = OD / depth_multiplier;
    const int multiplier = OD % depth_multiplier;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int input_row_start = OR * stride - pad_rows;
    const int input_col_start = OC * stride - pad_cols;
    const int input_row_end = input_row_start + filter_rows;
    const int input_col_end = input_col_start + filter_cols;

    T sum = 0;

    const int input_offset_temp = in_rows * OB;
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_row_end < in_rows && input_col_end < in_cols) {
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = input_row_start + f_r;
        const int filter_offset_temp = filter_cols * f_r;
        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = input_col_start + f_c;

          const int input_offset =
              in_d + in_depth * (in_c + in_cols * (in_r + input_offset_temp));
          const int filter_offset =
              multiplier +
              depth_multiplier * (in_d + in_depth * (f_c + filter_offset_temp));
          sum += ldg(input + input_offset) * ldg(filter + filter_offset);
        }
      }
    } else {
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = input_row_start + f_r;
        const int filter_offset_temp = filter_cols * f_r;
        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = input_col_start + f_c;
          if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
            const int in_c = input_col_start + f_c;

            const int input_offset =
                in_d + in_depth * (in_c + in_cols * (in_r + input_offset_temp));
            const int filter_offset =
                multiplier + depth_multiplier *
                                 (in_d + in_depth * (f_c + filter_offset_temp));
            sum += ldg(input + input_offset) * ldg(filter + filter_offset);
          }
        }
      }
    }
    output[thread_id] = sum;
  }
}

// A Cuda kernel to compute the depthwise convolution forward pass
// in NCHW format.
template <typename T>
__global__ void DepthwiseConv2dGPUKernelNCHW(const DepthwiseArgs args,
                                             const T* input, const T* filter,
                                             T* output, int num_outputs) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows = args.filter_rows;
  const int filter_cols = args.filter_cols;
  const int depth_multiplier = args.depth_multiplier;
  const int stride = args.stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_depth = args.out_depth;

  CUDA_1D_KERNEL_LOOP(thread_id, num_outputs) {
    // Compute the indexes of this thread in the output.
    //
    // We want coalesced reads so we make sure that each warp reads
    // a contiguous chunk of memory.
    //
    // THIS IS PROBABLY WRONG, we are not doing coalesced reads
    // into the input, because of the depth multiplier division...
    const int OC = thread_id % out_cols;
    const int OR = (thread_id / out_cols) % out_rows;
    const int OD = (thread_id / out_cols / out_rows) % out_depth;
    const int OB = thread_id / out_cols / out_rows / out_depth;

    // Compute the input depth and the index of depth multiplier
    // based off the output depth index that this thread is
    // computing n.
    const int in_d = OD / depth_multiplier;
    const int multiplier = OD % depth_multiplier;

    // Data is stored in the following format (let's assume we
    // flatten the height and width into one contiguous dimension
    // called "P".
    //
    // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
    //
    // Each row contains in_depth * in_rows * in_cols values
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
    const int input_offset_temp = (OB * in_depth + in_d) * (in_rows * in_cols);

    // Finally, we can iterate over the spatial dimensions and perform the
    // convolution, writing into the output at the end.
    //
    // We perform an additional optimization, where we can determine
    // whether the patch fits within the image indices statically, and
    // avoid boundary checking within the loop.
    const int input_row_start = OR * stride - pad_rows;
    const int input_col_start = OC * stride - pad_cols;
    const int input_row_end = input_row_start + filter_rows;
    const int input_col_end = input_col_start + filter_cols;

    T sum = 0;
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_row_end < in_rows && input_col_end < in_cols) {
      // Loop that doesn't need to check for boundary conditions.
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = input_row_start + f_r;
        const int filter_offset_temp = filter_cols * f_r;
        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = input_col_start + f_c;

          const int input_offset =
              (input_offset_temp) + (in_r * in_cols) + in_c;
          const int filter_offset =
              multiplier +
              depth_multiplier * (in_d + in_depth * (f_c + filter_offset_temp));
          sum += ldg(input + input_offset) * ldg(filter + filter_offset);
        }
      }
    } else {
      // Loop that needs to check for boundary conditions.
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = input_row_start + f_r;
        const int filter_offset_temp = filter_cols * f_r;
        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = input_col_start + f_c;
          // TODO(vrv): the in_r check can be done outside of this loop;
          // benchmark both methods to determine the better decision.
          if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
            const int in_c = input_col_start + f_c;

            // input_offset_temp indexes into the start of memory
            // where the spatial data starts.
            const int input_offset =
                (input_offset_temp) + (in_r * in_cols) + in_c;

            const int filter_offset =
                multiplier + depth_multiplier *
                                 (in_d + in_depth * (f_c + filter_offset_temp));
            sum += ldg(input + input_offset) * ldg(filter + filter_offset);
          }
        }
      }
    }

    output[thread_id] = sum;
  }
}

}  // namespace

// A simple launch pad to launch the Cuda kernel for depthwise convolution.
template <typename T>
struct DepthwiseConv2dGPULaunch {
  static void Run(const GPUDevice& d, const DepthwiseArgs args, const T* input,
                  const T* filter, T* output, TensorFormat data_format) {
    // In this kernel, each thread is computing the gradients from one element
    // in the out_backprop. Note that one element in the out_backprop can map
    // to multiple filter elements.
    const int num_outputs =
        args.batch * args.out_rows * args.out_cols * args.out_depth;
    CudaLaunchConfig config = GetCudaLaunchConfig(num_outputs, d);
    if (data_format == FORMAT_NHWC) {
      DepthwiseConv2dGPUKernelNHWC<T>
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
              args, input, filter, output, num_outputs);
    } else if (data_format == FORMAT_NCHW) {
      DepthwiseConv2dGPUKernelNCHW<T>
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
              args, input, filter, output, num_outputs);
    } else {
      assert(false);
    }
  }
};

template struct DepthwiseConv2dGPULaunch<float>;
template struct DepthwiseConv2dGPULaunch<double>;

// A Cuda kernel to compute the depthwise convolution backprop w.r.t. input.
template <typename T, int KNOWN_DEPTH_MULTIPLIER>
__global__ void DepthwiseConv2dBackpropInputGPUKernelNHWC(
    const DepthwiseArgs args, const T* out_backprop, const T* filter,
    T* in_backprop, int num_in_backprop) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows = args.filter_rows;
  const int filter_cols = args.filter_cols;
  const int depth_multiplier = KNOWN_DEPTH_MULTIPLIER == -1
                                   ? args.depth_multiplier
                                   : KNOWN_DEPTH_MULTIPLIER;
  const int stride = args.stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_depth = args.out_depth;

  CUDA_1D_KERNEL_LOOP(thread_id, num_in_backprop) {
    // Compute the indexes of this thread in the output.
    const int in_d = thread_id % in_depth;
    const int in_c = (thread_id / in_depth) % in_cols;
    const int in_r = (thread_id / in_depth / in_cols) % in_rows;
    const int b = thread_id / in_depth / in_cols / in_rows;

    T sum = 0;

    const int out_r_start =
        tf_max<int>(0, (in_r - filter_rows + pad_rows + stride) / stride);
    const int out_r_end = tf_min(out_rows - 1, (in_r + pad_rows) / stride);
    const int out_c_start =
        tf_max(0, (in_c - filter_cols + pad_cols + stride) / stride);
    const int out_c_end = tf_min(out_cols - 1, (in_c + pad_cols) / stride);

#pragma nounroll
    for (int out_r = out_r_start; out_r <= out_r_end; ++out_r) {
      const int f_r = in_r + pad_rows - out_r * stride;
      const int temp_out_backprop_offset =
          out_depth * out_cols * (out_r + out_rows * b);
      const int temp_filter_offset = filter_cols * f_r;
#pragma nounroll
      for (int out_c = out_c_start; out_c <= out_c_end; ++out_c) {
        const int f_c = in_c + pad_cols - out_c * stride;
        int filter_offset =
            depth_multiplier * (in_d + in_depth * (f_c + temp_filter_offset));
        const int out_backprop_offset =
            out_depth * out_c + temp_out_backprop_offset;
#pragma unroll 6
        for (int i = 0; i < depth_multiplier; ++i) {
          sum += ldg(out_backprop + out_backprop_offset +
                     in_d * depth_multiplier + i) *
                 ldg(filter + filter_offset + i);
        }
      }
    }
    const int in_backprop_offset =
        in_d + in_depth * (in_c + in_cols * (in_r + in_rows * b));
    in_backprop[in_backprop_offset] = sum;
  }
}

template <typename T>
__global__ void __launch_bounds__(1024)
    DepthwiseConv2dBackpropInputGPUKernelNCHW(const DepthwiseArgs args,
                                              const T* out_backprop,
                                              const T* filter, T* in_backprop,
                                              int num_in_backprop) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows = args.filter_rows;
  const int filter_cols = args.filter_cols;
  const int depth_multiplier = args.depth_multiplier;
  const int stride = args.stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_depth = args.out_depth;

  // TODO(vrv): Consider assigning threads to output and using
  // atomics for accumulation, similar to the filter case.
  CUDA_1D_KERNEL_LOOP(thread_id, num_in_backprop) {
    // Compute the indexes of this thread in the input.
    const int in_c = thread_id % in_cols;
    const int in_r = (thread_id / in_cols) % in_rows;
    const int in_d = (thread_id / in_cols / in_rows) % in_depth;
    const int b = thread_id / in_depth / in_cols / in_rows;

    T sum = 0;
    const int out_d_start = in_d * depth_multiplier;
    const int out_d_end = out_d_start + depth_multiplier;

    const int out_r_start =
        tf_max<int>(0, (in_r - filter_rows + pad_rows + stride) / stride);
    const int out_r_end = tf_min(out_rows - 1, (in_r + pad_rows) / stride);
    const int out_c_start =
        tf_max(0, (in_c - filter_cols + pad_cols + stride) / stride);
    const int out_c_end = tf_min(out_cols - 1, (in_c + pad_cols) / stride);

    UNROLL for (int out_d = out_d_start; out_d < out_d_end; ++out_d) {
      UNROLL for (int out_r = out_r_start; out_r <= out_r_end; ++out_r) {
        const int f_r = in_r + pad_rows - out_r * stride;
        const int filter_dm = out_d - out_d_start;

        const int temp_filter_offset = filter_cols * f_r;
        for (int out_c = out_c_start; out_c <= out_c_end; ++out_c) {
          const int f_c = in_c + pad_cols - out_c * stride;
          const int filter_offset =
              filter_dm + args.depth_multiplier *
                              (in_d + in_depth * (f_c + temp_filter_offset));

          const int out_backprop_offset =
              (b * out_depth * out_rows * out_cols) +
              (out_d * out_rows * out_cols) + (out_r * out_cols) + (out_c);

          sum += ldg(out_backprop + out_backprop_offset) *
                 ldg(filter + filter_offset);
        }
      }
    }
    const int in_backprop_offset = (b * in_rows * in_cols * in_depth) +
                                   (in_d * in_rows * in_cols) +
                                   (in_r * in_cols) + (in_c);
    in_backprop[in_backprop_offset] = sum;
  }
}

// A simple launch pad to launch the Cuda kernel for depthwise convolution.
template <typename T>
struct DepthwiseConv2dBackpropInputGPULaunch {
  static void Run(const GPUDevice& d, const DepthwiseArgs args,
                  const T* out_backprop, const T* filter, T* in_backprop,
                  TensorFormat data_format) {
    const int num_in_backprop =
        args.batch * args.in_rows * args.in_cols * args.in_depth;

    CudaLaunchConfig config = GetCudaLaunchConfig(num_in_backprop, d);
    // Increase block count for when there are more warps/SM than threads/SM.
    config.block_count *= 4;
    if (data_format == FORMAT_NHWC) {
      if (args.depth_multiplier == 1) {
        DepthwiseConv2dBackpropInputGPUKernelNHWC<T, 1>
            <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                args, out_backprop, filter, in_backprop, num_in_backprop);
      } else {
        DepthwiseConv2dBackpropInputGPUKernelNHWC<T, -1>
            <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
                args, out_backprop, filter, in_backprop, num_in_backprop);
      }
    } else if (data_format == FORMAT_NCHW) {
      DepthwiseConv2dBackpropInputGPUKernelNCHW<T>
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
              args, out_backprop, filter, in_backprop, num_in_backprop);
    } else {
      assert(false);
    }
  }
};

template struct DepthwiseConv2dBackpropInputGPULaunch<float>;
template struct DepthwiseConv2dBackpropInputGPULaunch<double>;

// A Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
template <typename T>
__global__ void DepthwiseConv2dBackpropFilterGPUKernelNHWC(
    const DepthwiseArgs args, const T* out_backprop, const T* input,
    T* filter_backprop, int num_out_backprop) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows = args.filter_rows;
  const int filter_cols = args.filter_cols;
  const int depth_multiplier = args.depth_multiplier;
  const int stride = args.stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_depth = args.out_depth;

  CUDA_1D_KERNEL_LOOP(thread_id, num_out_backprop) {
    // Compute the indexes of this thread in the output.
    const int out_d = thread_id % out_depth;
    const int out_c = (thread_id / out_depth) % out_cols;
    const int out_r = (thread_id / out_depth / out_cols) % out_rows;
    const int b = thread_id / out_depth / out_cols / out_rows;
    // Compute the input depth and the index of depth multiplier.
    const int in_d = out_d / depth_multiplier;
    const int dm = out_d % depth_multiplier;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_r_start = out_r * stride - pad_rows;
    const int in_c_start = out_c * stride - pad_cols;
    const int in_r_end = in_r_start + filter_rows;
    const int in_c_end = in_c_start + filter_cols;

    const int out_backprop_offset =
        out_d + out_depth * (out_c + out_cols * (out_r + out_rows * b));
    const T out_bp = ldg(out_backprop + out_backprop_offset);
    if (in_r_start >= 0 && in_c_start >= 0 && in_r_end < in_rows &&
        in_c_end < in_cols) {
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = in_r_start + f_r;
        // Avoid repeated computation.
        const int input_offset_temp = in_cols * (in_r + in_rows * b);
        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = in_c_start + f_c;

          const int input_offset = in_d + in_depth * (in_c + input_offset_temp);
          T partial_sum = ldg(input + input_offset) * out_bp;
          T* addr = filter_backprop +
                    (dm + depth_multiplier *
                              (in_d + in_depth * (f_c + filter_cols * f_r)));
          CudaAtomicAdd(addr, partial_sum);
        }
      }
    } else {
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = in_r_start + f_r;
        // Avoid repeated computation.
        const int input_offset_temp = in_cols * (in_r + in_rows * b);
        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = in_c_start + f_c;
          const int addr_temp = filter_cols * f_r;

          if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
            const int input_offset =
                in_d + in_depth * (in_c + input_offset_temp);
            T partial_sum = ldg(input + input_offset) * out_bp;
            T* addr =
                filter_backprop +
                (dm + depth_multiplier * (in_d + in_depth * (f_c + addr_temp)));
            // Potentially many threads can add to the same address so we have
            // to use atomic add here.
            // TODO(jmchen): If atomic add turns out to be slow, we can:
            // 1. allocate multiple buffers for the gradients (one for each
            // example in a batch, for example). This can reduce the
            // contention on the destination; 2. Have each thread compute one
            // gradient for an element in the filters. This should work well
            // when the input depth is big and filter size is not too small.
            CudaAtomicAdd(addr, partial_sum);
          }
        }
      }
    }
  }
}

// A Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
template <typename T>
__global__ void DepthwiseConv2dBackpropFilterGPUKernelNCHW(
    const DepthwiseArgs args, const T* out_backprop, const T* input,
    T* filter_backprop, int num_out_backprop) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows = args.filter_rows;
  const int filter_cols = args.filter_cols;
  const int depth_multiplier = args.depth_multiplier;
  const int stride = args.stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_depth = args.out_depth;

  CUDA_1D_KERNEL_LOOP(thread_id, num_out_backprop) {
    // Compute the indexes of this thread in the output.
    const int out_c = thread_id % out_cols;
    const int out_r = (thread_id / out_cols) % out_rows;
    const int out_d = (thread_id / out_cols / out_rows) % out_depth;

    const int b = thread_id / out_depth / out_cols / out_rows;
    // Compute the input depth and the index of depth multiplier.
    const int in_d = out_d / depth_multiplier;
    const int dm = out_d % depth_multiplier;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_r_start = out_r * stride - pad_rows;
    const int in_c_start = out_c * stride - pad_cols;
    const int in_r_end = in_r_start + filter_rows;
    const int in_c_end = in_c_start + filter_cols;

    const int out_backprop_offset = (b * out_depth * out_rows * out_cols) +
                                    (out_d * out_rows * out_cols) +
                                    (out_r * out_cols) + (out_c);

    const T out_bp = ldg(out_backprop + out_backprop_offset);
    if (in_r_start >= 0 && in_c_start >= 0 && in_r_end < in_rows &&
        in_c_end < in_cols) {
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = in_r_start + f_r;
        // Avoid repeated computation.
        const int input_offset_temp = (b * in_depth * in_rows * in_cols) +
                                      (in_d * in_rows * in_cols) +
                                      (in_r * in_cols);

        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = in_c_start + f_c;
          const int input_offset = input_offset_temp + in_c;
          T partial_sum = ldg(input + input_offset) * out_bp;
          T* addr = filter_backprop +
                    (dm + depth_multiplier *
                              (in_d + in_depth * (f_c + filter_cols * f_r)));
          CudaAtomicAdd(addr, partial_sum);
        }
      }
    } else {
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = in_r_start + f_r;
        // Avoid repeated computation.
        const int input_offset_temp = (b * in_depth * in_rows * in_cols) +
                                      (in_d * in_rows * in_cols) +
                                      (in_r * in_cols);
        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = in_c_start + f_c;
          const int addr_temp = filter_cols * f_r;

          if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
            const int input_offset = input_offset_temp + in_c;
            T partial_sum = ldg(input + input_offset) * out_bp;
            T* addr =
                filter_backprop +
                (dm + depth_multiplier * (in_d + in_depth * (f_c + addr_temp)));
            // Potentially many threads can add to the same address so we have
            // to use atomic add here.
            // TODO(jmchen): If atomic add turns out to be slow, we can:
            // 1. allocate multiple buffers for the gradients (one for each
            // example in a batch, for example). This can reduce the
            // contention on the destination; 2. Have each thread compute one
            // gradient for an element in the filters. This should work well
            // when the input depth is big and filter size is not too small.
            CudaAtomicAdd(addr, partial_sum);
          }
        }
      }
    }
  }
}

// A simple launch pad to launch the Cuda kernel for depthwise convolution.
template <typename T>
struct DepthwiseConv2dBackpropFilterGPULaunch {
  static void Run(const GPUDevice& d, const DepthwiseArgs args,
                  const T* out_backprop, const T* input, T* filter_backprop,
                  TensorFormat data_format) {
    // In this kernel, each thread is computing the gradients for one element in
    // the out_backprop.
    const int num_out_backprop =
        args.batch * args.out_rows * args.out_cols * args.out_depth;
    CudaLaunchConfig config = GetCudaLaunchConfig(num_out_backprop, d);

    if (data_format == FORMAT_NHWC) {
      DepthwiseConv2dBackpropFilterGPUKernelNHWC<T>
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
              args, out_backprop, input, filter_backprop, num_out_backprop);
    } else if (data_format == FORMAT_NCHW) {
      DepthwiseConv2dBackpropFilterGPUKernelNCHW<T>
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
              args, out_backprop, input, filter_backprop, num_out_backprop);
    } else {
      assert(false);
    }
  }
};

template struct DepthwiseConv2dBackpropFilterGPULaunch<float>;
template struct DepthwiseConv2dBackpropFilterGPULaunch<double>;
}  // namespace tensorflow
#endif  // GOOGLE_CUDA
