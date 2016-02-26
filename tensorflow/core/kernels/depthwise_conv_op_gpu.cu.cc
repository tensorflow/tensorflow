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

#include "tensorflow/core/kernels/depthwise_conv_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#define UNROLL _Pragma("unroll")

namespace tensorflow {

namespace {

typedef Eigen::GpuDevice GPUDevice;

// A Cuda kernel to compute the depthwise convolution.
template <typename T>
__global__ void DepthwiseConv2dGPUKernel(const DepthwiseArgs args,
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

    // Decide if all input is valid, if yes, we can skip the boundary checks for
    // each input.
    const int input_row_start = OR * stride - pad_rows;
    const int input_col_start = OC * stride - pad_cols;
    const int input_row_end = input_row_start + filter_rows;
    const int input_col_end = input_col_start + filter_cols;

    float sum = 0;
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_row_end < in_rows && input_col_end < in_cols) {
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          int in_r = input_row_start + f_r;
          int in_c = input_col_start + f_c;

          sum += input[in_d +
                       in_depth * (in_c + in_cols * (in_r + in_rows * OB))] *
                 filter[multiplier +
                        depth_multiplier *
                            (in_d + in_depth * (f_c + filter_cols * f_r))];
        }
      }
    } else {
      UNROLL for (int f_r = 0; f_r < filter_rows; ++f_r) {
        UNROLL for (int f_c = 0; f_c < filter_cols; ++f_c) {
          int in_r = input_row_start + f_r;
          int in_c = input_col_start + f_c;

          if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
            sum += input[in_d +
                         in_depth * (in_c + in_cols * (in_r + in_rows * OB))] *
                   filter[multiplier +
                          depth_multiplier *
                              (in_d + in_depth * (f_c + filter_cols * f_r))];
          }
        }
      }
    }
    output[OD + out_depth * (OC + out_cols * (OR + out_rows * OB))] = sum;
  }
}
}  // namespace

// A simple launch pad to launch the Cuda kernel for depthwise convolution.
template <typename T>
struct DepthwiseConv2dGPULaunch {
  void Run(const GPUDevice& d, const DepthwiseArgs args, const T* input,
           const T* filter, T* output) {
    const int num_outputs =
        args.batch * args.out_rows * args.out_cols * args.out_depth;
    CudaLaunchConfig config = GetCudaLaunchConfig(num_outputs, d);

    DepthwiseConv2dGPUKernel<
        T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        args, input, filter, output, num_outputs);
  }
};

template struct DepthwiseConv2dGPULaunch<float>;
template struct DepthwiseConv2dGPULaunch<double>;

}  // namespace tensorflow
#endif  // GOOGLE_CUDA
