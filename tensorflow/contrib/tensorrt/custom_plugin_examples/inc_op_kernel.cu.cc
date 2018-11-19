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

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

#include "tensorflow/contrib/tensorrt/custom_plugin_examples/inc_op_kernel.h"

#include <vector>

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op_kernel.h"
#include "cuda/include/cuda_runtime_api.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {
namespace tensorrt {

__global__ void VecInc(const float* vec, float inc, float* dest, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) dest[i] = vec[i] + inc;
}

void IncrementKernel(const float* d_input, float inc, float* d_output,
                     int count, cudaStream_t stream) {
  int threads_per_block = 256;
  int blocks_per_grid = (count + threads_per_block - 1) / threads_per_block;

  VecInc<<<threads_per_block, blocks_per_grid, 0, stream>>>(d_input, inc,
                                                            d_output, count);
}

// Note: this kernel definition is not needed in the plugin_test rule, but it is
// required for correctness of the TF program, i.e. if not using plugin or when
// run with trt optimization pass, the test should work.
class IncPluginTRT : public OpKernel {
 public:
  explicit IncPluginTRT(OpKernelConstruction* context) : OpKernel(context) {
    std::vector<float> inc_list;
    OP_REQUIRES_OK(context, context->GetAttr("inc", &inc_list));
    OP_REQUIRES(context, inc_list.size() == 1,
                errors::InvalidArgument(
                    "The increment list should contain single element."));
    inc_ = inc_list[0];
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const TensorShape& input_shape = input_tensor.shape();
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &output_tensor));
    const cudaStream_t* stream = CHECK_NOTNULL(
        reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                  ->stream()
                                                  ->implementation()
                                                  ->GpuStreamMemberHack()));
    IncrementKernel(input_tensor.flat<float>().data(), inc_,
                    output_tensor->flat<float>().data(),
                    input_shape.num_elements(), *stream);
  }

 private:
  float inc_;
};

REGISTER_KERNEL_BUILDER(Name("IncPluginTRT").Device(DEVICE_GPU), IncPluginTRT);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
