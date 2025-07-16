/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "absl/status/status.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/reshape_util.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename Device>
class SparseReshapeOp : public OpKernel {
 public:
  explicit SparseReshapeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_indices_in = context->input(0);
    const Tensor& input_shape_in = context->input(1);
    const Tensor& target_shape_in = context->input(2);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices_in.shape()),
                absl::InvalidArgumentError("Input must be a matrix."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape_in.shape()),
                absl::InvalidArgumentError("Input shape must be a vector."));
    OP_REQUIRES(context,
                input_indices_in.dim_size(1) == input_shape_in.dim_size(0),
                absl::InvalidArgumentError(
                    "Input tensor rank must match input shape length."));
    ReshapeSparseTensor<Device>(context, input_indices_in, input_shape_in,
                                target_shape_in, 0 /* output indices index */,
                                1 /* output shape index */);
  }
};

REGISTER_KERNEL_BUILDER(Name("SparseReshape").Device(DEVICE_CPU),
                        SparseReshapeOp<CPUDevice>)

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER_KERNEL_BUILDER(Name("SparseReshape")
                            .Device(DEVICE_GPU)
                            .HostMemory("input_shape")
                            .HostMemory("new_shape")
                            .HostMemory("output_shape"),
                        SparseReshapeOp<GPUDevice>)
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
