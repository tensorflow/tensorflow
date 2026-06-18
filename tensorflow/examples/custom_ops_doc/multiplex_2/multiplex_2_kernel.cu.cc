/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/examples/custom_ops_doc/multiplex_2/multiplex_2_kernel.h"

// Please use the appropriate namespace for your project
namespace tensorflow {
namespace custom_op_examples {

using GPUDevice = Eigen::GpuDevice;

// To support tensors containing different types (e.g. int32, float), one
// kernel per type is registered and is templatized by the "T" attr value.
// See go/tf-custom-ops-guide
#define REGISTER_KERNELS_GPU(type)                              \
  REGISTER_KERNEL_BUILDER(Name("Examples>MultiplexDense")       \
                              .Device(::tensorflow::DEVICE_GPU) \
                              .TypeConstraint<type>("T"),       \
                          MultiplexDenseOp<GPUDevice, type>)

REGISTER_KERNELS_GPU(bool);
REGISTER_KERNELS_GPU(Eigen::half);
REGISTER_KERNELS_GPU(float);
REGISTER_KERNELS_GPU(double);
REGISTER_KERNELS_GPU(int64);
REGISTER_KERNELS_GPU(complex64);
REGISTER_KERNELS_GPU(complex128);

#undef REGISTER_KERNELS_GPU

}  // namespace custom_op_examples
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
