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

#include "tensorflow/examples/custom_ops_doc/multiplex_2/multiplex_2_kernel.h"

// Please use the appropriate namespace for your project
namespace tensorflow {
namespace custom_op_examples {

using CPUDevice = Eigen::ThreadPoolDevice;

// To support tensors containing different types (e.g. int32, float), one
// kernel per type is registered and is templatized by the "T" attr value.
// See go/tf-custom-ops-guide
#define REGISTER_KERNELS_CPU(type)                              \
  REGISTER_KERNEL_BUILDER(Name("Examples>MultiplexDense")       \
                              .Device(::tensorflow::DEVICE_CPU) \
                              .TypeConstraint<type>("T"),       \
                          MultiplexDenseOp<CPUDevice, type>)
TF_CALL_ALL_TYPES(REGISTER_KERNELS_CPU);

#undef REGISTER_KERNELS_CPU

}  // namespace custom_op_examples
}  // namespace tensorflow
