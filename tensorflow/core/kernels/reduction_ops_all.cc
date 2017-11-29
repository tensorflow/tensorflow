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

#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(
    Name("All")
        .TypeConstraint<int32>("Tidx")
        .Device(DEVICE_CPU)
        .HostMemory("reduction_indices"),
    ReductionOp<CPUDevice, bool, Eigen::internal::AndReducer>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("All")
        .TypeConstraint<int32>("Tidx")
        .Device(DEVICE_GPU)
        .HostMemory("reduction_indices"),
    ReductionOp<GPUDevice, bool, Eigen::internal::AndReducer>);
#endif

}  // namespace tensorflow
