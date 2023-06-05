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

#include "tensorflow/core/kernels/linalg/qr_op_impl.h"

namespace tensorflow {

REGISTER_LINALG_OP_CPU("Qr", (QrOp<Eigen::half>), Eigen::half);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// We temporarily disable QR on GPU and register a fake GPU kernel for QR
// instead, which allows colocation constraints to not be violated. cl/211112318
// (https://partners.nvidia.com/bug/viewbug/2171459)

REGISTER_KERNEL_BUILDER(Name("Qr")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .HostMemory("input")
                            .HostMemory("q")
                            .HostMemory("r"),
                        QrOp<Eigen::half>);
#endif

}  // namespace tensorflow
