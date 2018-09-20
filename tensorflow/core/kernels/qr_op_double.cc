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

#include "tensorflow/core/kernels/qr_op_impl.h"

namespace tensorflow {

REGISTER_LINALG_OP("Qr", (QrOp<double>), double);

#if GOOGLE_CUDA
// We temporarily disable QR on GPU due to a bug in the QR implementation in
// cuSolver affecting older hardware. The cuSolver team is tracking the issue
// (https://partners.nvidia.com/bug/viewbug/2171459) and we will re-enable
// this feature when a fix is available.
REGISTER_KERNEL_BUILDER(Name("Qr")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T")
                            .HostMemory("input")
                            .HostMemory("q")
                            .HostMemory("r"),
                        QrOp<double>);
#endif

}  // namespace tensorflow
