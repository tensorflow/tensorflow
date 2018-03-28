/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.
#include "tensorflow/core/kernels/snapshot_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

#define REGISTER_KERNEL(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Snapshot").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      SnapshotOp<CPUDevice, TYPE>);

TF_CALL_POD_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SyclDevice;
#define REGISTER_SYCL_KERNEL(TYPE)                                    \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Snapshot").Device(DEVICE_SYCL).TypeConstraint<TYPE>("T"), \
      SnapshotOp<SyclDevice, TYPE>);

TF_CALL_POD_TYPES(REGISTER_SYCL_KERNEL);

#undef REGISTER_SYCL_KERNEL
#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
