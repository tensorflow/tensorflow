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

// See docs in ../ops/data_flow_ops.cc.

#include "tensorflow/core/kernels/stack.h"

#include <limits.h>
#include <atomic>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("Stack").Device(DEVICE_CPU), StackOp);
REGISTER_KERNEL_BUILDER(
    Name("Stack").Device(DEVICE_DEFAULT).HostMemory("handle"), StackOp);

REGISTER_KERNEL_BUILDER(Name("StackV2").Device(DEVICE_CPU), StackOp);
REGISTER_KERNEL_BUILDER(Name("StackV2")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("max_size")
                            .HostMemory("handle"),
                        StackOp);

REGISTER_KERNEL_BUILDER(Name("StackPush").Device(DEVICE_CPU),
                        TemplatedStackPushOp</*allow_swapping=*/false>);
REGISTER_KERNEL_BUILDER(Name("StackPushV2").Device(DEVICE_CPU),
                        TemplatedStackPushOp</*allow_swapping=*/false>);

REGISTER_KERNEL_BUILDER(Name("StackPop").Device(DEVICE_CPU), StackPopOp);
REGISTER_KERNEL_BUILDER(Name("StackPopV2").Device(DEVICE_CPU), StackPopOp);

#define REGISTER_DEFAULT_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(Name("StackPush")                               \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .TypeConstraint<type>("T"),                 \
                          TemplatedStackPushOp</*allow_swapping=*/true>); \
  REGISTER_KERNEL_BUILDER(Name("StackPushV2")                             \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .TypeConstraint<type>("T"),                 \
                          TemplatedStackPushOp</*allow_swapping=*/true>); \
  REGISTER_KERNEL_BUILDER(Name("StackPop")                                \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .TypeConstraint<type>("elem_type"),         \
                          StackPopOp);                                    \
  REGISTER_KERNEL_BUILDER(Name("StackPopV2")                              \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .TypeConstraint<type>("elem_type"),         \
                          StackPopOp);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL

// Special GPU kernels for int32 and bool.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_DEFAULT_HOST_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(Name("StackPush")                               \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .HostMemory("elem")                         \
                              .HostMemory("output")                       \
                              .TypeConstraint<type>("T"),                 \
                          TemplatedStackPushOp</*allow_swapping=*/true>); \
  REGISTER_KERNEL_BUILDER(Name("StackPushV2")                             \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .HostMemory("elem")                         \
                              .HostMemory("output")                       \
                              .TypeConstraint<type>("T"),                 \
                          TemplatedStackPushOp</*allow_swapping=*/true>); \
  REGISTER_KERNEL_BUILDER(Name("StackPop")                                \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .HostMemory("elem")                         \
                              .TypeConstraint<type>("elem_type"),         \
                          StackPopOp);                                    \
  REGISTER_KERNEL_BUILDER(Name("StackPopV2")                              \
                              .Device(DEVICE_DEFAULT)                     \
                              .HostMemory("handle")                       \
                              .HostMemory("elem")                         \
                              .TypeConstraint<type>("elem_type"),         \
                          StackPopOp);

REGISTER_DEFAULT_HOST_KERNEL(int32);
REGISTER_DEFAULT_HOST_KERNEL(bool);

#undef REGISTER_DEFAULT_HOST_KERNEL

REGISTER_KERNEL_BUILDER(Name("StackClose").Device(DEVICE_CPU), StackCloseOp);
REGISTER_KERNEL_BUILDER(
    Name("StackClose").Device(DEVICE_DEFAULT).HostMemory("handle"),
    StackCloseOp);
REGISTER_KERNEL_BUILDER(Name("StackCloseV2").Device(DEVICE_CPU), StackCloseOp);
REGISTER_KERNEL_BUILDER(
    Name("StackCloseV2").Device(DEVICE_DEFAULT).HostMemory("handle"),
    StackCloseOp);

}  // namespace tensorflow
