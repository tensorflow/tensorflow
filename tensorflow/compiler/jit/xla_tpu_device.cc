/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#if !defined(PLATFORM_GOOGLE)

#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {

// We automatically register this if we are building for open source. For
// Google platforms, we initialize these devices in other places.

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_TPU_NODE, XlaLocalLaunchOp, kTpuAllTypes);
REGISTER_XLA_COMPILE_KERNEL(DEVICE_TPU_NODE, XlaCompileOp, kTpuAllTypes);
REGISTER_XLA_RUN_KERNEL(DEVICE_TPU_NODE, XlaRunOp, kTpuAllTypes);

}  // namespace tensorflow

#endif  // PLATFORM_GOOGLE
