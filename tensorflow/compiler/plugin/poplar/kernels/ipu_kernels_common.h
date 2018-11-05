/* Copyright 2017 Graphcore Ltd
 */

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_IPU_KERNELS_COMMON_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_IPU_KERNELS_COMMON_H_

#define REGISTER_IPU_OP(OP_NAME, IMPL)                                     \
  REGISTER_KERNEL_BUILDER(Name(OP_NAME).Device(DEVICE_IPU_XLA_JIT), IMPL); \
  REGISTER_KERNEL_BUILDER(Name(OP_NAME).Device(DEVICE_XLA_IPU), IMPL);     \
  REGISTER_KERNEL_BUILDER(Name(OP_NAME).Device(DEVICE_XLA_IPU_REP), IMPL);

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_KERNELS_IPU_KERNELS_COMMON_H_
