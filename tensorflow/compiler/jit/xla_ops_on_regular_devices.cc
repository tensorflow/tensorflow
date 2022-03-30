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

// Register XlaXXX operations on regular CPU/GPU devices using
// `XlaCompileOnDemandOp`.
#include "tensorflow/compiler/jit/xla_compile_on_demand_op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

#define REGISTER_XLA_OPS_ON_DEVICE(DEVICE)                                     \
  REGISTER_KERNEL_BUILDER(Name("XlaConv")                                      \
                              .HostMemory("window_strides")                    \
                              .HostMemory("padding")                           \
                              .HostMemory("lhs_dilation")                      \
                              .HostMemory("rhs_dilation")                      \
                              .HostMemory("feature_group_count")               \
                              .Device(DEVICE),                                 \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaConvV2")                                    \
                              .HostMemory("window_strides")                    \
                              .HostMemory("padding")                           \
                              .HostMemory("lhs_dilation")                      \
                              .HostMemory("rhs_dilation")                      \
                              .HostMemory("feature_group_count")               \
                              .Device(DEVICE),                                 \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("XlaBroadcastHelper").HostMemory("broadcast_dims").Device(DEVICE),  \
      XlaCompileOnDemandOp);                                                   \
  REGISTER_KERNEL_BUILDER(Name("XlaSelfAdjointEig").Device(DEVICE),            \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaSvd").Device(DEVICE),                       \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaDot").Device(DEVICE),                       \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaDotV2").Device(DEVICE),                     \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("XlaDynamicSlice").HostMemory("size_indices").Device(DEVICE),       \
      XlaCompileOnDemandOp);                                                   \
  REGISTER_KERNEL_BUILDER(Name("XlaDynamicUpdateSlice").Device(DEVICE),        \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaIf").Device(DEVICE), XlaCompileOnDemandOp); \
  REGISTER_KERNEL_BUILDER(Name("XlaOptimizationBarrier").Device(DEVICE),       \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaPad")                                       \
                              .HostMemory("padding_low")                       \
                              .HostMemory("padding_high")                      \
                              .HostMemory("padding_interior")                  \
                              .Device(DEVICE),                                 \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaRecv").Device(DEVICE),                      \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaReduce").Device(DEVICE),                    \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaVariadicReduce").Device(DEVICE),            \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaVariadicReduceV2").Device(DEVICE),          \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaReduceWindow")                              \
                              .HostMemory("window_dimensions")                 \
                              .HostMemory("window_strides")                    \
                              .HostMemory("base_dilations")                    \
                              .HostMemory("window_dilations")                  \
                              .HostMemory("padding")                           \
                              .Device(DEVICE),                                 \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaSelectAndScatter")                          \
                              .HostMemory("window_dimensions")                 \
                              .HostMemory("window_strides")                    \
                              .HostMemory("padding")                           \
                              .Device(DEVICE),                                 \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaSend").Device(DEVICE),                      \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaSort").Device(DEVICE),                      \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaKeyValueSort").Device(DEVICE),              \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("XlaVariadicSort").HostMemory("dimension").Device(DEVICE),          \
      XlaCompileOnDemandOp);                                                   \
  REGISTER_KERNEL_BUILDER(Name("XlaWhile").Device(DEVICE),                     \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaDequantize").Device(DEVICE),                \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaEinsum").Device(DEVICE),                    \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaRngBitGenerator")                           \
                              .HostMemory("algorithm")                         \
                              .HostMemory("shape")                             \
                              .Device(DEVICE),                                 \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaSpmdShardToFullShape").Device(DEVICE),      \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaSharding").Device(DEVICE),                  \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(Name("XlaReplicaId").Device(DEVICE),                 \
                          XlaCompileOnDemandOp);                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("XlaGather").HostMemory("slice_sizes").Device(DEVICE),              \
      XlaCompileOnDemandOp);                                                   \
  REGISTER_KERNEL_BUILDER(Name("XlaScatter").Device(DEVICE),                   \
                          XlaCompileOnDemandOp);

REGISTER_XLA_OPS_ON_DEVICE(DEVICE_CPU);
REGISTER_XLA_OPS_ON_DEVICE(DEVICE_GPU);

}  // namespace tensorflow
