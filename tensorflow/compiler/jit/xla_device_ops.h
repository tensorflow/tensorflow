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

// Common kernel registrations for XLA devices.

#ifndef TENSORFLOW_COMPILER_JIT_XLA_DEVICE_OPS_H_
#define TENSORFLOW_COMPILER_JIT_XLA_DEVICE_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/constant_op.h"
#include "tensorflow/core/kernels/data/generator_dataset_op.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/kernels/data/optional_ops.h"
#include "tensorflow/core/kernels/data/prefetch_dataset_op.h"
#include "tensorflow/core/kernels/fifo_queue.h"
#include "tensorflow/core/kernels/function_ops.h"
#include "tensorflow/core/kernels/identity_op.h"
#include "tensorflow/core/kernels/resource_variable_ops.h"
#include "tensorflow/core/kernels/shape_ops.h"
#include "tensorflow/core/kernels/variable_ops.h"

namespace tensorflow {

// Dummy OpKernel, used for kernels assigned to an XLA device that should be
// compiled. Should never be called at runtime since such ops should be
// rewritten to a XlaLaunch op. If it is called, it means the placer placed an
// operator on an XLA device but the compiler did not compile it.
class XlaDeviceDummyOp : public OpKernel {
 public:
  explicit XlaDeviceDummyOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;
};

class XlaAssignVariableOp : public OpKernel {
 public:
  explicit XlaAssignVariableOp(OpKernelConstruction* c);
  void Compute(OpKernelContext* context) override;

 private:
  DataType dtype_;
};

#define REGISTER_XLA_LAUNCH_KERNEL(DEVICE, KERNEL, TYPES) \
  REGISTER_KERNEL_BUILDER(Name("XlaLaunch")               \
                              .Device(DEVICE)             \
                              .HostMemory("constants")    \
                              .HostMemory("resources"),   \
                          KERNEL);

#define REGISTER_XLA_COMPILE_KERNEL(DEVICE, KERNEL, TYPES)          \
  REGISTER_KERNEL_BUILDER(Name("_XlaCompile")                       \
                              .Device(DEVICE)                       \
                              .HostMemory("constants")              \
                              .HostMemory("key")                    \
                              .HostMemory("compilation_successful") \
                              .HostMemory("resources"),             \
                          KERNEL);

#define REGISTER_XLA_RUN_KERNEL(DEVICE, KERNEL, TYPES) \
  REGISTER_KERNEL_BUILDER(Name("_XlaRun").Device(DEVICE), KERNEL);

#define REGISTER_XLA_DEVICE_KERNELS(DEVICE, TYPES)                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Const").Device(DEVICE).TypeConstraint("dtype", TYPES),             \
      ConstantOp);                                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Identity").Device(DEVICE).TypeConstraint("T", TYPES), IdentityOp); \
                                                                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("VarHandleOp").Device(DEVICE).HostMemory("resource"), VarHandleOp); \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_VarHandlesOp").Device(DEVICE).HostMemory("resources"),            \
      ResourceHandlesOp<Var>);                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ReadVariableOp").Device(DEVICE).HostMemory("resource"),            \
      ReadVariableOp);                                                         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ReadVariablesOp").Device(DEVICE).HostMemory("resources"),         \
      ReadVariablesOp);                                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("DestroyResourceOp").Device(DEVICE).HostMemory("resource"),         \
      DestroyResourceOp);                                                      \
  REGISTER_KERNEL_BUILDER(Name("Shape")                                        \
                              .Device(DEVICE)                                  \
                              .HostMemory("output")                            \
                              .TypeConstraint<int32>("out_type")               \
                              .TypeConstraint("T", TYPES),                     \
                          ShapeOp<int32>);                                     \
  REGISTER_KERNEL_BUILDER(Name("Shape")                                        \
                              .Device(DEVICE)                                  \
                              .HostMemory("output")                            \
                              .TypeConstraint<int64>("out_type")               \
                              .TypeConstraint("T", TYPES),                     \
                          ShapeOp<int64>);                                     \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                                       \
                              .Device(DEVICE)                                  \
                              .HostMemory("output")                            \
                              .TypeConstraint<int32>("out_type")               \
                              .TypeConstraint("T", TYPES),                     \
                          ShapeNOp<int32>);                                    \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                                       \
                              .Device(DEVICE)                                  \
                              .HostMemory("output")                            \
                              .TypeConstraint<int64>("out_type")               \
                              .TypeConstraint("T", TYPES),                     \
                          ShapeNOp<int64>);                                    \
  REGISTER_KERNEL_BUILDER(Name("Size")                                         \
                              .Device(DEVICE)                                  \
                              .HostMemory("output")                            \
                              .TypeConstraint<int32>("out_type")               \
                              .TypeConstraint("T", TYPES),                     \
                          SizeOp<int32>);                                      \
  REGISTER_KERNEL_BUILDER(Name("Size")                                         \
                              .Device(DEVICE)                                  \
                              .HostMemory("output")                            \
                              .TypeConstraint<int64>("out_type")               \
                              .TypeConstraint("T", TYPES),                     \
                          SizeOp<int64>);                                      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Rank").Device(DEVICE).HostMemory("output").TypeConstraint("T",     \
                                                                      TYPES),  \
      RankOp);                                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("AssignVariableOp").Device(DEVICE).HostMemory("resource"),          \
      XlaAssignVariableOp);                                                    \
                                                                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("FIFOQueueV2").Device(DEVICE).HostMemory("handle"), FIFOQueueOp);   \
                                                                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(kArgOp).Device(DEVICE).TypeConstraint("T", TYPES), ArgOp);          \
  REGISTER_KERNEL_BUILDER(Name(kArgOp)                                         \
                              .Device(DEVICE)                                  \
                              .HostMemory("output")                            \
                              .TypeConstraint<ResourceHandle>("T"),            \
                          ArgOp);                                              \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(kArgOp).Device(DEVICE).TypeConstraint<Variant>("T"), ArgOp);        \
                                                                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(kRetOp).Device(DEVICE).TypeConstraint("T", TYPES), RetvalOp);       \
  REGISTER_KERNEL_BUILDER(Name(kRetOp)                                         \
                              .Device(DEVICE)                                  \
                              .TypeConstraint<ResourceHandle>("T")             \
                              .HostMemory("input"),                            \
                          RetvalOp);                                           \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(kDeviceRetOp).Device(DEVICE).TypeConstraint<int32>("T"), RetvalOp); \
                                                                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RemoteCall").Device(DEVICE).HostMemory("target"), RemoteCallOp);   \
                                                                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("GeneratorDataset").Device(DEVICE).HostMemory("handle"),            \
      data::GeneratorDatasetOp);                                               \
  REGISTER_KERNEL_BUILDER(Name("PrefetchDataset")                              \
                              .Device(DEVICE)                                  \
                              .HostMemory("buffer_size")                       \
                              .HostMemory("input_dataset")                     \
                              .HostMemory("handle"),                           \
                          data::PrefetchDatasetOp);                            \
                                                                               \
  REGISTER_KERNEL_BUILDER(Name("IteratorV2").Device(DEVICE),                   \
                          data::IteratorHandleOp);                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MakeIterator").Device(DEVICE).HostMemory("dataset"),               \
      data::MakeIteratorOp);                                                   \
  REGISTER_KERNEL_BUILDER(Name("AnonymousIterator").Device(DEVICE),            \
                          data::AnonymousIteratorHandleOp);                    \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("AnonymousIteratorV2").Device(DEVICE).HostMemory("deleter"),        \
      data::AnonymousIteratorHandleOp);                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("DeleteIterator").Device(DEVICE).HostMemory("deleter"),             \
      data::DeleteIteratorOp);                                                 \
  REGISTER_KERNEL_BUILDER(Name("IteratorGetNext").Device(DEVICE),              \
                          data::IteratorGetNextOp);                            \
  REGISTER_KERNEL_BUILDER(Name("IteratorGetNextAsOptional").Device(DEVICE),    \
                          data::IteratorGetNextAsOptionalOp);                  \
  REGISTER_KERNEL_BUILDER(Name("IteratorGetNextSync").Device(DEVICE),          \
                          data::IteratorGetNextSyncOp);                        \
  REGISTER_KERNEL_BUILDER(Name("IteratorToStringHandle")                       \
                              .Device(DEVICE)                                  \
                              .HostMemory("string_handle"),                    \
                          data::IteratorToStringHandleOp);                     \
  REGISTER_KERNEL_BUILDER(Name("IteratorFromStringHandleV2")                   \
                              .Device(DEVICE)                                  \
                              .HostMemory("string_handle"),                    \
                          data::IteratorFromStringHandleOp);                   \
  REGISTER_KERNEL_BUILDER(Name("OptionalNone").Device(DEVICE),                 \
                          data::OptionalNoneOp);                               \
  REGISTER_KERNEL_BUILDER(Name("OptionalFromValue").Device(DEVICE),            \
                          data::OptionalFromValueOp);                          \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("OptionalHasValue").Device(DEVICE).HostMemory("has_value"),         \
      data::OptionalHasValueOp);                                               \
  REGISTER_KERNEL_BUILDER(Name("OptionalGetValue").Device(DEVICE),             \
                          data::OptionalGetValueOp);                           \
  REGISTER_KERNEL_BUILDER(Name(FunctionLibraryDefinition::kArgOp)              \
                              .Device(DEVICE)                                  \
                              .HostMemory("output")                            \
                              .TypeConstraint<tstring>("T"),                   \
                          ArgOp);                                              \
  REGISTER_KERNEL_BUILDER(Name(FunctionLibraryDefinition::kRetOp)              \
                              .Device(DEVICE)                                  \
                              .TypeConstraint<tstring>("T")                    \
                              .HostMemory("input"),                            \
                          RetvalOp);

// TODO(b/118881356): currently we do not register the QueueEnqueueMany,
// QueueDequeueMany, or QueueDequeueUpTo kernels because they attempt to read
// and write the tensors they access in order to concatenate them into a batch.
// We would need either to call out to an XLA computation to perform the
// concatenation, or we would need to refactor those kernels so the splitting
// or merging is done in a separate operator that can be compiled.

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_DEVICE_OPS_H_
