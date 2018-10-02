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
#include "tensorflow/core/kernels/cast_op.h"
#include "tensorflow/core/kernels/constant_op.h"
#include "tensorflow/core/kernels/control_flow_ops.h"
#include "tensorflow/core/kernels/data/generator_dataset_op.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/kernels/data/prefetch_dataset_op.h"
#include "tensorflow/core/kernels/fifo_queue.h"
#include "tensorflow/core/kernels/function_ops.h"
#include "tensorflow/core/kernels/identity_n_op.h"
#include "tensorflow/core/kernels/identity_op.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/kernels/queue_op.h"
#include "tensorflow/core/kernels/resource_variable_ops.h"
#include "tensorflow/core/kernels/sendrecv_ops.h"
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

class XlaAssignVariableOp : public AsyncOpKernel {
 public:
  explicit XlaAssignVariableOp(OpKernelConstruction* c);
  void ComputeAsync(OpKernelContext* context, DoneCallback done) override;

 private:
  DataType dtype_;
};

#define REGISTER_XLA_LAUNCH_KERNEL(DEVICE, KERNEL, TYPES) \
  REGISTER_KERNEL_BUILDER(Name("XlaLaunch")               \
                              .Device(DEVICE)             \
                              .HostMemory("constants")    \
                              .HostMemory("resources"),   \
                          KERNEL);

#define REGISTER_XLA_COMPILE_KERNEL(DEVICE, KERNEL, TYPES) \
  REGISTER_KERNEL_BUILDER(Name("_XlaCompile")              \
                              .Device(DEVICE)              \
                              .HostMemory("constants")     \
                              .HostMemory("resources"),    \
                          KERNEL);

#define REGISTER_XLA_RUN_KERNEL(DEVICE, KERNEL, TYPES) \
  REGISTER_KERNEL_BUILDER(Name("_XlaRun").Device(DEVICE), KERNEL);

#define REGISTER_XLA_DEVICE_KERNELS(DEVICE, TYPES)                             \
  REGISTER_KERNEL_BUILDER(Name("_Send").Device(DEVICE), SendOp);               \
  REGISTER_KERNEL_BUILDER(Name("_Recv").Device(DEVICE), RecvOp);               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_HostSend").Device(DEVICE).HostMemory("tensor"), SendOp);          \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_HostRecv").Device(DEVICE).HostMemory("tensor"), RecvOp);          \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_HostCast").Device(DEVICE).HostMemory("x").HostMemory("y"),        \
      CpuCastOp);                                                              \
  REGISTER_KERNEL_BUILDER(Name("NoOp").Device(DEVICE), NoOp);                  \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Const").Device(DEVICE).TypeConstraint("dtype", TYPES),             \
      ConstantOp);                                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Identity").Device(DEVICE).TypeConstraint("T", TYPES), IdentityOp); \
  REGISTER_KERNEL_BUILDER(Name("IdentityN").Device(DEVICE), IdentityNOp);      \
  REGISTER_KERNEL_BUILDER(Name("Placeholder").Device(DEVICE), PlaceholderOp);  \
  REGISTER_KERNEL_BUILDER(Name("PlaceholderV2").Device(DEVICE),                \
                          PlaceholderOp);                                      \
                                                                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("VarHandleOp").Device(DEVICE).HostMemory("resource"),               \
      ResourceHandleOp<Var>);                                                  \
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
  REGISTER_KERNEL_BUILDER(Name("ControlTrigger").Device(DEVICE),               \
                          ControlTriggerOp);                                   \
  REGISTER_KERNEL_BUILDER(Name("Switch").Device(DEVICE).HostMemory("pred"),    \
                          SwitchOp);                                           \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Merge").Device(DEVICE).HostMemory("value_index"), MergeOp);        \
  REGISTER_KERNEL_BUILDER(Name("Enter").Device(DEVICE), EnterOp);              \
  REGISTER_KERNEL_BUILDER(Name("Exit").Device(DEVICE), ExitOp);                \
  REGISTER_KERNEL_BUILDER(Name("NextIteration").Device(DEVICE),                \
                          NextIterationOp);                                    \
  REGISTER_KERNEL_BUILDER(Name("LoopCond")                                     \
                              .Device(DEVICE)                                  \
                              .HostMemory("input")                             \
                              .HostMemory("output"),                           \
                          LoopCondOp);                                         \
                                                                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("QueueEnqueueV2").Device(DEVICE).HostMemory("handle"), EnqueueOp);  \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("QueueDequeueV2").Device(DEVICE).HostMemory("handle"), DequeueOp);  \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("QueueCloseV2").Device(DEVICE).HostMemory("handle"), QueueCloseOp); \
  REGISTER_KERNEL_BUILDER(Name("QueueSizeV2")                                  \
                              .Device(DEVICE)                                  \
                              .HostMemory("size")                              \
                              .HostMemory("handle"),                           \
                          QueueSizeOp);                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("QueueIsClosedV2").Device(DEVICE).HostMemory("handle"),             \
      QueueIsClosedOp);                                                        \
                                                                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("FIFOQueueV2").Device(DEVICE).HostMemory("handle"), FIFOQueueOp);   \
                                                                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(kArgOp).Device(DEVICE).HostMemory("output").TypeConstraint("T",     \
                                                                      TYPES),  \
      ArgOp);                                                                  \
  REGISTER_KERNEL_BUILDER(Name(kArgOp)                                         \
                              .Device(DEVICE)                                  \
                              .HostMemory("output")                            \
                              .TypeConstraint<ResourceHandle>("T"),            \
                          ArgOp);                                              \
                                                                               \
  REGISTER_KERNEL_BUILDER(Name(kRetOp)                                         \
                              .Device(DEVICE)                                  \
                              .TypeConstraint("T", TYPES)                      \
                              .HostMemory("input"),                            \
                          RetvalOp);                                           \
  REGISTER_KERNEL_BUILDER(Name(kRetOp)                                         \
                              .Device(DEVICE)                                  \
                              .TypeConstraint<ResourceHandle>("T")             \
                              .HostMemory("input"),                            \
                          RetvalOp);                                           \
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
  REGISTER_KERNEL_BUILDER(Name("IteratorGetNext").Device(DEVICE),              \
                          data::IteratorGetNextOp);                            \
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
  REGISTER_KERNEL_BUILDER(Name(FunctionLibraryDefinition::kArgOp)              \
                              .Device(DEVICE)                                  \
                              .HostMemory("output")                            \
                              .TypeConstraint<string>("T"),                    \
                          ArgOp);                                              \
  REGISTER_KERNEL_BUILDER(Name(FunctionLibraryDefinition::kRetOp)              \
                              .Device(DEVICE)                                  \
                              .TypeConstraint<string>("T")                     \
                              .HostMemory("input"),                            \
                          RetvalOp);

// TODO(phawkins): currently we do not register the QueueEnqueueMany,
// QueueDequeueMany, or QueueDequeueUpTo kernels because they attempt to read
// and write the tensors they access in order to concatenate them into a batch.
// We would need either to call out to an XLA computation to perform the
// concatenation, or we would need to refactor those kernels so the splitting
// or merging is done in a separate operator that can be compiled.

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_DEVICE_OPS_H_
