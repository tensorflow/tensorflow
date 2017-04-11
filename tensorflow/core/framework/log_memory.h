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

#ifndef TENSORFLOW_FRAMEWORK_LOG_MEMORY_H_
#define TENSORFLOW_FRAMEWORK_LOG_MEMORY_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// LogMemory contains methods for recording memory allocations and
// frees, associating each allocation with a step identified by a
// process-wide id. For now, logging is enabled whenever VLOG_IS_ON(1)
// for the log_memory module.
//
// Limitations: We don't log memory allocations by Eigen on the CPU
// since that would require major changes to plumb through to the
// Eigen::{DefaultDevice,ThreadPoolDevice} allocate and deallocate
// methods. We do log Eigen allocations on GPU since the plumbing was
// already in place.
class LogMemory {
 public:
  // Allocations sometimes happen outside any computation step, and
  // SpecialStepIds lists the ids used for those steps.
  enum SpecialStepIds {
    // Used when performing a just-in-time constant folding optimization.
    CONSTANT_FOLDING_STEP_ID = -1,
    // Used when constructing an Op kernel before executing a step.
    OP_KERNEL_CONSTRUCTION_STEP_ID = -2,
    // Used when allocating a tensor buffer from external code, e.g.,
    // the C API.
    EXTERNAL_TENSOR_ALLOCATION_STEP_ID = -3,
    // Used when allocating a buffer for network transfer.
    NETWORK_BUFFER_STEP_ID = -4,
    // Used when allocating a buffer to fill a Proto from the GPU.
    PROTO_BUFFER_STEP_ID = -5,
    // Used when allocating a Tensor where the caller has not indicated
    // the step.
    UNKNOWN_STEP_ID = -6,
  };

  static const string kLogMemoryLabel;

  // Test to see if memory logging is enabled. For now, logging is
  // enabled whenever VLOG_IS_ON(1) for the log_memory module.
  static bool IsEnabled();

  // Log the beginning of a step.
  static void RecordStep(int64 step_id, const string& handle);

  // Log a tensor buffer allocation. The name indicates which kernel
  // made the allocation. If the allocation is made through an
  // OpKernelContext the step_id indicates which step is executing,
  // otherwise step_id is one of the SpecialStepIds defined in
  // op_kernel.h, e.g. Op Kernel construction or an optimization pass
  // such as constant folding.
  static void RecordTensorAllocation(const string& kernel_name, int64 step_id,
                                     const Tensor& tensor);

  // Log a tensor buffer deallocation. The deallocation is triggered
  // when the buffer's refcount falls to zero, and the tracking
  // mechanism does not associate it with a particular step or
  // kernel. The allocation_id/allocator_name should match a
  // corresponding tensor previously passed in to
  // RecordTensorAllocation.
  static void RecordTensorDeallocation(int64 allocation_id,
                                       const string& allocator_name);

  // Log the use of a tensor as an output from a kernel.
  static void RecordTensorOutput(const string& kernel_name, int64 step_id,
                                 int index, const Tensor& tensor);

  // Log a "raw" allocation, which is just a buffer sized in
  // bytes. The Eigen allocator, and memory copies, record their
  // allocations this way, since they do not allocate TensorFlow
  // tensors. The operation is set to the OpKernel name if this is
  // called from within an Op execution, otherwise it indicates an
  // operation such as memcpy. The step_id if >=0 indicates which step
  // is executing, otherwise step_id is one of the SpecialStepIds
  // defined in op_kernel.h, e.g. Op Kernel construction or an
  // optimization pass such as constant folding.
  static void RecordRawAllocation(const string& operation, int64 step_id,
                                  size_t num_bytes, void* ptr,
                                  Allocator* allocator);

  // Log a "raw" deallocation of a buffer. When deferred is true, the
  // buffer won't be used again, but a GPU kernel may still be
  // enqueued using the buffer. A deferred deallocation should always
  // be followed by a matching non-deferred deallocation when the
  // buffer is actually returned and can be reused.
  static void RecordRawDeallocation(const string& operation, int64 step_id,
                                    void* ptr, Allocator* allocator,
                                    bool deferred);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_LOG_MEMORY_H_
