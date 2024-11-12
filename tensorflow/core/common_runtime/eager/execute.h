/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EXECUTE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EXECUTE_H_

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Utility function that executes a fully constructed EagerOperation.
// There are a few possible different combinations of how things can be
// executed:
//  - Async (the op context is configured to schedule asynchronously)
//    Eager execute should return quickly after scheduling this operation to
//    execute.
//  - Remote (the op device is on a remote task)
//    Eager execute will send an RPC to execute the op on a remote device.
//  Note that in the Async + Remote case, EagerExecute should still return
//  quickly, but it will schedule the op to be executed remotely.
//
// 'retvals' must point to a pre-allocated array of TensorHandle* and
// '*num_retvals' should be set to the size of this array. It is an error if
// the size of 'retvals' is less than the number of outputs. This call sets
// *num_retvals to the number of outputs.
absl::Status EagerExecute(EagerOperation* op, TensorHandle** retvals,
                          int* num_retvals);

// Low-level utility to execute the kernel specified by `kernel` on
// `kernel->device()`, with the inputs op_inputs, in the context 'ctx'.
absl::Status EagerKernelExecute(
    EagerContext* ctx, const absl::InlinedVector<TensorHandle*, 4>& op_inputs,
    const absl::optional<EagerFunctionParams>& eager_func_params,
    const core::RefCountPtr<KernelAndDevice>& kernel,
    GraphCollector* graph_collector, CancellationManager* cancellation_manager,
    absl::Span<TensorHandle*> retvals,
    const absl::optional<ManagedStackTrace>& stack_trace = {});

// Low-level utility to copy a tensor handle from one device to another. If
// successful, result TensorHandle will be populated. If the caller requests for
// the mirror flag, EagerCopyToDevice will attempt to add a mirror to the
// original handle and update *result to point to h. Since this is not
// guaranteed, callers should always use the value in *result.
absl::Status EagerCopyToDevice(TensorHandle* h, EagerContext* ctx,
                               EagerExecutor* executor, Device* device,
                               bool mirror, TensorHandle** result);

// Utility function that executes a fully constructed EagerOperation
// asynchronously on the local task. This function works differently from
// EagerExecute in several ways:
//  - It supports local execution only.
//  - It returns after launching the eager operation to run asynchronously.
//    Different from EagerExecute with async context that apends the operation
//    to the end of the eager executor schedule queue, this call bypasses the
//    executor logic and directly launches op execution. Ops running through
//    this call does NOT have an ordering and can be executed in parallel.
//  - It takes a StatusCallback which will be triggered after execution with the
//    execution status.
//
// Does not support custom device.
//
// 'retvals' must point to a pre-allocated array of TensorHandle* and
// '*num_retvals' should be set to the size of this array. It is an error if
// the size of 'retvals' is less than the number of outputs. This call sets
// *num_retvals to the number of outputs.
void EagerLocalExecuteAsync(EagerOperation* op, TensorHandle** retvals,
                            int* num_retvals, StatusCallback done);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EXECUTE_H_
