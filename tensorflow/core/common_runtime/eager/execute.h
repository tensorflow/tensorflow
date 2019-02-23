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

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

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
Status EagerExecute(
    EagerOperation* op,
    tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 2>* retvals,
    int* num_retvals);

// Low-level utility to execute the kernel specified by kernel on device
// 'device', with the inputs op_inputs, in the context 'ctx'.
Status EagerKernelExecute(EagerContext* ctx, Device* device,
                          const gtl::InlinedVector<TensorHandle*, 4>& op_inputs,
                          KernelAndDevice* kernel, NodeExecStats* maybe_stats,
                          StepStats* maybe_step_stats,
                          GraphCollector* graph_collector,
                          TensorHandle** retvals, int num_retvals);

// Low-level utility to copy a tensor handle from one device to another.
Status EagerCopyToDevice(TensorHandle* h, EagerContext* ctx,
                         const char* device_name, TensorHandle** result);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EXECUTE_H_
