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

#ifndef TENSORFLOW_CORE_TPU_TPU_EXECUTE_H_
#define TENSORFLOW_CORE_TPU_TPU_EXECUTE_H_

#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_node_context.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/tpu/kernels/tpu_executable_info.pb.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"

namespace tensorflow {

// Runs a TPU executable. `input_allocations` and `output_allocations` are
// non-owning pointers to the root buffers of each argument/result tuple.
// `output_shape` is the output shape of the XLA computation from which
// `program` was derived. If `session_module` is not nullptr, it will be filled
// with the input and output literals of the execution.
xla::StatusOr<xla::ExecutionOutput> TPUExecute(
    const TPUExecutableInfoProto& executable,
    const TPUHostTransferInfoProto& host_transfers,
    const xla::HloProto& hlo_metadata,
    std::vector<xla::ExecutionInput> arguments,
    const std::string& rendezvous_key_base, uint32 rng_seed,
    tpu::TpuNodeContext* node_context, xla::DeviceAssignment* device_assignment,
    CancellationManager* cancellation_manager, OpKernelContext* ctx,
    stream_executor::Stream* stream,
    stream_executor::Stream* host_to_device_stream,
    const XLA_TpuProgram* tpu_program);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_TPU_EXECUTE_H_
