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

#include "tensorflow/compiler/xla/service/gpu/miopen_conv_algorithm_picker.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "google/protobuf/any.pb.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"
#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_autotuning.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/scratch_allocator.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logger.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/proto/proto_utils.h"

namespace xla {
namespace gpu {

using absl::optional;
using se::DeviceMemoryBase;
using se::dnn::AlgorithmConfig;
using se::dnn::AlgorithmDesc;
using tensorflow::AutotuneResult;

StatusOr<tensorflow::AutotuneResult>
MiopenConvAlgorithmPicker::PickBestAlgorithmNoCache(
    const HloCustomCallInstruction& instr, se::DeviceMemoryAllocator* allocator,
    se::Stream* stream) {
  const auto device_ordinal = stream_exec_->device_ordinal();
  std::vector<se::DeviceMemoryBase> operand_buffers;
  se::DeviceMemoryBase result_buffer;

  ScratchAllocator input_output_allocator(device_ordinal, allocator);
  AllocateInitializeBuffers(instr, &input_output_allocator, stream,
                            &operand_buffers, &result_buffer);

  ScratchAllocator scratch_allocator(device_ordinal, allocator);
  se::dnn::ProfileResult profile_result;
  VLOG(3) << "Auto-tuning for " << instr.ToString();
  RunConvOptions options;
  options.profile_result = &profile_result;
  options.first_call_from_algorithm_picker = true;

  bool launch_ok =
      RunCudnnConv(&instr, absl::MakeSpan(operand_buffers), result_buffer,
                   &scratch_allocator, stream, options)
          .ok();

  AutotuneResult best_result;
  if (launch_ok && profile_result.is_valid()) {
    best_result.mutable_conv()->set_algorithm(
        profile_result.algorithm().algo_id());
    best_result.mutable_conv()->set_tensor_ops_enabled(
        profile_result.algorithm().tensor_ops_enabled());
    int64 scratch_bytes_used = scratch_allocator.TotalAllocatedBytes();
    best_result.set_scratch_bytes(scratch_bytes_used);
    *best_result.mutable_run_time() = tensorflow::proto_utils::ToDurationProto(
        absl::Milliseconds(profile_result.elapsed_time_in_ms()));

    return best_result;
  }

  return InternalError(
      "All algorithms tried for convolution %s failed.  Falling back to "
      "default algorithm.",
      instr.ToString());
}

Status MiopenConvAlgorithmPicker::AllocateInitializeBuffers(
    const HloCustomCallInstruction& instr,
    se::ScratchAllocator* input_output_allocator, se::Stream* stream,
    std::vector<se::DeviceMemoryBase>* operand_buffers,
    se::DeviceMemoryBase* result_buffer) {
  const auto initialize_buffer = [&stream](DeviceMemoryBase buffer) {
    // Although we don't have evidence this matters, zero out the buffers
    // before autotuning.  It's conceivable that using uninitialized memory as
    // the inputs might affect performance if e.g. the inputs contain
    // denormals, and this is easy enough.
    stream->ThenMemZero(&buffer, buffer.size());
  };

  // Allocate space for the input, filter, and output of the convolution.  We
  // use a ScratchAllocator for this instead of calling allocator_ directly so
  // that our allocations don't leak.
  for (const auto* operand : instr.operands()) {
    TF_ASSIGN_OR_RETURN(auto buffer,
                        input_output_allocator->AllocateBytes(
                            ShapeUtil::ByteSizeOf(operand->shape())));
    initialize_buffer(buffer);
    operand_buffers->push_back(buffer);
  }
  TF_ASSIGN_OR_RETURN(
      *result_buffer,
      input_output_allocator->AllocateBytes(
          ShapeUtil::ByteSizeOf(instr.shape().tuple_shapes(0))));
  initialize_buffer(*result_buffer);

  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
