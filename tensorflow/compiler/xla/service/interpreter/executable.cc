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

#include "tensorflow/compiler/xla/service/interpreter/executable.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/interpreter/executor.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace interpreter {

InterpreterExecutable::InterpreterExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloEvaluator> evaluator)
    : Executable(std::move(hlo_module), /*hlo_profile_printer_data=*/nullptr,
                 /*hlo_profile_index_map=*/nullptr),
      evaluator_(std::move(evaluator)) {}

InterpreterExecutable::~InterpreterExecutable() {}

StatusOr<ExecutionOutput> InterpreterExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ShapeTree<MaybeOwningDeviceMemory>> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();
  const se::Platform* platform = executor->platform();

  // Convert the ShapeTree to a ShapedBuffer. We do this so we can call
  // TransferManager methods below.
  std::vector<ShapedBuffer> argument_buffers;
  argument_buffers.reserve(arguments.size());
  for (const ShapeTree<MaybeOwningDeviceMemory>& arg : arguments) {
    argument_buffers.push_back(ShapedBuffer(arg.shape(), arg.shape(),
                                            /*platform=*/nullptr,
                                            /*device_ordinal=*/0));
    auto in_it = arg.begin();
    auto out_it = argument_buffers.back().buffers().begin();
    for (; in_it != arg.end(); ++in_it, ++out_it) {
      out_it->second = in_it->second.AsDeviceMemoryBase();
    }
  }

  VLOG(1) << "Execute " << module().name();
  if (VLOG_IS_ON(2)) {
    for (const auto& a : argument_buffers) {
      VLOG(2) << "-- argument " << a;
    }
  }

  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  const HloComputation* computation = module().entry_computation();
  if (computation->num_parameters() != arguments.size()) {
    return tensorflow::errors::Internal(
        "Mismatch between argument count and graph parameter count.");
  }

  // Check that the args have the right shape.
  for (int64 i = 0; i < computation->num_parameters(); ++i) {
    const auto& expected_shape = computation->parameter_instruction(i)->shape();
    const auto& actual_shape = argument_buffers[i].on_device_shape();
    if (!Shape::Equal().MinorToMajorOnlyInLayout()(expected_shape,
                                                   actual_shape)) {
      return InvalidArgument(
          "Shape mismatch on parameter %d.  Expected %s, but was %s.", i,
          ShapeUtil::HumanStringWithLayout(expected_shape),
          ShapeUtil::HumanStringWithLayout(actual_shape));
    }
  }

  TF_ASSIGN_OR_RETURN(TransferManager * transfer_manager,
                      TransferManager::GetForPlatform(platform));

  // Transform the ShapedBuffer arguments into literals which the evaluator
  // consumes.
  std::vector<Literal> arg_literals;
  for (int64 p = 0; p < computation->num_parameters(); ++p) {
    TF_ASSIGN_OR_RETURN(Literal arg_literal,
                        transfer_manager->TransferLiteralFromDevice(
                            run_options->stream(), argument_buffers[p]));
    arg_literals.push_back(std::move(arg_literal));
  }

  // Execute the graph using the HloEvaluator.
  Literal result_literal;
  {
    tensorflow::mutex_lock lock(evaluator_lock_);
    evaluator_->ResetVisitStates();
    TF_ASSIGN_OR_RETURN(result_literal,
                        evaluator_->Evaluate(*computation, arg_literals));
  }

  // Transform the result literal back into a ShapedBuffer.
  TF_ASSIGN_OR_RETURN(ScopedShapedBuffer result,
                      transfer_manager->AllocateScopedShapedBuffer(
                          result_literal.shape(), run_options->allocator(),
                          executor->device_ordinal()));
  TF_RETURN_IF_ERROR(transfer_manager->TransferLiteralToDevice(
      run_options->stream(), result_literal, result));

  uint64 end_micros = tensorflow::Env::Default()->NowMicros();

  ExecutionProfile* profile = run_options->run_options().execution_profile();
  if (profile) {
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    profile->set_compute_time_ns(std::max(nanoseconds, 1.0));
  }

  std::vector<se::OwningDeviceMemory> buffers_to_free;
  for (ShapeTree<MaybeOwningDeviceMemory>& argument : arguments) {
    for (std::pair<ShapeIndex, MaybeOwningDeviceMemory>& buffer : argument) {
      auto maybe_owning_buffer = buffer.second.Release();
      if (maybe_owning_buffer) {
        buffers_to_free.push_back(std::move(*maybe_owning_buffer));
      }
    }
  }
  return ExecutionOutput(std::move(result), std::move(buffers_to_free), {}, {});
}

/*static*/ int64 InterpreterExecutable::ShapeSizeBytes(const Shape& shape) {
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

}  // namespace interpreter
}  // namespace xla
