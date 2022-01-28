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

#include "tensorflow/compiler/xla/service/interpreter/executable_base.h"

#include <type_traits>
#include <vector>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace xla {
namespace interpreter {

InterpreterExecutableBase::InterpreterExecutableBase(
    std::unique_ptr<HloModule> hlo_module)
    : Executable(std::move(hlo_module), /*hlo_profile_printer_data=*/nullptr,
                 /*hlo_profile_index_map=*/nullptr) {}

StatusOr<ExecutionOutput> InterpreterExecutableBase::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();
  const se::Platform* platform = executor->platform();

  // Convert the ShapeTree to a ShapedBuffer. We do this so we can call
  // TransferManager methods below.
  std::vector<ShapedBuffer> argument_buffers;
  argument_buffers.reserve(arguments.size());
  int device_ordinal = run_options->device_ordinal();
  if (device_ordinal < 0) {
    device_ordinal = 0;
  }
  for (auto& argument : arguments) {
    const ShapeTree<MaybeOwningDeviceMemory>& buffers = argument.Buffers();
    argument_buffers.push_back(ShapedBuffer(buffers.shape(),
                                            /*device_ordinal=*/device_ordinal));
    auto in_it = buffers.begin();
    auto out_it = argument_buffers.back().buffers().begin();
    for (; in_it != buffers.end(); ++in_it, ++out_it) {
      out_it->second = in_it->second.AsDeviceMemoryBase();
    }
  }

  VLOG(1) << "Execute " << module().name();
  if (VLOG_IS_ON(2)) {
    for (const auto& a : argument_buffers) {
      VLOG(2) << "-- argument " << a;
    }
  }

  uint64_t start_micros = tensorflow::Env::Default()->NowMicros();

  const HloComputation* computation = module().entry_computation();
  if (computation->num_parameters() != arguments.size()) {
    return tensorflow::errors::Internal(
        "Mismatch between argument count and graph parameter count.");
  }

  // Check that the args have the right shape.
  for (int64_t i = 0; i < computation->num_parameters(); ++i) {
    const auto& expected_shape = computation->parameter_instruction(i)->shape();
    const auto& actual_shape = argument_buffers[i].on_device_shape();
    bool shape_match = true;
    if (expected_shape.is_dynamic()) {
      if (!ShapeUtil::DynamicArrayShapeIsCompatible(actual_shape,
                                                    expected_shape)) {
        shape_match = false;
      }
    } else if (!Shape::Equal().MinorToMajorOnlyInLayout()(expected_shape,
                                                          actual_shape)) {
      shape_match = false;
    }
    if (!shape_match) {
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
  const int64_t num_parameters = computation->num_parameters();
  arg_literals.reserve(num_parameters);
  for (int64_t p = 0; p < num_parameters; ++p) {
    TF_ASSIGN_OR_RETURN(Literal arg_literal,
                        transfer_manager->TransferLiteralFromDevice(
                            run_options->stream(), argument_buffers[p]));
    const auto& expected_shape = computation->parameter_instruction(p)->shape();
    if (expected_shape.is_dynamic()) {
      // Expand the input literal to expected shape.
      arg_literal = arg_literal.ToBoundedDynamic(expected_shape);
    }
    arg_literals.push_back(std::move(arg_literal));
  }

  TF_ASSIGN_OR_RETURN(Literal result_literal,
                      Evaluate(run_options, *computation, arg_literals));
  // Shrink the generated dynamic shape into static shape.
  result_literal = result_literal.ToStatic();

  // Transform the result literal back into a ShapedBuffer.
  TF_ASSIGN_OR_RETURN(ScopedShapedBuffer result_buffers,
                      transfer_manager->AllocateScopedShapedBuffer(
                          result_literal.shape(), run_options->allocator(),
                          executor->device_ordinal()));
  TF_RETURN_IF_ERROR(transfer_manager->TransferLiteralToDevice(
      run_options->stream(), result_literal, result_buffers));
  ExecutionOutput result(std::move(result_buffers));

  uint64_t end_micros = tensorflow::Env::Default()->NowMicros();

  ExecutionProfile* profile = run_options->run_options().execution_profile();
  if (profile) {
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    profile->set_compute_time_ns(std::max(nanoseconds, 1.0));
  }
  MarkToBeReleasedArguments(absl::MakeSpan(arguments), result);
  return std::move(result);
}

}  // namespace interpreter
}  // namespace xla
