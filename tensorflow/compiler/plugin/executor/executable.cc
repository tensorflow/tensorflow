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

#include "tensorflow/compiler/plugin/executor/executable.h"
#include "tensorflow/compiler/plugin/executor/executor.h"

#include "tensorflow/compiler/xla/service/hlo_evaluator.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace se = ::perftools::gputools;
namespace sep = ::perftools::gputools::executorplugin;

namespace xla {
namespace executorplugin {

ExecutorExecutable::ExecutorExecutable(std::unique_ptr<HloModule> hlo_module)
    : Executable(std::move(hlo_module), ShapeSizeBytes) {}

ExecutorExecutable::~ExecutorExecutable() {}

static se::DeviceMemoryBase AllocateSingleOutput(sep::ExecutorExecutor* executor,
                                                 const Literal& literal) {
  int64 size(xla::ShapeUtil::ByteSizeOf(literal.shape()));
  void* buf = executor->Allocate(size);
  const void* src = LiteralUtil::InternalData(literal);
  memcpy(buf, src, size);
  return se::DeviceMemoryBase(buf, size);
}

static se::DeviceMemoryBase AllocateOutputBuffer(sep::ExecutorExecutor* executor,
                                                 const Literal& literal) {
  const Shape& shape = literal.shape();
  if (shape.element_type() != xla::TUPLE) {
    return AllocateSingleOutput(executor, literal);
  } else {
    int64 size(xla::ShapeUtil::ByteSizeOf(shape, sizeof(void*)));
    void** buf = reinterpret_cast<void**>(executor->Allocate(size));
    for (int64 n = 0; n < xla::ShapeUtil::TupleElementCount(shape); n++) {
      se::DeviceMemoryBase out =
          AllocateSingleOutput(executor, literal.tuple_literals(n));
      *buf++ = out.opaque();
    }

    return se::DeviceMemoryBase(buf, size);
  }
}

StatusOr<se::DeviceMemoryBase> ExecutorExecutable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  se::Stream* stream = run_options->stream();

  VLOG(1) << "Execute " << module().name();
  if (VLOG_IS_ON(2)) {
    for (const auto& a : arguments) {
      VLOG(2) << "-- argument " << a.opaque();
    }
  }

  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  HloComputation* computation = module().entry_computation();
  if (computation->num_parameters() != arguments.size()) {
    return tensorflow::errors::Internal(
        "Mismatch between argument count and graph parameter count.");
  }

  // Create the arguments as an vector of XLA literals
  std::vector<std::unique_ptr<Literal>> arg_literals;
  std::vector<Literal*> arg_literals_ptrs;
  for (int64 p = 0; p < computation->num_parameters(); p++) {
    // Create the input literal for the parameter
    HloInstruction* param = computation->parameter_instruction(p);
    arg_literals.emplace_back(LiteralUtil::CreateFromShape(param->shape()));
    arg_literals_ptrs.push_back(arg_literals.back().get());

    // Copy in the data from the stream_executor buffers
    void* buffer = LiteralUtil::MutableInternalData(arg_literals.back().get());
    memcpy(buffer, arguments[p].opaque(),
           ShapeUtil::ByteSizeOf(param->shape()));
  }

  // Execute the graph using the evaluator
  HloEvaluator evaluator;
  std::unique_ptr<Literal> output;
  TF_ASSIGN_OR_RETURN(output,
                      evaluator.Evaluate(computation, arg_literals_ptrs));

  // Copy the result into the return buffer
  perftools::gputools::StreamExecutor* executor(stream->parent());
  sep::ExecutorExecutor* executorExecutor(
      static_cast<sep::ExecutorExecutor*>(executor->implementation()));

  se::DeviceMemoryBase ret =
      AllocateOutputBuffer(executorExecutor, *(output.get()));

  uint64 end_micros = tensorflow::Env::Default()->NowMicros();

  {
    tensorflow::mutex_lock lock(mutex_);
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    execution_profile_.set_compute_time_ns(std::max(nanoseconds, 1.0));
  }

  return ret;
}

StatusOr<std::unique_ptr<ShapedBuffer>> ExecutorExecutable::ExecuteOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  return tensorflow::errors::Unimplemented(
      "ExecuteOnStream is not yet supported on Executor.");
}

StatusOr<se::DeviceMemoryBase> ExecutorExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments) {
  return tensorflow::errors::Unimplemented(
      "ExecuteAsyncOnStream is not yet supported on Executor.");
}

/*static*/ int64 ExecutorExecutable::ShapeSizeBytes(const Shape& shape) {
  if (ShapeUtil::IsOpaque(shape)) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}


}  // namespace executorplugin
}  // namespace xla
