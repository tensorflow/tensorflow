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

#include "tensorflow/compiler/xla/client/local_client.h"

#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "llvm/ADT/Triple.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/service/source_map_util.h"
#include "tensorflow/compiler/xla/service/stream_pool.h"
#include "tensorflow/compiler/xla/status_macros.h"

using xla::source_map_util::InvalidParameterArgument;

namespace xla {

namespace {
StatusOr<StreamPool::Ptr> BorrowStreamForDevice(int device_ordinal,
                                                Backend* backend) {
  if (device_ordinal < 0) {
    device_ordinal = backend->default_device_ordinal();
  }
  return backend->BorrowStream(device_ordinal);
}
}  // namespace

LocalExecutable::LocalExecutable(std::unique_ptr<Executable> executable,
                                 Backend* backend,
                                 ExecutableBuildOptions build_options)
    : executable_(std::move(executable)),
      backend_(backend),
      build_options_(std::move(build_options)) {
  CHECK_GE(build_options_.device_ordinal(), 0)
      << "Must have a valid device ordinal that the executable was built for.";
}

Status LocalExecutable::ValidateExecutionOptions(
    const ExecutableRunOptions& run_options, const Backend& backend) {
  if (run_options.stream() != nullptr) {
    if (!run_options.stream()->ok()) {
      return InvalidArgument("stream is uninitialized or in an error state");
    }

    // Check stream matches service platform.
    const se::Platform* stream_platform =
        run_options.stream()->parent()->platform();
    if (stream_platform != backend_->platform()) {
      return InvalidArgument(
          "stream is for platform %s, but service targets platform %s",
          stream_platform->Name(), backend_->platform()->Name());
    }

    // Cannot specify device_ordinal with a stream. The stream determines these
    // values.
    if (run_options.device_ordinal() != -1) {
      return InvalidArgument(
          "cannot set both device ordinal and stream options in "
          "ExecutableRunOptions; the stream determines the device ordinal");
    }
  }

  // Verify that the device the executable was built for is equivalent
  // to the device it will run on.
  int run_device_ordinal = run_options.device_ordinal();
  if (run_device_ordinal == -1) {
    run_device_ordinal = run_options.stream() != nullptr
                             ? run_options.stream()->parent()->device_ordinal()
                             : backend_->default_device_ordinal();
  }
  TF_ASSIGN_OR_RETURN(bool devices_equivalent,
                      backend_->devices_equivalent(
                          run_device_ordinal, build_options_.device_ordinal()));
  if (!devices_equivalent) {
    TF_ASSIGN_OR_RETURN(se::StreamExecutor * run_executor,
                        backend_->stream_executor(run_device_ordinal));
    TF_ASSIGN_OR_RETURN(se::StreamExecutor * build_executor,
                        backend_->stream_executor(build_device_ordinal()));
    return InvalidArgument(
        "executable is built for device %s of type \"%s\"; cannot run it on "
        "device %s of type \"%s\"",
        backend_->device_name(build_device_ordinal()),
        build_executor->GetDeviceDescription().name(),
        backend_->device_name(run_device_ordinal),
        run_executor->GetDeviceDescription().name());
  }

  if (!run_options.allocator()) {
    return InvalidArgument("an allocator must be provided to ExecuteLocally");
  }

  if (run_options.allocator()->platform() != backend.platform()) {
    return InvalidArgument(
        "allocator platform (%s) does not match service platform (%s)",
        run_options.allocator()->platform()->Name(),
        backend.platform()->Name());
  }

  return Status::OK();
}

StatusOr<std::pair<ServiceExecutableRunOptions, StreamPool::Ptr>>
LocalExecutable::RunHelper(const absl::Span<const Shape* const> argument_shapes,
                           ExecutableRunOptions run_options) {
  const ComputationLayout& computation_layout =
      executable_->module_config().entry_computation_layout();

  // Check argument number, shapes, and layouts.
  const int argument_shapes_size = argument_shapes.size();
  if (argument_shapes_size != computation_layout.parameter_count()) {
    return InvalidArgument(
        "invalid number of arguments for computation: expected %d, got %u",
        computation_layout.parameter_count(), argument_shapes.size());
  }
  for (int i = 0, end = argument_shapes.size(); i < end; ++i) {
    // TODO(b/187081154): Compare tiling info also.
    if (!computation_layout.parameter_layout(i).MatchesLayoutInShape(
            *argument_shapes[i], /*minor_to_major_only=*/false,
            /*ignore_fully_empty_tiling=*/true)) {
      return InvalidParameterArgument(
          executable_.get(), i,
          "Argument does not match host shape or layout of computation "
          "parameter "
          "%d: want %s, got %s",
          i,
          ShapeUtil::HumanStringWithLayout(
              computation_layout.parameter_layout(i).shape()),
          ShapeUtil::HumanStringWithLayout(*argument_shapes[i]));
    }
  }

  TF_RETURN_IF_ERROR(ValidateExecutionOptions(run_options, *backend_));

  StreamPool::Ptr stream;
  if (run_options.stream() == nullptr) {
    // NB!  The lifetime of `stream` needs to match the lifetime of
    // `service_options` (otherwise we will end up using a returned stream in
    // ExecuteOnStreamWrapper), which is why it isn't declared in the inner "if"
    // scope.
    TF_ASSIGN_OR_RETURN(
        stream, BorrowStreamForDevice(run_options.device_ordinal(), backend_));
    run_options.set_stream(stream.get());
  }
  if (run_options.allocator() == nullptr) {
    run_options.set_allocator(backend_->memory_allocator());
  }

  // For local client execution on CPU backends:
  // *) The thread pool used for eigen CPU ops is from
  //    ExecutableRunOptions.eigen_intra_op_thread_pool.
  // *) The thread pool used for XLA CPU ops is from
  //    backend_->eigen_intra_op_thread_pool().
  ServiceExecutableRunOptions service_options(run_options,
                                              backend_->StreamBorrower());
  return std::make_pair(service_options, std::move(stream));
}

StatusOr<ScopedShapedBuffer> LocalExecutable::Run(
    const absl::Span<const ShapedBuffer* const> arguments,
    ExecutableRunOptions run_options) {
  std::vector<const Shape*> argument_shapes;
  argument_shapes.reserve(arguments.size());
  for (const ShapedBuffer* const arg : arguments) {
    argument_shapes.push_back(&arg->on_device_shape());
  }
  return AsyncCallAndBlockHostUntilDone<xla::ScopedShapedBuffer>(
      argument_shapes, run_options, [&](const ExecutableRunOptions& options) {
        return RunAsync(arguments, options);
      });
}

StatusOr<ExecutionOutput> LocalExecutable::Run(
    std::vector<ExecutionInput> arguments, ExecutableRunOptions run_options) {
  std::vector<const Shape*> argument_shapes;
  argument_shapes.reserve(arguments.size());
  for (const ExecutionInput& arg : arguments) {
    argument_shapes.push_back(&arg.shape());
  }
  return AsyncCallAndBlockHostUntilDone<ExecutionOutput>(
      argument_shapes, run_options, [&](const ExecutableRunOptions& options) {
        return RunAsync(argument_shapes, std::move(arguments), options);
      });
}

static std::shared_ptr<HloSnapshot> DumpArguments(
    const Backend* backend, const Executable* executable,
    const absl::Span<const ShapedBuffer* const> arguments, se::Stream* stream) {
  auto snapshot = std::make_shared<HloSnapshot>();
  snapshot->set_execution_platform(backend->platform()->Name());
  *snapshot->mutable_hlo() = *executable->hlo_proto();
  for (const ShapedBuffer* arg : arguments) {
    auto literal = std::make_shared<Literal>(arg->on_host_shape());
    backend->transfer_manager()->TransferLiteralFromDevice(
        stream, *arg, literal.get(), [snapshot, literal](Status status) {
          if (!status.ok()) {
            LOG(ERROR) << "TransferLiteralFromDevice for HLO snapshot inputs "
                          "failed: "
                       << status;
            return;
          }
          *snapshot->add_arguments() = literal->ToProto();
        });
  }
  return snapshot;
}

static void DumpOutputsAndSaveSnapshot(const Backend* backend,
                                       const ShapedBuffer& outputs,
                                       std::shared_ptr<HloSnapshot> snapshot,
                                       se::Stream* stream) {
  auto literal = std::make_shared<Literal>(outputs.on_host_shape());
  backend->transfer_manager()->TransferLiteralFromDevice(
      stream, outputs, literal.get(),
      [snapshot{std::move(snapshot)}, literal](Status status) {
        if (status.ok()) {
          *snapshot->mutable_result() = literal->ToProto();
        } else {
          LOG(ERROR)
              << "TransferLiteralFromDevice for HLO snapshot outputs failed: "
              << status;
        }
        DumpHloSnapshotIfEnabled(*snapshot, GetDebugOptionsFromFlags());
      });
}

StatusOr<ScopedShapedBuffer> LocalExecutable::RunAsync(
    const absl::Span<const ShapedBuffer* const> arguments,
    ExecutableRunOptions run_options) {
  std::vector<const Shape*> argument_shapes;
  argument_shapes.reserve(arguments.size());
  for (const ShapedBuffer* const arg : arguments) {
    argument_shapes.push_back(&arg->on_device_shape());
  }
  TF_ASSIGN_OR_RETURN(auto options_and_stream,
                      RunHelper(argument_shapes, run_options));
  se::Stream* stream = run_options.stream();

  std::shared_ptr<HloSnapshot> snapshot;
  if (executable_->dumping_snapshot()) {
    snapshot = DumpArguments(backend_, executable_.get(), arguments, stream);
  }

  TF_ASSIGN_OR_RETURN(ScopedShapedBuffer outputs,
                      executable_->ExecuteAsyncOnStreamWrapper(
                          &options_and_stream.first, arguments));

  // Transfer the outputs and save the snapshot to disk.
  if (snapshot) {
    DumpOutputsAndSaveSnapshot(backend_, outputs, std::move(snapshot), stream);
  }

  return std::move(outputs);
}

static ShapedBuffer MaybeOwningShapeTreeToShapedBuffer(
    const ShapeTree<MaybeOwningDeviceMemory>& tree, int device_ordinal) {
  ShapedBuffer result(tree.shape(), device_ordinal);
  auto it = tree.begin();
  auto out_it = result.buffers().begin();
  for (; it != tree.end(); ++it, ++out_it) {
    out_it->second = it->second.AsDeviceMemoryBase();
  }
  return result;
}

StatusOr<ExecutionOutput> LocalExecutable::RunAsync(
    absl::Span<Shape const* const> argument_host_shapes,
    std::vector<ExecutionInput> arguments, ExecutableRunOptions run_options) {
  if (argument_host_shapes.size() != arguments.size()) {
    return InvalidArgument(
        "Number of argument host shapes not equal to number of arguments (%d "
        "vs %d)",
        argument_host_shapes.size(), arguments.size());
  }
  TF_ASSIGN_OR_RETURN(auto options_and_stream,
                      RunHelper(argument_host_shapes, run_options));
  se::Stream* stream = run_options.stream();

  std::shared_ptr<HloSnapshot> snapshot;
  if (executable_->dumping_snapshot()) {
    std::vector<ShapedBuffer> shaped_buffers;
    std::vector<const ShapedBuffer*> shaped_buffer_ptrs;
    shaped_buffers.reserve(arguments.size());
    shaped_buffer_ptrs.reserve(arguments.size());
    for (size_t i = 0; i < arguments.size(); ++i) {
      shaped_buffers.push_back(MaybeOwningShapeTreeToShapedBuffer(
          arguments[i].Buffers(), stream->parent()->device_ordinal()));
      shaped_buffer_ptrs.push_back(&shaped_buffers.back());
    }

    snapshot =
        DumpArguments(backend_, executable_.get(), shaped_buffer_ptrs, stream);
  }

  TF_ASSIGN_OR_RETURN(ExecutionOutput outputs,
                      executable_->ExecuteAsyncOnStreamWrapper(
                          &options_and_stream.first, std::move(arguments)));

  // Transfer the outputs and save the snapshot to disk.
  if (snapshot) {
    DumpOutputsAndSaveSnapshot(backend_, outputs.Result(), std::move(snapshot),
                               stream);
  }

  return std::move(outputs);
}

StatusOr<ExecutionOutput> LocalExecutable::RunAsync(
    std::vector<ExecutionInput> arguments, ExecutableRunOptions run_options) {
  std::vector<const Shape*> argument_shapes;
  argument_shapes.reserve(arguments.size());
  for (const ExecutionInput& arg : arguments) {
    argument_shapes.push_back(&arg.shape());
  }
  return RunAsync(argument_shapes, std::move(arguments), run_options);
}

se::Platform* LocalClient::platform() const {
  return local_service_->backend().platform();
}

int LocalClient::device_count() const {
  return local_service_->backend().device_count();
}

bool LocalClient::device_ordinal_supported(int device_ordinal) const {
  return local_service_->backend().device_ordinal_supported(device_ordinal);
}

int LocalClient::default_device_ordinal() const {
  return local_service_->backend().default_device_ordinal();
}

const Backend& LocalClient::backend() const {
  return local_service_->backend();
}

Backend* LocalClient::mutable_backend() {
  return local_service_->mutable_backend();
}

static StatusOr<ExecutableBuildOptions> UpdateBuildOptions(
    const ExecutableBuildOptions& options, int default_device_ordinal) {
  ExecutableBuildOptions updated_options = options;
  if (options.device_ordinal() == -1) {
    updated_options.set_device_ordinal(default_device_ordinal);
    VLOG(3) << "Set device ordinal to default value of: "
            << updated_options.device_ordinal();
  }
  if (options.has_device_assignment()) {
    if (options.device_assignment().replica_count() != options.num_replicas()) {
      return InvalidArgument(
          "Mismatched number of replicas for device "
          "assignment and computation (%d vs %d).\n%s",
          options.device_assignment().replica_count(), options.num_replicas(),
          options.device_assignment().ToString());
    }
    if (options.device_assignment().computation_count() !=
        options.num_partitions()) {
      return InvalidArgument(
          "Mismatched number of partitions for device "
          "assignment and computation (%d vs %d).\n%s",
          options.device_assignment().computation_count(),
          options.num_partitions(), options.device_assignment().ToString());
    }
  }
  return updated_options;
}

StatusOr<std::vector<std::unique_ptr<LocalExecutable>>> LocalClient::Compile(
    const XlaComputation& computation,
    const absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& options) {
  TF_ASSIGN_OR_RETURN(ExecutableBuildOptions updated_options,
                      UpdateBuildOptions(options, default_device_ordinal()));
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<Executable>> executables,
                      local_service_->CompileExecutables(
                          computation, argument_layouts, updated_options));

  std::vector<std::unique_ptr<LocalExecutable>> local_executables;
  local_executables.reserve(executables.size());

  for (auto& executable : executables) {
    local_executables.push_back(absl::make_unique<LocalExecutable>(
        std::move(executable), local_service_->mutable_backend(),
        updated_options));
  }

  return std::move(local_executables);
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
LocalClient::CompileAheadOfTime(
    const XlaComputation& computation,
    const absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& options) {
  TF_ASSIGN_OR_RETURN(ExecutableBuildOptions updated_options,
                      UpdateBuildOptions(options, default_device_ordinal()));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
      local_service_->CompileAotResults(computation, argument_layouts,
                                        updated_options));

  return std::move(aot_results);
}

StatusOr<std::unique_ptr<LocalExecutable>> LocalClient::Load(
    const std::string& serialized_aot_result,
    const ExecutableBuildOptions& options) {
  TF_ASSIGN_OR_RETURN(ExecutableBuildOptions updated_options,
                      UpdateBuildOptions(options, default_device_ordinal()));
  TF_ASSIGN_OR_RETURN(
      se::StreamExecutor * executor,
      backend().stream_executor(updated_options.device_ordinal()));

  TF_ASSIGN_OR_RETURN(Compiler * compiler,
                      Compiler::GetForPlatform(platform()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::AotCompilationResult> aot_result,
      compiler->LoadAotCompilationResult(serialized_aot_result));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                      aot_result->LoadExecutable(compiler, executor));
  return absl::make_unique<LocalExecutable>(std::move(executable),
                                            local_service_->mutable_backend(),
                                            updated_options);
}

StatusOr<ScopedShapedBuffer> LocalClient::LiteralToShapedBuffer(
    const LiteralSlice& literal, int device_ordinal,
    se::DeviceMemoryAllocator* allocator) {
  if (allocator == nullptr) {
    allocator = backend().memory_allocator();
  }
  TF_ASSIGN_OR_RETURN(auto scoped_buffer,
                      backend().transfer_manager()->AllocateScopedShapedBuffer(
                          literal.shape(), allocator, device_ordinal));
  TF_ASSIGN_OR_RETURN(auto stream,
                      mutable_backend()->BorrowStream(device_ordinal));
  TF_RETURN_IF_ERROR(backend().transfer_manager()->TransferLiteralToDevice(
      stream.get(), literal, scoped_buffer));
  return std::move(scoped_buffer);
}

StatusOr<Literal> LocalClient::ShapedBufferToLiteral(
    const ShapedBuffer& shaped_buffer) {
  TF_ASSIGN_OR_RETURN(auto stream, mutable_backend()->BorrowStream(
                                       shaped_buffer.device_ordinal()));
  return backend().transfer_manager()->TransferLiteralFromDevice(stream.get(),
                                                                 shaped_buffer);
}

StatusOr<const ShapedBuffer*> LocalClient::GlobalDataToShapedBuffer(
    const GlobalDataHandle& data, int replica_number) {
  return local_service_->GlobalDataToShapedBuffer(data, replica_number);
}

Status LocalClient::TransferToInfeedLocal(const LiteralSlice& literal,
                                          int device_ordinal) {
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                      backend().stream_executor(device_ordinal));
  return backend().transfer_manager()->TransferLiteralToInfeed(executor,
                                                               literal);
}

Status LocalClient::TransferFromOutfeedLocal(int device_ordinal,
                                             MutableBorrowingLiteral literal) {
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                      backend().stream_executor(device_ordinal));
  return backend().transfer_manager()->TransferLiteralFromOutfeed(executor,
                                                                  literal);
}

StatusOr<int> LocalClient::ReplicaNumberToDeviceOrdinal(int replica_number) {
  return local_service_->ReplicaNumberToDeviceOrdinal(replica_number);
}

StatusOr<TransferToServerResponse> LocalClient::TransferToLocalServer(
    const ::xla::BorrowingLiteral& literal, int device_ordinal) {
  const ::xla::Shape& shape = literal.shape();

  TF_ASSIGN_OR_RETURN(::xla::ScopedShapedBuffer shaped_buffer,
                      backend().transfer_manager()->AllocateScopedShapedBuffer(
                          shape, backend().memory_allocator(), device_ordinal));
  TF_ASSIGN_OR_RETURN(auto stream,
                      mutable_backend()->BorrowStream(device_ordinal));
  TF_RETURN_IF_ERROR(backend().transfer_manager()->TransferLiteralToDevice(
      stream.get(), literal, shaped_buffer));
  std::vector<::xla::ScopedShapedBuffer> replicated_buffer;
  replicated_buffer.emplace_back(std::move(shaped_buffer));
  ::xla::TransferToServerResponse result;
  TF_ASSIGN_OR_RETURN(*result.mutable_data(),
                      local_service_->RegisterReplicatedBuffers(
                          std::move(replicated_buffer),
                          absl::StrCat("TransferToServer literal of shape ",
                                       ::xla::ShapeUtil::HumanString(shape))));

  return result;
}

}  // namespace xla
