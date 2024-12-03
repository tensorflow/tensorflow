/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/service.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/service/backend.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_layout.h"
#include "xla/service/computation_placer.h"
#include "xla/service/dump.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/dynamic_padder.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/stream_pool.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla {
namespace {

using absl::StrCat;
using absl::StrFormat;

// Records the arguments used to invoke a computation in an HloSnapshot proto.
absl::Status RecordArguments(
    const absl::Span<const ShapedBuffer* const> arguments, se::Stream* stream,
    TransferManager* transfer_manager, HloSnapshot* module) {
  module->clear_arguments();
  for (const ShapedBuffer* argument : arguments) {
    TF_ASSIGN_OR_RETURN(
        Literal literal,
        transfer_manager->TransferLiteralFromDevice(stream, *argument));
    *module->add_arguments() = literal.ToProto();
  }
  return absl::OkStatus();
}

// Records the result of a computation in a HloSnapshot proto.
absl::Status RecordResult(const ShapedBuffer& result, se::Stream* stream,
                          TransferManager* transfer_manager,
                          HloSnapshot* module) {
  module->clear_result();
  TF_ASSIGN_OR_RETURN(
      Literal literal,
      transfer_manager->TransferLiteralFromDevice(stream, result));
  *module->mutable_result() = literal.ToProto();
  return absl::OkStatus();
}

}  // namespace

ServiceOptions& ServiceOptions::set_platform(se::Platform* platform) {
  platform_ = platform;
  return *this;
}

se::Platform* ServiceOptions::platform() const { return platform_; }

ServiceOptions& ServiceOptions::set_number_of_replicas(int number_of_replicas) {
  number_of_replicas_ = number_of_replicas;
  return *this;
}

int ServiceOptions::number_of_replicas() const { return number_of_replicas_; }

ServiceOptions& ServiceOptions::set_intra_op_parallelism_threads(
    int num_threads) {
  intra_op_parallelism_threads_ = num_threads;
  return *this;
}

int ServiceOptions::intra_op_parallelism_threads() const {
  return intra_op_parallelism_threads_;
}

ServiceOptions& ServiceOptions::set_allowed_devices(
    const std::optional<std::set<int>>& allowed_devices) {
  allowed_devices_ = allowed_devices;
  return *this;
}

const std::optional<std::set<int>>& ServiceOptions::allowed_devices() const {
  return allowed_devices_;
}

Service::Service(const ServiceOptions& options,
                 std::unique_ptr<Backend> execute_backend)
    : options_(options),
      allocation_tracker_(execute_backend.get()),
      execute_backend_(std::move(execute_backend)) {
  CHECK_GT(options_.number_of_replicas(), 0);
  if (execute_backend_) {
    if (execute_backend_->device_count() > 0) {
      CHECK_GE(execute_backend_->device_count(), options_.number_of_replicas())
          << "Requested more replicas than there are devices.";
    }
    LOG(INFO) << StrFormat(
        "XLA service %p initialized for platform %s (this does not guarantee "
        "that XLA will be used). Devices:",
        this, execute_backend_->platform()->Name());
    auto stream_executors = execute_backend_->stream_executors();
    for (int i = 0; i < execute_backend_->device_count(); ++i) {
      se::StreamExecutor* executor = stream_executors.at(i);
      const auto& description = executor->GetDeviceDescription();
      LOG(INFO) << StrFormat("  StreamExecutor device (%d): %s, %s", i,
                             description.name(),
                             description.platform_version());
    }
  } else {
    VLOG(1) << "XLA compile-only service constructed";
  }
}

absl::StatusOr<ChannelHandle> Service::CreateChannelHandle(
    ChannelHandle::ChannelType type) {
  return channel_tracker_.NewChannel(type);
}

absl::Status Service::Unregister(const GlobalDataHandle& data) {
  return allocation_tracker_.Unregister(data);
}

// Deconstructs a previously-allocated global handle.
absl::StatusOr<std::vector<std::unique_ptr<GlobalData>>>
Service::DeconstructTuple(const GlobalData& data) {
  TF_ASSIGN_OR_RETURN(std::vector<GlobalDataHandle> elements,
                      allocation_tracker_.DeconstructTuple(data.handle()));
  std::vector<std::unique_ptr<GlobalData>> out;
  out.reserve(elements.size());
  for (GlobalDataHandle& element : elements) {
    out.push_back(std::make_unique<GlobalData>(this, element));
  }
  return out;
}

absl::Status Service::ValidateResultShape(const Shape& client_shape,
                                          const Shape& result_shape) {
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(client_shape));
  if (!ShapeUtil::Compatible(client_shape, result_shape)) {
    return InvalidArgument(
        "Shape used to set computation result layout %s is not compatible "
        "with result shape %s",
        ShapeUtil::HumanStringWithLayout(client_shape),
        ShapeUtil::HumanString(result_shape));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::vector<const ShapedBuffer*>>>
Service::ResolveAndValidateArguments(
    absl::Span<const GlobalData* const> arguments,
    absl::Span<se::StreamExecutor* const> stream_executors) const {
  CHECK_EQ(options_.number_of_replicas(), stream_executors.size());
  std::vector<std::vector<const ShapedBuffer*>> replicated_arguments;
  replicated_arguments.resize(options_.number_of_replicas());
  for (size_t i = 0; i < arguments.size(); ++i) {
    TF_ASSIGN_OR_RETURN(std::vector<const ShapedBuffer*> replicated_buffers,
                        allocation_tracker_.Resolve(arguments[i]->handle()));
    CHECK_EQ(options_.number_of_replicas(), replicated_buffers.size());
    for (int replica = 0; replica < options_.number_of_replicas(); ++replica) {
      const ShapedBuffer* shaped_buffer = replicated_buffers[replica];
      replicated_arguments[replica].push_back(shaped_buffer);
    }
  }
  return replicated_arguments;
}

absl::StatusOr<std::unique_ptr<HloModuleConfig>> Service::CreateModuleConfig(
    const ProgramShape& program_shape,
    absl::Span<const Shape* const> argument_shapes,
    const ExecutionOptions* execution_options,
    const AotCompilationOptions* aot_options) {
  int default_num_replicas = options_.number_of_replicas();
  std::optional<int> num_threads;
  if (execute_backend_ != nullptr &&
      execute_backend_->eigen_intra_op_thread_pool() != nullptr) {
    num_threads = execute_backend_->eigen_intra_op_thread_pool()->NumThreads();
  }

  return xla::CreateModuleConfig(program_shape, argument_shapes,
                                 execution_options, default_num_replicas,
                                 num_threads, aot_options);
}

absl::StatusOr<std::unique_ptr<HloModuleConfig>> Service::CreateModuleConfig(
    const ProgramShape& program_shape,
    absl::Span<const ShapedBuffer* const> arguments,
    const ExecutionOptions& execution_options,
    const AotCompilationOptions* aot_options) {
  std::vector<const Shape*> argument_shapes;
  for (const auto* arg : arguments) {
    argument_shapes.push_back(&arg->on_device_shape());
  }
  return CreateModuleConfig(program_shape, argument_shapes, &execution_options,
                            aot_options);
}

absl::StatusOr<std::vector<std::unique_ptr<Executable>>>
Service::BuildExecutables(
    const std::vector<const HloModuleProto*>& module_protos,
    std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
    Backend* backend, std::vector<std::vector<se::StreamExecutor*>> executors,
    const Compiler::CompileOptions& options, bool run_backend_only) {
  VLOG(1) << StrFormat("BuildExecutable on service %p", this);

  VLOG(1) << "Computations:";
  for (const HloModuleProto* proto : module_protos) {
    VLOG(1) << proto->name();
  }

  CHECK_EQ(module_protos.size(), module_configs.size());
  auto module_group =
      std::make_unique<HloModuleGroup>(module_protos[0]->name());
  for (int64_t i = 0, end = module_protos.size(); i < end; ++i) {
    const HloModuleProto* proto = module_protos[i];
    const HloModuleConfig& config = *module_configs[i];
    TF_ASSIGN_OR_RETURN(
        auto module, CreateModuleFromProto(*proto, config, run_backend_only));
    module->set_layout_canonicalization_callback(
        options.layout_canonicalization_callback);
    UpdateEntryComputationLayout(
        module.get(), std::bind(&Compiler::DefaultDeviceShapeRepresentation,
                                backend->compiler(), std::placeholders::_1));
    DumpHloModuleIfEnabled(*module, kBeforeOptimizationsDumpName);
    module_group->push_back(std::move(module));
  }

  std::vector<std::unique_ptr<Executable>> executables;
  if (!run_backend_only) {
    TF_ASSIGN_OR_RETURN(executables, backend->compiler()->Compile(
                                         std::move(module_group),
                                         std::move(executors), options));
  } else {
    auto modules = module_group->ConsumeModules();
    for (std::unique_ptr<HloModule>& module : modules) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                          backend->compiler()->RunBackend(
                              std::move(module), executors[0][0], options));
      executables.push_back(std::move(executable));
    }
  }

  return std::move(executables);
}

absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
Service::BuildAotResults(
    const std::vector<const HloModuleProto*>& module_protos,
    std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
    Backend* backend, std::vector<std::vector<se::StreamExecutor*>> executors,
    const Compiler::CompileOptions& options, bool run_backend_only) {
  VLOG(1) << StrFormat("BuildAotResults on service %p", this);

  VLOG(1) << "Computations:";
  for (const HloModuleProto* proto : module_protos) {
    VLOG(1) << proto->name();
  }

  CHECK_EQ(module_protos.size(), module_configs.size());
  auto module_group =
      std::make_unique<HloModuleGroup>(module_protos[0]->name());
  for (int64_t i = 0, end = module_protos.size(); i < end; ++i) {
    const HloModuleProto* proto = module_protos[i];
    const HloModuleConfig& config = *module_configs[i];
    TF_ASSIGN_OR_RETURN(
        auto module, CreateModuleFromProto(*proto, config, run_backend_only));
    DumpHloModuleIfEnabled(*module, kBeforeOptimizationsDumpName);
    module_group->push_back(std::move(module));
  }

  AotCompilationOptions aot_options(backend->compiler()->PlatformId());
  aot_options.set_executor(executors[0][0]);
  aot_options.set_device_allocator(options.device_allocator);
  aot_options.set_run_backend_only(run_backend_only);

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
      backend->compiler()->CompileAheadOfTime(std::move(module_group),
                                              aot_options));
  return std::move(aot_results);
}

absl::StatusOr<std::vector<GlobalDataHandle>>
Service::ExecuteParallelAndRegisterResult(
    absl::Span<Executable* const> executables,
    absl::Span<const std::vector<std::vector<const ShapedBuffer*>>> arguments,
    Backend* backend, absl::Span<const DeviceHandle> device_handles,
    absl::Span<const std::string> result_tags, ExecutionProfile* profile) {
  // Streams where the computation are launched, so we can wait on the streams
  // to complete.
  std::vector<StreamPool::Ptr> streams;

  // Global data handles for the computation results, one for each computation.
  std::vector<GlobalDataHandle> result_handles;

  // Device ID to stream executor, populated only with devices that are being
  // profiled.
  std::map<int64_t, se::Stream*> index_to_profiled_streams;

  // Build DeviceAssignment for all cores based on the provided device handles.
  DeviceAssignment device_assignment(options_.number_of_replicas(),
                                     executables.size());
  for (int64_t i = 0; i < executables.size(); i++) {
    TF_ASSIGN_OR_RETURN(auto replicas, Replicas(*backend, device_handles[i]));
    CHECK_EQ(replicas.size(), arguments[i].size());
    for (int64_t replica = 0, end = replicas.size(); replica < end; ++replica) {
      device_assignment(replica, i) = replicas[replica]->device_ordinal();
    }
  }

  for (int64_t i = 0, end = executables.size(); i < end; i++) {
    // Stream executors for the replicas of the current computation.
    TF_ASSIGN_OR_RETURN(auto replicas, Replicas(*backend, device_handles[i]));
    CHECK_EQ(replicas.size(), arguments[i].size());
    std::vector<ScopedShapedBuffer> result_buffers;
    const int64_t n = replicas.size();
    result_buffers.reserve(n);
    for (int64_t replica = 0; replica < n; ++replica) {
      TF_ASSIGN_OR_RETURN(StreamPool::Ptr stream,
                          backend->BorrowStream(replicas[replica]));
      streams.push_back(std::move(stream));

      if (replica == 0 &&
          executables[i]->module_config().debug_options().xla_hlo_profile() &&
          executables[i]->hlo_profiling_enabled()) {
        index_to_profiled_streams[i] = streams.back().get();
      }

      // Set up run options.
      ExecutableRunOptions options;
      options.set_stream(streams.back().get());
      options.set_allocator(backend->memory_allocator());
      options.set_intra_op_thread_pool(
          backend->eigen_intra_op_thread_pool_device());
      const DeviceAssignment* device_assignment_ptr = &device_assignment;
      if (executables[i]->module_config().has_static_device_assignment()) {
        device_assignment_ptr =
            &executables[i]->module_config().static_device_assignment();
      }
      options.set_device_assignment(device_assignment_ptr);
      // Use run-time profile information from execution_profile on the 0th
      // device.
      if (i == 0) {
        options.set_execution_profile(profile);
      }
      ServiceExecutableRunOptions run_options(
          options, backend->StreamBorrowerWithPriority());

      // Asynchronously launch the computation.
      TF_ASSIGN_OR_RETURN(ScopedShapedBuffer result,
                          executables[i]->ExecuteAsyncOnStream(
                              &run_options, arguments[i][replica]));

      result_buffers.push_back(std::move(result));
    }
    TF_ASSIGN_OR_RETURN(GlobalDataHandle handle,
                        allocation_tracker_.RegisterReplicatedBuffers(
                            std::move(result_buffers), result_tags[i]));
    result_handles.push_back(handle);
  }

  // Wait for all executions to complete.
  for (int64_t i = 0, end = streams.size(); i < end; ++i) {
    absl::Status block_status = streams[i]->BlockHostUntilDone();
    if (!block_status.ok()) {
      return Internal("failed to complete execution for stream %d: %s", i,
                      block_status.message());
    }
  }

  return result_handles;
}

absl::StatusOr<GlobalDataHandle> Service::ExecuteAndRegisterResult(
    Executable* executable,
    absl::Span<const std::vector<const ShapedBuffer*>> arguments,
    Backend* backend, const DeviceHandle& device_handle,
    const std::string& result_tag, ExecutionProfile* profile) {
  // Set up streams.
  std::vector<StreamPool::Ptr> streams;

  TF_ASSIGN_OR_RETURN(auto replicas, Replicas(*backend, device_handle));
  TF_RET_CHECK(!replicas.empty());
  for (se::StreamExecutor* executor : replicas) {
    TF_ASSIGN_OR_RETURN(StreamPool::Ptr stream,
                        backend->BorrowStream(executor));
    streams.push_back(std::move(stream));
  }

  DeviceAssignment device_assignment(options_.number_of_replicas(),
                                     /*computation_count=*/1);
  for (int64_t replica = 0; replica < replicas.size(); ++replica) {
    device_assignment(replica, 0) = replicas[replica]->device_ordinal();
  }
  const DeviceAssignment* device_assignment_ptr = &device_assignment;
  if (executable->module_config().has_static_device_assignment()) {
    device_assignment_ptr =
        &executable->module_config().static_device_assignment();
  }
  // Set up run options.
  std::vector<ServiceExecutableRunOptions> run_options;
  run_options.reserve(streams.size());
  for (const StreamPool::Ptr& stream : streams) {
    ExecutableRunOptions options;
    options.set_stream(stream.get());
    options.set_device_ordinal(stream->parent()->device_ordinal());
    options.set_local_device_count(backend->device_count());
    options.set_allocator(backend->memory_allocator());
    options.set_intra_op_thread_pool(
        backend->eigen_intra_op_thread_pool_device());
    options.set_device_assignment(device_assignment_ptr);
    options.set_execution_profile(profile);
    run_options.emplace_back(options, backend->StreamBorrowerWithPriority());
  }

  if (options_.number_of_replicas() == 1) {
    TF_ASSIGN_OR_RETURN(auto result, executable->ExecuteOnStreamWrapper(
                                         run_options.data(), arguments[0]));
    return allocation_tracker_.Register(std::move(result), result_tag);
  }

  // TODO(b/69985541): Support profiling also on this path.

  std::vector<absl::Span<const ShapedBuffer* const>> replicated_arguments;
  for (const auto& arg : arguments) {
    replicated_arguments.push_back(arg);
  }

  TF_ASSIGN_OR_RETURN(auto results, executable->ExecuteOnStreams(
                                        run_options, replicated_arguments));
  TF_RET_CHECK(!results.empty());
  return allocation_tracker_.RegisterReplicatedBuffers(std::move(results),
                                                       result_tag);
}

absl::StatusOr<std::vector<se::StreamExecutor*>> Service::GetExecutors(
    const ExecutionOptions& execution_options, int64_t requests_size,
    int64_t request_index) const {
  if (execution_options.device_handles().empty()) {
    return FailedPrecondition(
        "device handles must be given to execute parallel computations");
  }
  if (requests_size > 1 && execution_options.device_handles_size() > 1) {
    return InvalidArgument(
        "Parallel requests with multiple device handles is not supported. "
        "Found %d parallel requests, with request %d containing %d device "
        "handles.",
        requests_size, request_index, execution_options.device_handles_size());
  }
  std::vector<se::StreamExecutor*> executors;
  for (const auto& device_handle : execution_options.device_handles()) {
    TF_ASSIGN_OR_RETURN(auto replicas,
                        Replicas(*execute_backend_, device_handle));
    se::StreamExecutor* executor = replicas[0];
    CHECK(executor != nullptr);
    executors.push_back(executor);
  }
  return executors;
}

absl::StatusOr<std::vector<std::vector<const ShapedBuffer*>>>
Service::GetArguments(const ExecutionOptions& execution_options,
                      absl::Span<const GlobalData* const> arguments) const {
  // Resolve the allocations for the arguments of the computation, and create
  // a vector of device memory offsets for the arguments from the allocations.
  // In the case of partitioned computations, assume all arguments go on the
  // zeroth core.
  TF_ASSIGN_OR_RETURN(
      auto replicas,
      Replicas(*execute_backend_, execution_options.device_handles(0)));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<const ShapedBuffer*>> replicated_arguments,
      ResolveAndValidateArguments(arguments, replicas));
  return replicated_arguments;
}

absl::StatusOr<std::vector<std::unique_ptr<GlobalData>>>
Service::ExecuteGraphParallel(
    absl::Span<const XlaComputationInstance> computations) {
  VLOG(1) << "running execute-graph-parallel request";

  std::vector<std::vector<std::vector<const ShapedBuffer*>>> all_arguments;
  std::vector<std::vector<se::StreamExecutor*>> all_executors;
  std::vector<const HloModuleProto*> module_protos;
  std::vector<std::unique_ptr<HloModuleConfig>> module_configs;
  std::vector<std::string> computation_names;
  std::vector<DeviceHandle> device_handles;

  int num_requested_devices =
      std::accumulate(computations.begin(), computations.end(), 0,
                      [](int a, const XlaComputationInstance& r) -> int {
                        return a + r.execution_options.device_handles_size();
                      });

  if (num_requested_devices * options_.number_of_replicas() >
      execute_backend_->device_count()) {
    return FailedPrecondition(
        "there are not enough stream executors to execute %d computations",
        num_requested_devices);
  }

  for (int64_t i = 0; i < computations.size(); ++i) {
    const XlaComputationInstance& computation = computations[i];

    // Get the stream executor for the i'th computation. This stream executor
    // is one of the executors to run the replicated computation.
    const ExecutionOptions& execution_options = computation.execution_options;
    TF_RET_CHECK(computation.computation.proto().has_host_program_shape())
        << "program shape may not be empty";

    // Get the executors.
    TF_ASSIGN_OR_RETURN(
        std::vector<se::StreamExecutor*> executors,
        GetExecutors(execution_options, computations.size(), i));

    // Get the replicated arguments.
    TF_ASSIGN_OR_RETURN(
        std::vector<std::vector<const ShapedBuffer*>> replicated_arguments,
        GetArguments(execution_options, computation.arguments));

    for (auto& args : replicated_arguments) {
      for (auto& arg : args) {
        auto update_shape_with_empty_tiles = [this](
                                                 Shape* subshape,
                                                 const xla::ShapeIndex& index) {
          if (subshape->IsArray() && subshape->layout().tiles().empty()) {
            *subshape =
                execute_backend_->transfer_manager()->HostShapeToDeviceShape(
                    *subshape);
          }
        };
        ShapeUtil::ForEachMutableSubshape(
            const_cast<Shape*>(&arg->on_device_shape()),
            update_shape_with_empty_tiles);
      }
    }

    // Create an HloModuleConfig object for the computation, given the shape of
    // the program and the argument allocations. Here, we care only about the
    // shapes of the arguments, so, it is sufficient to use the arguments of
    // replica 0.
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModuleConfig> module_config,
        CreateModuleConfig(
            ProgramShape{computation.computation.proto().host_program_shape()},
            replicated_arguments.front(), computation.execution_options));
    VLOG(3)
        << "ExecuteGraphParallel created HloModuleConfig computation layout: "
        << module_config->entry_computation_layout().ToString();

    // Adds to the vectors to build and execute the computations after the loop.
    all_arguments.push_back(replicated_arguments);
    all_arguments.insert(all_arguments.end(), executors.size() - 1, {{}});
    module_protos.push_back(&computation.computation.proto());
    module_configs.push_back(std::move(module_config));
    computation_names.insert(computation_names.end(), executors.size(),
                             computation.computation.name());
    all_executors.push_back(executors);
    device_handles.insert(device_handles.end(),
                          execution_options.device_handles().begin(),
                          execution_options.device_handles().end());
  }

  // Build the HloModules and compile to generate the executables.
  //
  // TODO(jlebar): There's currently no way to pass a device allocator to
  // ExecuteGraphParallel, so we have to pass a null device_allocator below.
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<Executable>> executables,
                      BuildExecutables(module_protos, std::move(module_configs),
                                       execute_backend_.get(), all_executors,
                                       {/*device_allocator=*/nullptr}));
  std::vector<Executable*> executable_ptrs;
  executable_ptrs.reserve(executables.size());
  for (const auto& executable : executables) {
    executable_ptrs.push_back(executable.get());
  }

  std::vector<HloSnapshot> snapshots;
  snapshots.resize(executable_ptrs.size());
  for (int i = 0, end = executable_ptrs.size(); i < end; i++) {
    if (executable_ptrs[i]->dumping_snapshot()) {
      *snapshots[i].mutable_hlo() = *executable_ptrs[i]->hlo_proto();
      TF_ASSIGN_OR_RETURN(auto stream,
                          execute_backend_->BorrowStream(
                              all_executors[i][0]->device_ordinal()));
      TF_RETURN_IF_ERROR(RecordArguments(all_arguments[i].front(), stream.get(),
                                         execute_backend_->transfer_manager(),
                                         &snapshots[i]));
    }
  }

  // If we have multiple executables to run, execute them all in parallel.  But
  // if we only have one executable, execute it using the vanilla, non-parallel
  // call.
  //
  // We do this because the Client API uses ExecuteGraphParallel when it wants
  // to compile and run one computation without caching the executable, but not
  // all backends support the async StreamExecutor API required by
  // ExecuteParallelAndRegisterResult.
  //
  // TODO(b/122731460): Consolidate Execute{,Parallel}AndRegisterResult; they do
  // basically the same thing.
  ExecutionProfile profile;
  std::vector<GlobalDataHandle> outputs;
  absl::Status execution_status = absl::OkStatus();

  if (executable_ptrs.size() == 1) {
    absl::StatusOr<GlobalDataHandle> output_or_status =
        ExecuteAndRegisterResult(executable_ptrs[0], all_arguments[0],
                                 execute_backend_.get(), device_handles[0],
                                 computation_names[0], &profile);
    if (output_or_status.ok()) {
      outputs.push_back(std::move(output_or_status).value());
    } else {
      execution_status = output_or_status.status();
    }
  } else {
    absl::StatusOr<std::vector<GlobalDataHandle>> outputs_or_status =
        ExecuteParallelAndRegisterResult(executable_ptrs, all_arguments,
                                         execute_backend_.get(), device_handles,
                                         computation_names, &profile);
    if (outputs_or_status.ok()) {
      outputs = std::move(outputs_or_status).value();
    } else {
      execution_status = outputs_or_status.status();
    }
  }

  for (int64_t i = 0; i < computations.size(); ++i) {
    if (computations[i].execution_profile != nullptr) {
      *computations[i].execution_profile = profile;
    }
  }

  if (!execution_status.ok()) {
    // Execution failed so we don't have the results.  Dump the HLO snapshot
    // with just the program arguments.
    for (int i = 0, end = executable_ptrs.size(); i < end; i++) {
      DumpHloSnapshotIfEnabled(executable_ptrs[i]->module(), snapshots[i]);
    }
  }

  TF_RETURN_IF_ERROR(execution_status);

  std::vector<std::unique_ptr<GlobalData>> out;

  out.reserve(out.size());
  for (GlobalDataHandle& output : outputs) {
    out.push_back(std::make_unique<GlobalData>(this, output));
  }

  for (int i = 0, end = executable_ptrs.size(); i < end; i++) {
    Executable* executable = executable_ptrs[i];
    if (executable->dumping_snapshot()) {
      TF_ASSIGN_OR_RETURN(const ShapedBuffer* result_buffer,
                          allocation_tracker_.ResolveForReplica(outputs[i], 0));
      TF_ASSIGN_OR_RETURN(auto stream,
                          execute_backend_->BorrowStream(all_executors[i][0]));
      TF_RETURN_IF_ERROR(RecordResult(*result_buffer, stream.get(),
                                      execute_backend_->transfer_manager(),
                                      &snapshots[i]));
      DumpHloSnapshotIfEnabled(executable->module(), snapshots[i]);
    }
  }

  VLOG(1) << "successfully completed 'execute-graph-parallel' request";
  return out;
}

absl::StatusOr<std::vector<DeviceHandle>> Service::GetDeviceHandles(
    int64_t device_count) {
  const int64_t available_device_count = execute_backend_->device_count();
  const int64_t replica_count = options_.number_of_replicas();
  if (replica_count <= 0) {
    return FailedPrecondition("Replica count must be a positive integer");
  }
  if (available_device_count < device_count * replica_count) {
    return ResourceExhausted(
        "Requested logical device count (%d) with replica count (%d) exceeds "
        "the number of available physical devices on the target (%d)",
        device_count, replica_count, available_device_count);
  }

  std::vector<DeviceHandle> out;

  for (int64_t i = 0; i < device_count; ++i) {
    DeviceHandle device_handle;
    device_handle.set_handle(i);
    device_handle.set_device_count(device_count);
    out.push_back(device_handle);
  }
  return out;
}

absl::StatusOr<std::unique_ptr<Executable>> Service::BuildExecutable(
    const HloModuleProto& module_proto,
    std::unique_ptr<HloModuleConfig> module_config, Backend* backend,
    se::StreamExecutor* executor, const Compiler::CompileOptions& options,
    bool run_backend_only) {
  VLOG(1) << StrFormat(
      "BuildExecutable on service %p with serialized module proto: %s", this,
      module_proto.name());

  tsl::profiler::ScopedAnnotation annotation{[&] {
    // module's unique_id is not available yet
    return absl::StrFormat("XlaCompile:#module=%s#", module_proto.name());
  }};

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      CreateModuleFromProto(module_proto, *module_config, run_backend_only));
  UpdateEntryComputationLayout(
      module.get(), std::bind(&Compiler::DefaultDeviceShapeRepresentation,
                              backend->compiler(), std::placeholders::_1));
  DumpHloModuleIfEnabled(*module, kBeforeOptimizationsDumpName);

  std::unique_ptr<HloProto> hlo_proto_before_opt;
  if (!run_backend_only) {
    // Save proto state before optimizations if we want a snapshot.
    // When run_backend_only is enabled the post-optimization HLO will be the
    // same as the pre-optimization HLO.
    if (DumpingEnabledForHloModule(*module)) {
      hlo_proto_before_opt = std::make_unique<HloProto>(MakeHloProto(*module));
    }
    TF_ASSIGN_OR_RETURN(module, backend->compiler()->RunHloPasses(
                                    std::move(module), executor, options));
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      backend->compiler()->RunBackend(std::move(module), executor, options));

  const BufferAssignmentProto* buffer_assignment_proto_after_opt =
      executable->buffer_assignment_proto();

  // If dumping is enabled RunBackend(...) will emit a hlo_proto in the
  // executable. This contains the buffer_assignment that is only available
  // after RunBackend(). If hlo_proto_before_opt is not null, then we replace
  // its buffer_assignment with the one from after_opt and then store it into
  // the executable.
  if (hlo_proto_before_opt != nullptr &&
      buffer_assignment_proto_after_opt != nullptr) {
    CHECK(DumpingEnabledForHloModule(executable->module()));
    *hlo_proto_before_opt->mutable_buffer_assignment() =
        *buffer_assignment_proto_after_opt;
    executable->set_hlo_proto(std::move(hlo_proto_before_opt));
  }
  return executable;
}

absl::StatusOr<ExecutionHandle> Service::Compile(
    const XlaComputation& computation, absl::Span<const Shape> argument_shapes,
    const ExecutionOptions& execution_options) {
  VLOG(1) << "running compile request";

  if (!computation.proto().has_host_program_shape()) {
    return InvalidArgument("program shape may not be empty");
  }

  if (execution_options.device_handles_size() > 1) {
    return InvalidArgument(
        "The compile request does not support multiple device handles.");
  }

  std::vector<const Shape*> argument_shape_ptrs;
  for (const Shape& shape : argument_shapes) {
    argument_shape_ptrs.push_back(&shape);
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> module_config,
      CreateModuleConfig(ProgramShape{computation.proto().host_program_shape()},
                         argument_shape_ptrs, &execution_options));
  VLOG(3) << "Compile created HloModuleConfig computation layout: "
          << module_config->entry_computation_layout().ToString();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      BuildExecutable(computation.proto(), std::move(module_config),
                      execute_backend_.get(),
                      execute_backend_->default_stream_executor(),
                      {/*device_allocator=*/nullptr}));

  VLOG(1) << "successfully completed 'compile' request";
  return compilation_cache_.Insert(std::move(executable));
}

absl::StatusOr<std::unique_ptr<GlobalData>> Service::Execute(
    const ExecutionHandle& handle, absl::Span<GlobalData* const> arguments,
    ExecutionProfile* execution_profile) {
  VLOG(1) << "running execute request";

  TF_ASSIGN_OR_RETURN(std::shared_ptr<Executable> executable,
                      compilation_cache_.LookUp(handle));

  TF_ASSIGN_OR_RETURN(
      std::vector<se::StreamExecutor*> replicas,
      Replicas(*execute_backend_, SingleComputationDeviceHandle()));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<const ShapedBuffer*>> replicated_arguments,
      ResolveAndValidateArguments(arguments, replicas));

  // Check that the replicated_arguments has the same shape and layout as the
  // module config used when creating the executable.
  const int64_t num_module_args =
      executable->module_config().entry_computation_layout().parameter_count();
  if (num_module_args != arguments.size()) {
    return InvalidArgument(
        "The executable expects %lld arguments, but sees %lld.",
        num_module_args, arguments.size());
  }
  for (int64_t i = 0; i < num_module_args; i++) {
    const Shape& shape_module =
        executable->module_config().entry_computation_layout().parameter_shape(
            i);
    const Shape& shape_arg = replicated_arguments.front()[i]->on_device_shape();
    if (!ShapeUtil::Equal(shape_module, shape_arg)) {
      return InvalidArgumentStrCat(
          "The executable expects the ", i, "th argument in shape ",
          ShapeUtil::HumanStringWithLayout(shape_module), " but sees ",
          ShapeUtil::HumanStringWithLayout(shape_arg));
    }
  }

  TF_ASSIGN_OR_RETURN(auto stream,
                      execute_backend_->BorrowStream(
                          execute_backend_->default_stream_executor()));
  HloSnapshot snapshot;
  if (executable->dumping_snapshot()) {
    *snapshot.mutable_hlo() = *executable->hlo_proto();
    snapshot.set_execution_platform(execute_backend_->platform()->Name());
    TF_RETURN_IF_ERROR(
        RecordArguments(replicated_arguments.front(), stream.get(),
                        execute_backend_->transfer_manager(), &snapshot));
  }

  TF_ASSIGN_OR_RETURN(
      GlobalDataHandle output,
      ExecuteAndRegisterResult(
          executable.get(), replicated_arguments, execute_backend_.get(),
          SingleComputationDeviceHandle(),
          "result of " + executable->module().name(), execution_profile));

  if (executable->dumping_snapshot()) {
    TF_ASSIGN_OR_RETURN(const ShapedBuffer* result_buffer,
                        allocation_tracker_.ResolveForReplica(output, 0));
    TF_RETURN_IF_ERROR(RecordResult(*result_buffer, stream.get(),
                                    execute_backend_->transfer_manager(),
                                    &snapshot));
    DumpHloSnapshotIfEnabled(executable->module(), snapshot);
  }

  return std::make_unique<GlobalData>(this, output);
}

absl::StatusOr<Literal> Service::TransferToClient(
    const GlobalData& data, const Shape* shape_with_layout) {
  TF_ASSIGN_OR_RETURN(const ShapedBuffer* shaped_buffer,
                      allocation_tracker_.ResolveForReplica(data.handle(), 0));

  Shape return_shape;
  if (shape_with_layout) {
    return_shape = Shape(*shape_with_layout);
    if (!LayoutUtil::HasLayout(return_shape)) {
      return InvalidArgument("shape_with_layout must have layout if present.");
    }
    if (return_shape.has_layout() &&
        return_shape.layout().element_size_in_bits() != 0) {
      return InvalidArgument(
          "shape_with_layout cannot have layout's element_size_in_bits field "
          "set");
    }
  } else {
    return_shape = Shape(shaped_buffer->on_device_shape());
    if (return_shape.has_layout() &&
        return_shape.layout().element_size_in_bits() != 0) {
      // Literals do not support element_size_in_bits
      return_shape.mutable_layout()->set_element_size_in_bits(0);
    }
  }

  TF_ASSIGN_OR_RETURN(
      auto stream,
      execute_backend_->BorrowStream(shaped_buffer->physical_device_ordinal()));

  TF_ASSIGN_OR_RETURN(
      Literal result_literal,
      execute_backend_->transfer_manager()->TransferLiteralFromDevice(
          stream.get(), *shaped_buffer));

  if (LayoutUtil::LayoutsInShapesEqual(return_shape, result_literal.shape())) {
    return result_literal;
  }
  return result_literal.Relayout(return_shape);
}

absl::StatusOr<std::unique_ptr<GlobalData>> Service::TransferToServer(
    const LiteralSlice& literal_slice, const DeviceHandle* device_handle) {
  const Shape& shape = literal_slice.shape();
  std::vector<se::StreamExecutor*> replicas;
  if (device_handle) {
    TF_ASSIGN_OR_RETURN(replicas, Replicas(*execute_backend_, *device_handle));
  } else {
    TF_ASSIGN_OR_RETURN(
        replicas, Replicas(*execute_backend_, SingleComputationDeviceHandle()));
  }

  // Allocate memory in each replica and transfer the data to all replicas.
  std::vector<ScopedShapedBuffer> replicated_buffers;
  replicated_buffers.reserve(replicas.size());
  for (se::StreamExecutor* executor : replicas) {
    auto device_shape_representation_fn = [this](const Shape& shape) {
      return execute_backend_->compiler()->DefaultDeviceShapeRepresentation(
          shape);
    };
    TF_ASSIGN_OR_RETURN(
        ScopedShapedBuffer shaped_buffer,
        execute_backend_->transfer_manager()->AllocateScopedShapedBuffer(
            shape, execute_backend_->memory_allocator(),
            executor->device_ordinal(), device_shape_representation_fn));
    TF_ASSIGN_OR_RETURN(auto stream, execute_backend_->BorrowStream(executor));
    TF_RETURN_IF_ERROR(
        execute_backend_->transfer_manager()->TransferLiteralToDevice(
            stream.get(), literal_slice, shaped_buffer));
    replicated_buffers.emplace_back(std::move(shaped_buffer));
  }

  TF_ASSIGN_OR_RETURN(GlobalDataHandle out,
                      allocation_tracker_.RegisterReplicatedBuffers(
                          std::move(replicated_buffers),
                          StrCat("TransferToServer literal of shape ",
                                 ShapeUtil::HumanString(shape))));

  return std::make_unique<GlobalData>(this, out);
}

absl::Status Service::TransferToInfeed(const LiteralSlice& literal,
                                       int64_t replica_id,
                                       const DeviceHandle* device_handle) {
  const int64_t replica_count = options_.number_of_replicas();
  if (replica_id < 0 || replica_id >= replica_count) {
    return FailedPrecondition(
        "%s",
        StrCat("The replica_id=", replica_id,
               " on TransferToInfeedRequest not in range [0, replica_count=",
               replica_count, ")."));
  }

  se::StreamExecutor* executor;
  if (device_handle) {
    TF_ASSIGN_OR_RETURN(auto replicas,
                        Replicas(*execute_backend_, *device_handle));
    executor = replicas[replica_id];
  } else {
    TF_ASSIGN_OR_RETURN(
        auto replicas,
        Replicas(*execute_backend_, SingleComputationDeviceHandle()));
    executor = replicas[replica_id];
  }

  return execute_backend_->transfer_manager()->TransferLiteralToInfeed(executor,
                                                                       literal);
}

absl::StatusOr<Literal> Service::TransferFromOutfeed(
    const Shape* shape_with_layout, int64_t replica_id,
    const DeviceHandle* device_handle) {
  const int64_t replica_count = options_.number_of_replicas();
  if (replica_id < 0 || replica_id >= replica_count) {
    return FailedPrecondition(
        "The replica_id=%d on TransferFromOutfeedRequest not in range [0, %d)",
        replica_id, replica_count);
  }

  se::StreamExecutor* executor;
  if (device_handle) {
    TF_ASSIGN_OR_RETURN(auto replicas,
                        Replicas(*execute_backend_, *device_handle));
    executor = replicas[replica_id];
  } else {
    TF_ASSIGN_OR_RETURN(
        auto replicas,
        Replicas(*execute_backend_, SingleComputationDeviceHandle()));
    executor = replicas[replica_id];
  }

  auto literal = Literal::CreateFromShape(*shape_with_layout);

  TF_RETURN_IF_ERROR(
      execute_backend_->transfer_manager()->TransferLiteralFromOutfeed(
          executor, &literal));
  return literal;
}

absl::Status Service::ResetDevice() { return execute_backend_->ResetDevices(); }

absl::StatusOr<Literal> Service::ComputeConstantGraph(
    const XlaComputation& computation, const Layout* output_layout) {
  if (!computation.proto().has_host_program_shape()) {
    return InvalidArgument("program shape may not be empty");
  }
  if (computation.proto().host_program_shape().parameters_size() != 0) {
    return InvalidArgument(
        "constant computation may not depend on any parameters.");
  }

  ProgramShape program_shape(computation.proto().host_program_shape());
  TF_DCHECK_OK(ShapeUtil::ValidateShape(program_shape.result()));

  if (output_layout) {
    TF_RETURN_IF_ERROR(LayoutUtil::ValidateLayoutForShape(
        *output_layout, program_shape.result()));
  }

  HloModuleConfig config(program_shape);

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      CreateModuleFromProto(computation.proto(), config));
  DynamicPadder dynamic_padder;
  TF_RETURN_IF_ERROR(dynamic_padder.Run(module.get()).status());

  TF_ASSIGN_OR_RETURN(DynamicDimensionInference dynamic_dimension_inference,
                      DynamicDimensionInference::Run(module.get()));

  HloEvaluator evaluator;
  evaluator.set_dynamic_dimension_inference(&dynamic_dimension_inference);
  evaluator.set_custom_call_handler(
      [](const HloInstruction* custom_call,
         absl::Span<const Literal*> operands) -> absl::StatusOr<Literal> {
        if (custom_call->custom_call_target() == "SliceToDynamic") {
          auto result = operands[0]->Clone();
          for (int64_t i = 0; i < result.shape().rank(); ++i) {
            result.SetDynamicSize(i, operands[1 + i]->Get<int32_t>({}));
          }
          return result.ToStatic();
        }
        return Unimplemented("Custom call %s is not supported: %s",
                             custom_call->custom_call_target(),
                             custom_call->ToString());
      });
  TF_ASSIGN_OR_RETURN(auto result_literal, evaluator.Evaluate(*module, {}));

  // Since the result layout is non-effective to the Evaluator results, explicit
  // relayout here.
  //
  // TODO(b/77824332): Make HloEvaluator take care of the re-layout.
  if (output_layout) {
    result_literal = result_literal.Relayout(*output_layout);
  }
  return result_literal;
}

absl::StatusOr<Shape> Service::GetShape(const GlobalData& data) {
  TF_ASSIGN_OR_RETURN(const ShapedBuffer* buffer,
                      allocation_tracker_.ResolveForReplica(data.handle(), 0));
  return buffer->on_device_shape();
}

DeviceHandle Service::SingleComputationDeviceHandle() const {
  DeviceHandle device_handle;
  device_handle.set_handle(0);
  device_handle.set_device_count(1);
  return device_handle;
}

absl::StatusOr<std::vector<se::StreamExecutor*>> Service::Replicas(
    const Backend& backend, const DeviceHandle& device_handle) const {
  std::vector<se::StreamExecutor*> replicas;
  for (int replica = 0; replica < options_.number_of_replicas(); ++replica) {
    // From the computation placer, find out the device ids of the replicas for
    // the given device handle.
    TF_ASSIGN_OR_RETURN(
        int device_ordinal,
        backend.computation_placer()->DeviceId(replica, device_handle.handle(),
                                               options_.number_of_replicas(),
                                               device_handle.device_count()));
    TF_ASSIGN_OR_RETURN(auto executor, backend.stream_executor(device_ordinal));
    replicas.push_back(executor);
  }
  return replicas;
}

namespace {

// Releases a set of global data handles owned by the parent service
// interface.
void ReleaseHandles(Service* parent,
                    const absl::Span<const GlobalDataHandle> handles) {
  for (const GlobalDataHandle& handle : handles) {
    VLOG(1) << "Requesting to unregister " << handle.ShortDebugString();
    absl::Status status = parent->Unregister(handle);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to unregister handles: " << status
                   << "; continuing anyway...";
    }
  }
  VLOG(1) << "Done with request";
}

}  // namespace

GlobalData::GlobalData(Service* parent, GlobalDataHandle handle)
    : handle_(std::move(handle)), parent_(parent) {}

GlobalData::~GlobalData() {
  if (parent_ != nullptr) {
    ReleaseHandles(parent_, {handle_});
  }
}

/* static */ void GlobalData::Release(
    std::vector<std::unique_ptr<GlobalData>> instances) {
  absl::flat_hash_map<Service*, std::vector<GlobalDataHandle>>
      parent_handles_map;
  for (auto& instance : instances) {
    if (instance->parent_ != nullptr) {
      parent_handles_map[instance->parent_].push_back(instance->Release());
    }
  }
  for (auto& parent_handles : parent_handles_map) {
    ReleaseHandles(parent_handles.first, parent_handles.second);
  }
}

}  // namespace xla
