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

#include "tensorflow/compiler/xla/service/service.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/dynamic_dimension_inference.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/service/source_map_util.h"
#include "tensorflow/compiler/xla/service/stream_pool.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {
namespace {

using absl::StrCat;
using absl::StrFormat;

// Argument used when calling DumpHloModuleIfEnabled before optimizations are
// performed on an HloModule.
constexpr char kBeforeOptimizationsDumpName[] = "before_optimizations";

// Records the arguments used to invoke a computation in an HloSnapshot proto.
Status RecordArguments(const absl::Span<const ShapedBuffer* const> arguments,
                       se::Stream* stream, TransferManager* transfer_manager,
                       HloSnapshot* module) {
  module->clear_arguments();
  for (const ShapedBuffer* argument : arguments) {
    TF_ASSIGN_OR_RETURN(
        Literal literal,
        transfer_manager->TransferLiteralFromDevice(stream, *argument));
    *module->add_arguments() = literal.ToProto();
  }
  return Status::OK();
}

// Records the result of a computation in a HloSnapshot proto.
Status RecordResult(const ShapedBuffer& result, se::Stream* stream,
                    TransferManager* transfer_manager, HloSnapshot* module) {
  module->clear_result();
  TF_ASSIGN_OR_RETURN(
      Literal literal,
      transfer_manager->TransferLiteralFromDevice(stream, result));
  *module->mutable_result() = literal.ToProto();
  return Status::OK();
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
    const absl::optional<std::set<int>>& allowed_devices) {
  allowed_devices_ = allowed_devices;
  return *this;
}

const absl::optional<std::set<int>>& ServiceOptions::allowed_devices() const {
  return allowed_devices_;
}

/* static */ StatusOr<std::unique_ptr<Service>> Service::NewService(
    se::Platform* platform) {
  ServiceOptions default_options;
  default_options.set_platform(platform);
  return NewService(default_options);
}

/* static */ StatusOr<std::unique_ptr<Service>> Service::NewService(
    const ServiceOptions& options) {
  se::Platform* platform = options.platform();
  std::unique_ptr<Backend> execute_backend;
  if (platform == nullptr) {
    TF_ASSIGN_OR_RETURN(platform, PlatformUtil::GetDefaultPlatform());
  }
  BackendOptions backend_options;
  backend_options.set_platform(platform);
  backend_options.set_allowed_devices(options.allowed_devices());
  TF_ASSIGN_OR_RETURN(execute_backend, Backend::CreateBackend(backend_options));

  std::unique_ptr<Service> service(
      new Service(options, std::move(execute_backend)));
  return std::move(service);
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

Status Service::CreateChannelHandle(const CreateChannelHandleRequest* arg,
                                    CreateChannelHandleResponse* result) {
  TF_ASSIGN_OR_RETURN(*result->mutable_channel(),
                      channel_tracker_.NewChannel(arg->channel_type()));
  return Status::OK();
}

Status Service::Unregister(const UnregisterRequest* arg,
                           UnregisterResponse* result) {
  Status status;
  for (auto& data : arg->data()) {
    Status unregister_status = allocation_tracker_.Unregister(data);
    if (!unregister_status.ok() && status.ok()) {
      status = unregister_status;
    }
  }
  return status;
}

// Deconstructs a previously-allocated global handle.
Status Service::DeconstructTuple(const DeconstructTupleRequest* arg,
                                 DeconstructTupleResponse* result) {
  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDataHandle> elements,
      allocation_tracker_.DeconstructTuple(arg->tuple_handle()));

  for (auto& element : elements) {
    *result->add_element_handles() = element;
  }
  return Status::OK();
}

Status Service::ValidateResultShape(const Shape& client_shape,
                                    const Shape& result_shape) const {
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(client_shape));
  if (!ShapeUtil::Compatible(client_shape, result_shape)) {
    return InvalidArgument(
        "Shape used to set computation result layout %s is not compatible "
        "with result shape %s",
        ShapeUtil::HumanStringWithLayout(client_shape),
        ShapeUtil::HumanString(result_shape));
  }
  return Status::OK();
}

StatusOr<std::vector<std::vector<const ShapedBuffer*>>>
Service::ResolveAndValidateArguments(
    absl::Span<const GlobalDataHandle* const> arguments,
    absl::Span<se::StreamExecutor* const> stream_executors) const {
  CHECK_EQ(options_.number_of_replicas(), stream_executors.size());
  std::vector<std::vector<const ShapedBuffer*>> replicated_arguments;
  replicated_arguments.resize(options_.number_of_replicas());
  for (size_t i = 0; i < arguments.size(); ++i) {
    auto buffer_status = allocation_tracker_.Resolve(*arguments[i]);
    if (!buffer_status.ok()) {
      return Status(buffer_status.status().code(),
                    StrCat(buffer_status.status().error_message(), ", ",
                           "failed to resolve allocation for parameter ", i));
    }
    auto replicated_buffers = buffer_status.ValueOrDie();
    CHECK_EQ(options_.number_of_replicas(), replicated_buffers.size());
    for (int replica = 0; replica < options_.number_of_replicas(); ++replica) {
      const ShapedBuffer* shaped_buffer = replicated_buffers[replica];
      int replica_device_ordinal = stream_executors[replica]->device_ordinal();
      // Verify allocation is same platform and device as the execution.
      if (shaped_buffer->platform() != execute_backend_->platform() ||
          shaped_buffer->device_ordinal() != replica_device_ordinal) {
        return InvalidArgument(
            "argument %lu is on device %s:%d but computation will be executed "
            "on device %s",
            i, shaped_buffer->platform()->Name(),
            shaped_buffer->device_ordinal(),
            execute_backend_->device_name(replica_device_ordinal));
      }
      replicated_arguments[replica].push_back(shaped_buffer);
    }
  }
  return replicated_arguments;
}

StatusOr<std::unique_ptr<HloModuleConfig>> Service::CreateModuleConfig(
    const ProgramShape& program_shape,
    absl::Span<const Shape* const> argument_shapes,
    const ExecutionOptions* execution_options,
    const AotCompilationOptions* aot_options) {
  auto config = absl::make_unique<HloModuleConfig>(program_shape);
  ComputationLayout* computation_layout =
      config->mutable_entry_computation_layout();
  if (program_shape.parameters_size() != argument_shapes.size()) {
    return InvalidArgument("computation takes %d parameters, but %u given",
                           program_shape.parameters_size(),
                           argument_shapes.size());
  }
  for (int i = 0; i < argument_shapes.size(); ++i) {
    // Verify that shape of arguments matches the shape of the arguments in the
    // ProgramShape.
    if (!ShapeUtil::Compatible(*argument_shapes[i],
                               program_shape.parameters(i))) {
      return InvalidArgument(
          "Argument does not match shape of computation parameter %d: want "
          "%s, got %s",
          i, ShapeUtil::HumanString(program_shape.parameters(i)),
          ShapeUtil::HumanString(*argument_shapes[i]));
    }
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
            *argument_shapes[i]));
  }
  if (execution_options != nullptr &&
      execution_options->has_shape_with_output_layout()) {
    const Shape shape_with_output_layout(
        execution_options->shape_with_output_layout());
    TF_RETURN_IF_ERROR(
        ValidateResultShape(shape_with_output_layout, program_shape.result()));
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_result_layout()->CopyLayoutFromShape(
            shape_with_output_layout));
  } else {
    // If the result layout is not set, then choose the default.
    computation_layout->mutable_result_layout()->SetToDefaultLayout();
  }

  if (execution_options != nullptr) {
    if (execution_options->num_replicas() > 0) {
      config->set_replica_count(execution_options->num_replicas());
    } else {
      config->set_replica_count(options_.number_of_replicas());
    }
    if (execution_options->num_partitions() > 0) {
      config->set_num_partitions(execution_options->num_partitions());
    }
    config->set_seed(execution_options->seed());
    config->set_debug_options(execution_options->debug_options());
  } else {
    config->set_replica_count(options_.number_of_replicas());
    config->set_debug_options(GetDebugOptionsFromFlags());
  }

  if (execute_backend_ != nullptr &&
      execute_backend_->eigen_intra_op_thread_pool() != nullptr) {
    config->set_intra_op_parallelism_threads(
        execute_backend_->eigen_intra_op_thread_pool()->NumThreads());
  }

  if (execution_options != nullptr &&
      execution_options->has_device_assignment()) {
    TF_ASSIGN_OR_RETURN(
        auto device_assignment,
        DeviceAssignment::Deserialize(execution_options->device_assignment()));
    config->set_static_device_assignment(*device_assignment);
  }
  config->set_alias_passthrough_params(
      execution_options->alias_passthrough_params());

  if (aot_options != nullptr &&
      aot_options->fusion_config_collection() != FusionConfigCollection::kOff) {
    config->set_fusion_config_collection(
        aot_options->fusion_config_collection());
    *config->mutable_fusion_config() = aot_options->fusion_config();
  }

  return std::move(config);
}

StatusOr<std::unique_ptr<HloModuleConfig>> Service::CreateModuleConfig(
    const ProgramShape& program_shape,
    absl::Span<const ShapedBuffer* const> arguments,
    const ExecutionOptions& execution_options,
    const AotCompilationOptions* aot_options) {
  std::vector<const Shape*> argument_shapes;
  for (const auto* arg : arguments) {
    argument_shapes.push_back(&arg->on_host_shape());
  }
  return CreateModuleConfig(program_shape, argument_shapes, &execution_options,
                            aot_options);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> Service::BuildExecutables(
    const std::vector<const HloModuleProto*>& module_protos,
    std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
    Backend* backend, std::vector<std::vector<se::StreamExecutor*>> executors,
    se::DeviceMemoryAllocator* device_allocator) {
  VLOG(1) << StrFormat("BuildExecutable on service %p", this);

  // Dump computation proto state if flag is set.
  std::vector<std::unique_ptr<HloProto>> hlo_protos;
  for (int64 i = 0; i < module_protos.size(); ++i) {
    auto hlo_proto = absl::make_unique<HloProto>();
    *hlo_proto->mutable_hlo_module() = *module_protos[i];
    hlo_protos.push_back(std::move(hlo_proto));
  }

  VLOG(1) << "Computations:";
  for (const HloModuleProto* proto : module_protos) {
    VLOG(1) << proto->name();
  }

  CHECK_EQ(module_protos.size(), module_configs.size());
  auto module_group =
      absl::make_unique<HloModuleGroup>(module_protos[0]->name());
  for (int64 i = 0; i < module_protos.size(); ++i) {
    const HloModuleProto* proto = module_protos[i];
    const HloModuleConfig& config = *module_configs[i];
    TF_ASSIGN_OR_RETURN(auto module, CreateModuleFromProto(*proto, config));
    DumpHloModuleIfEnabled(*module, kBeforeOptimizationsDumpName);
    module_group->push_back(std::move(module));
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<Executable>> executables,
      backend->compiler()->Compile(std::move(module_group),
                                   std::move(executors), device_allocator));

  for (size_t i = 0; i < module_protos.size(); ++i) {
    const auto& debug_opts = module_configs[i]->debug_options();
    if (DumpingEnabledForHloModule(module_protos[i]->name(), debug_opts) &&
        debug_opts.xla_dump_hlo_snapshots()) {
      executables[i]->set_hlo_proto(std::move(hlo_protos[i]));
    }
  }

  return std::move(executables);
}

StatusOr<std::vector<GlobalDataHandle>>
Service::ExecuteParallelAndRegisterResult(
    absl::Span<Executable* const> executables,
    absl::Span<const std::vector<std::vector<const ShapedBuffer*>>> arguments,
    Backend* backend, absl::Span<const DeviceHandle> device_handles,
    absl::Span<const string> result_tags, ExecutionProfile* profile) {
  // Streams where the computation are launched, so we can wait on the streams
  // to complete.
  std::vector<StreamPool::Ptr> streams;
  std::vector<std::unique_ptr<se::Timer>> timers;

  // Global data handles for the computation results, one for each computation.
  std::vector<GlobalDataHandle> result_handles;

  // Device ID to stream executor, populated only with devices that are being
  // profiled.
  std::map<int64, se::Stream*> index_to_profiled_streams;

  // Build DeviceAssignment for all cores based on the provided device handles.
  DeviceAssignment device_assignment(options_.number_of_replicas(),
                                     executables.size());
  for (int64 i = 0; i < executables.size(); i++) {
    TF_ASSIGN_OR_RETURN(auto replicas, Replicas(*backend, device_handles[i]));
    CHECK_EQ(replicas.size(), arguments[i].size());
    for (int64 replica = 0; replica < replicas.size(); ++replica) {
      device_assignment(replica, i) = replicas[replica]->device_ordinal();
    }
  }

  for (int64 i = 0; i < executables.size(); i++) {
    // Stream executors for the replicas of the current computation.
    TF_ASSIGN_OR_RETURN(auto replicas, Replicas(*backend, device_handles[i]));
    CHECK_EQ(replicas.size(), arguments[i].size());
    std::vector<ScopedShapedBuffer> result_buffers;
    for (int64 replica = 0; replica < replicas.size(); ++replica) {
      TF_ASSIGN_OR_RETURN(StreamPool::Ptr stream,
                          backend->BorrowStream(replicas[replica]));
      streams.push_back(std::move(stream));

      if (replica == 0 && profile != nullptr) {
        timers.push_back(
            absl::make_unique<se::Timer>(streams.back()->parent()));
        streams.back()
            ->InitTimer(timers.back().get())
            .ThenStartTimer(timers.back().get());
        CHECK(timers.front() != nullptr);
      }

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
      options.set_device_assignment(&device_assignment);
      // Use run-time profile information from execution_profile on the 0th
      // device.
      if (i == 0) {
        options.set_execution_profile(profile);
      }
      ServiceExecutableRunOptions run_options(options,
                                              backend->StreamBorrower());

      // Asynchronously launch the computation.
      TF_ASSIGN_OR_RETURN(ScopedShapedBuffer result,
                          executables[i]->ExecuteAsyncOnStream(
                              &run_options, arguments[i][replica],
                              /*hlo_execution_profile=*/nullptr));

      if (replica == 0 && profile != nullptr) {
        streams.back()->ThenStopTimer(timers.back().get());
      }

      result_buffers.push_back(std::move(result));
    }
    TF_ASSIGN_OR_RETURN(GlobalDataHandle handle,
                        allocation_tracker_.RegisterReplicatedBuffers(
                            std::move(result_buffers), result_tags[i]));
    result_handles.push_back(handle);
  }

  // Wait for all executions to complete.
  for (int64 i = 0; i < streams.size(); ++i) {
    Status block_status = streams[i]->BlockHostUntilDone();
    if (!block_status.ok()) {
      return InternalError("failed to complete execution for stream %d: %s", i,
                           block_status.error_message());
    }
  }

  if (profile != nullptr) {
    CHECK(!timers.empty());
    std::vector<uint64> timer_nanoseconds;
    timer_nanoseconds.reserve(timers.size());
    for (auto& timer : timers) {
      timer_nanoseconds.push_back(timer->Nanoseconds());
    }
    uint64 nanoseconds =
        *std::max_element(timer_nanoseconds.begin(), timer_nanoseconds.end());

    // Overall execution time (in nanoseconds) from the executor timer.
    profile->set_compute_and_transfer_time_ns(nanoseconds);

    // TODO(b/28123297): On GPU we end up including transfer time in
    // the compute time this way. Instead, we should get the correct
    // value by measuring it. Setting the field here at least lets
    // benchmarks provide *some* value for GPU computations.
    //
    // TODO(b/28447609): The value in compute_and_transfer_time_ns is actually
    // the compute time without the transfer time, so this way we get the
    // correct compute time. We should instead have the correct value for
    // compute_and_transfer_time and set compute_time to the compute time.
    if (profile->compute_time_ns() == 0) {
      profile->set_compute_time_ns(profile->compute_and_transfer_time_ns());
    }
  }

  return result_handles;
}

StatusOr<GlobalDataHandle> Service::ExecuteAndRegisterResult(
    Executable* executable,
    absl::Span<const std::vector<const ShapedBuffer*>> arguments,
    Backend* backend, const DeviceHandle& device_handle,
    const string& result_tag, ExecutionProfile* profile) {
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
  for (int64 replica = 0; replica < replicas.size(); ++replica) {
    device_assignment(replica, 0) = replicas[replica]->device_ordinal();
  }

  // Set up run options.
  std::vector<ServiceExecutableRunOptions> run_options;
  for (const StreamPool::Ptr& stream : streams) {
    ExecutableRunOptions options;
    options.set_stream(stream.get());
    options.set_device_ordinal(stream->parent()->device_ordinal());
    options.set_allocator(backend->memory_allocator());
    options.set_intra_op_thread_pool(
        backend->eigen_intra_op_thread_pool_device());
    options.set_device_assignment(&device_assignment);
    options.set_execution_profile(profile);
    run_options.emplace_back(options, backend->StreamBorrower());
  }

  if (options_.number_of_replicas() == 1) {
    TF_ASSIGN_OR_RETURN(auto result, executable->ExecuteOnStreamWrapper(
                                         &run_options[0], arguments[0]));
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

StatusOr<std::vector<se::StreamExecutor*>> Service::GetExecutors(
    const ExecutionOptions& execution_options, int64 requests_size,
    int64 request_index) const {
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

StatusOr<std::vector<std::vector<const ShapedBuffer*>>> Service::GetArguments(
    const ExecutionOptions& execution_options,
    absl::Span<const GlobalDataHandle* const> arguments) const {
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

Status Service::ExecuteGraphParallel(const ExecuteGraphParallelRequest* arg,
                                     ExecuteParallelResponse* result) {
  VLOG(1) << "running execute-graph-parallel request";

  std::vector<std::vector<std::vector<const ShapedBuffer*>>> all_arguments;
  std::vector<std::vector<se::StreamExecutor*>> all_executors;
  std::vector<const HloModuleProto*> module_protos;
  std::vector<std::unique_ptr<HloModuleConfig>> module_configs;
  std::vector<string> computation_names;
  std::vector<DeviceHandle> device_handles;

  int num_requested_devices =
      std::accumulate(arg->requests().begin(), arg->requests().end(), 0,
                      [](int a, const ExecuteGraphRequest& r) -> int {
                        return a + r.execution_options().device_handles_size();
                      });
  if (num_requested_devices * options_.number_of_replicas() >
      execute_backend_->device_count()) {
    return FailedPrecondition(
        "there are not enough stream executors to execute %d computations",
        num_requested_devices);
  }

  for (int64 i = 0; i < arg->requests_size(); ++i) {
    // Get the stream executor for the i'th computation. This stream executor
    // is one of the executors to run the replicated computation.
    const ExecutionOptions& execution_options =
        arg->requests(i).execution_options();
    const ExecuteGraphRequest& request = arg->requests(i);
    TF_RET_CHECK(request.has_computation()) << "computations may not be empty";
    TF_RET_CHECK(request.computation().has_host_program_shape())
        << "programe shape may not be empty";

    // Get the executors.
    TF_ASSIGN_OR_RETURN(auto executors, GetExecutors(execution_options,
                                                     arg->requests_size(), i));

    // Get the replicated arguments.
    TF_ASSIGN_OR_RETURN(auto replicated_arguments,
                        GetArguments(execution_options, request.arguments()));

    // Create an HloModuleConfig object for the computation, given the shape of
    // the program and the argument allocations. Here, we care only about the
    // shapes of the arguments, so, it is sufficient to use the arguments of
    // replica 0.
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModuleConfig> module_config,
        CreateModuleConfig(
            ProgramShape{request.computation().host_program_shape()},
            replicated_arguments.front(), request.execution_options()));
    VLOG(3)
        << "ExecuteGraphParallel created HloModuleConfig computation layout: "
        << module_config->entry_computation_layout().ToString();

    // Adds to the vectors to build and execute the computations after the loop.
    all_arguments.push_back(replicated_arguments);
    all_arguments.insert(all_arguments.end(), executors.size() - 1, {{}});
    module_protos.push_back(&request.computation());
    module_configs.push_back(std::move(module_config));
    computation_names.insert(computation_names.end(), executors.size(),
                             request.computation().name());
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
                                       /*device_allocator=*/nullptr));
  std::vector<Executable*> executable_ptrs;
  executable_ptrs.reserve(executables.size());
  for (const auto& executable : executables) {
    executable_ptrs.push_back(executable.get());
  }

  std::vector<HloSnapshot> snapshots;
  snapshots.resize(executable_ptrs.size());
  for (int i = 0; i < executable_ptrs.size(); i++) {
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
  if (executable_ptrs.size() == 1) {
    TF_ASSIGN_OR_RETURN(
        auto output,
        ExecuteAndRegisterResult(executable_ptrs[0], all_arguments[0],
                                 execute_backend_.get(), device_handles[0],
                                 computation_names[0], &profile));
    outputs.push_back(std::move(output));
  } else {
    TF_ASSIGN_OR_RETURN(
        outputs, ExecuteParallelAndRegisterResult(
                     executable_ptrs, all_arguments, execute_backend_.get(),
                     device_handles, computation_names, &profile));
  }

  for (const GlobalDataHandle& output : outputs) {
    ExecuteResponse response;
    *response.mutable_output() = output;
    *response.mutable_profile() = profile;
    *result->add_responses() = response;
  }

  for (int i = 0; i < executable_ptrs.size(); i++) {
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
  return Status::OK();
}

Status Service::GetDeviceHandles(const GetDeviceHandlesRequest* arg,
                                 GetDeviceHandlesResponse* result) {
  const int64 available_device_count = execute_backend_->device_count();
  const int64 replica_count = options_.number_of_replicas();
  if (replica_count <= 0) {
    return FailedPrecondition("Replica count must be a positive integer");
  }
  if (available_device_count < arg->device_count() * replica_count) {
    return ResourceExhausted(
        "Requested logical device count (%d) with replica count (%d) exceeds "
        "the number of available physical devices on the target (%d)",
        arg->device_count(), replica_count, available_device_count);
  }

  for (int64 i = 0; i < arg->device_count(); ++i) {
    DeviceHandle device_handle;
    device_handle.set_handle(i);
    device_handle.set_device_count(arg->device_count());
    *result->add_device_handles() = device_handle;
  }

  return Status::OK();
}

StatusOr<std::unique_ptr<Executable>> Service::BuildExecutable(
    const HloModuleProto& module_proto,
    std::unique_ptr<HloModuleConfig> module_config, Backend* backend,
    se::StreamExecutor* executor, se::DeviceMemoryAllocator* device_allocator) {
  VLOG(1) << StrFormat(
      "BuildExecutable on service %p with serialized module proto: %s", this,
      module_proto.name());

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      CreateModuleFromProto(module_proto, *module_config));
  DumpHloModuleIfEnabled(*module, kBeforeOptimizationsDumpName);

  TF_ASSIGN_OR_RETURN(
      module, backend->compiler()->RunHloPasses(std::move(module), executor,
                                                device_allocator));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                      backend->compiler()->RunBackend(
                          std::move(module), executor, device_allocator));

  const auto& debug_opts = module_config->debug_options();
  if (DumpingEnabledForHloModule(module_proto.name(), debug_opts) &&
      debug_opts.xla_dump_hlo_snapshots()) {
    auto hlo_proto = absl::make_unique<HloProto>();
    *hlo_proto->mutable_hlo_module() = module_proto;
    executable->set_hlo_proto(std::move(hlo_proto));
  }

  return std::move(executable);
}

Status Service::Compile(const CompileRequest* arg, CompileResponse* result) {
  VLOG(1) << "running compile request";
  if (!arg->has_computation()) {
    return InvalidArgument("computations may not be empty");
  }
  if (!arg->computation().has_host_program_shape()) {
    return InvalidArgument("programe shape may not be empty");
  }

  if (arg->execution_options().device_handles_size() > 1) {
    return InvalidArgument(
        "The compile request does not support multiple device handles.");
  }

  std::vector<Shape> argument_shapes;
  argument_shapes.reserve(arg->input_shape_with_layout_size());
  std::vector<const Shape*> argument_shape_ptrs;
  for (const ShapeProto& shape_proto : arg->input_shape_with_layout()) {
    argument_shapes.push_back(Shape(shape_proto));
    argument_shape_ptrs.push_back(&argument_shapes.back());
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> module_config,
      CreateModuleConfig(ProgramShape{arg->computation().host_program_shape()},
                         argument_shape_ptrs, &arg->execution_options()));
  VLOG(3) << "Compile created HloModuleConfig computation layout: "
          << module_config->entry_computation_layout().ToString();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      BuildExecutable(arg->computation(), std::move(module_config),
                      execute_backend_.get(),
                      execute_backend_->default_stream_executor(),
                      /*device_allocator=*/nullptr));

  *result->mutable_handle() = compilation_cache_.Insert(std::move(executable));

  VLOG(1) << "successfully completed 'compile' request";
  return Status::OK();
}

Status Service::Execute(const ExecuteRequest* arg, ExecuteResponse* result) {
  VLOG(1) << "running execute request";
  if (!arg->has_handle()) {
    return InvalidArgument("execution handle should not be empty");
  }
  TF_ASSIGN_OR_RETURN(auto executable,
                      compilation_cache_.LookUp(arg->handle()));

  TF_ASSIGN_OR_RETURN(auto replicas, Replicas(*execute_backend_,
                                              SingleComputationDeviceHandle()));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<const ShapedBuffer*>> replicated_arguments,
      ResolveAndValidateArguments(arg->arguments(), replicas));

  // Check that the replicated_arguments has the same shape and layout as the
  // module config used when creating the exectuable.
  const int64 num_module_args =
      executable->module_config().entry_computation_layout().parameter_count();
  if (num_module_args != arg->arguments_size()) {
    return InvalidArgument(
        "The executable expects %lld arguments, but sees %lld.",
        num_module_args, arg->arguments_size());
  }
  for (int64 i = 0; i < num_module_args; i++) {
    const Shape& shape_module =
        executable->module_config().entry_computation_layout().parameter_shape(
            i);
    const Shape& shape_arg = replicated_arguments.front()[i]->on_host_shape();
    if (!ShapeUtil::Equal(shape_module, shape_arg)) {
      return InvalidArgumentStrCat(
          "The executable exepcts the ", i, "th argument in shape ",
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
      *result->mutable_output(),
      ExecuteAndRegisterResult(executable.get(), replicated_arguments,
                               execute_backend_.get(),
                               SingleComputationDeviceHandle(),
                               "result of " + executable->module().name(),
                               result->mutable_profile()));

  if (executable->dumping_snapshot()) {
    TF_ASSIGN_OR_RETURN(
        const ShapedBuffer* result_buffer,
        allocation_tracker_.ResolveForReplica(result->output(), 0));
    TF_RETURN_IF_ERROR(RecordResult(*result_buffer, stream.get(),
                                    execute_backend_->transfer_manager(),
                                    &snapshot));
    DumpHloSnapshotIfEnabled(executable->module(), snapshot);
  }

  VLOG(1) << "successfully completed 'execute' request";
  return Status::OK();
}

Status Service::WaitForExecution(const WaitForExecutionRequest* arg,
                                 WaitForExecutionResponse* result) {
  TF_ASSIGN_OR_RETURN(const auto execution,
                      execution_tracker_.Resolve(arg->execution()));

  TF_RETURN_IF_ERROR(execution->BlockUntilDone());

  *result->mutable_output() = execution->result();
  *result->mutable_profile() = execution->profile();

  TF_RETURN_IF_ERROR(execution_tracker_.Unregister(arg->execution()));
  VLOG(1) << "successfully completed 'wait-for-execution' request";
  return Status::OK();
}

Status Service::TransferToClient(const TransferToClientRequest* arg,
                                 TransferToClientResponse* result) {
  TF_ASSIGN_OR_RETURN(const ShapedBuffer* shaped_buffer,
                      allocation_tracker_.ResolveForReplica(arg->data(), 0));

  Shape return_shape;
  if (arg->has_shape_with_layout()) {
    return_shape = Shape(arg->shape_with_layout());
    if (!LayoutUtil::HasLayout(return_shape)) {
      return InvalidArgument("shape_with_layout must have layout if present.");
    }
  } else {
    return_shape = Shape(shaped_buffer->on_host_shape());
  }

  TF_ASSIGN_OR_RETURN(auto stream, execute_backend_->BorrowStream(
                                       shaped_buffer->device_ordinal()));

  TF_ASSIGN_OR_RETURN(
      Literal result_literal,
      execute_backend_->transfer_manager()->TransferLiteralFromDevice(
          stream.get(), *shaped_buffer));

  if (LayoutUtil::LayoutsInShapesEqual(return_shape, result_literal.shape())) {
    *result->mutable_literal() = result_literal.ToProto();
  } else {
    *result->mutable_literal() =
        result_literal.Relayout(return_shape).ToProto();
  }
  return Status::OK();
}

Status Service::TransferToServer(const TransferToServerRequest* arg,
                                 TransferToServerResponse* result) {
  TF_ASSIGN_OR_RETURN(Literal literal,
                      Literal::CreateFromProto(arg->literal()));
  const Shape& shape = literal.shape();

  std::vector<se::StreamExecutor*> replicas;
  if (arg->has_device_handle()) {
    TF_ASSIGN_OR_RETURN(replicas,
                        Replicas(*execute_backend_, arg->device_handle()));
  } else {
    TF_ASSIGN_OR_RETURN(
        replicas, Replicas(*execute_backend_, SingleComputationDeviceHandle()));
  }

  // Allocate memory in each replica and transfer the data to all replicas.
  std::vector<ScopedShapedBuffer> replicated_buffers;
  for (se::StreamExecutor* executor : replicas) {
    TF_ASSIGN_OR_RETURN(
        ScopedShapedBuffer shaped_buffer,
        execute_backend_->transfer_manager()->AllocateScopedShapedBuffer(
            shape, execute_backend_->memory_allocator(),
            executor->device_ordinal()));
    TF_ASSIGN_OR_RETURN(auto stream, execute_backend_->BorrowStream(executor));
    TF_RETURN_IF_ERROR(
        execute_backend_->transfer_manager()->TransferLiteralToDevice(
            stream.get(), literal, shaped_buffer));
    replicated_buffers.emplace_back(std::move(shaped_buffer));
  }
  TF_ASSIGN_OR_RETURN(*result->mutable_data(),
                      allocation_tracker_.RegisterReplicatedBuffers(
                          std::move(replicated_buffers),
                          StrCat("TransferToServer literal of shape ",
                                 ShapeUtil::HumanString(shape))));

  return Status::OK();
}

Status Service::TransferToInfeed(const TransferToInfeedRequest* arg,
                                 TransferToInfeedResponse* result) {
  const int64 replica_count = options_.number_of_replicas();
  if (arg->replica_id() < 0 || arg->replica_id() >= replica_count) {
    return FailedPrecondition(
        "%s",
        StrCat("The replica_id=", arg->replica_id(),
               " on TransferToInfeedRequest not in range [0, replica_count=",
               replica_count, ")."));
  }

  se::StreamExecutor* executor;
  if (arg->has_device_handle()) {
    TF_ASSIGN_OR_RETURN(auto replicas,
                        Replicas(*execute_backend_, arg->device_handle()));
    executor = replicas[arg->replica_id()];
  } else {
    TF_ASSIGN_OR_RETURN(
        auto replicas,
        Replicas(*execute_backend_, SingleComputationDeviceHandle()));
    executor = replicas[arg->replica_id()];
  }

  TF_ASSIGN_OR_RETURN(Literal literal,
                      Literal::CreateFromProto(arg->literal()));
  return execute_backend_->transfer_manager()->TransferLiteralToInfeed(executor,
                                                                       literal);
}

Status Service::TransferFromOutfeed(const TransferFromOutfeedRequest* arg,
                                    TransferFromOutfeedResponse* result) {
  const int64 replica_count = options_.number_of_replicas();
  if (arg->replica_id() < 0 || arg->replica_id() >= replica_count) {
    return FailedPrecondition(
        "The replica_id=%d on TransferFromOutfeedRequest not in range [0, %d)",
        arg->replica_id(), replica_count);
  }

  se::StreamExecutor* executor;
  if (arg->has_device_handle()) {
    TF_ASSIGN_OR_RETURN(auto replicas,
                        Replicas(*execute_backend_, arg->device_handle()));
    executor = replicas[arg->replica_id()];
  } else {
    TF_ASSIGN_OR_RETURN(
        auto replicas,
        Replicas(*execute_backend_, SingleComputationDeviceHandle()));
    executor = replicas[arg->replica_id()];
  }

  auto literal = Literal::CreateFromShape(Shape(arg->shape_with_layout()));

  TF_RETURN_IF_ERROR(
      execute_backend_->transfer_manager()->TransferLiteralFromOutfeed(
          executor, Shape(arg->shape_with_layout()), &literal));
  *result->mutable_literal() = literal.ToProto();
  return Status::OK();
}

Status Service::ResetDevice(const ResetDeviceRequest* arg,
                            ResetDeviceResponse* result) {
  return execute_backend_->ResetDevices();
}

Status Service::ComputeConstantGraph(const ComputeConstantGraphRequest* arg,
                                     ComputeConstantResponse* result) {
  if (!arg->has_computation()) {
    return InvalidArgument("computations may not be empty");
  }
  if (!arg->computation().has_host_program_shape()) {
    return InvalidArgument("program shape may not be empty");
  }
  if (arg->computation().host_program_shape().parameters_size() != 0) {
    return InvalidArgument(
        "constant computation may not depend on any parameters.");
  }

  ProgramShape program_shape(arg->computation().host_program_shape());
  TF_DCHECK_OK(ShapeUtil::ValidateShape(program_shape.result()));
  absl::optional<Layout> output_layout;
  if (arg->has_output_layout()) {
    output_layout = Layout::CreateFromProto(arg->output_layout());
    TF_RETURN_IF_ERROR(LayoutUtil::ValidateLayoutForShape(
        *output_layout, program_shape.result()));
  }

  HloModuleConfig config(program_shape);

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      CreateModuleFromProto(arg->computation(), config));

  TF_ASSIGN_OR_RETURN(DynamicDimensionInference dynamic_dimension_inference,
                      DynamicDimensionInference::Run(module.get()));

  HloEvaluator evaluator;
  evaluator.set_dynamic_dimension_inference(&dynamic_dimension_inference);
  TF_ASSIGN_OR_RETURN(auto result_literal, evaluator.Evaluate(*module, {}));

  // Since the result layout is non-effective to the Evaluator results, explicit
  // relayout here.
  //
  // TODO(b/77824332): Make HloEvaluator take care of the re-layout.
  if (output_layout.has_value()) {
    result_literal = result_literal.Relayout(*output_layout);
  }
  *result->mutable_literal() = result_literal.ToProto();

  return Status::OK();
}

Status Service::GetShape(const GetShapeRequest* arg, GetShapeResponse* result) {
  TF_ASSIGN_OR_RETURN(const ShapedBuffer* buffer,
                      allocation_tracker_.ResolveForReplica(arg->data(), 0));
  *result->mutable_shape() = buffer->on_host_shape().ToProto();
  return Status::OK();
}

Status Service::GetComputationGraphStats(
    const ComputationGraphStatsRequest* arg, ComputationStatsResponse* result) {
  if (!arg->has_computation()) {
    return InvalidArgument("Computations may not be empty.");
  }
  if (!arg->computation().has_host_program_shape()) {
    return InvalidArgument("Program shape may not be empty.");
  }

  HloModuleConfig config(ProgramShape{arg->computation().host_program_shape()});
  config.set_debug_options(arg->debug_options());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      CreateModuleFromProto(arg->computation(), config));
  DumpHloModuleIfEnabled(*module, kBeforeOptimizationsDumpName);

  // Run HLO analysis to get the computation statistics.
  HloCostAnalysis analysis(
      execute_backend_->compiler()->ShapeSizeBytesFunction());

  TF_RETURN_IF_ERROR(module->entry_computation()->Accept(&analysis));

  ComputationStats stats;
  stats.set_flop_count(analysis.flop_count());
  stats.set_transcendental_count(analysis.transcendental_count());
  *result->mutable_stats() = stats;
  return Status::OK();
}

DeviceHandle Service::SingleComputationDeviceHandle() const {
  DeviceHandle device_handle;
  device_handle.set_handle(0);
  device_handle.set_device_count(1);
  return device_handle;
}

StatusOr<std::vector<se::StreamExecutor*>> Service::Replicas(
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

}  // namespace xla
