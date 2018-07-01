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

#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
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
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

using ::tensorflow::strings::Printf;
using ::tensorflow::strings::StrCat;
using ::xla::source_map_util::InvalidParameterArgument;

namespace xla {

namespace {

// Records the arguments used to invoke a computation in an HloSnapshot proto.
Status RecordArguments(
    const tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    se::Stream* stream, TransferManager* transfer_manager,
    HloSnapshot* module) {
  module->clear_arguments();
  for (const ShapedBuffer* argument : arguments) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<Literal> literal,
        transfer_manager->TransferLiteralFromDevice(stream, *argument));
    *module->add_arguments() = literal->ToProto();
  }
  return Status::OK();
}

// Records the result of a computation in a HloSnapshot proto.
Status RecordResult(const ShapedBuffer& result, se::Stream* stream,
                    TransferManager* transfer_manager, HloSnapshot* module) {
  module->clear_result();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Literal> literal,
      transfer_manager->TransferLiteralFromDevice(stream, result));
  *module->mutable_result() = literal->ToProto();
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
    LOG(INFO) << Printf(
        "XLA service %p executing computations on platform %s. Devices:", this,
        execute_backend_->platform()->Name().c_str());
    for (int i = 0; i < execute_backend_->device_count(); ++i) {
      if (execute_backend_->device_ordinal_supported(i)) {
        se::StreamExecutor* executor =
            execute_backend_->stream_executor(i).ValueOrDie();
        const auto& description = executor->GetDeviceDescription();
        LOG(INFO) << Printf("  StreamExecutor device (%d): %s, %s", i,
                            description.name().c_str(),
                            description.platform_version().c_str());
      } else {
        LOG(INFO) << Printf("  StreamExecutor device (%d) not supported", i);
      }
    }
  } else {
    VLOG(1) << "XLA compile-only service constructed";
  }
}

Status Service::CreateChannelHandle(const CreateChannelHandleRequest* arg,
                                    CreateChannelHandleResponse* result) {
  *result->mutable_channel() = channel_tracker_.NewChannel();
  return Status::OK();
}

Status Service::Unregister(const UnregisterRequest* arg,
                           UnregisterResponse* result) {
  return allocation_tracker_.Unregister(arg->data());
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
        ShapeUtil::HumanStringWithLayout(client_shape).c_str(),
        ShapeUtil::HumanString(result_shape).c_str());
  }
  return Status::OK();
}

StatusOr<std::vector<std::vector<const ShapedBuffer*>>>
Service::ResolveAndValidateArguments(
    tensorflow::gtl::ArraySlice<const GlobalDataHandle*> arguments,
    tensorflow::gtl::ArraySlice<se::StreamExecutor*> stream_executors) {
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
            i, shaped_buffer->platform()->Name().c_str(),
            shaped_buffer->device_ordinal(),
            execute_backend_->device_name(replica_device_ordinal).c_str());
      }
      replicated_arguments[replica].push_back(shaped_buffer);
    }
  }
  return replicated_arguments;
}

StatusOr<std::unique_ptr<HloModuleConfig>> Service::CreateModuleConfig(
    const ProgramShape& program_shape,
    tensorflow::gtl::ArraySlice<const Shape*> argument_shapes,
    const ExecutionOptions* execution_options) {
  auto config = MakeUnique<HloModuleConfig>(program_shape);
  ComputationLayout* computation_layout =
      config->mutable_entry_computation_layout();
  if (program_shape.parameters_size() != argument_shapes.size()) {
    return InvalidArgument("computation takes %d parameters, but %zu given",
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
          i, ShapeUtil::HumanString(program_shape.parameters(i)).c_str(),
          ShapeUtil::HumanString(*argument_shapes[i]).c_str());
    }
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
            *argument_shapes[i]));
  }
  if (execution_options != nullptr &&
      execution_options->has_shape_with_output_layout()) {
    const auto& shape_with_output_layout =
        execution_options->shape_with_output_layout();
    TF_RETURN_IF_ERROR(
        ValidateResultShape(shape_with_output_layout, program_shape.result()));
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_result_layout()->CopyLayoutFromShape(
            shape_with_output_layout));
  } else {
    // If the result layout is not set, then choose the default.
    computation_layout->mutable_result_layout()->SetToDefaultLayout();
  }

  config->set_replica_count(options_.number_of_replicas());
  if (execution_options != nullptr) {
    config->set_seed(execution_options->seed());
    config->set_debug_options(execution_options->debug_options());
    config->set_resource_update_count(
        execution_options->resource_update_count());
  } else {
    config->set_debug_options(legacy_flags::GetDebugOptionsFromFlags());
  }

  if (execute_backend_ != nullptr &&
      execute_backend_->eigen_intra_op_thread_pool() != nullptr) {
    config->set_intra_op_parallelism_threads(
        execute_backend_->eigen_intra_op_thread_pool()->NumThreads());
  }
  return std::move(config);
}

StatusOr<std::unique_ptr<HloModuleConfig>> Service::CreateModuleConfig(
    const ProgramShape& program_shape,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    const ExecutionOptions& execution_options) {
  std::vector<const Shape*> argument_shapes;
  for (const auto* arg : arguments) {
    argument_shapes.push_back(&arg->on_host_shape());
  }
  return CreateModuleConfig(program_shape, argument_shapes, &execution_options);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> Service::BuildExecutables(
    const std::vector<const HloModuleProto*>& module_protos,
    std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
    Backend* backend, std::vector<std::vector<se::StreamExecutor*>> executors,
    DeviceMemoryAllocator* device_allocator) {
  VLOG(1) << Printf("BuildExecutable on service %p", this);

  // Dump computation proto state if flag is set.
  std::vector<std::unique_ptr<HloSnapshot>> hlo_snapshots;
  for (int64 i = 0; i < module_protos.size(); ++i) {
    const string& directory_path =
        module_configs[i]->debug_options().xla_dump_computations_to();
    const string& execution_directory_path =
        module_configs[i]->debug_options().xla_dump_executions_to();
    if (directory_path.empty() && execution_directory_path.empty()) {
      continue;
    }
    auto hlo_snapshot = MakeUnique<HloSnapshot>();
    *hlo_snapshot->mutable_hlo()->mutable_hlo_module() = *module_protos[i];
    if (!directory_path.empty()) {
      string filename =
          Printf("computation_%lld__%s", module_protos[i]->id(),
                 module_protos[i]->entry_computation_name().c_str());
      TF_RETURN_IF_ERROR(
          Executable::DumpToDirectory(directory_path, filename, *hlo_snapshot));
    }
    hlo_snapshots.push_back(std::move(hlo_snapshot));
  }

  VLOG(1) << "Computations:";
  for (const HloModuleProto* proto : module_protos) {
    VLOG(1) << proto->name();
  }

  CHECK_EQ(module_protos.size(), module_configs.size());
  std::vector<std::unique_ptr<HloModule>> modules;
  for (int64 i = 0; i < module_protos.size(); ++i) {
    const HloModuleProto* proto = module_protos[i];
    const HloModuleConfig& config = *module_configs[i];
    TF_ASSIGN_OR_RETURN(auto module,
                        HloModule::CreateFromProto(*proto, config));
    modules.push_back(std::move(module));
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<Executable>> executables,
      backend->compiler()->Compile(std::move(modules), std::move(executors),
                                   device_allocator));

  for (size_t i = 0; i < module_protos.size(); ++i) {
    if (!module_configs[i]->debug_options().xla_dump_executions_to().empty()) {
      executables[i]->set_hlo_snapshot(std::move(hlo_snapshots[i]));
    }
  }

  return std::move(executables);
}

StatusOr<std::vector<GlobalDataHandle>>
Service::ExecuteParallelAndRegisterResult(
    tensorflow::gtl::ArraySlice<Executable*> executables,
    tensorflow::gtl::ArraySlice<std::vector<std::vector<const ShapedBuffer*>>>
        arguments,
    Backend* backend, tensorflow::gtl::ArraySlice<DeviceHandle> device_handles,
    tensorflow::gtl::ArraySlice<string> result_tags,
    ExecutionProfile* profile) {
  // Streams where the computation are launched, so we can wait on the streams
  // to complete.
  std::vector<Pool<se::Stream>::SmartPtr> streams;
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
      TF_ASSIGN_OR_RETURN(Pool<se::Stream>::SmartPtr stream,
                          backend->BorrowStream(replicas[replica]));
      streams.push_back(std::move(stream));

      if (replica == 0 && profile != nullptr) {
        timers.emplace_back(new se::Timer(streams.back()->parent()));
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
      ServiceExecutableRunOptions run_options(options,
                                              backend->StreamBorrower());

      // Asynchronously launch the computation.
      TF_ASSIGN_OR_RETURN(ScopedShapedBuffer result,
                          executables[i]->ExecuteAsyncOnStream(
                              &run_options, arguments[i][replica]));

      if (replica == 0 && profile != nullptr) {
        streams.back()->ThenStopTimer(timers.back().get());
      }

      result_buffers.emplace_back(std::move(result));
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
      return InternalError("failed to complete execution for stream %lld: %s",
                           i, block_status.error_message().c_str());
    }
  }

  // For every stream that had profiling enabled, obtain and debug-dump the HLO
  // profile.
  for (auto& index_to_profiled_stream : index_to_profiled_streams) {
    int64 device = index_to_profiled_stream.first;
    se::Stream* stream = index_to_profiled_stream.second;
    Executable* executable = executables[device];
    const HloModule& module = executable->module();
    HloExecutionProfile hlo_profile(&executable->hlo_profile_printer_data(),
                                    &executable->hlo_profile_index_map());
    TF_RETURN_IF_ERROR(
        executable->PopulateExecutionProfile(&hlo_profile, stream));
    XLA_LOG_LINES(
        tensorflow::INFO,
        hlo_profile.ToString(streams[0]->parent()->GetDeviceDescription()));
    hlo_graph_dumper::MaybeDumpHloModule(module, "Service::Execute",
                                         &hlo_profile);
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

    // Merge in run-time profile information from execution_profile on the
    // zeroth device.
    profile->MergeFrom(executables[0]->execution_profile());

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
    const tensorflow::gtl::ArraySlice<std::vector<const ShapedBuffer*>>
        arguments,
    Backend* backend, const string& result_tag, ExecutionProfile* profile) {
  // Set up streams.
  std::vector<Pool<se::Stream>::SmartPtr> streams;

  TF_ASSIGN_OR_RETURN(auto replicas,
                      Replicas(*backend, SingleComputationDeviceHandle()));
  TF_RET_CHECK(!replicas.empty());
  for (se::StreamExecutor* executor : replicas) {
    TF_ASSIGN_OR_RETURN(Pool<se::Stream>::SmartPtr stream,
                        backend->BorrowStream(executor));
    streams.push_back(std::move(stream));
  }

  TF_ASSIGN_OR_RETURN(DeviceAssignment device_assignment,
                      backend->computation_placer()->AssignDevices(
                          options_.number_of_replicas(),
                          /*computation_count=*/1));

  // Set up run options.
  std::vector<ServiceExecutableRunOptions> run_options;
  for (const Pool<se::Stream>::SmartPtr& stream : streams) {
    ExecutableRunOptions options;
    options.set_stream(stream.get());
    options.set_device_ordinal(stream->parent()->device_ordinal());
    options.set_allocator(backend->memory_allocator());
    options.set_intra_op_thread_pool(
        backend->eigen_intra_op_thread_pool_device());
    options.set_device_assignment(&device_assignment);
    run_options.emplace_back(
        options, backend->StreamBorrower(),
        /*xla_intra_op_thread_pool=*/backend->eigen_intra_op_thread_pool());
  }

  if (options_.number_of_replicas() == 1) {
    TF_ASSIGN_OR_RETURN(
        auto result, executable->ExecuteOnStreamWrapper(&run_options[0],
                                                        profile, arguments[0]));
    return allocation_tracker_.Register(std::move(result), result_tag);
  }

  // TODO(b/69985541): Support profiling also on this path.

  std::vector<tensorflow::gtl::ArraySlice<const ShapedBuffer*>>
      replicated_arguments;
  for (const auto& arg : arguments) {
    replicated_arguments.emplace_back(arg);
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
        "Found %lld parallel requests, with request %lld containing %d device "
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
    tensorflow::gtl::ArraySlice<const GlobalDataHandle*> arguments) {
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
    TF_RET_CHECK(request.computation().has_program_shape())
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
        CreateModuleConfig(request.computation().program_shape(),
                           replicated_arguments.front(),
                           request.execution_options()));
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

  for (int i = 0; i < executable_ptrs.size(); i++) {
    if (executable_ptrs[i]->dumping_snapshot()) {
      TF_ASSIGN_OR_RETURN(auto stream,
                          execute_backend_->BorrowStream(
                              all_executors[i][0]->device_ordinal()));
      TF_RETURN_IF_ERROR(RecordArguments(all_arguments[i].front(), stream.get(),
                                         execute_backend_->transfer_manager(),
                                         executable_ptrs[i]->hlo_snapshot()));
    }
  }

  // Execute the generated executables in parallel and return the device
  // handles for each computation's output.
  ExecutionProfile profile;
  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDataHandle> outputs,
      ExecuteParallelAndRegisterResult(executable_ptrs, all_arguments,
                                       execute_backend_.get(), device_handles,
                                       computation_names, &profile));
  for (const GlobalDataHandle& output : outputs) {
    ExecuteResponse response;
    *response.mutable_output() = output;
    *response.mutable_profile() = profile;
    *result->add_responses() = response;
  }

  for (int i = 0; i < executable_ptrs.size(); i++) {
    if (executable_ptrs[i]->dumping_snapshot()) {
      TF_ASSIGN_OR_RETURN(const ShapedBuffer* result_buffer,
                          allocation_tracker_.ResolveForReplica(outputs[i], 0));
      TF_ASSIGN_OR_RETURN(auto stream,
                          execute_backend_->BorrowStream(all_executors[i][0]));
      TF_RETURN_IF_ERROR(RecordResult(*result_buffer, stream.get(),
                                      execute_backend_->transfer_manager(),
                                      executable_ptrs[i]->hlo_snapshot()));
      // Dump out the ith snapshot.
      TF_RETURN_IF_ERROR(executable_ptrs[i]->DumpHloSnapshot());
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
        "Requested device count (%lld) exceeds the number of available devices "
        "on the target (%lld)",
        arg->device_count(), available_device_count);
  }

  for (int64 i = 0; i < arg->device_count(); ++i) {
    DeviceHandle device_handle;
    device_handle.set_handle(i);
    device_handle.set_device_count(arg->device_count());
    *result->add_device_handles() = device_handle;
  }

  return Status::OK();
}

Status Service::ExecuteOneToN(const ExecuteGraphRequest* arg,
                              ExecuteResponse* result) {
  ExecuteGraphParallelRequest parallel_arg;
  *parallel_arg.add_requests() = *arg;
  ExecuteParallelResponse parallel_result;
  TF_RETURN_IF_ERROR(ExecuteGraphParallel(&parallel_arg, &parallel_result));
  return PickParallelResponse(parallel_result, result);
}

Status Service::PickParallelResponse(
    const ExecuteParallelResponse& parallel_result, ExecuteResponse* result) {
  // The "result device" selection is a bit hacky, but better than assuming it
  // is device 0. We have b/76035356 for restructuring the client API to clean
  // up the current asymmetries and support more functionalities.
  for (int64 i = 0; i < parallel_result.responses_size(); ++i) {
    TF_ASSIGN_OR_RETURN(const ShapedBuffer* buffer,
                        allocation_tracker_.ResolveForReplica(
                            parallel_result.responses(i).output(), 0));
    const Shape& shape = buffer->on_host_shape();
    if (!ShapeUtil::IsEmptyTuple(shape)) {
      *result = parallel_result.responses(i);
      VLOG(3) << "Fetching result from device " << i << ": "
              << ShapeUtil::HumanString(shape);
      return Status::OK();
    }
  }
  TF_RET_CHECK(parallel_result.responses_size() > 0);
  *result = parallel_result.responses(0);
  VLOG(1) << "Defaulting to device 0 result";
  return Status::OK();
}

StatusOr<std::unique_ptr<Executable>> Service::BuildExecutable(
    const HloModuleProto& module_proto,
    std::unique_ptr<HloModuleConfig> module_config, Backend* backend,
    se::StreamExecutor* executor, DeviceMemoryAllocator* device_allocator) {
  VLOG(1) << Printf(
      "BuildExecutable on service %p with serialized module proto: %s", this,
      module_proto.name().c_str());

  // Dump computation proto state if flag is set.
  auto hlo_snapshot = MakeUnique<HloSnapshot>();
  const string& directory_path =
      module_config->debug_options().xla_dump_computations_to();
  const string& execution_directory_path =
      module_config->debug_options().xla_dump_executions_to();
  if (!directory_path.empty() || !execution_directory_path.empty()) {
    *hlo_snapshot->mutable_hlo()->mutable_hlo_module() = module_proto;
    if (!directory_path.empty()) {
      string filename = Printf("computation_%lld__%s", module_proto.id(),
                               module_proto.entry_computation_name().c_str());
      TF_RETURN_IF_ERROR(
          Executable::DumpToDirectory(directory_path, filename, *hlo_snapshot));
    }
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      HloModule::CreateFromProto(module_proto, *module_config));

  TF_RETURN_IF_ERROR(MaybeDumpHloModule(*module));

  TF_ASSIGN_OR_RETURN(
      module, backend->compiler()->RunHloPasses(std::move(module), executor,
                                                device_allocator));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                      backend->compiler()->RunBackend(
                          std::move(module), executor, device_allocator));

  if (!execution_directory_path.empty()) {
    executable->set_hlo_snapshot(std::move(hlo_snapshot));
  }

  return std::move(executable);
}

Status Service::ExecuteGraph(const ExecuteGraphRequest* arg,
                             ExecuteResponse* result) {
  VLOG(1) << "running execute-graph request";

  if (!arg->has_computation()) {
    return InvalidArgument("computations may not be empty");
  }
  if (!arg->computation().has_program_shape()) {
    return InvalidArgument("programe shape may not be empty");
  }

  // If we received multiple device handles, we must partition the module.
  if (arg->execution_options().device_handles_size() > 1) {
    return ExecuteOneToN(arg, result);
  }

  TF_ASSIGN_OR_RETURN(auto replicas, Replicas(*execute_backend_,
                                              SingleComputationDeviceHandle()));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<const ShapedBuffer*>> replicated_arguments,
      ResolveAndValidateArguments(arg->arguments(), replicas));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModuleConfig> module_config,
                      CreateModuleConfig(arg->computation().program_shape(),
                                         replicated_arguments.front(),
                                         arg->execution_options()));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      BuildExecutable(arg->computation(), std::move(module_config),
                      execute_backend_.get(),
                      execute_backend_->default_stream_executor(),
                      /*device_allocator=*/nullptr));

  TF_ASSIGN_OR_RETURN(auto stream,
                      execute_backend_->BorrowStream(
                          execute_backend_->default_stream_executor()));
  if (executable->dumping_snapshot()) {
    executable->hlo_snapshot()->set_execution_platform(
        execute_backend_->platform()->Name());
    TF_RETURN_IF_ERROR(RecordArguments(
        replicated_arguments.front(), stream.get(),
        execute_backend_->transfer_manager(), executable->hlo_snapshot()));
  }

  TF_ASSIGN_OR_RETURN(
      *result->mutable_output(),
      ExecuteAndRegisterResult(
          executable.get(), replicated_arguments, execute_backend_.get(),
          "result of " + arg->computation().name(), result->mutable_profile()));

  if (executable->dumping_snapshot()) {
    TF_ASSIGN_OR_RETURN(
        const ShapedBuffer* result_buffer,
        allocation_tracker_.ResolveForReplica(result->output(), 0));
    TF_RETURN_IF_ERROR(RecordResult(*result_buffer, stream.get(),
                                    execute_backend_->transfer_manager(),
                                    executable->hlo_snapshot()));
    TF_RETURN_IF_ERROR(executable->DumpHloSnapshot());
  }

  VLOG(1) << "successfully completed 'execute-graph' request";
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

  const Shape* return_shape;
  if (arg->has_shape_with_layout()) {
    if (!LayoutUtil::HasLayout(arg->shape_with_layout())) {
      return InvalidArgument("shape_with_layout must have layout if present.");
    }
    return_shape = &arg->shape_with_layout();
  } else {
    return_shape = &shaped_buffer->on_host_shape();
  }

  TF_ASSIGN_OR_RETURN(auto stream, execute_backend_->BorrowStream(
                                       shaped_buffer->device_ordinal()));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Literal> result_literal,
      execute_backend_->transfer_manager()->TransferLiteralFromDevice(
          stream.get(), *shaped_buffer));

  if (LayoutUtil::LayoutsInShapesEqual(*return_shape,
                                       result_literal->shape())) {
    *result->mutable_literal() = result_literal->ToProto();
  } else {
    *result->mutable_literal() =
        result_literal->Relayout(*return_shape)->ToProto();
  }
  return Status::OK();
}

namespace {

// Creates a clone of the given shaped buffer with the given device ordinal. The
// shape and DeviceMemoryBase values of the clone are identical to the original.
std::unique_ptr<ShapedBuffer> CloneShapedBufferOnDevice(
    const ShapedBuffer& shaped_buffer, int device_ordinal) {
  auto clone = MakeUnique<ShapedBuffer>(
      shaped_buffer.on_host_shape(), shaped_buffer.on_device_shape(),
      shaped_buffer.platform(), device_ordinal);
  clone->buffers() = shaped_buffer.buffers();
  return clone;
}

}  // namespace

Status Service::TransferToServer(const TransferToServerRequest* arg,
                                 TransferToServerResponse* result) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Literal> literal,
                      Literal::CreateFromProto(arg->literal()));
  const Shape& shape = literal->shape();

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
            stream.get(), *literal, shaped_buffer));
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
               replica_count, ").")
            .c_str());
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

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Literal> literal,
                      Literal::CreateFromProto(arg->literal()));
  return execute_backend_->transfer_manager()->TransferLiteralToInfeed(
      executor, *literal);
}

Status Service::TransferFromOutfeed(const TransferFromOutfeedRequest* arg,
                                    TransferFromOutfeedResponse* result) {
  const int64 replica_count = options_.number_of_replicas();
  if (arg->replica_id() < 0 || arg->replica_id() >= replica_count) {
    return FailedPrecondition(
        "The replica_id=%lld on TransferFromOutfeedRequest not in range [0, "
        "%lld)",
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

  Literal literal;
  TF_RETURN_IF_ERROR(
      execute_backend_->transfer_manager()->TransferLiteralFromOutfeed(
          executor, arg->shape_with_layout(), &literal));
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
  if (!arg->computation().has_program_shape()) {
    return InvalidArgument("program shape may not be empty");
  }
  if (arg->computation().program_shape().parameters_size() != 0) {
    return InvalidArgument(
        "constant computation may not depend on any parameters.");
  }

  ProgramShape program_shape = arg->computation().program_shape();
  TF_DCHECK_OK(ShapeUtil::ValidateShape(program_shape.result()));
  if (arg->has_output_layout()) {
    TF_RETURN_IF_ERROR(LayoutUtil::ValidateLayoutForShape(
        arg->output_layout(), program_shape.result()));
  }

  HloModuleConfig config(program_shape);

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      HloModule::CreateFromProto(arg->computation(), config));

  HloEvaluator evaluator;
  TF_ASSIGN_OR_RETURN(auto result_literal,
                      evaluator.Evaluate<std::unique_ptr<Literal>>(
                          *module, /*arg_literals=*/{}));

  // Since the result layout is non-effective to the Evaluator results, explicit
  // relayout here.
  //
  // TODO(b/77824332): Make HloEvaluator take care of the re-layout.
  if (arg->has_output_layout()) {
    result_literal = result_literal->Relayout(arg->output_layout());
  }
  *result->mutable_literal() = result_literal->ToProto();

  return Status::OK();
}

Status Service::GetShape(const GetShapeRequest* arg, GetShapeResponse* result) {
  TF_ASSIGN_OR_RETURN(const ShapedBuffer* buffer,
                      allocation_tracker_.ResolveForReplica(arg->data(), 0));
  *result->mutable_shape() = buffer->on_host_shape();
  return Status::OK();
}

Status Service::GetComputationGraphStats(
    const ComputationGraphStatsRequest* arg, ComputationStatsResponse* result) {
  if (!arg->has_computation()) {
    return InvalidArgument("Computations may not be empty.");
  }
  if (!arg->computation().has_program_shape()) {
    return InvalidArgument("Program shape may not be empty.");
  }

  HloModuleConfig config(arg->computation().program_shape());
  config.set_debug_options(arg->debug_options());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      HloModule::CreateFromProto(arg->computation(), config));

  hlo_graph_dumper::MaybeDumpHloModule(*module,
                                       "computation statistics subject");

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

Status Service::MaybeDumpHloModule(const HloModule& module) const {
  const string xla_dump_unoptimized_hlo_proto_to =
      module.config().debug_options().xla_dump_unoptimized_hlo_proto_to();
  if (xla_dump_unoptimized_hlo_proto_to.empty()) {
    return Status::OK();
  }
  HloProto proto = MakeHloProto(module);
  return protobuf_util::DumpProtoToDirectory(
      proto, xla_dump_unoptimized_hlo_proto_to, module.name());
}

}  // namespace xla
