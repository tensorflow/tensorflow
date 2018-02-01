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
#include "tensorflow/compiler/xla/service/session.pb.h"
#include "tensorflow/compiler/xla/service/source_map_util.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

using ::tensorflow::strings::Printf;
using ::tensorflow::strings::StrCat;
using ::xla::source_map_util::InvalidParameterArgument;

namespace xla {

namespace {

// Records the arguments used to invoke a computation in a SessionModule
// proto.
tensorflow::Status RecordArguments(
    const tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    se::StreamExecutor* executor, TransferManager* transfer_manager,
    SessionModule* module) {
  module->clear_arguments();
  for (const ShapedBuffer* argument : arguments) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<Literal> literal,
        transfer_manager->TransferLiteralFromDevice(executor, *argument));
    *module->add_arguments() = literal->ToProto();
  }
  return tensorflow::Status::OK();
}

// Records the result of a computation in a SessionModule proto.
tensorflow::Status RecordResult(const ShapedBuffer& result,
                                se::StreamExecutor* executor,
                                TransferManager* transfer_manager,
                                SessionModule* module) {
  module->clear_result();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Literal> literal,
      transfer_manager->TransferLiteralFromDevice(executor, result));
  *module->mutable_result() = literal->ToProto();
  return tensorflow::Status::OK();
}

}  // namespace

ServiceOptions& ServiceOptions::set_platform(
    perftools::gputools::Platform* platform) {
  platform_ = platform;
  return *this;
}

perftools::gputools::Platform* ServiceOptions::platform() const {
  return platform_;
}

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
    perftools::gputools::Platform* platform) {
  ServiceOptions default_options;
  default_options.set_platform(platform);
  return NewService(default_options);
}

/* static */ StatusOr<std::unique_ptr<Service>> Service::NewService(
    const ServiceOptions& options) {
  perftools::gputools::Platform* platform = options.platform();
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

tensorflow::Status Service::Computation(const ComputationRequest* arg,
                                        ComputationResponse* result) {
  if (arg->name().empty()) {
    return InvalidArgument("computation request needs a name");
  }

  *result->mutable_computation() =
      computation_tracker_.NewComputation(arg->name());
  VLOG(1) << Printf("Created new computation %s on service %p, name %s",
                    result->computation().ShortDebugString().c_str(), this,
                    arg->name().c_str());
  return tensorflow::Status::OK();
}

tensorflow::Status Service::CreateChannelHandle(
    const CreateChannelHandleRequest* arg,
    CreateChannelHandleResponse* result) {
  *result->mutable_channel() = channel_tracker_.NewChannel();
  return tensorflow::Status::OK();
}

tensorflow::Status Service::Unregister(const UnregisterRequest* arg,
                                       UnregisterResponse* result) {
  return allocation_tracker_.Unregister(arg->data());
}

// Deconstructs a previously-allocated global handle.
tensorflow::Status Service::DeconstructTuple(const DeconstructTupleRequest* arg,
                                             DeconstructTupleResponse* result) {
  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDataHandle> elements,
      allocation_tracker_.DeconstructTuple(arg->tuple_handle()));

  for (auto& element : elements) {
    *result->add_element_handles() = element;
  }
  return tensorflow::Status::OK();
}

tensorflow::Status Service::ValidateResultShapeWithLayout(
    const Shape& shape_with_layout, const Shape& result_shape) const {
  if (!ShapeUtil::Compatible(shape_with_layout, result_shape)) {
    return InvalidArgument(
        "Shape used to set computation result layout %s is not compatible "
        "with result shape %s",
        ShapeUtil::HumanStringWithLayout(shape_with_layout).c_str(),
        ShapeUtil::HumanString(result_shape).c_str());
  }
  if (!LayoutUtil::HasLayout(shape_with_layout)) {
    return InvalidArgument(
        "Shape used to set computation result layout %s does not have layout",
        ShapeUtil::HumanStringWithLayout(shape_with_layout).c_str());
  }
  return ShapeUtil::ValidateShape(shape_with_layout);
}

StatusOr<std::vector<const ShapedBuffer*>> Service::ResolveAndValidateArguments(
    tensorflow::gtl::ArraySlice<const GlobalDataHandle*> arguments,
    int device_ordinal) {
  std::vector<const ShapedBuffer*> shaped_buffers;
  for (size_t i = 0; i < arguments.size(); ++i) {
    auto buffer_status = allocation_tracker_.Resolve(*arguments[i]);
    if (!buffer_status.ok()) {
      return Status(buffer_status.status().code(),
                    StrCat(buffer_status.status().error_message(), ", ",
                           "failed to resolve allocation for parameter ", i));
    }
    const ShapedBuffer* shaped_buffer = buffer_status.ValueOrDie();

    // Verify allocation is same platform and device as the execution.
    if (shaped_buffer->platform() != execute_backend_->platform() ||
        shaped_buffer->device_ordinal() != device_ordinal) {
      return InvalidArgument(
          "argument %lu is on device %s:%d but computation will be executed "
          "on device %s",
          i, shaped_buffer->platform()->Name().c_str(),
          shaped_buffer->device_ordinal(),
          execute_backend_->device_name(device_ordinal).c_str());
    }

    shaped_buffers.push_back(shaped_buffer);
  }
  return shaped_buffers;
}

StatusOr<std::unique_ptr<HloModuleConfig>> Service::CreateModuleConfig(
    const ProgramShape& program_shape,
    tensorflow::gtl::ArraySlice<const Shape*> argument_shapes,
    const ExecutionOptions* execution_options,
    const UserComputation& user_computation) {
  auto config = MakeUnique<HloModuleConfig>(program_shape);
  auto* computation_layout = config->mutable_entry_computation_layout();

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
      return InvalidParameterArgument(
          *user_computation.ParameterMetadata(i).value(),
          "Argument does not match shape of computation parameter %d: want %s, "
          "got %s",
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
    TF_RETURN_IF_ERROR(ValidateResultShapeWithLayout(shape_with_output_layout,
                                                     program_shape.result()));
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_result_layout()->CopyLayoutFromShape(
            shape_with_output_layout));
  } else {
    computation_layout->mutable_result_layout()->Clear();
  }

  config->set_replica_count(options_.number_of_replicas());
  if (execution_options != nullptr) {
    config->set_seed(execution_options->seed());
    config->set_debug_options(execution_options->debug_options());
    config->enable_hlo_profiling(
        execution_options->debug_options().xla_hlo_profile());
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
    const ExecutionOptions& execution_options,
    const UserComputation& user_computation) {
  std::vector<const Shape*> argument_shapes;
  for (const auto* arg : arguments) {
    argument_shapes.push_back(&arg->on_host_shape());
  }
  return CreateModuleConfig(program_shape, argument_shapes, &execution_options,
                            user_computation);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> Service::BuildExecutables(
    std::vector<VersionedComputationHandle> versioned_handles,
    std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
    Backend* backend,
    std::vector<std::vector<perftools::gputools::StreamExecutor*>> executors,
    DeviceMemoryAllocator* device_allocator) {
  VLOG(1) << Printf("BuildExecutable on service %p", this);

  // Dump computation proto state if flag is set.
  std::vector<std::unique_ptr<SessionModule>> session_modules;
  for (int64 i = 0; i < versioned_handles.size(); ++i) {
    const string& directory_path =
        module_configs[i]->debug_options().xla_dump_computations_to();
    const string& other_directory_path =
        module_configs[i]->debug_options().xla_dump_executions_to();
    if (directory_path.empty() && other_directory_path.empty()) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<SessionModule> session_module,
        computation_tracker_.SnapshotComputation(versioned_handles[i].handle));
    if (!directory_path.empty()) {
      string filename = Printf("computation_%lld__%s__version_%lld",
                               versioned_handles[i].handle.handle(),
                               session_module->entry().name().c_str(),
                               versioned_handles[i].version);
      TF_RETURN_IF_ERROR(Executable::DumpToDirectory(directory_path, filename,
                                                     *session_module));
      session_modules.push_back(std::move(session_module));
    }
  }

  VLOG(1) << "Computation handles:";
  for (const VersionedComputationHandle& versioned_handle : versioned_handles) {
    VLOG(1) << versioned_handle;
  }

  CHECK_EQ(versioned_handles.size(), module_configs.size());
  std::vector<std::unique_ptr<HloModule>> modules;
  for (int64 i = 0; i < versioned_handles.size(); ++i) {
    const VersionedComputationHandle& versioned_handle = versioned_handles[i];
    const HloModuleConfig& config = *module_configs[i];
    TF_ASSIGN_OR_RETURN(auto module,
                        computation_tracker_.BuildHloModule(
                            versioned_handle, config,
                            /*include_unreachable_instructions=*/true));
    modules.push_back(std::move(module));
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<Executable>> executables,
      backend->compiler()->Compile(std::move(modules), std::move(executors),
                                   device_allocator));

  for (size_t i = 0; i < versioned_handles.size(); ++i) {
    if (!module_configs[i]->debug_options().xla_dump_executions_to().empty()) {
      executables[i]->set_session_module(std::move(session_modules[i]));
    }
  }

  return std::move(executables);
}

StatusOr<std::unique_ptr<Executable>> Service::BuildExecutable(
    const VersionedComputationHandle& versioned_handle,
    std::unique_ptr<HloModuleConfig> module_config, Backend* backend,
    se::StreamExecutor* executor, DeviceMemoryAllocator* device_allocator) {
  VLOG(1) << Printf("BuildExecutable on service %p with handle %s", this,
                    versioned_handle.ToString().c_str());

  // Dump computation proto state if flag is set.
  std::unique_ptr<SessionModule> session_module;
  const string& directory_path =
      module_config->debug_options().xla_dump_computations_to();
  const string& other_directory_path =
      module_config->debug_options().xla_dump_executions_to();
  if (!directory_path.empty() || !other_directory_path.empty()) {
    TF_ASSIGN_OR_RETURN(
        session_module,
        computation_tracker_.SnapshotComputation(versioned_handle.handle));
    if (!directory_path.empty()) {
      string filename = Printf("computation_%lld__%s__version_%lld",
                               versioned_handle.handle.handle(),
                               session_module->entry().name().c_str(),
                               versioned_handle.version);
      TF_RETURN_IF_ERROR(Executable::DumpToDirectory(directory_path, filename,
                                                     *session_module));
    }
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      computation_tracker_.BuildHloModule(versioned_handle, *module_config,
                                          /*include_unreachable_instructions=*/
                                          true));

  TF_RETURN_IF_ERROR(MaybeDumpHloModule(*module));

  TF_ASSIGN_OR_RETURN(
      module, backend->compiler()->RunHloPasses(std::move(module), executor,
                                                device_allocator));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                      backend->compiler()->RunBackend(
                          std::move(module), executor, device_allocator));

  if (!other_directory_path.empty()) {
    executable->set_session_module(std::move(session_module));
  }

  return std::move(executable);
}

StatusOr<std::shared_ptr<Executable>> Service::BuildAndCacheExecutable(
    const VersionedComputationHandle& versioned_handle,
    std::unique_ptr<HloModuleConfig> module_config, Backend* backend,
    perftools::gputools::StreamExecutor* executor, ExecutionProfile* profile,
    DeviceMemoryAllocator* device_allocator) {
  std::shared_ptr<Executable> executable =
      compilation_cache_.LookUp(versioned_handle, *module_config);

  if (executable != nullptr) {
    // Executable found in the computation cache.
    if (profile != nullptr) {
      profile->set_compilation_cache_hit(true);
    }
    return executable;
  }

  uint64 start_micros =
      // Avoid reading the clock if we don't want timing info
      (profile != nullptr) ? tensorflow::Env::Default()->NowMicros() : 0;

  // Take a copy of the module config, as compilation introduces layouts where
  // layouts were optional before.
  HloModuleConfig original_module_config = *module_config;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable_unique_ptr,
      BuildExecutable(versioned_handle, std::move(module_config), backend,
                      executor, device_allocator));

  if (profile != nullptr) {
    uint64 end_micros = tensorflow::Env::Default()->NowMicros();
    uint64 milliseconds = (end_micros - start_micros) / 1000;
    profile->set_compilation_cache_hit(false);
    profile->set_compile_time_ms(milliseconds);
  }

  // Insert executable into the cache.
  return compilation_cache_.Insert(std::move(executable_unique_ptr),
                                   original_module_config);
}

StatusOr<std::vector<GlobalDataHandle>>
Service::ExecuteParallelAndRegisterResult(
    tensorflow::gtl::ArraySlice<Executable*> executables,
    tensorflow::gtl::ArraySlice<std::vector<const ShapedBuffer*>> arguments,
    Backend* backend, tensorflow::gtl::ArraySlice<DeviceHandle> device_handles,
    tensorflow::gtl::ArraySlice<string> result_tags,
    ExecutionProfile* profile) {
  // Streams where the computation are launched, so we can wait on the streams
  // to complete.
  std::vector<Pool<se::Stream>::SmartPtr> streams;
  std::vector<std::unique_ptr<perftools::gputools::Timer>> timers;

  // Global data handles for the computation results, one for each computation.
  std::vector<GlobalDataHandle> result_handles;

  // Device ID to stream executor, populated only with devices that are being
  // profiled.
  std::map<int64, se::Stream*> index_to_profiled_streams;

  TF_ASSIGN_OR_RETURN(DeviceAssignment device_assignment,
                      backend->computation_placer()->AssignDevices(
                          options_.number_of_replicas(), executables.size()));

  for (int64 i = 0; i < executables.size(); i++) {
    // Stream executors for the replicas of the current computation.
    TF_ASSIGN_OR_RETURN(auto replicas, Replicas(*backend, device_handles[i]));
    for (int64 replica = 0; replica < replicas.size(); ++replica) {
      TF_ASSIGN_OR_RETURN(Pool<se::Stream>::SmartPtr stream,
                          backend->BorrowStream(replicas[replica]));
      streams.push_back(std::move(stream));

      if (replica == 0 && profile != nullptr) {
        timers.emplace_back(
            new perftools::gputools::Timer(streams.back()->parent()));
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
      options.set_inter_op_thread_pool(backend->inter_op_thread_pool());
      options.set_intra_op_thread_pool(
          backend->eigen_intra_op_thread_pool_device());
      options.set_device_assignment(&device_assignment);
      ServiceExecutableRunOptions run_options(options,
                                              backend->StreamBorrower());

      // Asynchronously launch the computation.
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<ShapedBuffer> result,
          executables[i]->ExecuteAsyncOnStream(&run_options, arguments[i]));

      if (replica == 0 && profile != nullptr) {
        streams.back()->ThenStopTimer(timers.back().get());
      }

      // All replicas share the same device address for the result allocation,
      // so only one of the replicas need to register the result handle.
      if (replica == 0) {
        TF_ASSIGN_OR_RETURN(
            GlobalDataHandle handle,
            allocation_tracker_.Register(std::move(result), result_tags[i]));
        result_handles.push_back(handle);
      }
    }
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
        executable->PopulateExecutionProfile(&hlo_profile, stream->parent()));
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
    const tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    Backend* backend, perftools::gputools::StreamExecutor* executor,
    const string& result_tag, ExecutionProfile* profile) {
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
    options.set_inter_op_thread_pool(backend->inter_op_thread_pool());
    options.set_intra_op_thread_pool(
        backend->eigen_intra_op_thread_pool_device());
    options.set_device_assignment(&device_assignment);
    run_options.emplace_back(options, backend->StreamBorrower(),
                             backend->inter_op_thread_pool());
  }

  std::unique_ptr<ShapedBuffer> result;
  if (options_.number_of_replicas() == 1) {
    TF_ASSIGN_OR_RETURN(result, executable->ExecuteOnStreamWrapper(
                                    &run_options[0], profile, arguments));
  } else {
    // TODO(b/69985541): Support profiling also on this path.
    std::vector<tensorflow::gtl::ArraySlice<const ShapedBuffer*>>
        repeated_arguments(options_.number_of_replicas(), arguments);

    TF_ASSIGN_OR_RETURN(auto results, executable->ExecuteOnStreams(
                                          run_options, repeated_arguments));
    TF_RET_CHECK(!results.empty());
    result = std::move(results[0]);
  }
  return allocation_tracker_.Register(std::move(result), result_tag);
}

tensorflow::Status Service::SetReturnValue(const SetReturnValueRequest* arg,
                                           SetReturnValueResponse* results) {
  TF_ASSIGN_OR_RETURN(UserComputation * computation,
                      computation_tracker_.Resolve(arg->computation()));
  return computation->SetReturnValue(arg->operand());
}

tensorflow::Status Service::ExecuteParallel(const ExecuteParallelRequest* arg,
                                            ExecuteParallelResponse* result) {
  VLOG(1) << "running execute-parallel request: " << arg->ShortDebugString();

  std::vector<std::vector<const ShapedBuffer*>> all_arguments;
  std::vector<std::vector<perftools::gputools::StreamExecutor*>> all_executors;
  std::vector<VersionedComputationHandle> versioned_handles;
  std::vector<std::unique_ptr<HloModuleConfig>> module_configs;
  std::vector<string> computation_names;
  std::vector<DeviceHandle> device_handles;

  int num_requested_devices =
      std::accumulate(arg->requests().begin(), arg->requests().end(), 0,
                      [](int a, const ExecuteRequest& r) -> int {
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
    if (execution_options.device_handles().empty()) {
      return FailedPrecondition(
          "device handles must be given to execute parallel computations");
    }
    std::vector<perftools::gputools::StreamExecutor*> executors;
    for (const auto& device_handle : execution_options.device_handles()) {
      TF_ASSIGN_OR_RETURN(auto replicas,
                          Replicas(*execute_backend_, device_handle));
      se::StreamExecutor* executor = replicas[0];
      CHECK(executor != nullptr);
      executors.push_back(executor);
    }

    // Resolve the UserComputation object associated with the requested
    // computation and compute the program shape.
    const ExecuteRequest& request = arg->requests(i);
    TF_ASSIGN_OR_RETURN(UserComputation * user_computation,
                        computation_tracker_.Resolve(request.computation()));
    VersionedComputationHandle versioned_handle =
        user_computation->GetVersionedHandle();
    if (user_computation->request_count(versioned_handle.version) == 0) {
      return InvalidArgument("computations may not be empty");
    }

    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<const ProgramShape> program_shape,
        user_computation->ComputeProgramShape(versioned_handle.version));

    // Resolve the allocations for the arguments of the computation, and create
    // a vector of device memory offsets for the arguments from the allocations.
    // In the case of partitioned computations, assume all arguments go on the
    // zeroth core.
    TF_ASSIGN_OR_RETURN(
        std::vector<const ShapedBuffer*> arguments,
        ResolveAndValidateArguments(request.arguments(),
                                    executors[0]->device_ordinal()));

    // Create an HloModuleConfig object for the computation, given the shape of
    // the program and the argument allocations.
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModuleConfig> module_config,
        CreateModuleConfig(*program_shape, arguments,
                           request.execution_options(), *user_computation));
    VLOG(3) << "ExecuteParallel created HloModuleConfig computation layout: "
            << module_config->entry_computation_layout().ToString();

    // Adds to the vectors to build and execute the computations after the loop.
    all_arguments.push_back(arguments);
    all_arguments.insert(all_arguments.end(), executors.size() - 1, {});
    versioned_handles.push_back(versioned_handle);
    module_configs.push_back(std::move(module_config));
    computation_names.insert(computation_names.end(), executors.size(),
                             user_computation->name());
    all_executors.push_back(executors);
    device_handles.insert(device_handles.end(),
                          execution_options.device_handles().begin(),
                          execution_options.device_handles().end());
  }

  // Build the user computations into HloModules and compile to generate the
  // executables.
  //
  // TODO(jlebar): There's currently no way to pass a device allocator to
  // ExecuteParallel, so we have to pass a null device_allocator below.
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<Executable>> executables,
      BuildExecutables(versioned_handles, std::move(module_configs),
                       execute_backend_.get(), all_executors,
                       /*device_allocator=*/nullptr));
  std::vector<Executable*> executable_ptrs;
  executable_ptrs.reserve(executables.size());
  for (const auto& executable : executables) {
    executable_ptrs.push_back(executable.get());
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

  VLOG(1) << "successfully completed 'execute-parallel' request";
  return tensorflow::Status::OK();
}

tensorflow::Status Service::GetDeviceHandles(const GetDeviceHandlesRequest* arg,
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

  return tensorflow::Status::OK();
}

tensorflow::Status Service::Execute(const ExecuteRequest* arg,
                                    ExecuteResponse* result) {
  VLOG(1) << "running execute request: " << arg->ShortDebugString();

  TF_ASSIGN_OR_RETURN(UserComputation * user_computation,
                      computation_tracker_.Resolve(arg->computation()));

  VersionedComputationHandle versioned_handle =
      user_computation->GetVersionedHandle();

  if (user_computation->request_count(versioned_handle.version) == 0) {
    return InvalidArgument("computations may not be empty");
  }

  // If we received multiple device handles, we must partition the module.
  if (arg->execution_options().device_handles_size() > 1) {
    ExecuteParallelRequest parallel_arg;
    *parallel_arg.add_requests() = *arg;
    ExecuteParallelResponse parallel_result;
    TF_RETURN_IF_ERROR(ExecuteParallel(&parallel_arg, &parallel_result));
    TF_RET_CHECK(parallel_result.responses_size() > 0);
    *result = parallel_result.responses(0);
    return Status::OK();
  }

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const ProgramShape> program_shape,
      user_computation->ComputeProgramShape(versioned_handle.version));

  TF_ASSIGN_OR_RETURN(
      std::vector<const ShapedBuffer*> arguments,
      ResolveAndValidateArguments(arg->arguments(),
                                  execute_backend_->default_device_ordinal()));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> module_config,
      CreateModuleConfig(*program_shape, arguments, arg->execution_options(),
                         *user_computation));

  VLOG(3) << "Execute created HloModuleConfig computation layout: "
          << module_config->entry_computation_layout().ToString();

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<Executable> executable,
      BuildAndCacheExecutable(versioned_handle, std::move(module_config),
                              execute_backend_.get(),
                              execute_backend_->default_stream_executor(),
                              result->mutable_profile()));

  if (executable->dumping()) {
    executable->session_module()->set_execution_platform(
        execute_backend_->platform()->Name());
    TF_RETURN_IF_ERROR(RecordArguments(
        arguments, execute_backend_->default_stream_executor(),
        execute_backend_->transfer_manager(), executable->session_module()));
  }

  TF_ASSIGN_OR_RETURN(
      *result->mutable_output(),
      ExecuteAndRegisterResult(
          executable.get(), arguments, execute_backend_.get(),
          execute_backend_->default_stream_executor(),
          "result of " + user_computation->name(), result->mutable_profile()));

  if (executable->dumping()) {
    TF_ASSIGN_OR_RETURN(const ShapedBuffer* result_buffer,
                        allocation_tracker_.Resolve(result->output()));
    TF_RETURN_IF_ERROR(RecordResult(
        *result_buffer, execute_backend_->default_stream_executor(),
        execute_backend_->transfer_manager(), executable->session_module()));
    TF_RETURN_IF_ERROR(executable->DumpSessionModule());
  }

  VLOG(1) << "successfully completed 'execute' request";
  return tensorflow::Status::OK();
}

tensorflow::Status Service::ExecuteAsync(const ExecuteAsyncRequest* arg,
                                         ExecuteAsyncResponse* result) {
  VLOG(1) << "running execute-async request: " << arg->ShortDebugString();

  TF_ASSIGN_OR_RETURN(UserComputation * user_computation,
                      computation_tracker_.Resolve(arg->computation()));

  VersionedComputationHandle versioned_handle =
      user_computation->GetVersionedHandle();
  if (user_computation->request_count(versioned_handle.version) == 0) {
    return InvalidArgument("computations may not be empty");
  }

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const ProgramShape> program_shape,
      user_computation->ComputeProgramShape(versioned_handle.version));

  TF_ASSIGN_OR_RETURN(
      std::vector<const ShapedBuffer*> arguments,
      ResolveAndValidateArguments(arg->arguments(),
                                  execute_backend_->default_device_ordinal()));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> module_config,
      CreateModuleConfig(*program_shape, arguments, arg->execution_options(),
                         *user_computation));

  VLOG(3) << "ExecuteAsync created HloModuleConfig computation layout: "
          << module_config->entry_computation_layout().ToString();

  ExecutionProfile profile;

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<Executable> executable,
      BuildAndCacheExecutable(
          versioned_handle, std::move(module_config), execute_backend_.get(),
          execute_backend_->default_stream_executor(), &profile));

  TF_ASSIGN_OR_RETURN(auto replicas, Replicas(*execute_backend_,
                                              SingleComputationDeviceHandle()));
  TF_RET_CHECK(!replicas.empty());

  // Set up streams.
  std::vector<Pool<se::Stream>::SmartPtr> streams;

  for (se::StreamExecutor* executor : replicas) {
    TF_ASSIGN_OR_RETURN(Pool<se::Stream>::SmartPtr stream,
                        execute_backend_->BorrowStream(executor));
    streams.push_back(std::move(stream));
  }

  std::unique_ptr<ShapedBuffer> result_buffer;
  for (const Pool<se::Stream>::SmartPtr& stream : streams) {
    ExecutableRunOptions options;
    options.set_stream(stream.get());
    options.set_allocator(execute_backend_->memory_allocator());
    options.set_inter_op_thread_pool(execute_backend_->inter_op_thread_pool());
    options.set_intra_op_thread_pool(
        execute_backend_->eigen_intra_op_thread_pool_device());

    ServiceExecutableRunOptions service_options(
        options, execute_backend_->StreamBorrower());

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<ShapedBuffer> this_result_buffer,
        executable->ExecuteAsyncOnStream(&service_options, arguments));

    // Take the first result.
    if (result_buffer == nullptr) {
      result_buffer = std::move(this_result_buffer);
    }
  }

  TF_ASSIGN_OR_RETURN(
      GlobalDataHandle output,
      allocation_tracker_.Register(std::move(result_buffer),
                                   "result of " + user_computation->name()));

  *result->mutable_execution() = execution_tracker_.Register(
      execute_backend_.get(), std::move(streams), profile, output);
  streams.clear();

  VLOG(1) << "successfully completed 'execute-async' request";
  return tensorflow::Status::OK();
}

tensorflow::Status Service::WaitForExecution(const WaitForExecutionRequest* arg,
                                             WaitForExecutionResponse* result) {
  TF_ASSIGN_OR_RETURN(const auto execution,
                      execution_tracker_.Resolve(arg->execution()));

  TF_RETURN_IF_ERROR(execution->BlockUntilDone());

  *result->mutable_output() = execution->result();
  *result->mutable_profile() = execution->profile();

  TF_RETURN_IF_ERROR(execution_tracker_.Unregister(arg->execution()));
  VLOG(1) << "successfully completed 'wait-for-execution' request";
  return tensorflow::Status::OK();
}

tensorflow::Status Service::TransferToClient(const TransferToClientRequest* arg,
                                             TransferToClientResponse* result) {
  TF_ASSIGN_OR_RETURN(const ShapedBuffer* shaped_buffer,
                      allocation_tracker_.Resolve(arg->data()));

  const Shape* return_shape;
  if (arg->has_shape_with_layout()) {
    if (!LayoutUtil::HasLayout(arg->shape_with_layout())) {
      return InvalidArgument("shape_with_layout must have layout if present.");
    }
    return_shape = &arg->shape_with_layout();
  } else {
    return_shape = &shaped_buffer->on_host_shape();
  }

  TF_ASSIGN_OR_RETURN(
      se::StreamExecutor * executor,
      execute_backend_->stream_executor(shaped_buffer->device_ordinal()));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Literal> result_literal,
      execute_backend_->transfer_manager()->TransferLiteralFromDevice(
          executor, *shaped_buffer));

  if (LayoutUtil::LayoutsInShapesEqual(*return_shape,
                                       result_literal->shape())) {
    *result->mutable_literal() = result_literal->ToProto();
  } else {
    *result->mutable_literal() =
        result_literal->Relayout(*return_shape)->ToProto();
  }
  return tensorflow::Status::OK();
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

tensorflow::Status Service::TransferToServer(const TransferToServerRequest* arg,
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

  // All memory allocation is done on the first replica. The allocations in all
  // other replicas mirror the firsts'.
  int master_device_ordinal = replicas[0]->device_ordinal();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ShapedBuffer> shaped_buffer,
      execute_backend_->transfer_manager()->AllocateShapedBuffer(
          shape, execute_backend_->memory_allocator(), master_device_ordinal));

  // Transfer the data to the replicas.
  for (se::StreamExecutor* executor : replicas) {
    if (executor->device_ordinal() == master_device_ordinal) {
      TF_RETURN_IF_ERROR(
          execute_backend_->transfer_manager()->TransferLiteralToDevice(
              executor, *literal, *shaped_buffer));
    } else {
      // The replica is not the master. Create an cloned shaped buffer with
      // the replica's device ordinal. This is required because
      // TransferLiteralToDevice verifies that the device ordinal of the shaped
      // buffer matches that of the executor.
      std::unique_ptr<ShapedBuffer> clone =
          CloneShapedBufferOnDevice(*shaped_buffer, executor->device_ordinal());
      TF_RETURN_IF_ERROR(
          execute_backend_->transfer_manager()->TransferLiteralToDevice(
              executor, *literal, *clone));
    }
  }
  TF_ASSIGN_OR_RETURN(
      *result->mutable_data(),
      allocation_tracker_.Register(std::move(shaped_buffer),
                                   StrCat("TransferToServer literal of shape ",
                                          ShapeUtil::HumanString(shape))));

  return tensorflow::Status::OK();
}

tensorflow::Status Service::TransferToInfeed(const TransferToInfeedRequest* arg,
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

tensorflow::Status Service::TransferFromOutfeed(
    const TransferFromOutfeedRequest* arg,
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
  return tensorflow::Status::OK();
}

tensorflow::Status Service::ResetDevice(const ResetDeviceRequest* arg,
                                        ResetDeviceResponse* result) {
  return execute_backend_->ResetDevices();
}

tensorflow::Status Service::IsConstant(const IsConstantRequest* arg,
                                       IsConstantResponse* result) {
  TF_ASSIGN_OR_RETURN(UserComputation * user_computation,
                      computation_tracker_.Resolve(arg->computation()));

  VersionedComputationHandle versioned_handle =
      user_computation->GetVersionedHandleAtOperation(arg->operand());

  if (user_computation->request_count(versioned_handle.version) == 0) {
    return InvalidArgument("computations may not be empty");
  }

  TF_ASSIGN_OR_RETURN(
      bool is_constant,
      user_computation->IsConstant(arg->operand(), arg->num_parameters()));

  result->set_is_constant(is_constant);
  return tensorflow::Status::OK();
}

tensorflow::Status Service::ComputeConstant(const ComputeConstantRequest* arg,
                                            ComputeConstantResponse* result) {
  TF_ASSIGN_OR_RETURN(UserComputation * user_computation,
                      computation_tracker_.Resolve(arg->computation()));

  VersionedComputationHandle versioned_handle =
      user_computation->GetVersionedHandleAtOperation(arg->operand());

  if (user_computation->request_count(versioned_handle.version) == 0) {
    return InvalidArgument("computations may not be empty");
  }

  TF_ASSIGN_OR_RETURN(
      bool is_constant,
      user_computation->IsConstant(arg->operand(), arg->parameters_size()));
  if (!is_constant) {
    StatusOr<const OperationRequest*> op_request_status =
        user_computation->LookUpRequestForErrorReporting(arg->operand());
    string op_request_string = "<unknown operation>";
    if (op_request_status.ok()) {
      op_request_string = op_request_status.ValueOrDie()->ShortDebugString();
    }
    return InvalidArgument(
        "Operand to ComputeConstant depends on a parameter.\n\n"
        "  op requested for constant evaluation: %s\n\n"
        "This is an internal error that typically happens when the XLA user "
        "(e.g. TensorFlow) is attempting to determine a value that must be a "
        "compile-time constant (e.g. an array dimension) but it is not capable "
        "of being evaluated at XLA compile time.\n\n"
        "Please file a usability bug with the framework being used (e.g. "
        "TensorFlow).",
        op_request_string.c_str());
  }

  // We can't use ComputeProgramShape because it checks that all parameter
  // instructions are present and contiguous. Instead construct ProgramShape
  // directly.
  ProgramShape program_shape;
  TF_ASSIGN_OR_RETURN(*program_shape.mutable_result(),
                      user_computation->GetShape(arg->operand()));

  TF_DCHECK_OK(ShapeUtil::ValidateShape(program_shape.result()));

  ExecutionOptions execution_options = xla::CreateDefaultExecutionOptions();
  execution_options.mutable_debug_options()->set_xla_enable_fast_math(false);
  execution_options.mutable_debug_options()
      ->set_xla_eliminate_hlo_implicit_broadcast(true);
  *execution_options.mutable_shape_with_output_layout() =
      program_shape.result();

  Shape shape_with_output_layout(program_shape.result());
  if (arg->has_output_layout()) {
    TF_RETURN_IF_ERROR(LayoutUtil::ValidateLayoutForShape(
        arg->output_layout(), execution_options.shape_with_output_layout()));
    *execution_options.mutable_shape_with_output_layout()->mutable_layout() =
        arg->output_layout();
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModuleConfig> module_config,
                      CreateModuleConfig(program_shape, {}, execution_options,
                                         *user_computation));

  // Exclude dead parameter instructions for the purpose of computing constants.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      computation_tracker_.BuildHloModule(versioned_handle, *module_config,
                                          /*include_unreachable_instructions=*/
                                          false));

  std::vector<std::unique_ptr<Literal>> parameters(arg->parameters_size());
  for (int64 i = 0; i < arg->parameters_size(); ++i) {
    TF_ASSIGN_OR_RETURN(parameters[i],
                        Literal::CreateFromProto(arg->parameters(i)));
  }
  HloEvaluator evaluator;
  TF_ASSIGN_OR_RETURN(
      auto result_literal,
      evaluator.Evaluate<std::unique_ptr<Literal>>(*module, parameters));

  // Since the shape_with_output_layout option in ExecutionOption is
  // non-effective to the Evaluator results, explicit relayout here.
  if (arg->has_output_layout()) {
    result_literal = result_literal->Relayout(arg->output_layout());
  }
  *result->mutable_literal() = result_literal->ToProto();

  return tensorflow::Status::OK();
}

tensorflow::Status Service::GetShape(const GetShapeRequest* arg,
                                     GetShapeResponse* result) {
  TF_ASSIGN_OR_RETURN(const ShapedBuffer* buffer,
                      allocation_tracker_.Resolve(arg->data()));
  *result->mutable_shape() = buffer->on_host_shape();
  return tensorflow::Status::OK();
}

tensorflow::Status Service::GetComputationShape(
    const GetComputationShapeRequest* arg,
    GetComputationShapeResponse* result) {
  TF_ASSIGN_OR_RETURN(UserComputation * computation,
                      computation_tracker_.Resolve(arg->computation()));

  VersionedComputationHandle versioned_handle =
      computation->GetVersionedHandle();

  TF_ASSIGN_OR_RETURN(auto program_shape, computation->ComputeProgramShape(
                                              versioned_handle.version));
  *result->mutable_program_shape() = *program_shape;
  return tensorflow::Status::OK();
}

tensorflow::Status Service::GetLocalShape(const GetLocalShapeRequest* arg,
                                          GetLocalShapeResponse* result) {
  TF_ASSIGN_OR_RETURN(UserComputation * computation,
                      computation_tracker_.Resolve(arg->computation()));

  TF_ASSIGN_OR_RETURN(*result->mutable_shape(),
                      computation->GetShape(arg->operand()));
  return tensorflow::Status::OK();
}

tensorflow::Status Service::GetComputationStats(
    const ComputationStatsRequest* arg, ComputationStatsResponse* result) {
  TF_ASSIGN_OR_RETURN(UserComputation * user_computation,
                      computation_tracker_.Resolve(arg->computation()));

  VersionedComputationHandle versioned_handle =
      user_computation->GetVersionedHandle();

  HloModuleConfig config;
  config.set_debug_options(arg->debug_options());
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      computation_tracker_.BuildHloModule(versioned_handle, config));

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
  return tensorflow::Status::OK();
}

template <typename RequestT, typename ResponseT>
tensorflow::Status Service::AddInstruction(
    const RequestT* arg, ResponseT* result,
    const std::function<StatusOr<ComputationDataHandle>(UserComputation*)>&
        adder) {
  TF_ASSIGN_OR_RETURN(UserComputation * computation,
                      computation_tracker_.Resolve(arg->computation()));

  TF_ASSIGN_OR_RETURN(*result->mutable_output(), adder(computation));
  return tensorflow::Status::OK();
}

tensorflow::Status Service::Op(const OpRequest* arg, OpResponse* result) {
  TF_ASSIGN_OR_RETURN(UserComputation * computation,
                      computation_tracker_.Resolve(arg->computation()));
  StatusOr<ComputationDataHandle> handle_status;

  switch (arg->op_case()) {
    case OpRequest::kBatchNormTrainingRequest:
      handle_status = computation->AddBatchNormTrainingInstruction(
          arg->batch_norm_training_request());
      break;
    case OpRequest::kBatchNormInferenceRequest:
      handle_status = computation->AddBatchNormInferenceInstruction(
          arg->batch_norm_inference_request());
      break;
    case OpRequest::kBatchNormGradRequest:
      handle_status = computation->AddBatchNormGradInstruction(
          arg->batch_norm_grad_request());
      break;
    case OpRequest::kBinaryOpRequest:
      handle_status =
          computation->AddBinaryInstruction(arg->binary_op_request());
      break;
    case OpRequest::kBroadcastRequest:
      handle_status =
          computation->AddBroadcastInstruction(arg->broadcast_request());
      break;
    case OpRequest::kCallRequest: {
      TF_ASSIGN_OR_RETURN(
          UserComputation * to_apply,
          computation_tracker_.Resolve(arg->call_request().to_apply()));
      handle_status =
          computation->AddCallInstruction(arg->call_request(), *to_apply);
      break;
    }
    case OpRequest::kConcatenateRequest:
      handle_status =
          computation->AddConcatenateInstruction(arg->concatenate_request());
      break;
    case OpRequest::kConditionalRequest: {
      TF_ASSIGN_OR_RETURN(UserComputation * true_computation,
                          computation_tracker_.Resolve(
                              arg->conditional_request().true_computation()));
      TF_ASSIGN_OR_RETURN(UserComputation * false_computation,
                          computation_tracker_.Resolve(
                              arg->conditional_request().false_computation()));
      handle_status = computation->AddConditionalInstruction(
          arg->conditional_request(), *true_computation, *false_computation);
      break;
    }
    case OpRequest::kConstantRequest:
      handle_status =
          computation->AddConstantInstruction(arg->constant_request());
      break;
    case OpRequest::kConvertRequest:
      handle_status =
          computation->AddConvertInstruction(arg->convert_request());
      break;
    case OpRequest::kBitcastConvertRequest:
      handle_status = computation->AddBitcastConvertInstruction(
          arg->bitcast_convert_request());
      break;
    case OpRequest::kConvolveRequest:
      handle_status =
          computation->AddConvolveInstruction(arg->convolve_request());
      break;
    case OpRequest::kCrossReplicaSumRequest:
      handle_status = computation->AddCrossReplicaSumInstruction(
          arg->cross_replica_sum_request());
      break;
    case OpRequest::kCustomCallRequest:
      handle_status =
          computation->AddCustomCallInstruction(arg->custom_call_request());
      break;
    case OpRequest::kDotRequest:
      handle_status = computation->AddDotInstruction(arg->dot_request());
      break;
    case OpRequest::kDynamicSliceRequest:
      handle_status =
          computation->AddDynamicSliceInstruction(arg->dynamic_slice_request());
      break;
    case OpRequest::kDynamicUpdateSliceRequest:
      handle_status = computation->AddDynamicUpdateSliceInstruction(
          arg->dynamic_update_slice_request());
      break;
    case OpRequest::kFftRequest:
      handle_status = computation->AddFftInstruction(arg->fft_request());
      break;
    case OpRequest::kGetTupleElementRequest:
      handle_status = computation->AddGetTupleElementInstruction(
          arg->get_tuple_element_request());
      break;
    case OpRequest::kInfeedRequest:
      handle_status = computation->AddInfeedInstruction(arg->infeed_request());
      break;
    case OpRequest::kOutfeedRequest:
      TF_RETURN_IF_ERROR(
          computation->AddOutfeedInstruction(arg->outfeed_request()));
      return tensorflow::Status::OK();
    case OpRequest::kMapRequest: {
      TF_ASSIGN_OR_RETURN(
          UserComputation * to_apply,
          computation_tracker_.Resolve(arg->map_request().to_apply()));
      handle_status =
          computation->AddMapInstruction(arg->map_request(), *to_apply);
      break;
    }
    case OpRequest::kPadRequest:
      handle_status = computation->AddPadInstruction(arg->pad_request());
      break;
    case OpRequest::kParameterRequest:
      handle_status =
          computation->AddParameterInstruction(arg->parameter_request());
      break;
    case OpRequest::kReduceRequest: {
      TF_ASSIGN_OR_RETURN(
          UserComputation * to_apply,
          computation_tracker_.Resolve(arg->reduce_request().to_apply()));
      handle_status =
          computation->AddReduceInstruction(arg->reduce_request(), *to_apply);
      break;
    }
    case OpRequest::kReducePrecisionRequest: {
      handle_status = computation->AddReducePrecisionInstruction(
          arg->reduce_precision_request());
      break;
    }
    case OpRequest::kReduceWindowRequest: {
      TF_ASSIGN_OR_RETURN(UserComputation * to_apply,
                          computation_tracker_.Resolve(
                              arg->reduce_window_request().to_apply()));
      handle_status = computation->AddReduceWindowInstruction(
          arg->reduce_window_request(), *to_apply);
      break;
    }
    case OpRequest::kReshapeRequest:
      handle_status =
          computation->AddReshapeInstruction(arg->reshape_request());
      break;
    case OpRequest::kReverseRequest:
      handle_status =
          computation->AddReverseInstruction(arg->reverse_request());
      break;
    case OpRequest::kRngRequest:
      handle_status = computation->AddRngInstruction(arg->rng_request());
      break;
    case OpRequest::kSelectAndScatterRequest: {
      TF_ASSIGN_OR_RETURN(UserComputation * select,
                          computation_tracker_.Resolve(
                              arg->select_and_scatter_request().select()));
      TF_ASSIGN_OR_RETURN(UserComputation * scatter,
                          computation_tracker_.Resolve(
                              arg->select_and_scatter_request().scatter()));
      handle_status = computation->AddSelectAndScatterInstruction(
          arg->select_and_scatter_request(), *select, *scatter);
      break;
    }
    case OpRequest::kSliceRequest:
      handle_status = computation->AddSliceInstruction(arg->slice_request());
      break;
    case OpRequest::kTernaryOpRequest:
      handle_status =
          computation->AddTernaryInstruction(arg->ternary_op_request());
      break;
    case OpRequest::kTraceRequest:
      return computation->AddTraceInstruction(arg->trace_request());
    case OpRequest::kTransposeRequest:
      handle_status =
          computation->AddTransposeInstruction(arg->transpose_request());
      break;
    case OpRequest::kUnaryOpRequest:
      handle_status = computation->AddUnaryInstruction(arg->unary_op_request());
      break;
    case OpRequest::kVariadicOpRequest:
      handle_status =
          computation->AddVariadicInstruction(arg->variadic_op_request());
      break;
    case OpRequest::kWhileRequest: {
      TF_ASSIGN_OR_RETURN(
          UserComputation * condition,
          computation_tracker_.Resolve(arg->while_request().condition()));
      TF_ASSIGN_OR_RETURN(
          UserComputation * body,
          computation_tracker_.Resolve(arg->while_request().body()));
      handle_status = computation->AddWhileInstruction(arg->while_request(),
                                                       *condition, *body);
      break;
    }
    case OpRequest::kSendRequest: {
      TF_RETURN_IF_ERROR(
          channel_tracker_.RegisterSend(arg->send_request().channel_handle()));
      TF_RETURN_IF_ERROR(computation->AddSendInstruction(arg->send_request()));
      return tensorflow::Status::OK();
    }
    case OpRequest::kRecvRequest: {
      TF_RETURN_IF_ERROR(
          channel_tracker_.RegisterRecv(arg->recv_request().channel_handle()));
      handle_status = computation->AddRecvInstruction(arg->recv_request());
      break;
    }
    case OpRequest::OP_NOT_SET:
      return InvalidArgument("XLA service received OpRequest with OP_NOT_SET");
    default:
      return InvalidArgument("Unsupported operation in XLA service");
  }
  TF_ASSIGN_OR_RETURN(*result->mutable_output(), handle_status);

  // We set the debug metadata here, because we slice off part of the OpRequest
  // proto in the above switch statement.
  TF_ASSIGN_OR_RETURN(ComputationDataHandle handle, handle_status);
  TF_RETURN_IF_ERROR(computation->SetOpMetadata(handle, arg->metadata()));
  if (arg->has_sharding()) {
    TF_RETURN_IF_ERROR(computation->SetOpSharding(handle, arg->sharding()));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status Service::SnapshotComputation(
    const SnapshotComputationRequest* arg,
    SnapshotComputationResponse* result) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<SessionModule> module,
      computation_tracker_.SnapshotComputation(arg->computation()));

  result->set_allocated_module(module.release());

  return tensorflow::Status::OK();
}

tensorflow::Status Service::LoadComputationSnapshot(
    const LoadComputationSnapshotRequest* arg,
    LoadComputationSnapshotResponse* result) {
  TF_ASSIGN_OR_RETURN(*result->mutable_computation(),
                      computation_tracker_.LoadSessionModule(arg->module()));
  return tensorflow::Status::OK();
}

DeviceHandle Service::SingleComputationDeviceHandle() const {
  DeviceHandle device_handle;
  device_handle.set_handle(0);
  device_handle.set_device_count(1);
  return device_handle;
}

StatusOr<std::vector<perftools::gputools::StreamExecutor*>> Service::Replicas(
    const Backend& backend, const DeviceHandle& device_handle) const {
  std::vector<perftools::gputools::StreamExecutor*> replicas;
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
