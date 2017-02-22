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

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/legacy_flags/service_flags.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/service/session.pb.h"
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
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

using ::tensorflow::strings::Printf;
using ::tensorflow::strings::StrCat;

namespace xla {

namespace {

// Copies the contents of an Allocation into a Literal proto.
tensorflow::Status LiteralFromAllocation(const Allocation* allocation,
                                         const Shape& literal_shape,
                                         Literal* literal) {
  TF_ASSIGN_OR_RETURN(
      se::StreamExecutor * executor,
      allocation->backend()->stream_executor(allocation->device_ordinal()));
  return allocation->backend()->transfer_manager()->TransferLiteralFromDevice(
      executor, allocation->device_memory(), allocation->shape(), literal_shape,
      literal);
}

// Records the arguments used to invoke a computation in a SessionModule
// proto.
tensorflow::Status RecordArguments(
    const tensorflow::gtl::ArraySlice<const Allocation*> arg_allocations,
    SessionModule* module) {
  module->clear_arguments();
  for (const Allocation* allocation : arg_allocations) {
    TF_RETURN_IF_ERROR(LiteralFromAllocation(allocation, allocation->shape(),
                                             module->add_arguments()));
  }
  return tensorflow::Status::OK();
}

// Records the result of a computation in a SessionModule proto.
tensorflow::Status RecordResult(const Allocation* result_allocation,
                                SessionModule* module) {
  module->clear_result();
  return LiteralFromAllocation(result_allocation, result_allocation->shape(),
                               module->mutable_result());
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
  TF_ASSIGN_OR_RETURN(
      execute_backend,
      Backend::CreateBackend(platform, options.number_of_replicas()));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Backend> compute_constant_backend,
                      CreateComputeConstantBackend());
  std::unique_ptr<Service> service(new Service(
      std::move(execute_backend), std::move(compute_constant_backend)));
  return std::move(service);
}

/* static */ StatusOr<std::unique_ptr<Backend>>
Service::CreateComputeConstantBackend() {
  TF_ASSIGN_OR_RETURN(std::vector<se::Platform*> platforms,
                      PlatformUtil::GetSupportedPlatforms());
  for (auto* platform : platforms) {
    if (platform->id() == se::host::kHostPlatformId) {
      return Backend::CreateBackend(platform, /*replica_count=*/1);
    }
  }
  return NotFound("CPU platform not found");
}

/* static */ void Service::DumpExecutedHlo(const HloModule& module,
                                           const string& label,
                                           const HloExecutionProfile* profile) {
  VLOG(2) << "module name = " << module.name();
  legacy_flags::ServiceFlags* flags = legacy_flags::GetServiceFlags();
  if (!flags->xla_generate_hlo_graph.empty() &&
      RE2::PartialMatch(module.name(), flags->xla_generate_hlo_graph)) {
    hlo_graph_dumper::DumpGraph(*module.entry_computation(), label,
                                flags->xla_hlo_graph_addresses,
                                flags->xla_hlo_graph_layout, profile);
  }
  if (!flags->xla_log_hlo_text.empty() &&
      RE2::PartialMatch(module.name(), flags->xla_log_hlo_text)) {
    LOG(INFO) << "HLO for module " << module.name();
    LOG(INFO) << "Label: " << label;
    XLA_LOG_LINES(2, module.ToString());
  }
  if (!flags->xla_dump_hlo_text_to.empty()) {
    hlo_graph_dumper::DumpText(module, label, flags->xla_dump_hlo_text_to);
  }
}

/* static */ Compiler::HloDumper Service::MakeHloDumper() {
  return [](const HloModule& module, const string& label) {
    return DumpExecutedHlo(module, label, /*profile=*/nullptr);
  };
}

Service::Service(std::unique_ptr<Backend> execute_backend,
                 std::unique_ptr<Backend> compute_constant_backend)
    : execute_backend_(std::move(execute_backend)),
      compute_constant_backend_(std::move(compute_constant_backend)) {
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
}

tensorflow::Status Service::Computation(const ComputationRequest* arg,
                                        ComputationResponse* result) {
  if (arg->name().empty()) {
    return InvalidArgument("computation request needs a name");
  }

  *result->mutable_computation() =
      computation_tracker_.NewComputation(arg->name());
  VLOG(1) << Printf("Created new computation %s on service %p",
                    result->computation().ShortDebugString().c_str(), this);
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

StatusOr<std::vector<const Allocation*>> Service::ResolveAndValidateArguments(
    tensorflow::gtl::ArraySlice<const GlobalDataHandle*> arguments,
    const Backend* backend, int device_ordinal) {
  std::vector<const Allocation*> allocations;
  for (int i = 0; i < arguments.size(); ++i) {
    auto allocation_status = allocation_tracker_.Resolve(*arguments[i]);
    if (!allocation_status.ok()) {
      return Status(allocation_status.status().code(),
                    StrCat(allocation_status.status().error_message(), ", ",
                           "failed to resolve allocation for parameter ", i));
    }
    const Allocation* allocation = allocation_status.ValueOrDie();

    // Verify allocation is same platform and device as the execution.
    if (allocation->backend() != backend ||
        allocation->device_ordinal() != device_ordinal) {
      return InvalidArgument(
          "argument %d is on device %s but computation will be executed "
          "on device %s",
          i,
          allocation->backend()
              ->device_name(allocation->device_ordinal())
              .c_str(),
          backend->device_name(device_ordinal).c_str());
    }

    allocations.push_back(allocation);
  }
  return allocations;
}

StatusOr<std::unique_ptr<HloModuleConfig>> Service::CreateModuleConfig(
    const ProgramShape& program_shape,
    tensorflow::gtl::ArraySlice<const Allocation*> arguments,
    const ExecutionOptions& execution_options) {
  auto module_config = MakeUnique<HloModuleConfig>(program_shape);
  auto* computation_layout = module_config->mutable_entry_computation_layout();

  if (program_shape.parameters_size() != arguments.size()) {
    return InvalidArgument("computation takes %d parameters, but %zu given",
                           program_shape.parameters_size(), arguments.size());
  }

  for (int i = 0; i < arguments.size(); ++i) {
    // Verify that shape of arguments matches the shape of the arguments in the
    // ProgramShape.
    if (!ShapeUtil::Compatible(arguments[i]->shape(),
                               program_shape.parameters(i))) {
      return InvalidArgument(
          "computation expects parameter %d to have shape %s, given shape %s",
          i, ShapeUtil::HumanString(program_shape.parameters(i)).c_str(),
          ShapeUtil::HumanString(arguments[i]->shape()).c_str());
    }
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
            arguments[i]->shape()));
  }
  if (!execution_options.has_shape_with_output_layout()) {
    computation_layout->mutable_result_layout()->Clear();
  } else {
    const auto& shape_with_output_layout =
        execution_options.shape_with_output_layout();
    TF_RETURN_IF_ERROR(ValidateResultShapeWithLayout(shape_with_output_layout,
                                                     program_shape.result()));
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_result_layout()->CopyLayoutFromShape(
            shape_with_output_layout));
  }

  legacy_flags::ServiceFlags* flags = legacy_flags::GetServiceFlags();
  if (flags->xla_hlo_profile) {
    module_config->enable_hlo_profiling(true);
  }

  module_config->set_replica_count(execute_backend_->Replicas().size());
  module_config->set_fast_math_disabled(execution_options.disable_fast_math());
  module_config->set_seed(execution_options.seed());

  return std::move(module_config);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> Service::BuildExecutables(
    std::vector<VersionedComputationHandle> versioned_handles,
    std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
    Backend* backend,
    std::vector<perftools::gputools::StreamExecutor*> executors) {
  VLOG(1) << Printf("BuildExecutable on service %p", this);

  // Dump computation proto state if flag is set.
  std::vector<std::unique_ptr<SessionModule>> session_modules;
  legacy_flags::ServiceFlags* flags = legacy_flags::GetServiceFlags();
  const string& directory_path = flags->xla_dump_computations_to;
  const string& other_directory_path = flags->xla_dump_executions_to;
  if ((!directory_path.empty() || !other_directory_path.empty())) {
    for (int64 i = 0; i < versioned_handles.size(); ++i) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<SessionModule> session_module,
                          computation_tracker_.SnapshotComputation(
                              versioned_handles[i].handle));
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
  }

  VLOG(1) << "Computation handles:";
  for (const VersionedComputationHandle& versioned_handle : versioned_handles) {
    VLOG(1) << versioned_handle;
  }

  std::vector<std::unique_ptr<HloModule>> modules;
  for (const VersionedComputationHandle& versioned_handle : versioned_handles) {
    TF_ASSIGN_OR_RETURN(auto module,
                        computation_tracker_.BuildHloModule(
                            versioned_handle,
                            /*include_unreachable_instructions=*/true));
    modules.push_back(std::move(module));
  }

  Compiler::HloDumper hlo_dumper = MakeHloDumper();
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<Executable>> executables,
                      backend->compiler()->Compile(
                          std::move(modules), std::move(module_configs),
                          hlo_dumper, std::move(executors)));

  if (!other_directory_path.empty()) {
    for (int64 i = 0; i < versioned_handles.size(); ++i) {
      executables[i]->set_session_module(std::move(session_modules[i]));
    }
  }

  return std::move(executables);
}

StatusOr<std::unique_ptr<Executable>> Service::BuildExecutable(
    const VersionedComputationHandle& versioned_handle,
    std::unique_ptr<HloModuleConfig> module_config,
    bool executable_for_compute_constant,
    const tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
        arguments,
    Backend* backend, se::StreamExecutor* executor) {
  VLOG(1) << Printf("BuildExecutable on service %p with handle %s", this,
                    versioned_handle.ToString().c_str());

  // Dump computation proto state if flag is set.
  std::unique_ptr<SessionModule> session_module;
  legacy_flags::ServiceFlags* flags = legacy_flags::GetServiceFlags();
  const string& directory_path = flags->xla_dump_computations_to;
  const string& other_directory_path = flags->xla_dump_executions_to;
  if (!executable_for_compute_constant &&
      (!directory_path.empty() || !other_directory_path.empty())) {
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
      computation_tracker_.BuildHloModule(versioned_handle,
                                          /*include_unreachable_instructions=*/
                                          !executable_for_compute_constant));

  Compiler::HloDumper hlo_dumper = MakeHloDumper();
  if (executable_for_compute_constant &&
      !flags->xla_hlo_graph_for_compute_constant) {
    hlo_dumper = [](const HloModule&, const string&) {};
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      backend->compiler()->Compile(std::move(module), std::move(module_config),
                                   hlo_dumper, executor));

  if (!other_directory_path.empty()) {
    executable->set_session_module(std::move(session_module));
  }

  return std::move(executable);
}

StatusOr<std::shared_ptr<Executable>> Service::BuildAndCacheExecutable(
    const VersionedComputationHandle& versioned_handle,
    std::unique_ptr<HloModuleConfig> module_config,
    const tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
        arguments,
    Backend* backend, perftools::gputools::StreamExecutor* executor,
    ExecutionProfile* profile) {
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
      BuildExecutable(versioned_handle, std::move(module_config),
                      /*executable_for_compute_constant=*/false, arguments,
                      execute_backend_.get(), executor));

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
    tensorflow::gtl::ArraySlice<
        std::vector<perftools::gputools::DeviceMemoryBase>>
        arguments,
    Backend* backend,
    tensorflow::gtl::ArraySlice<perftools::gputools::StreamExecutor*> executors,
    tensorflow::gtl::ArraySlice<string> result_tags) {
  // TODO(b/33943292): Support for replication when using multiple computations.
  TF_RET_CHECK(backend->Replicas().size() == 1);

  // Set up streams.
  std::vector<Pool<se::Stream>::SmartPtr> streams;

  for (se::StreamExecutor* executor : executors) {
    TF_ASSIGN_OR_RETURN(Pool<se::Stream>::SmartPtr stream,
                        backend->BorrowStream(executor));
    streams.push_back(std::move(stream));
  }

  // Set up run options.
  std::vector<ExecutableRunOptions> run_options;
  for (const Pool<se::Stream>::SmartPtr& stream : streams) {
    run_options.emplace_back();
    auto& options = run_options.back();
    options.set_stream(stream.get());
    options.set_allocator(backend->memory_allocator());
    options.set_inter_op_thread_pool(backend->inter_op_thread_pool());
    options.set_intra_op_thread_pool(
        backend->eigen_intra_op_thread_pool_device());
  }

  // Asynchronously launch all executables.
  std::vector<GlobalDataHandle> result_handles;
  for (int64 i = 0; i < executables.size(); i++) {
    TF_ASSIGN_OR_RETURN(
        perftools::gputools::DeviceMemoryBase result,
        executables[i]->ExecuteAsyncOnStream(&run_options[i], arguments[i]));
    result_handles.push_back(allocation_tracker_.Register(
        backend, executors[i]->device_ordinal(), result,
        executables[i]->result_shape(), result_tags[i]));
  }

  // Wait for all executions to complete.
  for (int64 i = 0; i < result_handles.size(); ++i) {
    if (!streams[i]->BlockHostUntilDone()) {
      return InternalError("failed to complete execution for stream %lld", i);
    }
  }

  return result_handles;
}

StatusOr<GlobalDataHandle> Service::ExecuteAndRegisterResult(
    Executable* executable,
    const tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
        arguments,
    Backend* backend, perftools::gputools::StreamExecutor* executor,
    const string& result_tag, ExecutionProfile* profile) {
  TF_RET_CHECK(!backend->Replicas().empty());

  // Set up streams.
  std::vector<Pool<se::Stream>::SmartPtr> streams;

  for (se::StreamExecutor* executor : backend->Replicas()) {
    TF_ASSIGN_OR_RETURN(Pool<se::Stream>::SmartPtr stream,
                        backend->BorrowStream(executor));
    streams.push_back(std::move(stream));
  }

  // Set up run options.
  std::vector<ExecutableRunOptions> run_options;
  for (const Pool<se::Stream>::SmartPtr& stream : streams) {
    run_options.emplace_back();
    auto& options = run_options.back();
    options.set_stream(stream.get());
    options.set_allocator(backend->memory_allocator());
    options.set_inter_op_thread_pool(backend->inter_op_thread_pool());
    options.set_intra_op_thread_pool(
        backend->eigen_intra_op_thread_pool_device());
  }

  perftools::gputools::DeviceMemoryBase result;
  if (backend->Replicas().size() == 1) {
    TF_ASSIGN_OR_RETURN(
        result, ExecuteOnStreamWrapper<StatusOr<se::DeviceMemoryBase>>(
                    executable, &run_options[0], profile,
                    [&arguments](Executable* executable,
                                 const ExecutableRunOptions* run_options,
                                 HloExecutionProfile* hlo_execution_profile) {
                      return executable->ExecuteOnStream(run_options, arguments,
                                                         hlo_execution_profile);
                    }));
  } else {
    std::vector<
        tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>>
        repeated_arguments(backend->Replicas().size(), arguments);

    TF_ASSIGN_OR_RETURN(
        auto results,
        executable->ExecuteOnStreams(run_options, repeated_arguments));
    TF_RET_CHECK(!results.empty());
    result = results[0];
  }
  return allocation_tracker_.Register(backend, executor->device_ordinal(),
                                      result, executable->result_shape(),
                                      result_tag);
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

  std::vector<std::vector<se::DeviceMemoryBase>> all_arguments;
  std::vector<perftools::gputools::StreamExecutor*> executors;
  std::vector<VersionedComputationHandle> versioned_handles;
  std::vector<std::unique_ptr<HloModuleConfig>> module_configs;
  std::vector<string> computation_names;

  if (arg->requests_size() > execute_backend_->stream_executors().size()) {
    return FailedPrecondition(
        "there are not enough stream executors to execute %d computations",
        arg->requests_size());
  }

  for (int64 i = 0; i < arg->requests_size(); ++i) {
    // Get the stream executor on which the computation will run. Select the
    // specific device if requested, otherwise select the i'th device from the
    // list of available stream executors.
    se::StreamExecutor* executor;
    if (arg->requests(i).has_device_handle()) {
      executor =
          execute_backend_
              ->stream_executors()[arg->requests(i).device_handle().handle()];
    } else {
      executor = execute_backend_->stream_executors()[i];
    }
    CHECK(executor != nullptr);

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
    TF_ASSIGN_OR_RETURN(
        std::vector<const Allocation*> arg_allocations,
        ResolveAndValidateArguments(request.arguments(), execute_backend_.get(),
                                    executor->device_ordinal()));
    std::vector<se::DeviceMemoryBase> arguments;
    for (const Allocation* allocation : arg_allocations) {
      arguments.push_back(allocation->device_memory());
    }

    // Create an HloModuleConfig object for the computation, given the shape of
    // the program and the argument allocations.
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModuleConfig> module_config,
                        CreateModuleConfig(*program_shape, arg_allocations,
                                           request.execution_options()));
    VLOG(3) << "ExecuteParallel created HloModuleConfig computation layout: "
            << module_config->entry_computation_layout().ToString();

    // Adds to the vectors to build and execute the computations after the loop.
    all_arguments.push_back(arguments);
    versioned_handles.push_back(versioned_handle);
    module_configs.push_back(std::move(module_config));
    computation_names.push_back(user_computation->name());
    executors.push_back(executor);
  }

  // Build the user computations into HloModules and compile to generate the
  // executables.
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<Executable>> executables,
      BuildExecutables(versioned_handles, std::move(module_configs),
                       execute_backend_.get(), executors));
  std::vector<Executable*> executable_ptrs;
  for (const auto& executable : executables) {
    executable_ptrs.push_back(executable.get());
  }

  // Execute the generated executables in parallel and return the device
  // handles for each computation's output.
  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDataHandle> outputs,
      ExecuteParallelAndRegisterResult(executable_ptrs, all_arguments,
                                       execute_backend_.get(), executors,
                                       computation_names));
  for (const GlobalDataHandle& output : outputs) {
    ExecuteResponse response;
    *response.mutable_output() = output;
    *result->add_responses() = response;
  }

  VLOG(1) << "successfully completed 'execute-parallel' request";
  return tensorflow::Status::OK();
}

tensorflow::Status Service::GetDeviceHandles(const GetDeviceHandlesRequest* arg,
                                             GetDeviceHandlesResponse* result) {
  const int64 available_device_count =
      execute_backend_->stream_executors().size();
  const int64 replicas = execute_backend_->Replicas().size();
  if (available_device_count < arg->device_count() * replicas) {
    return ResourceExhausted(
        "Requested device count (%lld) exceeds the number of available devices "
        "on the target (%lld)",
        arg->device_count(), available_device_count);
  }

  for (int64 i = 0; i < arg->device_count(); ++i) {
    DeviceHandle device_handle;
    device_handle.set_handle(
        execute_backend_->stream_executors()[i * replicas]->device_ordinal());
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

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const ProgramShape> program_shape,
      user_computation->ComputeProgramShape(versioned_handle.version));

  TF_ASSIGN_OR_RETURN(
      std::vector<const Allocation*> arg_allocations,
      ResolveAndValidateArguments(arg->arguments(), execute_backend_.get(),
                                  execute_backend_->default_device_ordinal()));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModuleConfig> module_config,
                      CreateModuleConfig(*program_shape, arg_allocations,
                                         arg->execution_options()));

  VLOG(3) << "Execute created HloModuleConfig computation layout: "
          << module_config->entry_computation_layout().ToString();

  std::vector<se::DeviceMemoryBase> arguments;
  for (const Allocation* allocation : arg_allocations) {
    arguments.push_back(allocation->device_memory());
  }

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<Executable> executable,
      BuildAndCacheExecutable(versioned_handle, std::move(module_config),
                              arguments, execute_backend_.get(),
                              execute_backend_->default_stream_executor(),
                              result->mutable_profile()));

  if (executable->dumping()) {
    executable->session_module()->set_execution_platform(
        execute_backend_->platform()->Name());
    TF_RETURN_IF_ERROR(
        RecordArguments(arg_allocations, executable->session_module()));
  }

  TF_ASSIGN_OR_RETURN(
      *result->mutable_output(),
      ExecuteAndRegisterResult(
          executable.get(), arguments, execute_backend_.get(),
          execute_backend_->default_stream_executor(),
          "result of " + user_computation->name(), result->mutable_profile()));

  if (executable->dumping()) {
    TF_ASSIGN_OR_RETURN(const Allocation* result_allocation,
                        allocation_tracker_.Resolve(result->output()));
    TF_RETURN_IF_ERROR(
        RecordResult(result_allocation, executable->session_module()));
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
      std::vector<const Allocation*> arg_allocations,
      ResolveAndValidateArguments(arg->arguments(), execute_backend_.get(),
                                  execute_backend_->default_device_ordinal()));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModuleConfig> module_config,
                      CreateModuleConfig(*program_shape, arg_allocations,
                                         arg->execution_options()));

  VLOG(3) << "ExecuteAsync created HloModuleConfig computation layout: "
          << module_config->entry_computation_layout().ToString();

  std::vector<se::DeviceMemoryBase> arguments;
  for (const Allocation* allocation : arg_allocations) {
    arguments.push_back(allocation->device_memory());
  }

  ExecutionProfile profile;

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<Executable> executable,
      BuildAndCacheExecutable(versioned_handle, std::move(module_config),
                              arguments, execute_backend_.get(),
                              execute_backend_->default_stream_executor(),
                              &profile));

  TF_RET_CHECK(!execute_backend_->Replicas().empty());
  // Set up streams.
  std::vector<Pool<se::Stream>::SmartPtr> streams;

  for (se::StreamExecutor* executor : execute_backend_->Replicas()) {
    TF_ASSIGN_OR_RETURN(Pool<se::Stream>::SmartPtr stream,
                        execute_backend_->BorrowStream(executor));
    streams.push_back(std::move(stream));
  }

  perftools::gputools::DeviceMemoryBase result_data;
  for (const Pool<se::Stream>::SmartPtr& stream : streams) {
    ExecutableRunOptions options;
    options.set_stream(stream.get());
    options.set_allocator(execute_backend_->memory_allocator());
    options.set_inter_op_thread_pool(execute_backend_->inter_op_thread_pool());
    options.set_intra_op_thread_pool(
        execute_backend_->eigen_intra_op_thread_pool_device());

    TF_ASSIGN_OR_RETURN(perftools::gputools::DeviceMemoryBase this_result_data,
                        executable->ExecuteAsyncOnStream(&options, arguments));

    // Take the first result.
    if (result_data == nullptr) {
      result_data = this_result_data;
    }
  }

  auto output = allocation_tracker_.Register(
      execute_backend_.get(), execute_backend_->default_device_ordinal(),
      result_data, executable->result_shape(),
      "result of " + user_computation->name());

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
  TF_ASSIGN_OR_RETURN(const Allocation* allocation,
                      allocation_tracker_.Resolve(arg->data()));

  const Shape* literal_shape;
  if (arg->has_shape_with_layout()) {
    if (!LayoutUtil::HasLayout(arg->shape_with_layout())) {
      return InvalidArgument("shape_with_layout must have layout if present.");
    }
    literal_shape = &arg->shape_with_layout();
  } else {
    literal_shape = &allocation->shape();
  }

  return LiteralFromAllocation(allocation, *literal_shape,
                               result->mutable_literal());
}

tensorflow::Status Service::TransferToServer(const TransferToServerRequest* arg,
                                             TransferToServerResponse* result) {
  const Literal& literal = arg->literal();
  const Shape& shape = literal.shape();

  if (ShapeUtil::IsTuple(shape) && execute_backend_->Replicas().size() > 1) {
    // TODO(b/32990684): Tuple transfers to host end up allocating further
    // buffers - implement that correctly.
    return Unimplemented(
        "Tuple transfers to the device not supported with replication.");
  }

  se::StreamExecutor* stream_executor;
  if (arg->has_device_handle()) {
    TF_ASSIGN_OR_RETURN(
        stream_executor,
        execute_backend_->stream_executor(arg->device_handle().handle()));
  } else {
    stream_executor = execute_backend_->default_stream_executor();
  }

  // Allocate memory on the device, using the stream executor. The size of the
  // allocation is obtained by examining the shape of the literal passed from
  // the client. An allocation handle is returned in the response.
  int64 allocation_size =
      execute_backend_->transfer_manager()->GetByteSizeRequirement(shape);

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase allocation,
                      execute_backend_->memory_allocator()->Allocate(
                          stream_executor->device_ordinal(), allocation_size));

  *result->mutable_data() = allocation_tracker_.Register(
      execute_backend_.get(), stream_executor->device_ordinal(), allocation,
      shape, StrCat("TransferToServer literal of size ", allocation_size));

  TF_ASSIGN_OR_RETURN(
      auto replicas,
      execute_backend_->Replicas(stream_executor->device_ordinal()));
  for (se::StreamExecutor* executor : replicas) {
    TF_RETURN_IF_ERROR(
        execute_backend_->transfer_manager()->TransferLiteralToDevice(
            executor, literal, &allocation));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status Service::TransferToInfeed(const TransferToInfeedRequest* arg,
                                             TransferToInfeedResponse* result) {
  const int64 replica_count = execute_backend_->Replicas().size();
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
    TF_ASSIGN_OR_RETURN(
        auto replicas,
        execute_backend_->Replicas(arg->device_handle().handle()));
    executor = replicas[arg->replica_id()];
  } else {
    executor = execute_backend_->Replicas()[arg->replica_id()];
  }

  return execute_backend_->transfer_manager()->TransferLiteralToInfeed(
      executor, arg->literal());
}

tensorflow::Status Service::ResetDevice(const ResetDeviceRequest* arg,
                                        ResetDeviceResponse* result) {
  return execute_backend_->ResetDevices();
}

tensorflow::Status Service::TransferToClientInProcess(
    const TransferToClientInProcessRequest* arg,
    TransferToClientInProcessResponse* result) {
  TF_RETURN_IF_ERROR(CheckRunsInClientProcess("TransferToClientInProcess"));

  TF_ASSIGN_OR_RETURN(const Allocation* allocation,
                      allocation_tracker_.Resolve(arg->data()));

  void* buffer = reinterpret_cast<void*>(arg->buffer());
  int64 size = ShapeUtil::ByteSizeOf(allocation->shape());
  TF_ASSIGN_OR_RETURN(
      se::StreamExecutor * executor,
      allocation->backend()->stream_executor(allocation->device_ordinal()));

  return allocation->backend()->transfer_manager()->TransferBufferFromDevice(
      executor, allocation->device_memory(), size, buffer);
}

tensorflow::Status Service::TransferToServerInProcess(
    const TransferToServerInProcessRequest* arg,
    TransferToServerInProcessResponse* result) {
  TF_RETURN_IF_ERROR(CheckRunsInClientProcess("TransferToServerInProcess"));

  const Shape& shape = arg->shape();

  if (ShapeUtil::IsTuple(shape) && execute_backend_->Replicas().size() > 1) {
    // TODO(b/32990684): Tuple transfers to host end up allocating further
    // buffers - implement that correctly.
    return Unimplemented(
        "Tuple transfers to the device not supported with replication.");
  }

  if (!LayoutUtil::HasLayout(shape)) {
    return InvalidArgument("shape must have layout");
  }

  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(shape));

  const void* buffer = reinterpret_cast<const void*>(arg->buffer());

  // Allocate memory on the device, using the stream executor. The size of the
  // allocation is obtained by examining the shape of the literal passed from
  // the client. An allocation handle is returned in the response.
  int64 allocation_size =
      execute_backend_->transfer_manager()->GetByteSizeRequirement(shape);
  se::StreamExecutor* stream_executor =
      execute_backend_->default_stream_executor();

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase allocation,
                      execute_backend_->memory_allocator()->Allocate(
                          stream_executor->device_ordinal(), allocation_size));

  *result->mutable_data() = allocation_tracker_.Register(
      execute_backend_.get(), stream_executor->device_ordinal(), allocation,
      shape, StrCat("TransferToServer literal of size ", allocation_size));

  for (se::StreamExecutor* executor : execute_backend_->Replicas()) {
    TF_RETURN_IF_ERROR(
        execute_backend_->transfer_manager()->TransferBufferToDevice(
            executor, allocation_size, buffer, &allocation));
  }
  return tensorflow::Status::OK();
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

  TF_ASSIGN_OR_RETURN(bool is_constant,
                      user_computation->IsConstant(arg->operand()));

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

  TF_ASSIGN_OR_RETURN(bool is_constant,
                      user_computation->IsConstant(arg->operand()));

  if (!is_constant) {
    return InvalidArgument("Operand to ComputeConstant depends on parameter.");
  }

  // We can't use ComputeProgramShape because it checks that all parameter
  // instructions are present and contiguous. Instead construct ProgramShape
  // directly.
  ProgramShape program_shape;
  TF_ASSIGN_OR_RETURN(*program_shape.mutable_result(),
                      user_computation->GetShape(arg->operand()));

  TF_DCHECK_OK(ShapeUtil::ValidateShape(program_shape.result()));

  ExecutionOptions execution_options;
  execution_options.set_disable_fast_math(true);
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
                      CreateModuleConfig(program_shape, {}, execution_options));

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<Executable> executable,
      BuildExecutable(versioned_handle, std::move(module_config),
                      /*executable_for_compute_constant=*/true,
                      /*arguments=*/{}, compute_constant_backend_.get(),
                      compute_constant_backend_->default_stream_executor()));

  TF_ASSIGN_OR_RETURN(
      *result->mutable_output(),
      ExecuteAndRegisterResult(
          executable.get(), /*arguments=*/{}, compute_constant_backend_.get(),
          compute_constant_backend_->default_stream_executor(),
          "constant computed from " + user_computation->name(),
          /*profile=*/nullptr));
  return tensorflow::Status::OK();
}

tensorflow::Status Service::GetShape(const GetShapeRequest* arg,
                                     GetShapeResponse* result) {
  TF_ASSIGN_OR_RETURN(const Allocation* allocation,
                      allocation_tracker_.Resolve(arg->data()));
  *result->mutable_shape() = allocation->shape();
  return tensorflow::Status::OK();
}

tensorflow::Status Service::GetComputationShape(
    const GetComputationShapeRequest* arg,
    GetComputationShapeResponse* result) {
  TF_ASSIGN_OR_RETURN(UserComputation * computation,
                      computation_tracker_.Resolve(arg->computation()));

  VersionedComputationHandle versioned_handle =
      computation->GetVersionedHandle();

  TF_ASSIGN_OR_RETURN(
      auto program_shape,
      computation->ComputeProgramShape(versioned_handle.version));
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

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      computation_tracker_.BuildHloModule(versioned_handle));

  MakeHloDumper()(*module, "computation statistics subject");

  // Run HLO analysis to get the computation statistics.
  HloCostAnalysis analysis([this](const Shape& shape) {
    return execute_backend_->compiler()->ShapeSizeBytes(shape);
  });

  TF_RETURN_IF_ERROR(
      module->entry_computation()->root_instruction()->Accept(&analysis));

  ComputationStats stats;
  stats.set_flop_count(analysis.flop_count());
  stats.set_transcendental_count(analysis.transcendental_count());
  *result->mutable_stats() = stats;
  return tensorflow::Status::OK();
}

tensorflow::Status Service::CheckRunsInClientProcess(
    const string& method_name) const {
  if (runs_in_client_process_) {
    return tensorflow::Status::OK();
  } else {
    return FailedPrecondition(
        "%s only supported if service runs in the same process as the client",
        method_name.c_str());
  }
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
  StatusOr<ComputationDataHandle> handle;

  switch (arg->op_case()) {
    case OpRequest::kBinaryOpRequest:
      handle = computation->AddBinaryInstruction(arg->binary_op_request());
      break;
    case OpRequest::kBroadcastRequest:
      handle = computation->AddBroadcastInstruction(arg->broadcast_request());
      break;
    case OpRequest::kCallRequest: {
      TF_ASSIGN_OR_RETURN(
          UserComputation * to_apply,
          computation_tracker_.Resolve(arg->call_request().to_apply()));
      handle = computation->AddCallInstruction(arg->call_request(), *to_apply);
      break;
    }
    case OpRequest::kConcatenateRequest:
      handle =
          computation->AddConcatenateInstruction(arg->concatenate_request());
      break;
    case OpRequest::kConstantRequest:
      handle = computation->AddConstantInstruction(arg->constant_request());
      break;
    case OpRequest::kConvertRequest:
      handle = computation->AddConvertInstruction(arg->convert_request());
      break;
    case OpRequest::kConvolveRequest:
      handle = computation->AddConvolveInstruction(arg->convolve_request());
      break;
    case OpRequest::kCrossReplicaSumRequest:
      handle = computation->AddCrossReplicaSumInstruction(
          arg->cross_replica_sum_request());
      break;
    case OpRequest::kCustomCallRequest:
      handle =
          computation->AddCustomCallInstruction(arg->custom_call_request());
      break;
    case OpRequest::kDynamicSliceRequest:
      handle =
          computation->AddDynamicSliceInstruction(arg->dynamic_slice_request());
      break;
    case OpRequest::kDynamicUpdateSliceRequest:
      handle = computation->AddDynamicUpdateSliceInstruction(
          arg->dynamic_update_slice_request());
      break;
    case OpRequest::kGetTupleElementRequest:
      handle = computation->AddGetTupleElementInstruction(
          arg->get_tuple_element_request());
      break;
    case OpRequest::kInfeedRequest:
      handle = computation->AddInfeedInstruction(arg->infeed_request());
      break;
    case OpRequest::kOutfeedRequest:
      TF_RETURN_IF_ERROR(
          computation->AddOutfeedInstruction(arg->outfeed_request()));
      return tensorflow::Status::OK();
    case OpRequest::kMapRequest: {
      TF_ASSIGN_OR_RETURN(
          UserComputation * to_apply,
          computation_tracker_.Resolve(arg->map_request().to_apply()));
      handle = computation->AddMapInstruction(arg->map_request(), *to_apply);
      break;
    }
    case OpRequest::kPadRequest:
      handle = computation->AddPadInstruction(arg->pad_request());
      break;
    case OpRequest::kParameterRequest:
      handle = computation->AddParameterInstruction(arg->parameter_request());
      break;
    case OpRequest::kReduceRequest: {
      TF_ASSIGN_OR_RETURN(
          UserComputation * to_apply,
          computation_tracker_.Resolve(arg->reduce_request().to_apply()));
      handle =
          computation->AddReduceInstruction(arg->reduce_request(), *to_apply);
      break;
    }
    case OpRequest::kReduceWindowRequest: {
      TF_ASSIGN_OR_RETURN(UserComputation * to_apply,
                          computation_tracker_.Resolve(
                              arg->reduce_window_request().to_apply()));
      handle = computation->AddReduceWindowInstruction(
          arg->reduce_window_request(), *to_apply);
      break;
    }
    case OpRequest::kReshapeRequest:
      handle = computation->AddReshapeInstruction(arg->reshape_request());
      break;
    case OpRequest::kReverseRequest:
      handle = computation->AddReverseInstruction(arg->reverse_request());
      break;
    case OpRequest::kRngRequest:
      handle = computation->AddRngInstruction(arg->rng_request());
      break;
    case OpRequest::kSelectAndScatterRequest: {
      TF_ASSIGN_OR_RETURN(UserComputation * select,
                          computation_tracker_.Resolve(
                              arg->select_and_scatter_request().select()));
      TF_ASSIGN_OR_RETURN(UserComputation * scatter,
                          computation_tracker_.Resolve(
                              arg->select_and_scatter_request().scatter()));
      handle = computation->AddSelectAndScatterInstruction(
          arg->select_and_scatter_request(), *select, *scatter);
      break;
    }
    case OpRequest::kSliceRequest:
      handle = computation->AddSliceInstruction(arg->slice_request());
      break;
    case OpRequest::kTernaryOpRequest:
      handle = computation->AddTernaryInstruction(arg->ternary_op_request());
      break;
    case OpRequest::kTraceRequest:
      return computation->AddTraceInstruction(arg->trace_request());
    case OpRequest::kUnaryOpRequest:
      handle = computation->AddUnaryInstruction(arg->unary_op_request());
      break;
    case OpRequest::kVariadicOpRequest:
      handle = computation->AddVariadicInstruction(arg->variadic_op_request());
      break;
    case OpRequest::kWhileRequest: {
      TF_ASSIGN_OR_RETURN(
          UserComputation * condition,
          computation_tracker_.Resolve(arg->while_request().condition()));
      TF_ASSIGN_OR_RETURN(
          UserComputation * body,
          computation_tracker_.Resolve(arg->while_request().body()));
      handle = computation->AddWhileInstruction(arg->while_request(),
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
      handle = computation->AddRecvInstruction(arg->recv_request());
      break;
    }
    default:
      return InvalidArgument("Unsupported operation");
  }
  TF_ASSIGN_OR_RETURN(*result->mutable_output(), handle);
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

}  // namespace xla
