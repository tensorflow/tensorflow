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

#include "tensorflow/compiler/xla/service/local_service.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/legacy_flags/service_flags.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/computation_tracker.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/service/user_computation.h"
#include "tensorflow/compiler/xla/service/versioned_computation_handle.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace se = ::perftools::gputools;

namespace xla {

LocalExecuteOptions& LocalExecuteOptions::set_platform(
    perftools::gputools::Platform* platform) {
  platform_ = platform;
  return *this;
}

perftools::gputools::Platform* LocalExecuteOptions::platform() const {
  return platform_;
}

LocalExecuteOptions& LocalExecuteOptions::set_device_ordinal(
    int device_ordinal) {
  device_ordinal_ = device_ordinal;
  return *this;
}

int LocalExecuteOptions::device_ordinal() const { return device_ordinal_; }

LocalExecuteOptions& LocalExecuteOptions::set_allocator(
    DeviceMemoryAllocator* allocator) {
  allocator_ = allocator;
  return *this;
}

DeviceMemoryAllocator* LocalExecuteOptions::allocator() const {
  return allocator_;
}

LocalExecuteOptions& LocalExecuteOptions::set_stream(
    perftools::gputools::Stream* stream) {
  stream_ = stream;
  return *this;
}

perftools::gputools::Stream* LocalExecuteOptions::stream() const {
  return stream_;
}

LocalExecuteOptions& LocalExecuteOptions::set_execution_profile(
    ExecutionProfile* profile) {
  profile_ = profile;
  return *this;
}

ExecutionProfile* LocalExecuteOptions::execution_profile() const {
  return profile_;
}

LocalExecuteOptions& LocalExecuteOptions::set_result_layout(
    const Shape& shape_with_layout) {
  has_result_shape_with_layout_ = true;
  result_shape_with_layout_ = shape_with_layout;
  return *this;
}

const Shape* LocalExecuteOptions::result_layout() const {
  return has_result_shape_with_layout_ ? &result_shape_with_layout_ : nullptr;
}

/* static */ StatusOr<std::unique_ptr<LocalService>> LocalService::NewService(
    perftools::gputools::Platform* platform) {
  ServiceOptions default_options;
  default_options.set_platform(platform);
  return NewService(default_options);
}

/* static */ StatusOr<std::unique_ptr<LocalService>> LocalService::NewService(
    const ServiceOptions& options) {
  perftools::gputools::Platform* platform = options.platform();
  if (platform == nullptr) {
    TF_ASSIGN_OR_RETURN(platform, PlatformUtil::GetDefaultPlatform());
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Backend> backend,
      Backend::CreateBackend(platform, options.number_of_replicas()));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Backend> compute_constant_backend,
                      CreateComputeConstantBackend());
  std::unique_ptr<LocalService> service(new LocalService(
      std::move(backend), std::move(compute_constant_backend)));
  return std::move(service);
}

LocalService::LocalService(std::unique_ptr<Backend> execute_backend,
                           std::unique_ptr<Backend> compute_constant_backend)
    : Service(std::move(execute_backend), std::move(compute_constant_backend)) {
  runs_in_client_process_ = true;
}

tensorflow::Status LocalService::ResolveArguments(
    const tensorflow::gtl::ArraySlice<const GlobalDataHandle*> arguments,
    int device_ordinal,
    std::vector<perftools::gputools::DeviceMemoryBase>* argument_ptrs) {
  TF_ASSIGN_OR_RETURN(std::vector<const Allocation*> arg_allocations,
                      ResolveAndValidateArguments(
                          arguments, execute_backend_.get(), device_ordinal));
  argument_ptrs->resize(arg_allocations.size());
  for (int i = 0; i < arguments.size(); ++i) {
    const Allocation& allocation = *arg_allocations[i];
    (*argument_ptrs)[i] = allocation.device_memory();
  }
  return tensorflow::Status::OK();
}

namespace {
// Returns the space required to allocate a shape. If
// allocate_space_for_deep_copy the space includes all sub-buffers of
// a tuple.
int64 RequiredSpace(const Shape& shape, bool allocate_space_for_deep_copy,
                    TransferManager* transfer_manager) {
  int64 size = 0;
  // TODO(b/33492279) remove once no devices represent result tuples as
  // contiguous buffers.
  if (allocate_space_for_deep_copy) {
    TF_CHECK_OK(ShapeUtil::ForEachSubshape(
        shape, [&size, transfer_manager](const Shape& subshape,
                                         const ShapeIndex& /*index*/) {
          size += transfer_manager->GetByteSizeRequirement(subshape);
          return tensorflow::Status::OK();
        }));
  }
  return size;
}
}  // namespace

StatusOr<GlobalDataHandle> LocalService::AllocateBufferOnDevice(
    const Shape& shape, int device_ordinal, bool allocate_space_for_deep_copy) {
  int64 allocation_size = RequiredSpace(shape, allocate_space_for_deep_copy,
                                        execute_backend_->transfer_manager());

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase allocation,
                      execute_backend_->memory_allocator()->Allocate(
                          device_ordinal, allocation_size));

  return allocation_tracker_.Register(
      execute_backend_.get(), device_ordinal, allocation, shape,
      tensorflow::strings::StrCat("AllocateBufferOnDevice of size ",
                                  allocation_size));
}

StatusOr<std::unique_ptr<ShapedBuffer>> LocalService::ExecuteLocally(
    const ComputationHandle& computation,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    const LocalExecuteOptions& options) {
  return ExecuteLocallyInternal(computation, arguments, options,
                                /*preallocated_result_buffer=*/nullptr);
}

tensorflow::Status LocalService::ExecuteLocally(
    const ComputationHandle& computation,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    const LocalExecuteOptions& options, ShapedBuffer* result_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ShapedBuffer> null_buffer,
      ExecuteLocallyInternal(computation, arguments, options, result_buffer));
  // Because the result is written into result_buffer, a null ShapedBuffer
  // pointer should have been returned.
  CHECK_EQ(nullptr, null_buffer.get());
  return tensorflow::Status::OK();
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
LocalService::CompileAheadOfTime(
    const tensorflow::gtl::ArraySlice<AheadOfTimeComputationInstance>
        computations,
    const AotCompilationOptions& options) {
  std::vector<std::unique_ptr<HloModule>> hlo_modules;
  std::vector<std::unique_ptr<HloModuleConfig>> module_configs;
  for (const AheadOfTimeComputationInstance& instance : computations) {
    TF_ASSIGN_OR_RETURN(UserComputation * user_computation,
                        computation_tracker_.Resolve(instance.computation));
    VersionedComputationHandle versioned_handle =
        user_computation->GetVersionedHandle();

    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                        computation_tracker_.BuildHloModule(
                            versioned_handle,
                            /*include_unreachable_instructions=*/true));
    hlo_modules.push_back(std::move(hlo_module));

    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<const ProgramShape> program_shape,
        user_computation->ComputeProgramShape(versioned_handle.version));

    module_configs.push_back(MakeUnique<HloModuleConfig>(*program_shape));
    HloModuleConfig* module_config = module_configs.back().get();
    auto* computation_layout =
        module_config->mutable_entry_computation_layout();
    for (int i = 0; i < instance.argument_layouts.size(); ++i) {
      const Shape& argument_layout = *instance.argument_layouts[i];
      if (ShapeUtil::IsTuple(argument_layout)) {
        return Unimplemented("tuple arguments not supported yet");
      }
      TF_RETURN_IF_ERROR(
          computation_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
              argument_layout));
    }
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_result_layout()->CopyLayoutFromShape(
            *instance.result_layout));
  }

  return execute_backend_->compiler()->CompileAheadOfTime(
      std::move(hlo_modules), std::move(module_configs), MakeHloDumper(),
      options);
}

tensorflow::Status LocalService::ValidateExecuteOptions(
    const ProgramShape& program_shape,
    tensorflow::gtl::ArraySlice<const Shape*> argument_layouts,
    const LocalExecuteOptions& options,
    const ShapedBuffer* preallocated_result_buffer) {
  if (argument_layouts.size() != program_shape.parameters_size()) {
    return InvalidArgument(
        "invalid number of arguments for computation: expected %d, got %zu",
        program_shape.parameters_size(), argument_layouts.size());
  }

  if (options.stream()) {
    if (!options.stream()->ok()) {
      return InvalidArgument("stream is uninitialized or in an error state");
    }

    // Check stream matches service platform.
    const se::Platform* stream_platform =
        options.stream()->parent()->platform();
    if (stream_platform != execute_backend_->platform()) {
      return InvalidArgument(
          "stream is for platform %s, but service targets platform %s",
          stream_platform->Name().c_str(),
          execute_backend_->platform()->Name().c_str());
    }

    // Cannot specify platform or device_ordinal with a stream. The stream
    // determines these values.
    if (options.device_ordinal() >= 0) {
      return InvalidArgument(
          "cannot set both device ordinal and stream options in "
          "LocalExecuteOptions; the stream determines the device ordinal");
    }
    if (options.platform()) {
      return InvalidArgument(
          "cannot set both platform and stream options in "
          "LocalExecuteOptions; the stream determines the platform");
    }
  }
  if (options.platform() &&
      options.platform() != execute_backend_->platform()) {
    return InvalidArgument(
        "service platform (%s) does not match platform set in "
        "LocalExecuteOptions (%s)",
        execute_backend_->platform()->Name().c_str(),
        options.platform()->Name().c_str());
  }

  // TODO(cwhipkey): validate the thread pool provided?

  if (!options.allocator()) {
    return InvalidArgument("an allocator must be provided to ExecuteLocally");
  }

  if (options.allocator()->platform() != execute_backend_->platform()) {
    return InvalidArgument(
        "allocator platform (%s) does not match service platform (%s)",
        options.allocator()->platform()->Name().c_str(),
        execute_backend_->platform()->Name().c_str());
  }

  if (preallocated_result_buffer != nullptr) {
    if (options.result_layout()) {
      return InvalidArgument(
          "cannot set both result ShapedBuffer and result layout; the result "
          "ShapedBuffer determines the result layout");
    }
    if (!ShapeUtil::Compatible(preallocated_result_buffer->shape(),
                               program_shape.result())) {
      return InvalidArgument(
          "result ShapedBuffer of shape %s not compatible with computation "
          "result shape %s",
          ShapeUtil::HumanString(preallocated_result_buffer->shape()).c_str(),
          ShapeUtil::HumanString(program_shape.result()).c_str());
    }
  }
  if (options.result_layout()) {
    TF_RETURN_IF_ERROR(ValidateResultShapeWithLayout(*options.result_layout(),
                                                     program_shape.result()));
  }

  // Check that all argument layouts are valid and the right shape.
  for (int i = 0; i < argument_layouts.size(); ++i) {
    const Shape& argument_shape = *argument_layouts[i];
    TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(argument_shape));
    if (!ShapeUtil::Compatible(argument_shape, program_shape.parameters(i))) {
      return InvalidArgument(
          "invalid argument shape for argument %d, expected %s, got %s", i,
          ShapeUtil::HumanString(program_shape.parameters(i)).c_str(),
          ShapeUtil::HumanString(argument_shape).c_str());
    }
  }

  return tensorflow::Status::OK();
}

StatusOr<std::unique_ptr<ShapedBuffer>> LocalService::ExecuteLocallyInternal(
    const ComputationHandle& computation,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    const LocalExecuteOptions& options,
    ShapedBuffer* preallocated_result_buffer) {
  TF_ASSIGN_OR_RETURN(UserComputation * user_computation,
                      computation_tracker_.Resolve(computation));
  VersionedComputationHandle versioned_handle =
      user_computation->GetVersionedHandle();

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const ProgramShape> program_shape,
      user_computation->ComputeProgramShape(versioned_handle.version));

  // Determine device ordinal the computation will run on.
  int device_ordinal;
  if (options.device_ordinal() >= 0) {
    device_ordinal = options.device_ordinal();
  } else if (options.stream()) {
    device_ordinal = options.stream()->parent()->device_ordinal();
  } else {
    device_ordinal = execute_backend_->default_device_ordinal();
  }

  // Check that all arguments are on the right platform and device ordinal.
  std::vector<const Shape*> argument_layouts(arguments.size());
  for (int i = 0; i < arguments.size(); ++i) {
    auto argument = arguments[i];
    if (argument->platform() != execute_backend_->platform() ||
        argument->device_ordinal() != device_ordinal) {
      return InvalidArgument(
          "computation to run on device %s but argument %d is on "
          "device %s:%d",
          execute_backend_->device_name(device_ordinal).c_str(), i,
          argument->platform()->Name().c_str(), argument->device_ordinal());
    }
    argument_layouts[i] = &argument->shape();
  }

  TF_RETURN_IF_ERROR(ValidateExecuteOptions(
      *program_shape, argument_layouts, options, preallocated_result_buffer));

  // Construct computation layout from the argument layouts.
  auto module_config = MakeUnique<HloModuleConfig>(*program_shape);
  module_config->set_has_hybrid_result(true);
  module_config->set_replica_count(execute_backend_->Replicas().size());
  std::vector<perftools::gputools::DeviceMemoryBase> argument_buffers;
  auto* computation_layout = module_config->mutable_entry_computation_layout();
  for (int i = 0; i < arguments.size(); ++i) {
    const ShapedBuffer* argument = arguments[i];
    if (ShapeUtil::IsTuple(argument->shape())) {
      return Unimplemented("tuple arguments not supported yet");
    }
    argument_buffers.push_back(argument->buffer(/*index=*/{}));
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
            argument->shape()));
  }
  if (options.result_layout()) {
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_result_layout()->CopyLayoutFromShape(
            *options.result_layout()));
  } else if (preallocated_result_buffer != nullptr) {
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_result_layout()->CopyLayoutFromShape(
            preallocated_result_buffer->shape()));
  } else {
    computation_layout->mutable_result_layout()->SetToDefaultLayout();
  }

  ExecutableRunOptions run_options;
  run_options.set_allocator(options.allocator());
  run_options.set_inter_op_thread_pool(
      execute_backend_->inter_op_thread_pool());
  run_options.set_intra_op_thread_pool(
      execute_backend_->eigen_intra_op_thread_pool_device());

  // "stream" owns the stream used for execution if no stream is given.
  Backend::StreamPtr stream;
  if (options.stream()) {
    run_options.set_stream(options.stream());
  } else {
    se::StreamExecutor* stream_executor;
    if (options.device_ordinal() >= 0) {
      TF_ASSIGN_OR_RETURN(
          stream_executor,
          execute_backend_->stream_executor(options.device_ordinal()));
    } else {
      stream_executor = execute_backend_->default_stream_executor();
    }
    TF_ASSIGN_OR_RETURN(stream,
                        execute_backend_->BorrowStream(stream_executor));
    run_options.set_stream(stream.get());
  }

  ExecutionProfile* profile = options.execution_profile();
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<Executable> executable,
      BuildAndCacheExecutable(versioned_handle, std::move(module_config),
                              argument_buffers, execute_backend_.get(),
                              run_options.stream()->parent(), profile));

  if (preallocated_result_buffer == nullptr) {
    return Service::ExecuteOnStreamWrapper<
        StatusOr<std::unique_ptr<ShapedBuffer>>>(
        executable.get(), &run_options, profile,
        [&arguments](Executable* executable,
                     const ExecutableRunOptions* run_options,
                     HloExecutionProfile* hlo_execution_profile) {
          return executable->ExecuteOnStream(run_options, arguments,
                                             hlo_execution_profile);
        });
  } else {
    TF_RETURN_IF_ERROR(Service::ExecuteOnStreamWrapper<tensorflow::Status>(
        executable.get(), &run_options, profile,
        [&arguments, preallocated_result_buffer](
            Executable* executable, const ExecutableRunOptions* run_options,
            HloExecutionProfile* hlo_execution_profile) {
          return executable->ExecuteOnStream(run_options, arguments,
                                             preallocated_result_buffer,
                                             hlo_execution_profile);
        }));
    // To satisfy the return value type, Return a null ShapedBuffer pointer.
    return std::unique_ptr<ShapedBuffer>();
  }
}

StatusOr<std::unique_ptr<Executable>> LocalService::CompileExecutable(
    const ComputationHandle& computation,
    const tensorflow::gtl::ArraySlice<const Shape*> argument_layouts,
    const Shape* result_layout, int device_ordinal, bool has_hybrid_result) {
  TF_ASSIGN_OR_RETURN(UserComputation * user_computation,
                      computation_tracker_.Resolve(computation));
  VersionedComputationHandle versioned_handle =
      user_computation->GetVersionedHandle();

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const ProgramShape> program_shape,
      user_computation->ComputeProgramShape(versioned_handle.version));

  // Validate incoming layouts.
  if (argument_layouts.size() != program_shape->parameters_size()) {
    return InvalidArgument(
        "invalid number of arguments for computation: expected %d, got %zu",
        program_shape->parameters_size(), argument_layouts.size());
  }
  for (int i = 0; i < argument_layouts.size(); ++i) {
    const Shape& argument_shape = *argument_layouts[i];
    TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(argument_shape));
    if (!ShapeUtil::Compatible(argument_shape, program_shape->parameters(i))) {
      return InvalidArgument(
          "invalid argument shape for argument %d, expected %s, got %s", i,
          ShapeUtil::HumanString(program_shape->parameters(i)).c_str(),
          ShapeUtil::HumanString(argument_shape).c_str());
    }
  }
  if (result_layout != nullptr) {
    TF_RETURN_IF_ERROR(
        ValidateResultShapeWithLayout(*result_layout, program_shape->result()));
  }

  // Construct computation layout from the argument layouts.
  auto module_config = MakeUnique<HloModuleConfig>(*program_shape);
  module_config->set_has_hybrid_result(has_hybrid_result);
  module_config->set_replica_count(execute_backend_->Replicas().size());
  legacy_flags::ServiceFlags* flags = legacy_flags::GetServiceFlags();
  if (flags->xla_hlo_profile) {
    module_config->enable_hlo_profiling(true);
  }
  auto* computation_layout = module_config->mutable_entry_computation_layout();
  for (int i = 0; i < argument_layouts.size(); ++i) {
    const Shape& shape = *argument_layouts[i];
    if (ShapeUtil::IsTuple(shape)) {
      return Unimplemented("tuple arguments not supported yet");
    }
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
            shape));
  }
  if (result_layout != nullptr) {
    TF_RETURN_IF_ERROR(
        computation_layout->mutable_result_layout()->CopyLayoutFromShape(
            *result_layout));
  } else {
    computation_layout->mutable_result_layout()->SetToDefaultLayout();
  }

  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                      execute_backend_->stream_executor(device_ordinal));

  std::vector<perftools::gputools::DeviceMemoryBase> argument_buffers(
      argument_layouts.size());
  return BuildExecutable(versioned_handle, std::move(module_config),
                         /*executable_for_compute_constant=*/false,
                         argument_buffers, execute_backend_.get(), executor);
}

}  // namespace xla
