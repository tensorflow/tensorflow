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
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace se = ::perftools::gputools;

namespace xla {

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

    // Dump computation proto state if flag is set.
    legacy_flags::ServiceFlags* flags = legacy_flags::GetServiceFlags();
    const string& directory_path = flags->xla_dump_computations_to;
    if (!directory_path.empty()) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<SessionModule> session_module,
          computation_tracker_.SnapshotComputation(versioned_handle.handle));
      string filename = tensorflow::strings::StrCat(
          "computation_", versioned_handle.handle.handle(), "__",
          session_module->entry().name(), "__version_",
          versioned_handle.version);
      TF_RETURN_IF_ERROR(Executable::DumpToDirectory(directory_path, filename,
                                                     *session_module));
    }

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
    if (flags->xla_hlo_profile) {
      module_config->enable_hlo_profiling(true);
    }
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
