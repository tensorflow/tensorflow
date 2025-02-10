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

#include "xla/service/local_service.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/service/backend.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_layout.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/local_service_utils.h"
#include "xla/service/platform_util.h"
#include "xla/service/service.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {

/* static */ absl::StatusOr<std::unique_ptr<LocalService>>
LocalService::NewService(const ServiceOptions& options) {
  se::Platform* platform = options.platform();
  if (platform == nullptr) {
    TF_ASSIGN_OR_RETURN(platform, PlatformUtil::GetDefaultPlatform());
  }

  BackendOptions backend_options;
  backend_options.set_platform(platform)
      .set_intra_op_parallelism_threads(options.intra_op_parallelism_threads())
      .set_allowed_devices(options.allowed_devices());

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Backend> backend,
                      Backend::CreateBackend(backend_options));

  std::unique_ptr<LocalService> service(
      new LocalService(options, std::move(backend)));
  return std::move(service);
}

LocalService::LocalService(const ServiceOptions& options,
                           std::unique_ptr<Backend> execute_backend)
    : Service(options, std::move(execute_backend)) {}

absl::StatusOr<std::vector<std::unique_ptr<Executable>>>
LocalService::CompileExecutables(
    const XlaComputation& computation,
    const absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& build_options) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> module_config,
      GetHloModuleConfig(computation, argument_layouts, build_options,
                         &options_, execute_backend_.get()));

  VLOG(3) << "Computation Layout: "
          << module_config->entry_computation_layout().ToString();

  TF_ASSIGN_OR_RETURN(
      se::StreamExecutor * executor,
      execute_backend_->stream_executor(build_options.device_ordinal()));

  // TODO(cjfj): Investigate why there are a couple of test failures when the
  // single partition computations are built using `BuildExecutables`, fix it,
  // and remove this special case (provided the performance if similar).
  const Compiler::CompileOptions compile_options{
      build_options.device_allocator(),
      build_options.compile_thread_pool(),
      build_options.layout_canonicalization_callback(),
      false,
      {},
      {build_options.key_value_store(), build_options.process_index(),
       build_options.process_count()}};
  if (build_options.num_partitions() == 1) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<Executable> executable,
        BuildExecutable(computation.proto(), std::move(module_config),
                        execute_backend_.get(), executor, compile_options,
                        build_options.run_backend_only()));
    std::vector<std::unique_ptr<Executable>> executables;
    executables.push_back(std::move(executable));
    return executables;
  } else {
    std::vector<std::unique_ptr<HloModuleConfig>> module_configs;
    module_configs.push_back(std::move(module_config));
    // BuildExecutables uses the executors length to determine the number of
    // cores per module, but otherwise only uses the first executor.
    std::vector<se::StreamExecutor*> executors(build_options.num_partitions(),
                                               executor);

    return BuildExecutables(
        /*module_protos=*/{&computation.proto()}, std::move(module_configs),
        execute_backend_.get(), {executors}, compile_options,
        build_options.run_backend_only());
  }
}

absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
LocalService::CompileAotResults(
    const XlaComputation& computation,
    const absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& build_options) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> module_config,
      GetHloModuleConfig(computation, argument_layouts, build_options,
                         &options_, execute_backend_.get()));

  TF_ASSIGN_OR_RETURN(
      se::StreamExecutor * executor,
      execute_backend_->stream_executor(build_options.device_ordinal()));

  std::vector<std::unique_ptr<HloModuleConfig>> module_configs;
  module_configs.push_back(std::move(module_config));
  // BuildAotResults uses the executors length to determine the number of
  // cores per module, but otherwise only uses the first executor.
  std::vector<se::StreamExecutor*> executors(build_options.num_partitions(),
                                             executor);

  return BuildAotResults(
      /*module_protos=*/{&computation.proto()}, std::move(module_configs),
      execute_backend_.get(), {executors},
      Compiler::CompileOptions{build_options.device_allocator(),
                               build_options.compile_thread_pool()},
      build_options.run_backend_only());
}

absl::StatusOr<int> LocalService::ReplicaNumberToDeviceOrdinal(
    int replica_number) {
  return backend().computation_placer()->DeviceId(
      replica_number, /*computation=*/0, options_.number_of_replicas(),
      /*computation_count=*/1);
}

absl::StatusOr<const ShapedBuffer*> LocalService::GlobalDataToShapedBuffer(
    const GlobalDataHandle& data, int replica_number) {
  TF_ASSIGN_OR_RETURN(auto buffers, allocation_tracker_.Resolve(data));
  if (replica_number >= buffers.size()) {
    return InvalidArgument(
        "replica_number %d out of range; must be less than num_replicas = %u.",
        replica_number, buffers.size());
  }
  return buffers[replica_number];
}

absl::StatusOr<GlobalDataHandle> LocalService::RegisterReplicatedBuffers(
    std::vector<ScopedShapedBuffer> replicated_buffers,
    const std::string& tag) {
  return allocation_tracker_.RegisterReplicatedBuffers(
      std::move(replicated_buffers), tag);
}

}  // namespace xla
