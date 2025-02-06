/* Copyright 2020 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/tpu/c_api_conversions.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/proto_helper.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_executable.h"
#include "xla/stream_executor/tpu/tpu_executor.h"
#include "xla/stream_executor/tpu/tpu_executor_api.h"
#include "xla/stream_executor/tpu/tpu_executor_c_api.h"
#include "xla/stream_executor/tpu/tpu_platform_id.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

using ::stream_executor::tpu::ExecutorApiFn;

class TpuCompiler : public Compiler {
 public:
  TpuCompiler() { compiler_ = ExecutorApiFn()->TpuCompiler_NewFn(); }
  ~TpuCompiler() override { ExecutorApiFn()->TpuCompiler_FreeFn(compiler_); }

  stream_executor::Platform::Id PlatformId() const override {
    return tensorflow::tpu::GetTpuPlatformId();
  }

  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module,
      stream_executor::StreamExecutor* executor,
      const CompileOptions& options) override {
    XLA_HloModule hlo_module;
    auto cleanup = absl::MakeCleanup([&hlo_module]() {
      stream_executor::tpu::SerializedProto_Free(hlo_module.proto);
      ApiConverter::Destroy(&hlo_module.module_config);
    });
    hlo_module.module_config = ApiConverter::ToC(module->config());
    hlo_module.proto = stream_executor::tpu::SerializeProto(module->ToProto());
    auto allocator = ApiConverter::ToC(options.device_allocator);
    XLA_HloModule result;
    StatusHelper status;
    ExecutorApiFn()->TpuCompiler_RunHloPassesFn(
        compiler_, &hlo_module,
        static_cast<stream_executor::tpu::TpuExecutor*>(executor)
            ->se_executor(),
        &allocator, &result, status.c_status);
    if (!status.ok()) {
      return status.status();
    }
    HloModuleProto result_proto =
        stream_executor::tpu::DeserializeProto<HloModuleProto>(result.proto);
    stream_executor::tpu::SerializedProto_Free(result.proto);
    return HloModule::CreateFromProto(result_proto, module->config());
  }

  absl::StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module,
      stream_executor::StreamExecutor* executor,
      const CompileOptions& options) override {
    XLA_HloModule hlo_module;
    auto cleanup = absl::MakeCleanup([&hlo_module]() {
      stream_executor::tpu::SerializedProto_Free(hlo_module.proto);
      ApiConverter::Destroy(&hlo_module.module_config);
    });
    SE_Executable* result;
    hlo_module.module_config = ApiConverter::ToC(module->config());
    hlo_module.proto = stream_executor::tpu::SerializeProto(module->ToProto());
    auto allocator = ApiConverter::ToC(options.device_allocator);

    StatusHelper status;
    ExecutorApiFn()->TpuCompiler_RunBackendFn(
        compiler_, &hlo_module,
        static_cast<stream_executor::tpu::TpuExecutor*>(executor)
            ->se_executor(),
        &allocator, &result, status.c_status);
    if (!status.ok()) {
      return status.status();
    }

    std::unique_ptr<Executable> exec =
        std::make_unique<TpuExecutable>(result, std::move(module));
    return exec;
  }

  absl::StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<stream_executor::StreamExecutor*>> stream_exec,
      const CompileOptions& options) override {
    XLA_HloModuleGroup se_module_group;
    se_module_group.proto =
        stream_executor::tpu::SerializeProto(module_group->ToProto());
    se_module_group.module_config =
        new XLA_HloModuleConfig[module_group->size()];
    int module_group_size = module_group->size();
    auto cleanup_config =
        absl::MakeCleanup([&se_module_group, module_group_size]() {
          for (auto i = 0; i < module_group_size; ++i) {
            ApiConverter::Destroy(&se_module_group.module_config[i]);
          }
          delete[] se_module_group.module_config;
        });
    for (int i = 0; i < module_group->size(); ++i) {
      const auto& config = module_group->module(i).config();
      se_module_group.module_config[i] = ApiConverter::ToC(config);
    }
    std::vector<SE_StreamExecutorList> se_lists(stream_exec.size());
    std::vector<std::vector<SE_StreamExecutor*>> se_lists_storage;
    for (int i = 0; i < stream_exec.size(); ++i) {
      se_lists[i].count = stream_exec[i].size();
      se_lists_storage.emplace_back(stream_exec[i].size());
      se_lists[i].exec = se_lists_storage.back().data();
      for (int j = 0; j < stream_exec[i].size(); ++j) {
        se_lists[i].exec[j] =
            static_cast<stream_executor::tpu::TpuExecutor*>(stream_exec[i][j])
                ->se_executor();
      }
    }

    SE_DeviceMemoryAllocator allocator =
        ApiConverter::ToC(options.device_allocator);

    SE_Executable** se_executables = new SE_Executable*[module_group->size()];

    StatusHelper status;

    ExecutorApiFn()->TpuCompiler_CompileFn(
        compiler_, &se_module_group, se_lists.data(), stream_exec.size(),
        &allocator, se_executables, status.c_status);

    if (!status.ok()) {
      return status.status();
    }

    std::vector<std::unique_ptr<Executable>> executables;
    for (int i = 0; i < module_group->size(); ++i) {
      // We get the HloModule from the compiled executable, rather than reusing
      // the input module from 'module_group', in case the module changed in
      // some way. For example, if the computation is automatically partitioned
      // via XLA, the executable's module may have different input/output shapes
      // than the input module.
      XLA_HloModule c_module =
          ExecutorApiFn()->TpuExecutable_HloModuleFn(se_executables[i]);
      auto cleanup_c_module = absl::MakeCleanup(
          [&c_module]() { ApiConverter::Destroy(&c_module); });
      TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                          ApiConverter::FromC(c_module));
      std::shared_ptr<HloModule> module_shared(module.release());
      executables.emplace_back(std::make_unique<TpuExecutable>(
          se_executables[i], std::move(module_shared)));
    }

    stream_executor::tpu::SerializedProto_Free(se_module_group.proto);
    delete[] se_executables;

    return executables;
  }

  // Compiles the HLO module group for ahead-of-time execution.  This is
  // intended for use in static compilation.
  absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& options) override {
    return Unimplemented("This compiler does not support CompileAheadOfTime.");
  }

  // Returns a function that computes the size in bytes of the logical
  // buffer that contains a shape.
  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override {
    return [this](const xla::Shape& shape) {
      XLA_Shape c_shape;
      ApiConverter::ToC(shape, &c_shape);
      int64_t bytes =
          ExecutorApiFn()->TpuCompiler_ShapeSizeFn(compiler_, &c_shape);
      ApiConverter::Destroy(&c_shape);
      return bytes;
    };
  }

  Shape DefaultDeviceShapeRepresentation(const Shape& shape) const override {
    XLA_Shape host_shape, device_shape;
    ApiConverter::ToC(shape, &host_shape);
    ExecutorApiFn()->TpuCompiler_DefaultDeviceShapeRepresentationFn(
        compiler_, &host_shape, &device_shape);
    ApiConverter::Destroy(&host_shape);
    Shape result = ApiConverter::FromC(&device_shape);
    ApiConverter::Destroy(&device_shape);
    return result;
  }

 private:
  Tpu_Compiler* compiler_;
};

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      tensorflow::tpu::GetTpuPlatformId(),
      []() { return std::make_unique<TpuCompiler>(); });
  return true;
}

static bool module_initialized = InitModule();

}  // namespace
}  // namespace xla
