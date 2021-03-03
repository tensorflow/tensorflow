/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_group.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_executable_interface.h"
#include "tensorflow/stream_executor/tpu/tpu_executor.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_id.h"
#include "tensorflow/stream_executor/tpu/tpu_stream.h"

namespace ApiConverter {
static SE_ExecutableRunOptions ToC(
    const xla::ServiceExecutableRunOptions& options) {
  SE_ExecutableRunOptions se_options;
  se_options.allocator = ApiConverter::ToC(options.run_options().allocator());
  se_options.device_ordinal = options.run_options().device_ordinal();
  if (options.run_options().host_to_device_stream() != nullptr) {
    se_options.host_to_device_stream =
        static_cast<tensorflow::tpu::TpuStream*>(
            options.run_options().host_to_device_stream()->implementation())
            ->se_stream();
  } else {
    se_options.host_to_device_stream = nullptr;
  }

  if (options.run_options().device_assignment() != nullptr) {
    xla::DeviceAssignmentProto dev_assign_proto;
    options.run_options()
        .device_assignment()
        ->Serialize(&dev_assign_proto)
        .IgnoreError();
    se_options.device_assignment =
        stream_executor::tpu::SerializeProto(dev_assign_proto);
  } else {
    se_options.device_assignment.bytes = nullptr;
    se_options.device_assignment.size = 0;
  }

  se_options.rng_seed = options.run_options().rng_seed();
  se_options.run_id = options.run_options().run_id().ToInt();
  se_options.launch_id = options.run_options().launch_id();

  CHECK_EQ(options.run_options().then_execute_function(), nullptr)
      << "ThenExecuteFunction not supported by this platform.";

  auto impl =
      const_cast<stream_executor::Stream*>(options.stream())->implementation();
  se_options.stream =
      static_cast<tensorflow::tpu::TpuStream*>(impl)->se_stream();
  return se_options;
}
}  // namespace ApiConverter

namespace xla {

namespace {

using ::tensorflow::tpu::ExecutorApiFn;

class TpuExecutable : public TpuExecutableInterface {
 public:
  TpuExecutable(SE_Executable* se_executable,
                std::shared_ptr<HloModule> hlo_module)
      : TpuExecutableInterface(std::move(hlo_module)),
        se_executable_(se_executable) {}

  ~TpuExecutable() override {
    ExecutorApiFn()->TpuExecutable_FreeFn(se_executable_);
  }

  StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile) override {
    SE_ExecutableRunOptions se_run_options = ApiConverter::ToC(*run_options);
    SE_ExecutionInput** se_args = new SE_ExecutionInput*[arguments.size()];
    for (int i = 0; i < arguments.size(); ++i) {
      auto& arg = arguments[i];
      se_args[i] = new SE_ExecutionInput;

      ApiConverter::ToC(arg.shape(), &se_args[i]->shape_tree.shape);
      auto* arg_buffers = arg.MutableBuffers();
      absl::InlinedVector<SE_MaybeOwningDeviceMemory, 2> se_buffers;
      for (auto& pair : *arg_buffers) {
        bool aliased = arg.unowned_indices().count(pair.first) > 0;
        se_buffers.push_back(ApiConverter::ToC(pair.second, aliased));
      }
      se_args[i]->shape_tree.buffers =
          new SE_MaybeOwningDeviceMemory[se_buffers.size()];
      for (int j = 0; j < se_buffers.size(); ++j) {
        se_args[i]->shape_tree.buffers[j] = se_buffers[j];
      }

      ApiConverter::ToC(arg.shape(), &se_args[i]->dynamic_shape);
      const auto& unowned_indices = arg.unowned_indices();
      se_args[i]->unowned_indices_size = unowned_indices.size();
      se_args[i]->unowned_indices = new XLA_ShapeIndex[unowned_indices.size()];
      int j = 0;
      for (auto& idx : unowned_indices) {
        se_args[i]->unowned_indices[j] = ApiConverter::ToC(idx);
        ++j;
      }
    }
    SE_ExecutionOutput se_execution_output;
    StatusHelper status;
    ExecutorApiFn()->TpuExecutable_ExecuteAsyncOnStreamFn(
        se_executable_, &se_run_options, se_args, arguments.size(), nullptr,
        &se_execution_output, status.c_status);

    if (se_run_options.device_assignment.bytes != nullptr) {
      stream_executor::tpu::SerializedProto_Free(
          se_run_options.device_assignment);
    }
    for (int i = 0; i < arguments.size(); ++i) {
      ApiConverter::Free(&se_args[i]->shape_tree.shape);
      ApiConverter::Free(&se_args[i]->dynamic_shape);
      delete[] se_args[i]->unowned_indices;
      delete[] se_args[i]->shape_tree.buffers;
      delete se_args[i];
    }
    delete[] se_args;

    if (!status.ok()) {
      return status.status();
    }

    xla::ScopedShapedBuffer result(
        ApiConverter::FromC(&se_execution_output.result),
        run_options->stream()->parent()->GetAllocator());
    ApiConverter::Free(&se_execution_output.result);

    ExecutionOutput output(std::move(result));
    for (int i = 0; i < se_execution_output.aliased_indices_size; ++i) {
      output.AddAliasedIndex(
          ApiConverter::FromC(&se_execution_output.aliased_indices[i]));
    }
    ExecutorApiFn()->TpuExecutable_FreeXlaShapeIndexArrayFn(
        se_execution_output.aliased_indices);

    for (int i = 0; i < se_execution_output.to_be_released_size; ++i) {
      output.AddToBeReleased(
          ApiConverter::FromC(&se_execution_output.to_be_released[i],
                              run_options->stream()->parent()->GetAllocator())
              .Release()
              .value());
    }
    ExecutorApiFn()->TpuExecutable_FreeMaybeOwningDeviceMemoryArrayFn(
        se_execution_output.to_be_released);

    return output;
  }

  absl::string_view fingerprint() const override {
    const char* data;
    size_t size;
    ExecutorApiFn()->TpuExecutable_FingerprintFn(se_executable_, &data, &size);
    return absl::string_view(data, size);
  }

 private:
  Status LoadProgramAndEnqueueToStream(
      const ServiceExecutableRunOptions& run_options,
      absl::Span<const stream_executor::DeviceMemoryBase> arguments,
      stream_executor::DeviceMemoryBase result,
      absl::optional<stream_executor::DeviceMemoryBase>
          cross_program_prefetch_addr) override {
    LOG(FATAL) << "LoadProgramAndEnqueueToStream unimplemented";
  }

  Shape HostShapeToDeviceShape(const Shape& host_shape) override {
    LOG(FATAL) << "HostShapeToDeviceShape unimplemented";
  }

  int64 ShapeSize(const Shape& shape) override {
    LOG(FATAL) << "ShapeSize unimplemented";
  }

  SE_Executable* se_executable_;
};

class TpuCompiler : public Compiler {
 public:
  TpuCompiler() { compiler_ = ExecutorApiFn()->TpuCompiler_NewFn(); }
  ~TpuCompiler() override { ExecutorApiFn()->TpuCompiler_FreeFn(compiler_); }

  stream_executor::Platform::Id PlatformId() const override {
    return tensorflow::tpu::GetTpuPlatformId();
  }

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module,
      stream_executor::StreamExecutor* executor,
      const CompileOptions& options) override {
    XLA_HloModule hlo_module;
    auto cleanup = xla::MakeCleanup([&hlo_module]() {
      stream_executor::tpu::SerializedProto_Free(hlo_module.proto);
      ApiConverter::Free(&hlo_module.module_config);
    });
    hlo_module.module_config = ApiConverter::ToC(module->config());
    hlo_module.proto = stream_executor::tpu::SerializeProto(module->ToProto());
    auto allocator = ApiConverter::ToC(options.device_allocator);
    XLA_HloModule result;
    StatusHelper status;
    ExecutorApiFn()->TpuCompiler_RunHloPassesFn(
        compiler_, &hlo_module,
        static_cast<tensorflow::tpu::TpuExecutor*>(executor->implementation())
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

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module,
      stream_executor::StreamExecutor* executor,
      const CompileOptions& options) override {
    XLA_HloModule hlo_module;
    auto cleanup = xla::MakeCleanup([&hlo_module]() {
      stream_executor::tpu::SerializedProto_Free(hlo_module.proto);
      ApiConverter::Free(&hlo_module.module_config);
    });
    SE_Executable* result;
    hlo_module.module_config = ApiConverter::ToC(module->config());
    hlo_module.proto = stream_executor::tpu::SerializeProto(module->ToProto());
    auto allocator = ApiConverter::ToC(options.device_allocator);

    StatusHelper status;
    ExecutorApiFn()->TpuCompiler_RunBackendFn(
        compiler_, &hlo_module,
        static_cast<tensorflow::tpu::TpuExecutor*>(executor->implementation())
            ->se_executor(),
        &allocator, &result, status.c_status);
    if (!status.ok()) {
      return status.status();
    }

    std::unique_ptr<Executable> exec =
        absl::make_unique<TpuExecutable>(result, std::move(module));
    return exec;
  }

  StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
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
        xla::MakeCleanup([&se_module_group, module_group_size]() {
          for (auto i = 0; i < module_group_size; ++i) {
            ApiConverter::Free(&se_module_group.module_config[i]);
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
        se_lists[i].exec[j] = static_cast<tensorflow::tpu::TpuExecutor*>(
                                  stream_exec[i][j]->implementation())
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
      auto cleanup_c_module =
          xla::MakeCleanup([&c_module]() { ApiConverter::Free(&c_module); });
      TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                          ApiConverter::FromC(c_module));
      std::shared_ptr<HloModule> module_shared(module.release());
      executables.emplace_back(absl::make_unique<TpuExecutable>(
          se_executables[i], std::move(module_shared)));
    }

    stream_executor::tpu::SerializedProto_Free(se_module_group.proto);
    delete[] se_executables;

    return executables;
  }

  // Compiles the HLO module group for ahead-of-time execution.  This is
  // intended for use in static compilation.
  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
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
      int64 bytes =
          ExecutorApiFn()->TpuCompiler_ShapeSizeFn(compiler_, &c_shape);
      ApiConverter::Free(&c_shape);
      return bytes;
    };
  }

 private:
  Tpu_Compiler* compiler_;
};

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      tensorflow::tpu::GetTpuPlatformId(),
      []() { return absl::make_unique<TpuCompiler>(); });
  return true;
}

static bool module_initialized = InitModule();

}  // namespace
}  // namespace xla
