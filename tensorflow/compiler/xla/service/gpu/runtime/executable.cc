/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime/executable.h"

#include <memory>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "tensorflow/compiler/xla/mlir/transforms/runtime/compilation_pipeline_gpu.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/runtime/jit_executable.h"

namespace xla {
namespace gpu {

using ::xla::runtime::Executable;
using ::xla::runtime::JitExecutable;

GpuRuntimeExecutable::GpuRuntimeExecutable(
    std::vector<int64_t> buffer_sizes,
    std::unique_ptr<JitExecutable> jit_executable, DebugOptions debug_options)
    : buffer_sizes_(std::move(buffer_sizes)),
      executable_(std::move(jit_executable)),
      debug_options_(std::move(debug_options)) {}

GpuRuntimeExecutable::GpuRuntimeExecutable(
    std::vector<int64_t> buffer_sizes,
    std::unique_ptr<Executable> aot_executable, DebugOptions debug_options)
    : buffer_sizes_(std::move(buffer_sizes)),
      executable_(std::move(aot_executable)),
      debug_options_(std::move(debug_options)) {}

//===---------------------------------------------------------------------===///
// Compile Xla program lowered to runtime dialects to Gpu runtime executable.
//===---------------------------------------------------------------------===///

/*static*/ StatusOr<std::unique_ptr<GpuRuntimeExecutable>>
GpuRuntimeExecutable::Create(std::unique_ptr<GpuRuntimeProgram> program) {
  // Options for the default XLA Runtim compilation pipeline.
  runtime::CompilationPipelineOptions copts;

  // Populate mapping from XLA (SE) enums/structs type id to symbol names.
  copts.populate_type_id_names = PopulateXlaGpuTypeIdNames;

  // For passing LMHLO attributes as XLA (SE) enums/structs to custom calls.
  copts.populate_attr_encodings = PopulateLmhloToXlaAttrEncoding;

  // Options for constructing XLA runtime JitExecutable.
  JitExecutable::Options opts;
  opts.specialization = JitExecutable::Specialization::kDisabled;
  opts.compiler.register_dialects =
      runtime::RegisterDefaultXlaGpuRuntimeDialects;

  // Register XLA Gpu runtime custom calls with the linker.
  opts.compiler.symbols_binding = runtime::ToSymbolsBinding(
      PopulateXlaGpuCustomCalls, PopulateXlaGpuTypeIdNames);

  // We just use the default compilation pipeline provided by the XLA runtime.
  // Alternatively instead of having a separate Xla Runtime program (LMHLO
  // lowered to canonical dialects), we can assemble a pipeline that will
  // compile starting from the LMHLO dialect. However this intermediate step
  // helps with debugging, by materializing IR with XLA runtime custom calls.
  opts.compiler.create_compilation_pipeline =
      [copts](xla::runtime::PassManager& passes) {
        runtime::CreateDefaultXlaGpuRuntimeCompilationPipeline(passes, copts);
      };

  // TODO(b/241296710): LLVM optimizations interact badly with the memory
  // loads and stores pattern generated in very large XLA programs, and can
  // take minutes to run. Currently we do not expect any expensive code
  // running on the host, so we can safely disable optimization passes.
  opts.compiler.jit_code_opt_level = llvm::CodeGenOpt::None;

  // Instantiate new JitExecutable from the MLIR source.
  auto jit_executable =
      JitExecutable::Instantiate(program->module, program->entry_point, opts);
  if (!jit_executable.ok())
    return InternalError("Failed to compile XLA Runtime program: %s",
                         jit_executable.status().message());

  return std::unique_ptr<GpuRuntimeExecutable>(new GpuRuntimeExecutable(
      std::move(program->buffer_sizes),
      std::make_unique<JitExecutable>(std::move(*jit_executable)),
      std::move(program->debug_options)));
}

//===---------------------------------------------------------------------===///
// Constructs Gpu runtime executable from AOT compiled runtime artifact.
//===---------------------------------------------------------------------===///

/*static*/ StatusOr<std::unique_ptr<GpuRuntimeExecutable>>
GpuRuntimeExecutable::Create(absl::Span<const int64_t> buffer_sizes,
                             Executable executable,
                             DebugOptions debug_options) {
  return std::unique_ptr<GpuRuntimeExecutable>(new GpuRuntimeExecutable(
      std::vector<int64_t>(buffer_sizes.begin(), buffer_sizes.end()),
      std::make_unique<Executable>(std::move(executable)),
      std::move(debug_options)));
}

//===---------------------------------------------------------------------===///

Executable& GpuRuntimeExecutable::executable() {
  if (auto* jit = std::get_if<std::unique_ptr<JitExecutable>>(&executable_)) {
    return *(*jit)->DefaultExecutable();
  }
  return *std::get<std::unique_ptr<Executable>>(executable_);
}

StatusOr<std::string_view> GpuRuntimeExecutable::GetObjFile() const {
  const auto* jit = std::get_if<std::unique_ptr<JitExecutable>>(&executable_);
  if (!jit) return InternalError("ObjFile is not available");

  if (auto obj_file = (*jit)->DefaultExecutable()->obj_file())
    return std::string_view(obj_file->getBuffer());

  return InternalError("gpu runtime executable didn't save the obj file");
}

StatusOr<std::string_view> GpuRuntimeExecutable::GetMlirModule() const {
  const auto* jit = std::get_if<std::unique_ptr<JitExecutable>>(&executable_);
  if (!jit) return InternalError("MLIR module is not available");

  return (*jit)->mlir_module();
}

}  // namespace gpu
}  // namespace xla
