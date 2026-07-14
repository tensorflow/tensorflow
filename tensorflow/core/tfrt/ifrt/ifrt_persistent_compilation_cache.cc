/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/ifrt/ifrt_persistent_compilation_cache.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace tensorflow {
namespace ifrt_serving {

absl::StatusOr<xla::ifrt::LoadedExecutableRef>
IfrtPersistentCompilationCache::LookupLoadedExecutableOrCreate(
    std::unique_ptr<xla::ifrt::HloProgram> hlo_program,
    xla::ifrt::DeviceListRef device_list,
    const xla::CompileOptions& xla_compile_options,
    const std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>&
        loaded_host_callbacks,
    xla::ifrt::Client* client,
    absl::AnyInvocable<absl::StatusOr<xla::ifrt::LoadedExecutableRef>(
        std::unique_ptr<xla::ifrt::Program> program,
        std::unique_ptr<xla::ifrt::CompileOptions> options)>
        value_fn) {
  // No persistent cache implemented, compile directly.
  auto ifrt_xla_compile_options =
      std::make_unique<xla::ifrt::XlaCompileOptions>(
          xla_compile_options, std::move(device_list), loaded_host_callbacks);
  return value_fn(std::move(hlo_program), std::move(ifrt_xla_compile_options));
}

absl::StatusOr<Tf2HloResult>
IfrtPersistentCompilationCache::LookupTf2HloResultOrCreate(
    Tf2HloArg tf2hlo_arg, TfToHloCompiler* tf_to_hlo_compiler) {
  // No tf2xla persistent cache is implemented, compile directly.
  return tf_to_hlo_compiler->CompileTfToHlo(tf2hlo_arg);
}

}  // namespace ifrt_serving
}  // namespace tensorflow
