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
#ifndef TENSORFLOW_CORE_TFRT_IFRT_IFRT_PERSISTENT_COMPILATION_CACHE_H_
#define TENSORFLOW_CORE_TFRT_IFRT_IFRT_PERSISTENT_COMPILATION_CACHE_H_

#include <memory>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/program.h"
#include "xla/tsl/concurrency/ref_count.h"
namespace tensorflow {
namespace ifrt_serving {

class IfrtPersistentCompilationCache {
 public:
  IfrtPersistentCompilationCache() = default;
  virtual ~IfrtPersistentCompilationCache() = default;

  // The implementation of this API should be thread-safe. It generates a key
  // for looking up the executable in the persistent cache and it will return
  // the LoadedExecutable if hits cache. Otherwise, it will call the `value_fn`
  // to generate and return the LoadedExecutable.
  virtual absl::StatusOr<std::unique_ptr<xla::ifrt::LoadedExecutable>>
  LookupLoadedExecutableOrCreate(
      std::unique_ptr<xla::ifrt::HloProgram> hlo_program,
      tsl::RCReference<xla::ifrt::DeviceList> device_list,
      const xla::CompileOptions& xla_compile_options,
      const std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>&
          loaded_host_callbacks,
      xla::ifrt::Client* client,
      absl::AnyInvocable<
          absl::StatusOr<std::unique_ptr<xla::ifrt::LoadedExecutable>>(
              std::unique_ptr<xla::ifrt::Program> program,
              std::unique_ptr<xla::ifrt::CompileOptions> options)>
          value_fn);

  // The implementation of this API should be thread-safe. It generates a key
  // for looking up the Tf2HloResult in the persistent cache and it will return
  // the Tf2HloResult if hits cache. Otherwise, it will call the `value_fn` to
  // generate and return the Tf2HloResult.
  virtual absl::StatusOr<Tf2HloResult> LookupTf2HloResultOrCreate(
      mlir::ModuleOp mlir_module, absl::string_view main_func,
      absl::Span<const DtypeAndShape> dtypes_and_shapes,
      tsl::RCReference<xla::ifrt::DeviceList> device_list,
      xla::ifrt::Client* client,
      absl::AnyInvocable<absl::StatusOr<Tf2HloResult>()> value_fn);

  virtual bool IsXlaCompilationCacheEnabled() const { return false; }
  virtual bool IsTf2HloCompilationCacheEnabled() const { return false; }
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_PERSISTENT_COMPILATION_CACHE_H_
