/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_GPU_AOT_COMPILATION_RESULT_H_
#define XLA_SERVICE_GPU_GPU_AOT_COMPILATION_RESULT_H_

#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "google/protobuf/arena.h"
#include "riegeli/bytes/reader.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiled_module.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"

namespace xla::gpu {

namespace internal {
struct ArenaAllocatedGpuExecutableProto {
  ArenaAllocatedGpuExecutableProto(
      std::unique_ptr<google::protobuf::Arena> absl_nonnull arena,
      GpuExecutableProto* absl_nonnull proto)
      : arena(std::move(arena)), proto(proto) {}

  std::unique_ptr<google::protobuf::Arena> absl_nonnull arena;
  GpuExecutableProto* absl_nonnull proto;
};
}  // namespace internal

// `AotCompilationResult` implementation for GPU, containing a serialized
// `GpuExecutable`.
//
// Unlike `LegacyGpuAotCompilationResult`, this result contains the entire
// optimized executable, including the Thunks, as opposed to just the optimized
// HLO.
class GpuAotCompilationResult : public CompiledModule {
 public:
  static absl::StatusOr<std::unique_ptr<GpuAotCompilationResult>> FromProto(
      GpuExecutableProto executable_proto);

  // Creates a `GpuAotCompilationResult` from a serialized result (i.e. as it
  // would be returned by `SerializeAsString`).
  static absl::StatusOr<std::unique_ptr<GpuAotCompilationResult>>
  FromSerialized(std::unique_ptr<riegeli::Reader> reader);

  absl::StatusOr<std::string> SerializeAsString() const final;

  absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable() && final {
    return absl::UnimplementedError(
        "LoadExecutable without parameters not supported");
  }

  absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      se::Platform::Id platform_id,
      const se::DeviceDescription& device_description) &&
      final;

  const HloModule* optimized_module() const final { return hlo_module_.get(); }

  std::shared_ptr<HloModule> shared_optimized_module() final {
    return hlo_module_;
  }

 private:
  const GpuExecutableProto& GetExecutableProto() const;

  explicit GpuAotCompilationResult(
      std::variant<internal::ArenaAllocatedGpuExecutableProto,
                   GpuExecutableProto>
          gpu_executable_proto,
      std::shared_ptr<HloModule> hlo_module)
      : gpu_executable_proto_(std::move(gpu_executable_proto)),
        hlo_module_(std::move(hlo_module)) {}

  std::variant<internal::ArenaAllocatedGpuExecutableProto, GpuExecutableProto>
      gpu_executable_proto_;
  std::shared_ptr<HloModule> hlo_module_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_GPU_AOT_COMPILATION_RESULT_H_
