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

#ifndef XLA_SERVICE_GPU_LEGACY_GPU_AOT_COMPILATION_RESULT_H_
#define XLA_SERVICE_GPU_LEGACY_GPU_AOT_COMPILATION_RESULT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// Represents the legacy result of a GPU AOT compilation.
//
// This result primarily contains the optimized HLO module. Executables loaded
// from this result can bypass the HLO optimization passes, since this result
// already contains the optimized HLO.
//
// This class is considered legacy and is expected to be replaced by a
// new AOT result type as part of the runtime split. The new type will
// encapsulate the compilation up to the Thunks generation stage.
class LegacyGpuAotCompilationResult : public AotCompilationResult {
 public:
  static absl::StatusOr<std::unique_ptr<LegacyGpuAotCompilationResult>>
  FromModule(const HloModule* hlo_module,
             const BufferAssignment* buffer_assignment,
             absl::string_view asm_text, absl::Span<const uint8_t> binary,
             const BinaryMap& dnn_compiled_graphs, int pointer_size);

  static absl::StatusOr<std::unique_ptr<LegacyGpuAotCompilationResult>>
  FromString(const std::string& serialized, int pointer_size);

  absl::StatusOr<std::string> SerializeAsString() const override;

  absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      Compiler* compiler, const se::StreamExecutor* stream_exec) &&
      override;

  const HloModule* optimized_module() const override { return module_.get(); }
  std::unique_ptr<HloModule> consume_optimized_module() override {
    return std::move(module_);
  }

  absl::StatusOr<std::unique_ptr<BufferAssignment>> buffer_assignment()
      const override;

  const GpuExecutableProto& GetGpuExecutableProto() const { return proto_; }

 private:
  LegacyGpuAotCompilationResult(std::unique_ptr<HloModule> module,
                                GpuExecutableProto proto, int pointer_size)
      : module_(std::move(module)),
        proto_(std::move(proto)),
        pointer_size_(pointer_size) {}

  std::unique_ptr<HloModule> module_;
  GpuExecutableProto proto_;
  int pointer_size_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_LEGACY_GPU_AOT_COMPILATION_RESULT_H_
