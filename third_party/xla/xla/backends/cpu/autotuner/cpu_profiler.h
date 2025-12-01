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

#ifndef XLA_BACKENDS_CPU_AUTOTUNER_CPU_PROFILER_H_
#define XLA_BACKENDS_CPU_AUTOTUNER_CPU_PROFILER_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/literal.h"
#include "xla/service/executable.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/shaped_buffer.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

struct LiteralBackedCpuBuffers : public InputBuffers {
  std::vector<Literal> backing_literals;
  std::vector<MaybeOwningDeviceAddress> buffers;
};

class CpuProfiler : public Profiler {
 public:
  static std::unique_ptr<Profiler> Create(ProfileOptions options);

  absl::StatusOr<std::unique_ptr<InputBuffers>> CreateInputBuffers(
      const Executable* executable) override;

  absl::StatusOr<ProfileResult> Profile(Executable* executable,
                                        const InputBuffers& buffers) override;

  absl::Status CheckInputBuffers(InputBuffers& buffers) override {
    return absl::UnimplementedError("InputBuffers check for CPU.");
  }

  absl::Status CheckOutputBuffer(ScopedShapedBuffer& output,
                                 ScopedShapedBuffer& reference,
                                 float rtol) override {
    return absl::UnimplementedError("Output buffer check for CPU.");
  }

 protected:
  explicit CpuProfiler(ProfileOptions options) : options_(options) {}

  absl::Status Execute(Executable* executable,
                       absl::Span<const MaybeOwningDeviceAddress> buffers,
                       ExecutionProfile* profile);

 private:
  ProfileOptions options_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_AUTOTUNER_CPU_PROFILER_H_
