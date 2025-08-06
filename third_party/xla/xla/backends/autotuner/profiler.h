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

#ifndef XLA_BACKENDS_AUTOTUNER_PROFILER_H_
#define XLA_BACKENDS_AUTOTUNER_PROFILER_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/service/executable.h"
#include "xla/service/shaped_buffer.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

struct ProfileOptions {
  // Padding around the buffers to check for out-of-bounds reads/writes.
  int redzone_padding_bytes = 0;
  // Whether to initialize the buffers with random data or leave them
  // uninitialized.
  bool should_init_buffers = false;
  // Whether to populate the output_buffer in the ProfileResult with the result
  // of the execution. This is to avoid data copies if the caller doesn't need
  // the output buffer.
  bool should_populate_output_buffer = true;
};

struct ProfileResult {
  absl::Duration duration = absl::ZeroDuration();
  std::optional<ScopedShapedBuffer> output_buffer = std::nullopt;
};

struct InputBuffers {
  virtual ~InputBuffers() = default;
};

// Interface to run and profile XLA compiled executables for autotuning,
// also manages buffers for the executables.
class Profiler {
 public:
  virtual ~Profiler() = default;

  // Profiles a single executable.
  virtual absl::StatusOr<ProfileResult> Profile(
      std::unique_ptr<Executable> executable) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<InputBuffers> buffers,
                        CreateInputBuffers(executable.get()));
    return Profile(executable.get(), *buffers);
  }

  // Profiles multiple executables with shared buffers. This guarantees that
  // the provided executables have same arguments. This is important for
  // autotuning as we run same instruction with different configs.
  // Note that an executable can still fail during runtime even if it compiled
  // successfully, which is why the return type is a vector of StatusOr.
  virtual absl::StatusOr<std::vector<absl::StatusOr<ProfileResult>>>
  ProfileWithSharedBuffers(
      std::vector<std::unique_ptr<Executable>> executables) {
    std::vector<absl::StatusOr<ProfileResult>> results;
    if (executables.empty()) {
      return results;
    }
    TF_ASSIGN_OR_RETURN(std::unique_ptr<InputBuffers> buffers,
                        CreateInputBuffers(executables[0].get()));
    for (auto& executable : executables) {
      results.push_back(Profile(executable.get(), *buffers));
    }
    return results;
  }

  // Creates Input buffers for a given executable on the device. The buffers
  // are created with the same shape as the input parameters of the executable.
  virtual absl::StatusOr<std::unique_ptr<InputBuffers>> CreateInputBuffers(
      const Executable* executable) = 0;

  // Profiles a single executable with the provided buffers. The buffers
  // must be created by calling CreateInputBuffers from the same profiler.
  virtual absl::StatusOr<ProfileResult> Profile(
      Executable* executable, const InputBuffers& buffers) = 0;

  // Check for red-zone errors i.e out-of-bounds reads/writes to InputBuffers.
  // This is a no-op if redzone padding bytes are not set in the options.
  // The buffers are refreshed if needed, so that the buffers can be reused
  // for subsequent profiling.
  virtual absl::Status CheckInputBuffers(InputBuffers& buffers) = 0;

  // Compare the on device output buffer against the reference output buffer.
  // Returns ok if the buffers have the same content, otherwise returns an
  // error.
  virtual absl::Status CheckOutputBuffer(ScopedShapedBuffer& output,
                                         ScopedShapedBuffer& reference,
                                         float rtol) = 0;
};
}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_PROFILER_H_
