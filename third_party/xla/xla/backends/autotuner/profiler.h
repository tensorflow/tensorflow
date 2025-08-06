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
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/service/executable.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

struct ProfileOptions {
  // Padding around the buffers to check for out-of-bounds reads/writes.
  int redzone_padding_bytes;
  // Whether to initialize the buffers with random data or leave them
  // uninitialized.
  bool should_init_buffers;
};

struct ProfileResult {
  absl::Duration duration;
};

// Interface to run and profile XLA compiled executables for autotuning.
class Profiler {
 public:
  virtual ~Profiler() = default;

  // Profiles a single executable.
  virtual absl::StatusOr<ProfileResult> Profile(
      std::unique_ptr<Executable> executable) {
    std::vector<std::unique_ptr<Executable>> executables;
    executables.push_back(std::move(executable));
    TF_ASSIGN_OR_RETURN(std::vector<ProfileResult> results,
                        ProfileWithSharedBuffers(std::move(executables)));
    return results[0];
  }

  // Profiles multiple executables with shared buffers. This guarantees that
  // the provided executables have same arguments. This is important for
  // autotuning as we run same instruction with different configs.
  virtual absl::StatusOr<std::vector<ProfileResult>> ProfileWithSharedBuffers(
      std::vector<std::unique_ptr<Executable>> executables) = 0;
};
}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_PROFILER_H_
