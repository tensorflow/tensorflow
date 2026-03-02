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

#ifndef XLA_SERVICE_COMPILED_MODULE_H_
#define XLA_SERVICE_COMPILED_MODULE_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/platform.h"

namespace stream_executor {
class DeviceDescription;

}  // namespace stream_executor

namespace xla {

class Executable;

// Abstract superclass describing the result of an ahead-of-time compilation.
class CompiledModule {
 public:
  virtual ~CompiledModule() = default;

  virtual absl::StatusOr<std::string> SerializeAsString() const = 0;

  virtual absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable() && = 0;

  virtual absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      stream_executor::Platform::Id platform_id,
      const stream_executor::DeviceDescription& device_description) && = 0;

  virtual absl::StatusOr<std::unique_ptr<BufferAssignment>> buffer_assignment()
      const {
    return absl::UnimplementedError("buffer_assignment is not supported.");
  }

  // Returns the optimized HLO module if one was computed and the implementation
  // supports it.
  virtual const HloModule* optimized_module() const = 0;
  virtual std::shared_ptr<HloModule> shared_optimized_module() = 0;
};

}  // namespace xla

#endif  // XLA_SERVICE_COMPILED_MODULE_H_
