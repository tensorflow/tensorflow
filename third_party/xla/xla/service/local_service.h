/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_LOCAL_SERVICE_H_
#define XLA_SERVICE_LOCAL_SERVICE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/service/backend.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/service.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Service implementation that extends the XLA Service to leverage running
// in the same process as the client.
class LocalService : public Service {
 public:
  // Factory for creating a LocalService.
  static absl::StatusOr<std::unique_ptr<LocalService>> NewService(
      const ServiceOptions& options);

  // Builds Executables with the given XlaComputation, argument layouts and
  // options. If result_layout is non-null, then the executable is compiled to
  // produce a result of the given layout.  If device_allocator is non-null,
  // then the compiler may use it to allocate temp space on the device.  The
  // compiler is responsible for freeing any memory it allocates this way.
  absl::StatusOr<std::vector<std::unique_ptr<Executable>>> CompileExecutables(
      const XlaComputation& computation,
      const absl::Span<const Shape* const> argument_layouts,
      const ExecutableBuildOptions& build_options);

  // Same as CompileExecutables() above, but return AotCompilationResult objects
  // (instead of Executable objects), which can be persisted to later load
  // Executable objects.
  absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAotResults(const XlaComputation& computation,
                    const absl::Span<const Shape* const> argument_layouts,
                    const ExecutableBuildOptions& build_options);

  // Returns the device ordinal that corresponds to the given replica number.
  //
  // This returns an error if there is not a one-to-one correspondence of
  // replicas to device ordinals, but is useful as a short term mechanism for
  // the "easy" case where a single replica is a single device.
  absl::StatusOr<int> ReplicaNumberToDeviceOrdinal(int replica_number);

  // Converts a GlobalDataHandle into a pointer to a ShapedBuffer that's valid
  // as long as the handle is valid.
  absl::StatusOr<const ShapedBuffer*> GlobalDataToShapedBuffer(
      const GlobalDataHandle& data, int replica_number);

  // Registers a vector of shaped buffers of device memory, one per replica, and
  // returns a corresponding handle that can be used for talking to XLA clients.
  absl::StatusOr<GlobalDataHandle> RegisterReplicatedBuffers(
      std::vector<ScopedShapedBuffer> replicated_buffers,
      const std::string& tag);

 private:
  explicit LocalService(const ServiceOptions& options,
                        std::unique_ptr<Backend> backend);
  LocalService(const LocalService&) = delete;
  void operator=(const LocalService&) = delete;
};

}  // namespace xla

#endif  // XLA_SERVICE_LOCAL_SERVICE_H_
