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

#ifndef XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_EXECUTABLE_H_
#define XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_EXECUTABLE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/core/host_offloading/host_offloading_allocator.h"
#include "xla/core/host_offloading/host_offloading_buffer.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"

namespace xla {

// Host offloading executable is a base class for executables that can be
// invoked by a TPU program on a host attached to a device. It can wrap
// arbitrary function that reads arguments from host memory and writes results
// back to host memory, but in practice it's implemented as an HLO module
// compiled to executable via the XLA:CPU.
//
// Key properties of host offloading executables:
//
//   (1) All parameters passed as available heap buffers (no async values), and
//       it's up to the caller to ensure that parameters stay alive while
//       the offloading executable might be running.
//
//   (2) It is a destination-passing style executable that does not allocate
//       memory for results, but instead writes to the buffers provided by the
//       caller. It can allocate temporary scratch space if needed.
//
//   (3) Parameter and result buffers can alias. However host offloading
//       executable doesn't need to know anything about buffers that own
//       the memory that was passed to them, it does not matter if host
//       memory is backed by a TpuHostBuffer or an std::vector<std::byte>.
//
// This class gives TPU compiler and runtime a flexibility to choose the
// strategy for compiling and executing offloaded executables.
class HostOffloadingExecutable {
 public:
  using ExecuteEvent = tsl::Chain;

  virtual ~HostOffloadingExecutable() = default;

  struct ExecuteOptions {
    int32_t device_index;
    int32_t launch_id;
    const ExecuteContext* context;
  };

  // Executes host offloading executable with given `parameters` and writes
  // to `result` buffer(s).
  // Returns an async value that signals when the execution is done. The value
  // should be waited on (e.x. with tsl::BlockUntilReady) before reading the
  // buffer(s) in `result`.
  virtual tsl::AsyncValueRef<ExecuteEvent> Execute(
      absl::Span<const ShapeTree<HostOffloadingBuffer>> parameters,
      const ShapeTree<HostOffloadingBuffer>& result,
      const ExecuteOptions& execute_options) = 0;

  // Host offloading executable name (for debugging and tracing).
  virtual absl::string_view name() const = 0;

  // Program shape of the executable (entry computation function signature)
  virtual const ProgramShape& program_shape() const = 0;

  virtual bool needs_layout_conversion() const { return false; }
};
}  // namespace xla

#endif  // XLA_CORE_HOST_OFFLOADING_HOST_OFFLOADING_EXECUTABLE_H_
